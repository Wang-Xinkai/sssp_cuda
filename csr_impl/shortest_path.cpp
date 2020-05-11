#include "shortest_path.h"
#include <limits>
#include <stdio.h>
#include <iostream>
using namespace std;
#include "hip/hip_runtime.h"

#include <rocprim/rocprim.hpp>
#define BLOCK_SZ 512
#define max_float 0x7f800000
#define WARP_SIZE 64

#define errCheck(cmd)                                                                \
  {                                                                                  \
    hipError_t error = cmd;                                                          \
    if (error != hipSuccess)                                                         \
    {                                                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, \
              __FILE__, __LINE__);                                                   \
      exit(EXIT_FAILURE);                                                            \
    }                                                                                \
  }

template <typename T>
void printD(T *data, int size)
{
  T *temp = new T[size];
  hipMemcpy(temp, data, size * sizeof(T), hipMemcpyDeviceToHost);
  for (int i = 0; i < size; i++)
  {
    cout << temp[i] << "\t";
    if ((i + 1) % 4 == 0)
    {
      cout << "\n";
    }
  }
}
__device__ float fatomicMin(float *addr, float value)
{
  float old = *addr, assumed;
  if (old <= value)
    return old;
  do
  {
    assumed = old;
    old = atomicCAS((unsigned int *)addr, __float_as_int(assumed),
                    __float_as_int(fminf(value, assumed)));
  } while (old != assumed);
  return old;
}
__global__ void get_nnz(int num_node, int *vex, float *arc, int *row_nnz)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_node)
  {
    int nnz = 0;
    for (int i = 0; i < num_node; i++)
    {
      // if (arc[tid * num_node + i] != 0)
      if (arc[tid * num_node + i] > 0.0)
      {
        ++nnz;
      }
    }
    row_nnz[tid] = nnz;
  }
}
__global__ void dense_to_csr(int num_node, int *vex, float *arc, int *row_nnz,
                             int *csr_ptr, int *csr_col, float *csr_val)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_node)
  {
    int nnz = 0;
    for (size_t i = 0; i < num_node; i++)
    {
      // if (arc[tid * num_node + i] != 0)
      if (arc[tid * num_node + i] > 0.0)
      {
        csr_col[csr_ptr[tid] + nnz] = i;
        csr_val[csr_ptr[tid] + nnz] = arc[tid * num_node + i];
        ++nnz;
      }
    }
  }
}
__global__ void dmemset(float *ptr, int size)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size * size)
  {
    if (ptr[tid] == -1)
      ptr[tid] = max_float;
  }
  if (tid < size)
  {
    ptr[tid * size + tid] = 0.0;
  }
}
__global__ void dmemset2(float *ptr, int size)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size * size)
  {
    ptr[tid] = max_float;
  }
  if (tid < size)
  {
    ptr[tid * size + tid] = 0.0;
  }
}
__global__ void _check(int num_node, int *row_nnz, char *finished_d)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_node)
  {
    if (row_nnz[tid] != 0)
    {
      *finished_d = false;
    }
  }
}
template <typename T>
__global__ void _mset(T *data, int n, T val)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n)
  {
    data[tid] = val;
  }
}
__global__ void _init2f(int num_node, int *data)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_node)
  {
    data[tid * num_node] = tid;
  }
}
struct Csr
{
  int nnz;
  int *csr_ptr, *csr_col;
  float *csr_val;
  void delate()
  {
    // free(csr_ptr);
    // free(csr_col);
    // free(csr_val);
  }
};
__global__ void _initf(int num_node, int *data, int *nnz, Csr csr)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_node)
  {
    for (size_t i = 0; i < csr.csr_ptr[tid + 1] - csr.csr_ptr[tid]; i++)
    {
      data[tid * num_node + i] = csr.csr_col[csr.csr_ptr[tid] + i];
    }
    nnz[tid] = csr.csr_ptr[tid + 1] - csr.csr_ptr[tid];
  }
}

struct Frontier
{
  int num_node;
  int *data, *row_nnz;
  char finished, *finished_d;
  __host__ void init(int n)
  {
    num_node = n;
    errCheck(hipMalloc(&data, num_node * num_node * sizeof(int)));
    errCheck(hipMalloc(&row_nnz, num_node * sizeof(int)));
    errCheck(hipMalloc(&finished_d, sizeof(int)));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset), dim3(num_node * num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, data, num_node * num_node, 0);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset), dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, row_nnz, num_node, 0);
  }
  // __host__ void init2f()
  // {
  //   hipLaunchKernelGGL(_init2f, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, data);
  //   hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset<int>), dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, row_nnz, num_node, 1);
  // }
  __host__ void init2f(Csr csr)
  {
    // hipLaunchKernelGGL(_init2f, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, data);
    hipLaunchKernelGGL(_initf, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, data, row_nnz, csr);
    // hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset<int>), dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, row_nnz, num_node, 1);
  }
  __host__ bool check()
  {
    hipMemset(finished_d, 1, sizeof(char));
    hipLaunchKernelGGL(_check, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, row_nnz,
                       finished_d);
    hipMemcpy(&finished, finished_d, sizeof(char), hipMemcpyDeviceToHost);
    bool tmp = (finished == 1) ? true : false;
    return tmp;
  }
  __host__ void reset()
  {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset), dim3(num_node * num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, data, num_node * num_node, 0);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset), dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, row_nnz, num_node, 0);
  }
  __device__ void append(int i, int j)
  {
    data[i * num_node + row_nnz[i]] = j;
    row_nnz[i]++;
  }
};

void scan(int *input, int *output, int size)
{
  size_t temporary_storage_size_bytes;
  void *temporary_storage_ptr = nullptr;
  // Get required size of the temporary storage
  rocprim::inclusive_scan(
      temporary_storage_ptr, temporary_storage_size_bytes,
      input, output, size, rocprim::plus<int>());
  // allocate temporary storage
  hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
  // perform scan
  rocprim::inclusive_scan(
      temporary_storage_ptr, temporary_storage_size_bytes,
      input, output, size, rocprim::plus<int>());
  hipFree(temporary_storage_ptr);
}

Csr transfer_csr(int num_node, int *vex, float *arc)
{
  int *row_nnz;
  int *csr_ptr, *csr_col;
  float *csr_val;
  errCheck(hipMalloc(&csr_ptr, (num_node + 1) * sizeof(int)));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset), dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, csr_ptr, num_node, 0);
  errCheck(hipMalloc(&row_nnz, num_node * sizeof(int)));
  hipLaunchKernelGGL(get_nnz, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, vex, arc,
                     row_nnz);
  scan(row_nnz, csr_ptr + 1, num_node);
  int nnz = 0;
  hipMemcpy(&nnz, csr_ptr + num_node, sizeof(int), hipMemcpyDeviceToHost);
  long long mem_usage = (nnz + num_node * num_node) * sizeof(float) + (nnz + num_node * num_node) * sizeof(int);
  if ()
  {
    errCheck(hipMalloc(&csr_col, nnz * sizeof(int)));
    errCheck(hipMalloc(&csr_val, nnz * sizeof(float)));
  }
  else
  {
    errCheck(hipMallocManaged(&csr_col, nnz * sizeof(int)));
    errCheck(hipMallocManaged(&csr_val, nnz * sizeof(float)));
  }
  hipLaunchKernelGGL(dense_to_csr, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, vex, arc, row_nnz, csr_ptr, csr_col, csr_val);
  Csr csr;
  csr.nnz = nnz;
  csr.csr_ptr = csr_ptr;
  csr.csr_col = csr_col;
  csr.csr_val = csr_val;
  return csr;
}

__global__ void compute(Csr csr, int *path_node, float *shortLenTable,
                        int num_node, Frontier frontier, Frontier frontier2)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  int warpid = tid / WARP_SIZE;
  if (warpid < num_node)
  {
    for (size_t i = 0; i < frontier.row_nnz[warpid]; i++)
    {
      int dst = frontier.data[warpid * num_node + i];
      for (size_t j = csr.csr_ptr[dst] + tid % WARP_SIZE;
           j < csr.csr_ptr[dst + 1]; j += WARP_SIZE)
      {
        int dst2 = csr.csr_col[j];
        if (shortLenTable[warpid * num_node + dst2] >
            shortLenTable[warpid * num_node + dst] +
                shortLenTable[dst * num_node + dst2])
        {
          fatomicMin(&shortLenTable[warpid * num_node + dst2],
                     shortLenTable[warpid * num_node + dst] +
                         shortLenTable[dst * num_node + dst2]);
          path_node[warpid * num_node + dst2] = dst;
          frontier2.append(warpid, dst2);
        }
      }
    }
  }
}
void shortestPath_floyd(int num_node, int *vex, float *arc, int *path_node,
                        float *shortLenTable)
{
  int *vex_d;
  float *arc_d;
  errCheck(hipMalloc(&arc_d, num_node * num_node * sizeof(float)));
  errCheck(hipMemcpy(arc_d, arc, num_node * num_node * sizeof(float),
                     hipMemcpyHostToDevice));

  Csr csr = transfer_csr(num_node, vex_d, arc_d);
  hipLaunchKernelGGL(dmemset, dim3(num_node * num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, arc_d,
                     num_node);
  float *shortLenTable_d;
  errCheck(hipMalloc(&shortLenTable_d, num_node * num_node * sizeof(float)));
  errCheck(hipMemcpy(shortLenTable_d, arc_d, num_node * num_node * sizeof(float),
                     hipMemcpyDeviceToDevice));
  errCheck(hipFree(arc_d));
  // hipLaunchKernelGGL(dmemset2, dim3(num_node * num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, shortLenTable_d,
  //                    num_node);

  Frontier frontier, frontier2;
  frontier.init(num_node);
  frontier.init2f(csr);
  frontier2.init(num_node);

  int *path_node_d;
  errCheck(hipMalloc(&path_node_d, num_node * num_node * sizeof(int)));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset), dim3(num_node * num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, path_node_d, num_node * num_node, -1);

  bool finished = false;
  int itr = 0;
  do
  {
    // cout << "shortLenTable for itr" << itr << endl;
    // printD(shortLenTable_d, num_node * num_node);
    if (itr % 2 == 0)
    {
      hipLaunchKernelGGL(compute, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, csr, path_node_d, shortLenTable_d, num_node, frontier, frontier2);
      frontier.reset();
      finished = frontier2.check();
      itr++;
    }
    else
    {
      hipLaunchKernelGGL(compute, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, csr, path_node_d, shortLenTable_d, num_node, frontier2, frontier);
      frontier2.reset();
      finished = frontier.check();
      itr++;
    }
  } while (!finished);
  printf("itr: %d\n", itr);
  // cout << "path_node"<<endl;
  // printD(path_node_d, num_node * num_node);

  // if (path_node == nullptr)
  path_node = (int *)malloc(num_node * num_node * sizeof(int));
  // if (shortLenTable == nullptr)
  shortLenTable = (float *)malloc(num_node * num_node * sizeof(float));
  // cout << "malloced" << endl;
  errCheck(hipMemcpy(shortLenTable, shortLenTable_d, num_node * num_node * sizeof(float),
                     hipMemcpyDeviceToHost));
  errCheck(hipMemcpy(path_node, path_node_d, num_node * num_node * sizeof(int),
                     hipMemcpyDeviceToHost));
  csr.delate();
  errCheck(hipFree(vex_d));
  errCheck(hipFree(shortLenTable_d));
}