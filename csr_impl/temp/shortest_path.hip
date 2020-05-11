// #include "scan2.cu"
#include "shortest_path.h"
#include <limits>
#include <stdio.h>
#include <iostream>
using namespace std;
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "hip/hip_runtime.h"

#define BLOCK_SZ 512
#define max_float 0x7f800000
// #define WARP_SIZE 64
// #ifdef __CUDACC__
// #define WARP_SIZE 32
// #define max_float __int_as_float(0x7f800000)
// #endif
// #ifdef __HIP__
#define WARP_SIZE 64
// #endif

// void shortestPath_Dijkstra(int num_node, int *vex, float *arc, int v0,
//                            int *path_node, float *shortLenTable) {}
template <typename T>
void printD(T *data, int size)
{
  T *temp = new T[size];
  // cudaMalloc(&temp, size * sizeof(T));
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
// __device__ __forceinline__ int warpReduceSum(int val)
// {
//   for (int offset = 32 >> 1; offset > 0; offset >>= 1)
//     val += __shfl_down_sync(0xFFFFFFFF, val, offset);
//   return val;
// }
__global__ void get_nnz(int num_node, int *vex, float *arc, int *row_nnz)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_node)
  {
    int nnz = 0;
    for (int i = 0; i < num_node; i++)
    {
      if (arc[tid * num_node + i] != 0)
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
  // __shared__ int temp[];
  if (tid < num_node)
  {
    int nnz = 0;
    for (size_t i = 0; i < num_node; i++)
    {
      if (arc[tid * num_node + i] != 0)
      {
        csr_col[csr_ptr[tid] + nnz] = i;
        csr_val[csr_ptr[tid] + nnz] = arc[tid * num_node + i];
        ++nnz;
      }
    }
  }
}
// template <typename T>
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
struct Csr
{
  int nnz;
  int *csr_ptr, *csr_col;
  float *csr_val;
};
Csr transfer_csr(int num_node, int *vex, float *arc)
{
  int *row_nnz;
  int *csr_ptr, *csr_col;
  float *csr_val;

  hipMalloc(&csr_ptr, (num_node + 1) * sizeof(int));
  hipMalloc(&row_nnz, num_node * sizeof(int));
  hipLaunchKernelGGL(get_nnz, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, vex, arc,
                                                 row_nnz);
  // full_block_scan_d(row_nnz, csr_ptr, num_node);
  thrust::inclusive_scan(thrust::device, row_nnz, row_nnz + num_node, csr_ptr + 1);
  // cout<<"csr.csr_ptr \n";
  // printD(csr_ptr,11);
  int nnz = 0;
  hipMemcpy(&nnz, csr_ptr + num_node, sizeof(int), hipMemcpyDeviceToHost);
  // cout<<"\nnnz"<<nnz<<" \n";
  hipMalloc(&csr_col, nnz * sizeof(int));
  hipMalloc(&csr_val, nnz * sizeof(float));
  hipLaunchKernelGGL(dense_to_csr, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, vex, arc, row_nnz, csr_ptr, csr_col, csr_val);
  Csr csr;
  csr.nnz = nnz;
  csr.csr_ptr = csr_ptr;
  csr.csr_col = csr_col;
  csr.csr_val = csr_val;
  return csr;
}
__global__ void _init2f(int num_node, int *data)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_node)
  {
    data[tid * num_node] = tid;
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

struct Frontier
{
  int num_node;
  int *data, *row_nnz;
  // thrust::device_ptr<int> data;
  // thrust::device_ptr<int> row_nnz;
  char finished, *finished_d;

  __host__ void init(int n)
  {
    num_node = n;
    hipMalloc(&data, num_node * num_node * sizeof(int));
    hipMalloc(&row_nnz, num_node * sizeof(int));
    hipMalloc(&finished_d, sizeof(int));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset), dim3(num_node * num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, data, num_node * num_node, 0);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset), dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, row_nnz, num_node, 0);
  }
  __host__ void init2f()
  {
    hipLaunchKernelGGL(_init2f, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, num_node, data);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_mset<int>), dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, row_nnz, num_node, 1);
  }
  __host__ bool check()
  {
    hipMemset(finished_d, 1, sizeof(char));
    // cout << "row_nnz \n";
    // printD(row_nnz, num_node);
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
    // cudaMemset(data, 0, num_node * num_node * sizeof(int));
    // cudaMemset(row_nnz, 0, num_node * sizeof(int));
    // thrust::fill(thrust::device, data, data +num_node*num_node, 0);
    // thrust::fill(thrust::device, row_nnz, row_nnz + num_node, 0);
  }
  __device__ void append(int i, int j)
  {
    data[i * num_node + row_nnz[i]] = j;
    row_nnz[i]++;
  }
};
__device__ void updatePath() {}
__global__ void compute(Csr csr, int *path_node, float *shortLenTable, float *val,
                        int num_node, Frontier frontier, Frontier frontier2)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  int warpid = tid / WARP_SIZE;
  // int laneid = tid % WARP_SIZE;
  if (warpid < num_node)
  {
    // for (size_t laneid = tid % WARP_SIZE; i < frontier.row_nnz[warpid];
    // i+=WARP_SIZE)
    for (size_t i = 0; i < frontier.row_nnz[warpid]; i++)
    {
      int dst = frontier.data[warpid * num_node + i];
      // for (size_t j = 0; j < csr.csr_ptr[dst]; j++)
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
          frontier2.append(warpid, dst2);
          updatePath();
        }
      }
    }
  }
}
__global__ void compute_first(Csr csr, int *path_node, float *shortLenTable, float *val,
                              int num_node, Frontier frontier, Frontier frontier2)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  int warpid = tid / WARP_SIZE;
  // int laneid = tid % WARP_SIZE;
  if (warpid < num_node)
  {
    // for (size_t laneid = tid % WARP_SIZE; i < frontier.row_nnz[warpid];
    // i+=WARP_SIZE)
    for (size_t i = 0; i < frontier.row_nnz[warpid]; i++)
    {
      int dst = frontier.data[warpid * num_node + i];
      // for (size_t j = 0; j < csr.csr_ptr[dst]; j++)
      for (size_t j = csr.csr_ptr[dst] + tid % WARP_SIZE;
           j < csr.csr_ptr[dst + 1]; j += WARP_SIZE)
      {
        int dst2 = csr.csr_col[j];
        if (shortLenTable[warpid * num_node + dst2] >
            val[warpid * num_node + dst] +
                val[dst * num_node + dst2])
        {
          fatomicMin(&shortLenTable[warpid * num_node + dst2],
                     val[warpid * num_node + dst] +
                         val[dst * num_node + dst2]);
          frontier2.append(warpid, dst2);
          updatePath();
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
  // cudaMalloc(&vex_d, num_node * num_node * sizeof(int));
  hipMalloc(&arc_d, num_node * num_node * sizeof(float));
  // cudaMemcpy(vex_d, vex, num_node * num_node * sizeof(int),
  //            cudaMemcpyHostToDevice);
  hipMemcpy(arc_d, arc, num_node * num_node * sizeof(float),
             hipMemcpyHostToDevice);
  printD(arc_d, num_node * num_node);
  Csr csr = transfer_csr(num_node, vex_d, arc_d);
  float *shortLenTable_d;
  hipMalloc(&shortLenTable_d, num_node * num_node * sizeof(float));
  // cudaMemcpy(shortLenTable_d, arc, num_node * num_node * sizeof(float), cudaMemcpyHostToDevice);
  hipLaunchKernelGGL(dmemset, dim3(num_node * num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, arc_d,
                                                            num_node);
  hipLaunchKernelGGL(dmemset2, dim3(num_node * num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, shortLenTable_d,
                                                             num_node);
  // thrust::fill(thrust::device, shortLenTable_d, shortLenTable_d+ num_node*num_node, 1.1);
  cout << "shortLenTable \n";
  printD(shortLenTable_d, num_node * num_node);
  Frontier frontier, frontier2;
  frontier.init(num_node);
  frontier.init2f();
  frontier2.init(num_node);
  cout << "frontier \n";
  printD(frontier.data, num_node * num_node);
  cout << "frontier nnz \n";
  printD(frontier.row_nnz, num_node);
  bool finished = false;
  int itr = 0;
  do
  {
    if (itr % 2 == 0)
    {
      if (itr == 0)
        hipLaunchKernelGGL(compute_first, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, csr, path_node, shortLenTable_d, arc_d, num_node, frontier, frontier2);
      else
        hipLaunchKernelGGL(compute, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, csr, path_node, shortLenTable_d, arc_d, num_node, frontier, frontier2);
      frontier.reset();
      finished = frontier2.check();
      itr++;
      // cout << "frontier2 nnz \n";
      // printD(frontier2.row_nnz, num_node);
      // cout << "frontier2 \n";
      // printD(frontier2.data, num_node * num_node);
    }
    else
    {
      hipLaunchKernelGGL(compute, dim3(num_node / BLOCK_SZ + 1), dim3(BLOCK_SZ), 0, 0, csr, path_node, shortLenTable_d, arc_d, num_node, frontier2, frontier);
      frontier2.reset();
      finished = frontier.check();
      itr++;
      // cout << "frontier nnz \n";
      // printD(frontier.row_nnz, num_node);
      // cout << "frontier \n";
      // printD(frontier.data, num_node * num_node);
    }
    cout << "shortLenTable for itr" << itr << " \n";
    printD(shortLenTable_d, num_node * num_node);
  } while ((!finished) && (itr < 5));
  printf("itr: %d\n", itr);
  // cout << "shortLenTable \n";
  // printD(shortLenTable_d, num_node * num_node);
  hipMalloc(&shortLenTable, num_node * num_node * sizeof(float));
}