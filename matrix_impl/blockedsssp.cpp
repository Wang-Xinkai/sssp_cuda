#include <algorithm>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <cmath>
#include "hip/hip_runtime.h"

using namespace std;

#define INF (1<<22)
#define BLOCK_SIZE 512
#define _DTH hipMemcpyDeviceToHost
#define _HTD hipMemcpyHostToDevice

#define errCheck(cmd)                                                                
  {                                                                                  
    hipError_t error = cmd;                                                          
    if (error != hipSuccess)                                                         
    {                                                                                
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, 
              __FILE__, __LINE__);                                                   
      exit(EXIT_FAILURE);                                                            
    }                                                                                
  }

__global__ void funct1(int k, float* x, int* qx, int n) {

    __shared__ float dBlock[1024];
    __shared__ float QBlock[1024];
    int i = (threadIdx.x >> 5);
    int j = threadIdx.x & 31;

    int index1 = (k * 32 + i) * n + k * 32 + j;
    dBlock[threadIdx.x] = x[index1];
    QBlock[threadIdx.x] = qx[index1];
    int k1 = k * 32;

    for (int l = 0; l < 32; l++) {
        __syncthreads();
        float temp2 = dBlock[(i << 5) + l] + dBlock[(l << 5) + j];
        if (dBlock[threadIdx.x] > temp2) {
            dBlock[threadIdx.x] = temp2;
            QBlock[threadIdx.x] = l + k1;
        }
    }
    x[index1] = dBlock[threadIdx.x];
    qx[index1] = QBlock[threadIdx.x];
}

__global__ void funct2(int k, float* x, int* qx, int n) {
    if (blockIdx.y == 0) {

        int i = (threadIdx.x >> 5);
        int j = threadIdx.x & 31;
        int k1 = k * 32;
        __shared__ float dBlock[1024];
        __shared__ float QcBlock[1024];
        __shared__ float cBlock[1024];
        dBlock[threadIdx.x] = x[(k1 + i) * n + k1 + j];
        int add = 0;

        if (blockIdx.x >= k) { //jumping over central block
            add = 1;
        }

        int index1 = (k1 + i) * n + (blockIdx.x + add)*32 + j;
        cBlock[threadIdx.x] = x[index1];
        QcBlock[threadIdx.x] = qx[index1];

        for (int l = 0; l < 32; l++) {
            __syncthreads();
            float temp2 = dBlock[i * 32 + l] + cBlock[l * 32 + j];
            if (cBlock[threadIdx.x] > temp2) {
                cBlock[threadIdx.x] = temp2;
                QcBlock[threadIdx.x] = l + k1;
            }
        }
        x[index1] = cBlock[threadIdx.x];
        qx[index1] = QcBlock[threadIdx.x];

    } else {

        int i = (threadIdx.x >> 5);
        int j = threadIdx.x & 31;
        int k1 = k * 32;
        __shared__ float dBlock[1024];
        __shared__ float QcBlock[1024];
        __shared__ float cBlock[1024];
        dBlock[threadIdx.x] = x[(k1 + i) * n + k1 + j];
        int add = 0;

        if (blockIdx.x >= k) { //jumping over central block        
            add = 1;
        }

        int index1 = ((blockIdx.x + add)*32 + i) * n + k1 + j;
        cBlock[threadIdx.x] = x[index1];
        QcBlock[threadIdx.x] = qx[index1];

        for (int l = 0; l < 32; l++) {
            __syncthreads();
            float temp2 = cBlock[i * 32 + l] + dBlock[l * 32 + j];

            if (cBlock[threadIdx.x] > temp2) {
                cBlock[threadIdx.x] = temp2;

                QcBlock[threadIdx.x] = l + k1;
            }
        }
        x[index1] = cBlock[threadIdx.x];
        qx[index1] = QcBlock[threadIdx.x];
    }
}

__global__ void funct3(int k, float* x, int* qx, int n) {
    int i = (threadIdx.x >> 5);
    int j = threadIdx.x & 31;
    int k1 = k * 32;
    int addx = 0;
    int addy = 0;

    __shared__ float dyBlock[1024];
    __shared__ float dxBlock[1024];
    __shared__ float QcBlock[1024];
    __shared__ float cBlock[1024];

    if (blockIdx.x >= k) {
        addx = 1;
    }
    if (blockIdx.y >= k) {
        addy = 1;
    }

    dxBlock[threadIdx.x] = x[ ((k << 5) + i) * n + ((blockIdx.y + addy) << 5) + j];
    dyBlock[threadIdx.x] = x[ (((blockIdx.x + addx) << 5) + i) * n + (k << 5) + j];
    int index1 = (((blockIdx.x + addx) << 5) + i) * n + ((blockIdx.y + addy) << 5) + j;
    cBlock[threadIdx.x] = x[index1];
    QcBlock[threadIdx.x] = qx[index1];

    for (int l = 0; l < 32; l++) {
        __syncthreads();
        float temp2 = dyBlock[i * 32 + l] + dxBlock[l * 32 + j];
        if (cBlock[threadIdx.x] > temp2) {
            cBlock[threadIdx.x] = temp2;
            QcBlock[threadIdx.x] = l + k1;
        }
    }
    x[index1] = cBlock[threadIdx.x];
    qx[index1] = QcBlock[threadIdx.x];
}

void shortestPath_floyd(int num_nodes, int *vex, float *arc, int *path_nodes, float *shortLenTable){
    float *dG;
	int *dP;
	int numBytesFloat=num_nodes*num_nodes*sizeof(float);
	int numBytesInt = num_nodes * num_nodes * sizeof(int);
    errCheck(hipMalloc((float **)&dG, numBytesFloat));
    errCheck(hipMalloc((int **)&dP,numBytesInt));
    errCheck(hipMemcpy(dG,arc,numBytesFloat,_HTD));
	errCheck(hipMemcpy(dP,path_nodes,numBytesInt,_HTD));
    dim3 bk2(n / 32 - 1, 2);
    dim3 bk3(n / 32 - 1, n / 32 - 1);
    int gputhreads = 1024;
    for (k = 0; k < n / 32; k++) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(funct1), dim3(1), dim3(gputhreads), 0, 0, k, dG, dP, num_nodes);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(funct2), dim3(bk2), dim3(gputhreads), 0, 0, k, dG, dP, num_nodes);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(funct3), dim3(bk3), dim3(gputhreads), 0, 0, k, dG, dP, num_nodes);
    }
    hipDeviceSynchronize();
    errCheck(hipMemcpy(shortLenTable,dG,numBytesFloat,_DTH));
	errCheck(hipMemcpy(path_nodes,dP,numBytesInt,_DTH));
	errCheck(hipFree(dG));
	errCheck(hipFree(dP));
}