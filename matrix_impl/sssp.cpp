#include <algorithm>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <cmath>
#include "hip/hip_runtime.h"
#include "../shortest_path.h"

using namespace std;

#define INF (1<<22)
#define BLOCK_SIZE 512
#define _DTH hipMemcpyDeviceToHost
#define _HTD hipMemcpyHostToDevice

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

//GPU kernel/functions forward declaration
__global__ void _GPU_Floyd_kernel(int k, float *G,float *P, int N);
void _GPU_Floyd(float *H_G, float *H_Gpath, const int N);

__global__ void _GPU_Floyd_kernel(int k, int *G,int *P, int N){//G will be the adjacency matrix, P will be path matrix
	int col=blockIdx.x*blockDim.x + threadIdx.x;
    if(col>=N)
        return;
	int idx=N*blockIdx.y+col;

	__shared__ float best;
	if(threadIdx.x==0)
		best=G[N*blockIdx.y+k];
	__syncthreads();
    if(best==INF)
        return;
	int tmp_b=G[k*N+col];
    if(tmp_b==INF)
        return;
	float cur=best+tmp_b;
	if(cur<G[idx]){
		G[idx]=cur;
		P[idx]=k;
	}
}
void shortestPath_floyd(int num_nodes, int *vex, float *arc, int *path_nodes, float *shortLenTable){
	float *dG;
	int *dP;
	int numBytesFloat=num_nodes*num_nodes*sizeof(float);
	int numBytesInt = num_nodes * num_nodes * sizeof(int);
	errCheck(hipMalloc((float **)&dG,numBytesFloat));
	errCheck(hipMalloc((int **)&dP,numBytesInt));
	errCheck(hipMemcpy(dG,arc,numBytesFloat,_HTD));
	errCheck(hipMemcpy(dP,path_nodes,numBytesInt,_HTD));

	dim3 dimGrid((num_nodes+BLOCK_SIZE-1)/BLOCK_SIZE,num_nodes);

	for(int k=0;k<num_nodes;k++){//main loop
		hipLaunchKernelGGL(HIP_KERNEL_NAME(_GPU_Floyd_kernel), dim3(dimGrid), dim3(BLOCK_SIZE), 0, 0, k,dG,dP,num_nodes);
		hipDeviceSynchronize();
	}
    cout << "finish computing.\n";
	errCheck(hipMemcpy(shortLenTable,dG,numBytesFloat,_DTH));
	errCheck(hipMemcpy(path_nodes,dP,numBytesInt,_DTH));
    cout << "finishg copy.\n";
	errCheck(hipFree(dG));
	errCheck(hipFree(dP));
}
