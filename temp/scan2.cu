// #include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <time.h>

using namespace std;

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		int aa = 0;
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		scanf("%d", &aa);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void Kernel(int *input, int *output, int* results, int space,bool direction,int step,int steps)
{
	//2D to 1D INDEXING
	int tix = threadIdx.x + blockDim.x*blockIdx.x;
	int tiy = threadIdx.y + blockDim.y*blockIdx.y;
	int tid = tix + gridDim.x*blockDim.x*tiy; 

	// space =      GAP BETWEEN NEIGHBORS
	// direction =  FROM INPUT TO OUTPUT OR FROM OUTPUT TO INPUT
	// step =       CURRENT PROCESSING STEP
	// steps =      TOTAL AMOUNT OF STEPS NEEDED TO CALCULATE SCAN
	int res = 0;
	if (direction){
		//OUTPUT -> INPUT
		if (tid < space){
			res = output[tid]; //ONLY REWRITE TO CORRECT MEMORY ADDRESS
			input[tid] = res;
		}
		else{
			res = output[tid] + output[tid - space];
			input[tid] = res;
		}
	}else {
		//INPUT -> OUTPUT
		if (tid<space){
			res = input[tid]; //ONLY REWRITE TO CORRECT MEMORY ADDRESS
			output[tid] = res;
		}
		else{
			res = input[tid] + input[tid - space];
			output[tid] = res;
		}
	}

	//THE FINAL STEP: WRITE RESULTS INTO CORRECT LOCATION ON GPU
	if (step == (steps - 1)){
		results[tid] = res;
	}
}

void scan(int* dev_input,int* dev_results,int N)
{
	//INIT
	// const int N = 131072; // 131072; //4194304;   //65536;  //4194304;
	
	int* dev_output;
	
    HANDLE_ERROR(cudaMalloc((void**)&dev_output, N*sizeof(int)));

	//DEFINE DIMENSION
	dim3 THREADS_PER_BLOCK(1024, 1, 1);
	dim3 BLOCKS_PER_GRID(1, 1, 1);
	if (N > 65536){
		BLOCKS_PER_GRID = dim3(64, N / 65536, 1);
	}else{
		BLOCKS_PER_GRID = dim3(N / 1024, 1, 1);
	}

	//LAUNCH KERNELS IN LOOP
	int space = 1;
	int steps = static_cast<int>(log2(static_cast<float>(N)));
	for (size_t step = 0; step<steps; step++){
		bool direction = (step % 2) !=0;  
		Kernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(dev_input, dev_output, dev_results, space, direction, step, steps);
		space = space * 2;

	}
	cudaFree(dev_output);
}
