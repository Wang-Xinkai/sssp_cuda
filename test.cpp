#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include "shortest_path.h"
#include "hip/hip_runtime.h"
using namespace std;
class Timer
{
private:
  timeval StartingTime;

public:
  void Start()
  {
    gettimeofday(&StartingTime, NULL);
  }
  float Finish()
  {
    timeval PausingTime, ElapsedTime;
    gettimeofday(&PausingTime, NULL);
    timersub(&PausingTime, &StartingTime, &ElapsedTime);
    float d = ElapsedTime.tv_sec * 1000.0 + ElapsedTime.tv_usec / 1000.0;
    Start();
    return d;
  }
};
template <typename T>
void printH(T *data, int size, int len = 10)
{
  for (int i = 0; i < size; i++)
  {
    cout << data[i] << "\t";
    if ((i + 1) % len == 0)
    {
      cout << "\n";
    }
  }
}
// just to warpup GPU

__global__ void warpup(float *data)
{
  long tid = blockDim.x * blockIdx.x + threadIdx.x;
  tid++;
}

int main()
{
  hipSetDevice(0);
  hipDeviceReset();
  float *tmp;
  hipMalloc(&tmp, sizeof(float));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(warpup), dim3(100), dim3(1024), 0, 0,tmp);
  // hipFree(tmp);
  // cout << "warpup!" << endl;
  srand(3);
  Timer timer;
  int sizes[4]{80, 1024, 5760, 40000};
  for (size_t ii = 0; ii < 4; ii++)
  {
    int size = sizes[ii];
    printf("peraring data %d\n", size);
    cout << endl;
    int *vex = new int[size * size];
    float *arc = new float[size * size];
    int *path_node;
    float *shortLenTable;
    for (size_t j = 0; j < size * size; j++)
    {
        arc[j] =
            static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1));
    }
    for (size_t k = 0; k < size; k++)
    {
      arc[k * size + k] = 0.0; //对角为自身
    }
    cout<<"computing "<<endl;
    timer.Start();
    shortestPath_floyd(size, vex, arc, path_node, shortLenTable);
    printf("shortestPath_floyd for size %d matrix in %f (ms)\n\n", size, timer.Finish());
  }
}