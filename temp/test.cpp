#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include "../shortest_path.h"
#define INF (1<<22)
int main()
{
  srand(static_cast<unsigned>(1));
  // for (size_t i = 10; i < 20; i *= 2)
  {
    size_t i=4;
    printf("testing %ld\n", i);
    int *vex = new int[i * i];
    float *arc;
    int *path_node;
    float *shortLenTable;
    arc = new float[i * i]{0,2,6,4,INF,0,3,INF,7,INF,-0,1,5,INF,12,0};
    // for (size_t j = 0; j < i * i; j++)
    // {
    //   arc[j] =-1;
    //   if (j % 3 != 0)
    //     arc[j] =
    //         static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100));
    // }
    // for (size_t j = 0; j < i; j++)
    // {
    //   arc[j * i + j] = 0.0;
    // }
    shortestPath_floyd(i, vex, arc, path_node, shortLenTable);
  }
}
