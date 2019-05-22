#include <stdio.h>
#include <iostream>




__global__
void mandel_iter(int maxiter, int *d_res, int width, int height)
{
    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int X = offset % width;
    int Y = (offset-X)/width;

    int idx = width * Y + X;

    float x0 = ((float)X / width) * (3.5) - 2.5;
    float y0 = ((float)Y /height) * (2) - 1;

    float x = 0.0f;
    float y = 0.0f;

    int iter = 0;
    float xtemp;
    while((x *x + y *y <= 4.0f) && (iter < maxiter)){
      xtemp = x * x - y * y + x0;
      y = 2.0f * x * y + y0;
      x = xtemp;
      iter++;
    }

    d_res[idx] = iter;
}


int main(void)
{
    int HEIGHT = 1200;
    int WIDTH = 800;

    int N = WIDTH * HEIGHT;
    int *d_res;
    int *iter_res;
    iter_res = (int*)malloc(N*sizeof(int));

    cudaMalloc(&d_res, N*sizeof(int));

   dim3 blockDims(512,1,1);
   dim3 gridDims((unsigned int) ceil((double)(WIDTH*HEIGHT/blockDims.x)), 1, 1 );


    mandel_iter<<<gridDims, blockDims>>>(N, d_res, WIDTH, HEIGHT);

    cudaMemcpy(iter_res, d_res, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++){
      if (iter_res[i] != 0)
        std::cout << iter_res[i] << std::endl;
    }

    cudaFree(d_res);
    free(iter_res);

    return 0;
}
