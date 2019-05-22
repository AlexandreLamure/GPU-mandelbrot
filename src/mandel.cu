#include <stdio.h>

__global__
void mandel_iter(int maxiter, int *d_res)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    d_res[i] = maxiter;
}


int main(void)
{
    int N = 100
    int *d_res;
    int *iter_res;
    iter_res = (int*)malloc(N*sizeof(int));

    cudaMalloc(&d_res, N*sizeof(int));

    mandel_iter<<<(N+255)/256, 256>>>(N, d_res);

    cudaMemcpy(iter_res, d_res, N*sizeof(int), cudaMemcpyDeviceToHost);


    
    cudaFree(d_res);
    free(iter_res);

    return 0;
}