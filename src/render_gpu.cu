#include <stdio.h>
#include <iostream>

#include "render_gpu.hpp"

#include "heat_lut.hpp"


__global__
void mandel_iter(int *iter_matrix, int width, int height, int n_iterations)
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
    while((x *x + y *y <= 4.0f) && (iter < n_iterations)){
      xtemp = x * x - y * y + x0;
      y = 2.0f * x * y + y0;
      x = xtemp;
      iter++;
    }

    iter_matrix[idx] = iter;
}


void GPURenderer::render_gpu(uint8_t* buffer,
                             int width,
                             int height,
                             std::ptrdiff_t stride,
                             int n_iterations)
{
    //int *histogram_cu;
    int *histogram = new int[n_iterations];
    for (int i = 0; i < n_iterations; ++i)
        histogram[i] = 0;

    int N = width * height;
    int *iter_matrix_cu;
    int *iter_matrix = new int[N];

    float total = 0.f;

    cudaMalloc(&iter_matrix_cu, N*sizeof(int));

   dim3 nb_blocks(ceil(float(height)/32),(float(width)/1),1);
   dim3 threads_per_block(32, 1, 1);

    mandel_iter<<<nb_blocks, threads_per_block>>>(iter_matrix_cu,
                                                  //histogram_cu,
                                                  width, height,
                                                  n_iterations);

    cudaMemcpy(iter_matrix_cu, iter_matrix, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        if (iter_matrix[i] != 0)
            std::cout << iter_matrix[i] << std::endl;
    }

    cudaFree(iter_matrix_cu);

    rgba8_t *hue = new rgba8_t[n_iterations + 1];
    for (int i = 0; i < n_iterations + 1; ++i)
        hue[i] = rgba8_t{0, 0, 0, 255};
    float tmp = (float)histogram[0] / total;
    hue[0] = heat_lut(tmp);
    for (int i = 1; i < n_iterations; ++i)
    {
        tmp = tmp + ((float)histogram[i] / total);
        hue[i] = heat_lut(tmp);
    }
 
    auto buffer_down = buffer + stride * (height - 1);
    for (int Py = 0; Py < height / 2; ++Py)
    {
        rgba8_t* lineptr_top = reinterpret_cast<rgba8_t*>(buffer);
        rgba8_t* lineptr_bottom = reinterpret_cast<rgba8_t*>(buffer_down);
        for (int Px = 0; Px < width; ++Px)
        {
            lineptr_top[Px] = hue[iter_matrix[Py * width + Px]];
            lineptr_bottom[Px] = hue[iter_matrix[Py * width + Px]];
        }
        buffer += stride;
        buffer_down -= stride;
    }

    delete[] iter_matrix;
}
