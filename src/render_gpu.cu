#include <stdio.h>
#include <iostream>

#include "render_gpu.hpp"

#include "heat_lut.hpp"


__global__
void mandel_iter(int *iter_matrix, int width, int height, int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width or Y >= height)
      return;

    int idx = width * Y + X;

    float x0 = ((3.5 / ((float)width - 1.0)) * (float)X) - 2.5; //((float)X / width) * (3.5) - 2.5;
    float y0 = ((2.0 / ((float)height - 1.0)) * (float)Y) - 1.0; //((float)Y /height) * (2) - 1;

    float x = 0.0f;
    float y = 0.0f;

    int iter = 0;
    float tmp1 = (x0 - 0.25f) * (x0 - 0.25f);
    float tmp2 = y0 * y0;
    float q = tmp1 + tmp2;
    if (q * (q + (x0 - 0.25f)) < (0.25f * tmp2))
        iter = n_iterations;
    else
    {
        float xtemp;
        while(x *x + y *y < 4 && iter < n_iterations)
        {
          xtemp = x * x - y * y + x0;
          y = 2.0f * x * y + y0;
          x = xtemp;
          iter++;
        }
        /*
        if (iter != n_iterations)
        {
            histogram[iter] += 2;
            total += 2;
        }
        */
    }

    iter_matrix[idx] = iter;
}


void GPURenderer::render_gpu(uint8_t* buffer,
                             int width,
                             int height,
                             std::ptrdiff_t stride,
                             int n_iterations)
{
    int *histogram = new int[n_iterations];
    for (int i = 0; i < n_iterations; ++i)
        histogram[i] = 0;
    //int *histogram_cu;

    int N = width * height;
    int *iter_matrix = new int[N];
    int *iter_matrix_cu;
    cudaMalloc(&iter_matrix_cu, N*sizeof(int));

    float total = 0.f;
    dim3 nb_blocks(width/128 + (width % 128 != 0), height/2,1);
    dim3 threads_per_block(128, 1, 1);

    mandel_iter<<< nb_blocks, threads_per_block>>>(iter_matrix_cu,
                                                  //histogram_cu,
                                                  width, height,
                                                  n_iterations);

    cudaMemcpy(iter_matrix, iter_matrix_cu, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(iter_matrix_cu);

    for (int Py = 0; Py < height / 2; ++Py)
    {
        for (int Px = 0; Px < width; ++Px)
        {
            int iter = iter_matrix[Py * width + Px];
            if (iter != n_iterations)
            {
                histogram[iter] += 2;
                total += 2;
            }
        }
    }

    //for (int i = 0; i < n_iterations; ++i)
    //    std::cout << histogram[i] << std::endl;

    rgba8_t *hue = new rgba8_t[n_iterations + 1];
    for (int i = 0; i < n_iterations + 1; ++i)
        hue[i] = rgba8_t{0, 0, 0, 255};
    float tmp = (float)histogram[0] / total;
    hue[0] = heat_lut(tmp);
    for (int i = 1; i < n_iterations - 1; ++i)
    {
        tmp = tmp + ((float)histogram[i] / total);
        hue[i] = heat_lut(tmp);
    }
    hue[n_iterations - 1] = rgba8_t{ 0, 0, 0, 255};
 
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
