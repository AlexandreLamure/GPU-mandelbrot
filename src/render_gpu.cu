#include <stdio.h>
#include <iostream>

#include "render_gpu.hpp"

#include "heat_lut.hpp"


constexpr static auto NB_THREADS = 128;


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
    }

    iter_matrix[idx] = iter;
}

__global__
void buffer_fill(rgba8_t *hue, int *iter_matrix, int width, int height,
                 rgba8_t* buffer, rgba8_t* buffer_down)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    auto offset = width * Y;
    auto pixel_top = buffer + offset;
    auto pixel_down = buffer_down - offset;

    pixel_top[X] = hue[iter_matrix[Y * width + X]];
    pixel_down[X] = hue[iter_matrix[Y * width + X]];
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

    int N = width * height;
    int N2 = width * (height / 2 + 2);
    int *iter_matrix = new int[N2];
    int *iter_matrix_cu;
    cudaMalloc(&iter_matrix_cu, N2*sizeof(int));

    float total = 0.f;
    dim3 nb_blocks(width/NB_THREADS + (width % NB_THREADS != 0), height/2+1,1);
    dim3 threads_per_block(NB_THREADS, 1, 1);

    mandel_iter<<< nb_blocks, threads_per_block>>>(iter_matrix_cu,
                                                   width, height,
                                                   n_iterations);

    cudaMemcpy(iter_matrix, iter_matrix_cu, N2*sizeof(int), cudaMemcpyDeviceToHost);

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
    hue[n_iterations - 1] = rgba8_t{0, 0, 0, 255};


    rgba8_t *hue_cu;
    cudaMalloc(&hue_cu, (n_iterations + 1) * sizeof(rgba8_t));
    cudaMemcpy(hue_cu, hue, (n_iterations + 1) * sizeof(rgba8_t), cudaMemcpyHostToDevice);

    rgba8_t *buffer_cu;
    cudaMalloc(&buffer_cu, N*sizeof(rgba8_t));
    rgba8_t *buffer_down_cu = buffer_cu + width * (height - 1);

    nb_blocks = dim3(width/NB_THREADS + (width % NB_THREADS != 0), height/2,1);
    threads_per_block = dim3(NB_THREADS, 1, 1);

    buffer_fill<<< nb_blocks, threads_per_block>>>(hue_cu, iter_matrix_cu,
                                                   width, height,
                                                   buffer_cu, buffer_down_cu);

    cudaMemcpy(buffer, buffer_cu, N*sizeof(rgba8_t), cudaMemcpyDeviceToHost);

    delete[] iter_matrix;
    cudaFree(iter_matrix_cu);
    cudaFree(buffer_cu);
}
