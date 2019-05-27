#include "render_mt.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <omp.h>
#include <mutex>

#include "heat_lut.hpp"

void render_mt(uint8_t* buffer,
               int width,
               int height,
               std::ptrdiff_t stride,
               int n_iterations)
{
    std::mutex m;
    int *histogram = new int[n_iterations];
    for (int i = 0; i < n_iterations; ++i)
    histogram[i] = 0;
    int *iter_matrix = new int[height / 2 * width];

    float *y0 = new float[height];
    for (int i = 0; i < height; ++i)
        y0[i] =  ((2.0 / ((float)height - 1.0)) * (float)i) - 1.0; //y0 = scaled y coordinate of pixel (scaled to lie in the Mandelbrot Y scale (-1, 1))

    float *x0 = new float[width];
    for (int i = 0; i < width; ++i)
        x0[i] = ((3.5 / ((float)width - 1.0)) * (float)i) - 2.5; //x0 = scaled x coordinate of pixel (scaled to lie in the Mandelbrot X scale (-2.5, 1))

    float total = 0.0;
    #pragma omp parallel for //private(x0, y0, histogram, iter_matrix)
    for (int Py = 0; Py < height / 2; ++Py)
    {
        int histogram_copy[n_iterations];
        for (int i = 0; i < n_iterations; ++i)
            histogram_copy[i] = 0;

        for (int Px = 0; Px < width; ++Px)
        {
            float x = 0.0;
            float y = 0.0;
            int iteration = 0;
            float tmp1 = ((float)x0[Px] - 0.25f) * ((float)x0[Px] - 0.25f);
            float tmp2 = (float)y0[Py] * (float)y0[Py];
            float q = tmp1 + tmp2;
            if (q * (q + (x0[Px] - 0.25f)) < (0.25f * tmp2))
                iteration = n_iterations;
            else
            {
                while (x*x + y*y < 4 && iteration < n_iterations)
                {
                    float xtemp = x*x - y*y + x0[Px];
                    y = 2*x*y + y0[Py];
                    x = xtemp;
                    iteration = iteration + 1;
                }
                if (iteration != n_iterations)
                    histogram_copy[iteration] += 2;
            }
            iter_matrix[Py * width + Px] = iteration;
        }
        m.lock();
        for (int i = 0; i < n_iterations; ++i)
        {
            histogram[i] += histogram_copy[i];
            total += histogram_copy[i];
        }
        m.unlock();
    }

    rgba8_t *hue = new rgba8_t[n_iterations];

    for (int i = 0; i < n_iterations; ++i)
        hue[i] = rgba8_t{0, 0, 0, 255};
    float tmp = (float)histogram[0] / total;
    hue[0] = heat_lut(tmp);
    for (int i = 1; i < n_iterations; ++i)
    {
        tmp = tmp + ((float)histogram[i] / total);
        hue[i] = heat_lut(tmp);
    }

    auto buffer_down = buffer + stride * (height - 1);

    #pragma omp parallel for
    for (int Py = 0; Py < height / 2; ++Py)
    {
        auto buffer_top_copy = buffer + stride * Py;
        auto buffer_down_copy = buffer_down - stride * Py;
        rgba8_t* lineptr_top = reinterpret_cast<rgba8_t*>(buffer_top_copy);
        rgba8_t* lineptr_bottom = reinterpret_cast<rgba8_t*>(buffer_down_copy);
        for (int Px = 0; Px < width; ++Px)
        {
            if (iter_matrix[Py * width + Px] != n_iterations)
            {
                lineptr_top[Px] = hue[iter_matrix[Py * width + Px]];
                lineptr_bottom[Px] = hue[iter_matrix[Py * width + Px]];
            }
        }
    }

    delete[] iter_matrix;
    delete[] histogram;
    delete[] y0;
    delete[] x0;
}
