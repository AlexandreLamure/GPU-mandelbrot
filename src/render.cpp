#include "render.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

#if defined(__GNUC__)
#define QUOTE(name) #name
#else
#define QUOTE(name) name
#endif

#define STR(macro) QUOTE(macro)
static constexpr const char* kKernelFilename = STR(KERNEL_SOURCE_FILE);


struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

rgba8_t heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgba8_t{0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgba8_t{0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgba8_t{r, 255, 0, 255};
  }
  else
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgba8_t{255, b, 0, 255};
  }
}


void render_cpu(std::byte* buffer,
                int width,
                int height,
                std::ptrdiff_t stride,
                int n_iterations)
{
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
  for (int Py = 0; Py < height; ++Py)
  {
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
        {
          histogram[iteration] += 2;
          total += 2;
        }
      }

      iter_matrix[Py * width + Px] = iteration;
    }
  }

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
  delete[] histogram;
}


struct GPURenderer::data_t
{
  int test;
};


GPURenderer::GPURenderer()
{
    std::cout << "creating object" << std::endl;
}

GPURenderer::~GPURenderer()
{
}

constexpr int kRGBASize = 4;


void
GPURenderer::execute(std::byte* buffer,
                     int width,
                     int height,
                     std::ptrdiff_t stride,
                     int n_iterations)
{
    return;
}
