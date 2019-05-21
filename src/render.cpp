#include "render.hpp"

#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl.hpp>
#include <spdlog/spdlog.h>
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
  std::vector<cl::Device> devices;
  cl::Program program;
  cl::Context ctx;
};


GPURenderer::GPURenderer()
{
  m_data = std::make_unique<data_t>();
  spdlog::info("OpenCL - Initialization");


  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0)
  {
    spdlog::error("OpenCL - No platform detected");
    std::abort();
  }

  cl::Platform default_platform = platforms[0];
  spdlog::info("Using default platform: {}", default_platform.getInfo<CL_PLATFORM_NAME>());


  cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
  m_data->ctx = cl::Context(CL_DEVICE_TYPE_DEFAULT, properties);
  m_data->devices = m_data->ctx.getInfo<CL_CONTEXT_DEVICES>();

  // Read Sources
  std::string src;
  {
    std::ifstream t(kKernelFilename);
    if (!t.good())
    {
      spdlog::error("Unable to load the Kernel file: {}", kKernelFilename);
      std::abort();
    }
    std::stringstream buffer;
    buffer << t.rdbuf();
    src = buffer.str();
  }


  m_data->program = cl::Program(m_data->ctx, src);
  if (m_data->program.build(m_data->devices) != CL_SUCCESS)
  {
    spdlog::error("OpenCL - Error building kernel: {}",  m_data->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_data->devices[0]));
    std::abort();
  }
  spdlog::info("OpenCL - Kernel Loaded. ALL GOOD!");
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
  // Create a device global memory
  cl::Buffer buf(m_data->ctx, CL_MEM_READ_WRITE, height * width * kRGBASize);
  cl::CommandQueue queue(m_data->ctx, m_data->devices[0]);

  // Iteration computation
  {
    // Get the kernel
    cl::Kernel ker(m_data->program, "demo");

    // Set args
    ker.setArg(0, buf);

    queue.enqueueNDRangeKernel(
      ker,
      cl::NullRange,              // offset
      cl::NDRange(height, width), // Global size
      cl::NullRange);             // Local size
  }

  // Copy back the data
  if (stride == width * kRGBASize)
  {
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, width * height * kRGBASize, buffer);
  }
  else
  {
    // handle this case correctly
    std::abort();
  }

  //std::byte* a = buffer;
  //std::cout << (int)a[0] << ' ' << (int)a[1] << ' ' << (int)a[2] << ' ' << (int)a[3] << "\n";
}
