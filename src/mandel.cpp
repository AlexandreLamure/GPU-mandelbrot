#include <cstddef>
#include <memory>

#include <png.h>

#include <CLI/CLI.hpp>
#include "render_cpu.hpp"
#include "render_gpu.hpp"

void write_png(const uint8_t* buffer,
               int width,
               int height,
               int stride,
               const char* filename)
{
  png_structp png_ptr =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }
  FILE* fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(buffer));
    buffer += stride;
  }
  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, nullptr);
  fclose(fp);
}


// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  std::string filename = "output.png";
  std::string mode = "GPU";
  int width = 1200;
  int height = 800;

  CLI::App app{"mandel"};
  app.add_option("-o", filename, "Output image");
  app.add_option("width", width, "width of the output image");
  app.add_option("height", height, "height of the output image");
  app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

      CLI11_PARSE(app, argc, argv);

  // Create buffer
  constexpr int kRGBASize = 4;
  int stride = width * kRGBASize;
  auto buffer = std::make_unique<uint8_t[]>(height * stride);

  // Rendering

  if (mode == "CPU")
  {
    render_cpu(buffer.get(), width, height, stride);
  }
  else if (mode == "GPU")
  {
    GPURenderer r;
    r.render_gpu(buffer.get(), width, height, stride);
  }

  // Save
  write_png(buffer.get(), width, height, stride, filename.c_str());
  return 0;
}

