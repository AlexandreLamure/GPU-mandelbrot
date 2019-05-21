#pragma once
#include <cstddef>
#include <memory>


/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
void render_cpu(std::byte* buffer,
                int width,
                int height,
                std::ptrdiff_t stride,
                int n_iterations = 100);

class GPURenderer
{
public:
  GPURenderer();
  ~GPURenderer();


  /// \param buffer The RGBA24 image buffer
  /// \param width Image width
  /// \param height Image height
  /// \param stride Number of bytes between two lines
  /// \param n_iterations Number of iterations maximal to decide if a point
  ///                     belongs to the mandelbrot set.
  void execute(std::byte* buffer,
               int width,
               int height,
               std::ptrdiff_t stride,
               int n_iterations = 100);

private:
  struct data_t;
  std::unique_ptr<data_t> m_data;
};



