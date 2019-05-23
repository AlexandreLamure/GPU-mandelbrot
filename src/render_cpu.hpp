#pragma once
#include <cstddef>
#include <memory>


/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
void render_cpu(uint8_t* buffer,
                int width,
                int height,
                std::ptrdiff_t stride,
                int n_iterations = 100);
