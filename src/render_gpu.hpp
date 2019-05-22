//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

#endif //GPGPU_RENDER_GPU_HPP

#include <cstddef>


class GPURenderer {




public:

    GPURenderer(){};

    void execute(std::byte* buffer, int width, int height, std::ptrdiff_t stride, int n_iterations = 100);

};