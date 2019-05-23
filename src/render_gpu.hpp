//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP


class GPURenderer {
public:
    GPURenderer(){};
    void render_gpu(uint8_t* buffer,
                    int width, int height,
                    std::ptrdiff_t stride,
                    int n_iterations = 100);

};


#endif //GPGPU_RENDER_GPU_HPP
