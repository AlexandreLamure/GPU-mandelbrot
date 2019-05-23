#include <stdio.h>
#include <iostream>

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


__global__
void mandel_iter(int maxiter, int *d_res, int width, int height)
{
    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int X = offset % width;
    int Y = (offset-X)/width;

    int idx = width * Y + X;

    float x0 = ((float)X / width) * (3.5) - 2.5;
    float y0 = ((float)Y /height) * (2) - 1;

    float x = 0.0f;
    float y = 0.0f;

    int iter = 0;
    float xtemp;
    while((x *x + y *y <= 4.0f) && (iter < maxiter)){
      xtemp = x * x - y * y + x0;
      y = 2.0f * x * y + y0;
      x = xtemp;
      iter++;
    }

    d_res[idx] = iter;
}


int main(void)
{
    int HEIGHT = 1200;
    int WIDTH = 800;

    int N = WIDTH * HEIGHT;
    int *d_res;
    int *iter_res;
    iter_res = (int*)malloc(N*sizeof(int));

    cudaMalloc(&d_res, N*sizeof(int));

   dim3 blockDims(512,1,1);
   dim3 gridDims((unsigned int) ceil((double)(WIDTH*HEIGHT/blockDims.x)), 1, 1 );


    mandel_iter<<<gridDims, blockDims>>>(N, d_res, WIDTH, HEIGHT);

    cudaMemcpy(iter_res, d_res, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++){
      if (iter_res[i] != 0)
        std::cout << iter_res[i] << std::endl;
    }

    cudaFree(d_res);
    free(iter_res);

    return 0;
}
