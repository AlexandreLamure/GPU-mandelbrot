#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

kernel
void demo(global uchar4 * buf)
{
  int height = get_global_size(0);
  int width = get_global_size(1);
  int Y = get_global_id(0);
  int X = get_global_id(1);
  int idx = width * Y + X;

  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  __global uchar4* image = (__global uchar4*) buf;

  float x = (X * X + Y * Y) / (float) (width * width + height * height);

  if (x < x0)
  {
    uint g = (uint) (x / x0 * 255);
    image[idx] = (uchar4) (0, g, 255, 255);
  }
  else if (x < x1)
  {
    uint b = (uint) ((x1 - x) / x0 * 255);
    image[idx] = (uchar4) (0, 255, b, 255);
  }
  else if (x < x2)
  {
    uint r = (uint) ((x - x1) / x0 * 255);
    image[idx] = (uchar4) (r, 255, 0, 255);
  }
  else if (x < 1)
  {
    uint b = (uint) ((1.f - x) / x0 * 255);
    image[idx] = (uchar4) (255, b, 0, 255);
  }
  else // x == 1
  {
    image[idx] = (uchar4) (0, 0, 0, 255);
  }
}
