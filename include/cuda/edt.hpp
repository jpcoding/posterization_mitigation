#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <float.h>

#include <cassert>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <system_error>
#include <vector>

template <typename Type>
void writefile2(const char* file, Type* data, size_t num_elements)
{
  std::ofstream fout(file, std::ios::binary);
  fout.write(reinterpret_cast<const char*>(&data[0]), num_elements * sizeof(Type));
  fout.close();
}

// width , height  are planes
// depth is the len
// first dimension
__global__ void edt_depth(
    int* d_output, size_t stride, uint rank, uint d, uint len, uint width, uint height,
    size_t width_stride, size_t height_stride)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height
  if (x >= width || y >= height) return;

  // supposed d_output has been initialized

  int* d_output_start = d_output + x * width_stride + y * height_stride;
  int l = -1, ii, maxl, idx1, idx2, jj;
  // __shared__ int f[512][3];
  // __shared__ int g[512];
  int f[512][3];
  int g[512];

  // extern __shared__ int shared_mem[];  // Flexible shared memory
  // int* g = shared_mem + threadIdx.y * blockDim.x * 512 + threadIdx.x * 512;

  // __shared__ int shared_coor[3*16*16];
  // int tid = threadIdx.y * blockDim.x + threadIdx.x;

  // int *coor = shared_coor + tid * 3;
  int coor[3];
  coor[d] = 0;
  coor[(d + 1) % 3] = x;
  coor[(d + 2) % 3] = y;
  // __syncthreads();

  for (ii = 0; ii < len; ii++) {
    for (jj = 0; jj < rank; jj++) {
      f[ii][jj] = d_output_start[ii * stride + jj];
      // d_output_start[ii * stride + jj] = coor[jj];
    }
  }
  // __syncthreads();
  // return;

  for (ii = 0; ii < len; ii++) {
    if (f[ii][0] >= 0) {
      int fd = f[ii][d];
      int wR = 0.0;
      for (jj = 0; jj < rank; jj++) {
        if (jj != d) {
          int tw = (f[ii][jj] - coor[jj]);
          wR += tw * tw;
        }
      }
      while (l >= 1) {
        int a, b, c, uR = 0.0, vR = 0.0, f1;
        idx1 = g[l];
        f1 = f[idx1][d];
        idx2 = g[l - 1];
        a = f1 - f[idx2][d];
        b = fd - f1;
        c = a + b;
        for (jj = 0; jj < rank; jj++) {
          if (jj != d) {
            int cc = coor[jj];
            int tu = f[idx2][jj] - cc;
            int tv = f[idx1][jj] - cc;
            uR += tu * tu;
            vR += tv * tv;
          }
        }
        if (c * vR - b * uR - a * wR - a * b * c <= 0.0) { break; }
        --l;
      }
      ++l;
      g[l] = ii;
    }
  }
  maxl = l;
  if (maxl >= 0) {
    l = 0;
    for (ii = 0; ii < len; ii++) {
      int delta1 = 0.0, t;
      for (jj = 0; jj < rank; jj++) {
        t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];

        delta1 += t * t;
      }
      while (l < maxl) {
        int delta2 = 0.0;
        for (jj = 0; jj < rank; jj++) {
          t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];

          delta2 += t * t;
        }
        if (delta1 <= delta2) break;
        delta1 = delta2;
        ++l;
      }
      idx1 = g[l];
      for (jj = 0; jj < rank; jj++) d_output_start[ii * stride + jj] = f[idx1][jj];
    }
  }
}

__global__ void edt_depth_chunck(
    int* d_output, size_t stride, uint rank, uint d, uint len_total, uint width, uint height,
    size_t width_stride, size_t height_stride)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height
  uint z_chunk = blockIdx.z;                       // which chunk in depth this block processes

  if (x >= width || y >= height) return;
  // uint len = len_total / 4;  // len_total is the total depth, gridDim.z is the number of chunks  
  uint len = len_total / gridDim.z;  // len_total is the total depth, gridDim.z is the number of chunks 
  // each chunk processes len depth slices
  uint z_offset = z_chunk * len;
  int* base_ptr = d_output + x * width_stride + y * height_stride;

  int coor[3];
  coor[d] = 0;
  coor[(d + 1) % 3] = x;
  coor[(d + 2) % 3] = y;
  int l = -1, ii, maxl, idx1, idx2, jj;
  int f[256][3];
  int g[256];

  for (int ii = 0; ii < len; ii++) {
    int z = z_offset + ii;
    int* out_ptr = base_ptr + z * stride;
    for (int jj = 0; jj < rank; jj++) {
      f[ii][jj] = out_ptr[jj];
    }
  }

  // copy from buffer to main memory 
  {
    
  }

  
  
  for (ii = 0; ii < len ; ii++) {
    if (f[ii][0] >= 0) {
      int fd = f[ii][d];
      int wR = 0.0;
      for (jj = 0; jj < rank; jj++) {
        if (jj != d) {
          int tw = (f[ii][jj] - coor[jj]);
          wR += tw * tw;
        }
      }
      while (l >= 1) {
        int a, b, c, uR = 0.0, vR = 0.0, f1;
        idx1 = g[l];
        f1 = f[idx1][d];
        idx2 = g[l - 1];
        a = f1 - f[idx2][d];
        b = fd - f1;
        c = a + b;
        for (jj = 0; jj < rank; jj++) {
          if (jj != d) {
            int cc = coor[jj];
            int tu = f[idx2][jj] - cc;
            int tv = f[idx1][jj] - cc;
            uR += tu * tu;
            vR += tv * tv;
          }
        }
        if (c * vR - b * uR - a * wR - a * b * c <= 0.0) { break; }
        --l;
      }
      ++l;
      g[l] = ii;
    }
  }
  maxl = l;
  if (maxl >= 0) {
    l = 0;
    for (ii = z_offset; ii < len+z_offset; ii++) {
      int delta1 = 0.0, t;
      for (jj = 0; jj < rank; jj++) {
        t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];
        delta1 += t * t;
      }
      while (l < maxl) {
        int delta2 = 0.0;
        for (jj = 0; jj < rank; jj++) {
          t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];
          delta2 += t * t;
        }
        if (delta1 <= delta2) break;
        delta1 = delta2;
        ++l;
      }
      idx1 = g[l];
      int* out_ptr = base_ptr + ii * stride;
      for (jj = 0; jj < rank; jj++) { out_ptr[jj] = f[idx1][jj]; }
    }
  }
}

__global__ void edt_depth_bf(
    int* d_output, char* boundary, uint d, uint width, uint height, uint depth, size_t* strides)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height
  uint z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;

  uint coords[3] = {x, y, z};
  uint dims[3] = {width, height, depth};
  uint data_strides[3] = {1, width, width * height};
  char working_direction = d;
  char plane_dim1 = (d + 1) % 3;
  char plane_dim2 = (d + 2) % 3;
  size_t output_line_start_offset =
      coords[plane_dim1] * strides[plane_dim1] + coords[plane_dim2] * strides[plane_dim2];
  size_t idx = x + y * width + z * width * height;

  size_t start_offset = coords[plane_dim1] * data_strides[plane_dim1] +
                        coords[plane_dim2] * data_strides[plane_dim2];

  // d_output[idx*3] = z;
  // d_output[idx*3+1] = y;
  // d_output[idx*3+2] = x;
  // return;

  if (boundary[idx] == 1) { return; }
  int temp_coord[3] = {-1, -1, -1};
  int min_distance = width * width + height * height + depth * depth;
  for (int i = 0; i < dims[d]; i++) {
    int* cur_out_put = &d_output[(start_offset + i * data_strides[d]) * 3];
    if (*cur_out_put != -1) {
      int dz = cur_out_put[0] - z;
      int dy = cur_out_put[1] - y;
      int dx = cur_out_put[2] - x;
      int distance = dz * dz + dy * dy + dx * dx;
      if (distance < min_distance) {
        min_distance = distance;
        temp_coord[0] = cur_out_put[0];
        temp_coord[1] = cur_out_put[1];
        temp_coord[2] = cur_out_put[2];
      }
    }
  }
  if (temp_coord[0] != -1) {
    d_output[idx * 3] = temp_coord[0];
    d_output[idx * 3 + 1] = temp_coord[1];
    d_output[idx * 3 + 2] = temp_coord[2];
  }
}

__global__ void init_edt_3d(
    char* input, int* output, char b_tag, int rank, uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;  // slowest dimension
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;

  if (input[idx] == b_tag) {
    output[idx * rank] = z;
    output[idx * rank + 1] = y;
    output[idx * rank + 2] = x;
  }
  else {
    output[idx * rank] = -1;
    // output[idx*rank + 1] = 0;
    // output[idx*rank + 2] = 0;
  }
}

__global__ void calculate_distance(
    int* output, float* distance, int rank, uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  double d = 0;
  d += (output[idx * rank] - z) * (output[idx * rank] - z);
  d += (output[idx * rank + 1] - y) * (output[idx * rank + 1] - y);
  d += (output[idx * rank + 2] - x) * (output[idx * rank + 2] - x);
  distance[idx] = sqrt(d);
}

void edt_3d(char* d_boundary, int* index, float* distance, uint width, uint height, uint depth)
{
  size_t width_stride = 3;
  size_t height_stride = width * 3;
  size_t depth_stride = width * height * 3;

  size_t size = width * height * depth;

  dim3 block(8, 8, 8);
  dim3 grid(
      (width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
      (depth + block.z - 1) / block.z);
  dim3 block1(32, 32);
  dim3 grid1((height + block1.x - 1) / block1.x, (width + block1.y - 1) / block1.y);
  dim3 block3(32, 32);
  dim3 grid3((width + block3.x - 1) / block3.x, (depth + block3.y - 1) / block3.y);
  dim3 block2(32, 32);
  dim3 grid2((depth + block2.x - 1) / block2.x, (height + block2.y - 1) / block2.y);

  size_t shared_mem_size = block1.x * block1.y * 512 * sizeof(int);

  init_edt_3d<<<grid, block>>>(d_boundary, index, (char)1, (int)3, width, height, depth);
  cudaDeviceSynchronize();
  edt_depth<<<grid1, block1>>>(
      index, depth_stride, 3, 0, depth, height, width, height_stride, width_stride);
  cudaDeviceSynchronize();
  edt_depth<<<grid3, block3>>>(
      index, height_stride, 3, 1, height, width, depth, width_stride, depth_stride);
  cudaDeviceSynchronize();
  edt_depth<<<grid2, block2>>>(
      index, width_stride, 3, 2, width, depth, height, depth_stride, height_stride);
  cudaDeviceSynchronize();
  calculate_distance<<<grid, block>>>(index, distance, 3, width, height, depth);
  cudaDeviceSynchronize();

  // if (0) {
  //   std::vector<float> distance_host(size, 0);
  //   cudaMemcpy(
  //       distance_host.data(), distance, size * sizeof(float), cudaMemcpyDeviceToHost);
  //   writefile("distance_ne.dat", distance_host.data(), size);
  // }
}

void edt_3d_chunck(
    char* d_boundary, int* index, float* distance, uint width, uint height, uint depth)
{
  size_t width_stride = 3;
  size_t height_stride = width * 3;
  size_t depth_stride = width * height * 3;

  size_t size = width * height * depth;

  dim3 block(8, 8, 8);
  dim3 grid(
      (width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
      (depth + block.z - 1) / block.z);

  int num_chunks = 4; 


  // limit number of threads in a block

  dim3 block1(32, 32, 1);
  dim3 grid1(
      (width + block1.x - 1) / block1.x, (height + block1.y - 1) / block1.y,
      num_chunks);  

  dim3 block3(32, 32, 1);
  dim3 grid3(
      (width + block3.x - 1) / block3.x, (depth + block3.y - 1) / block3.y,
      num_chunks);
  dim3 block2(32, 32, 1);
  dim3 grid2(
      (depth + block2.x - 1) / block2.x, (height + block2.y - 1) / block2.y,
      num_chunks);

  init_edt_3d<<<grid, block>>>(d_boundary, index, (char)1, (int)3, width, height, depth);
  cudaDeviceSynchronize();

  edt_depth_chunck<<<grid1, block1>>>(
      index, depth_stride, 3, 0, depth, height, width, height_stride, width_stride);
  cudaDeviceSynchronize();

  edt_depth_chunck<<<grid3, block3>>>(
      index, height_stride, 3, 1, height, width, depth, width_stride, depth_stride);
  cudaDeviceSynchronize();

  edt_depth_chunck<<<grid2, block2>>>(
      index, width_stride, 3, 2, width, depth, height, depth_stride, height_stride);
  cudaDeviceSynchronize();

  calculate_distance<<<grid, block>>>(index, distance, 3, width, height, depth);
  cudaDeviceSynchronize();

  // if (0) {
  //   std::vector<float> distance_host(size, 0);
  //   cudaMemcpy(
  //       distance_host.data(), distance, size * sizeof(float), cudaMemcpyDeviceToHost);
  //   writefile("distance_ne.dat", distance_host.data(), size);
  // }
}

// search the current dimension and fins the cloest boundary point.
// the current dimension is preset to zero if it's not a boundary point
// if its a boundary point, no need to search, simply return. // or juse use bounday to
// check?(could be less efficient)
void edt_3d_bf(char* d_boundary, int* index, float* distance, uint width, uint height, uint depth)
{
  size_t width_stride = 3;
  size_t height_stride = width * 3;
  size_t depth_stride = width * height * 3;

  size_t strides[3] = {width_stride, height_stride, depth_stride};

  dim3 block(8, 8, 16);
  dim3 grid(
      (width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
      (depth + block.z - 1) / block.z);

  init_edt_3d<<<grid, block>>>(d_boundary, index, (char)1, (int)3, width, height, depth);
  cudaDeviceSynchronize();
  edt_depth_bf<<<grid, block>>>(index, d_boundary, 0, width, height, depth, strides);
  cudaDeviceSynchronize();
  edt_depth_bf<<<grid, block>>>(index, d_boundary, 1, width, height, depth, strides);
  cudaDeviceSynchronize();
  edt_depth_bf<<<grid, block>>>(index, d_boundary, 2, width, height, depth, strides);
  cudaDeviceSynchronize();
  calculate_distance<<<grid, block>>>(index, distance, 3, width, height, depth);
  cudaDeviceSynchronize();
}