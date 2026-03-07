#ifndef EDT_PROPAGATE_HPP
#define EDT_PROPAGATE_HPP
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <float.h>
#include <pthread.h>

#include <cmath>
#include <csignal>
#include <cstddef>
#include <iostream>
#include <system_error>
#include <vector>

#include "edt.hpp"
#include "utils/file_utils.hpp"

__global__ void edt_depth(
    int* d_output, char* boundary, size_t stride, uint rank, uint d, uint len, uint width,
    uint height, size_t width_stride, size_t height_stride)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height
  if (x >= width || y >= height) return;

  // supposed d_output has been initialized

  int* d_output_start = d_output + x * width_stride + y * height_stride;
  char* d_boundary_start = boundary + x * width_stride / 3 + y * height_stride / 3;
  int l = -1, ii, maxl, idx1, idx2, jj;
  // __shared__ int f[512][3];
  // __shared__ int g[512];
  __shared__ int f[512][3];
  int g_len = 0;

  // __shared__ int shared_coor[3*16*16];
  // int tid = threadIdx.y * blockDim.x + threadIdx.x;

  // int *coor = shared_coor + tid * 3;
  int coor[3];
  coor[d] = 0;
  coor[(d + 1) % 3] = x;
  coor[(d + 2) % 3] = y;

  for (ii = 0; ii < len; ii++) {
    if (d_boundary_start[ii * stride / 3] == 1) {
      for (jj = 0; jj < rank; jj++) { f[ii][jj] = d_output_start[ii * stride + jj]; }
      g_len++;
    }
  }

  if(g_len == 0) return; 
  int temp_coord[3] = {-1, -1, -1}; 
  for (ii = 0; ii < len; ii++) {
    coor[d] = ii; 
    int min = INT_MAX;
    for (int jj = 0; jj< g_len; jj++)
    {
      int distance = 0; 
      for (int kk = 0; kk < rank; kk++)
      {
        distance += (f[jj][kk] - coor[kk]) * (f[jj][kk] - coor[kk]);
      }
      if (distance < min)
      {
        min = distance; 
        temp_coord[0] = f[jj][0];
        temp_coord[1] = f[jj][1];
        temp_coord[2] = f[jj][2];
      }
    }






  }





}

void edt_3d_prop(
    char* d_boundary, int* index, float* distance, uint width, uint height, uint depth)
{
  size_t width_stride = 3;
  size_t height_stride = width * 3;
  size_t depth_stride = width * height * 3;

  size_t size = width * height * depth;
  size_t strides[3] = {width_stride, height_stride, depth_stride};

  dim3 block(8, 8, 16);
  dim3 grid(
      (width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
      (depth + block.z - 1) / block.z);

  init_edt_3d<<<grid, block>>>(d_boundary, index, (char)1, (int)3, width, height, depth);
  cudaDeviceSynchronize();

  dim3 depthGrid(width, height, 1);
  dim3 depthThreads(1, 1, 1);

  edt_depth_bf2<<<depthGrid, depthThreads>>>(index, d_boundary, 0, width, height, depth, strides);
  cudaDeviceSynchronize();
  edt_depth_bf<<<grid, block>>>(index, d_boundary, 1, width, height, depth, strides);
  cudaDeviceSynchronize();
  edt_depth_bf<<<grid, block>>>(index, d_boundary, 2, width, height, depth, strides);
  cudaDeviceSynchronize();

  calculate_distance<<<grid, block>>>(index, distance, 3, width, height, depth);
  cudaDeviceSynchronize();
}

#endif  // EDT_PROPAGATE_HPP