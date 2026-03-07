#ifndef compensation
#define compensation
#include <cuda_runtime.h>
#include <type_traits>
#include "boundary_cuda.hpp" 

// input: quantization index [0 as the middle point]
// output: the boundary of the quantization index
// b_tag: the boundary tag
// rank: the number of dimensions

// given the quantization index map and the boundary map
// calculate the sign of the edges

template <typename T_data_sign>
__device__ char get_sign(T_data_sign data) {
  char sign = (char)(((double)data > 0.0) - ((double)data < 0.0));
  return sign;
}



__global__ void get_sign_map(
    int* d_quant_inds, char* d_boundary, char* d_sign_map, int rank, int width, int height,
    int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fastest dimension depth 
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // slowest dimension width
  if (x >= width || y >= height || z >= depth) return;
  size_t stride_x = 1; 
  size_t stride_y = width;
  size_t stride_z = width * height;
  size_t strides[3] = {stride_x, stride_y, stride_z}; 
  int dims[3] = {width, height, depth}; // fastest to slowest 
  int idx = x*stride_x + y*stride_y + z*stride_z; // fast to slowest 
  if (d_boundary[idx] == 0) return;  // if this is not a boundary point, return
  int cur_quant_index = d_quant_inds[idx]; // the current quantization index
  int tx, ty, tz;  
  char signs[6] = {0, 0, 0, 0, 0, 0};
  int distances[6] = {0, 0, 0, 0, 0, 0};
  int gradient[3] = {0,0,0}; 
  double max_grad = -1; 
  gradient[0] = abs(d_quant_inds[idx + stride_x] - d_quant_inds[idx - stride_x]);
  gradient[1] = abs(d_quant_inds[idx + stride_y] - d_quant_inds[idx - stride_y]);
  gradient[2] = abs(d_quant_inds[idx + stride_z] - d_quant_inds[idx - stride_z]);
  for (int i = 0; i < 3; i++)
  {
    if ((gradient[i]) > max_grad)
    {
      max_grad = (gradient[i])*1.0;
    }
  }
  max_grad = max_grad * 0.5; 
  if(max_grad >= 1.0)
  {
    d_sign_map[idx] = 0 ; 
    return; 
  } 


  // first dimension 
  int index; 

  index = 2;
  tx = x - 1; ty = y; tz = z; 
  while (tx >0)
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(cur_quant_index - d_quant_inds[cur_idx]);
      break; 
    }
    tx--;
  }
  distances[index] = x - tx -1;

  index = 3; 
  tx = x + 1; ty = y; tz = z;
  while (tx < dims[0])
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(d_quant_inds[cur_idx] -cur_quant_index);
      break; 
    }
    tx++;
  }
  distances[index] = tx - x - 1;

  // second dimension
  index = 0; 
  tx = x; ty = y - 1; tz = z;
  while (ty > 0)
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(cur_quant_index - d_quant_inds[cur_idx]);
      break; 
    }
    ty--;
  }
  distances[index] = y - ty - 1;

  index = 1; 
  tx = x; ty = y + 1; tz = z;
  while (ty < dims[1])
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(d_quant_inds[cur_idx] - cur_quant_index);
      break; 
    }
    ty++;
  }
  distances[index] = ty - y - 1;

  // third dimension
  index = 4;
  tx = x; ty = y; tz = z - 1;
  while (tz > 0)
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(cur_quant_index - d_quant_inds[cur_idx]);
      break; 
    }
    tz--;
  }
  distances[index] = z - tz - 1;

  index = 5;
  tx = x; ty = y; tz = z + 1;
  while (tz < dims[2] )
  {
    int cur_idx = tx*strides[0] + ty * strides[1] + tz * strides[2]; 
    if (d_quant_inds[cur_idx] != cur_quant_index)
    {
      signs[index] = get_sign(d_quant_inds[cur_idx] - cur_quant_index);
      break; 
    }
    tz++;
  }
  distances[index] = tz - z - 1;

  // calculate the sign map
  char sign = 0;
  int direction  = 0; 
  int min_distance = distances[0];  
  for (int i = 0; i < 6; i++)
  {
    if (distances[i] < min_distance)
    {
      min_distance = distances[i];
      direction = i;
    }
  }
  sign = (direction % 2 == 0) ? -1.0f : 1.0f; 
  sign = sign * signs[direction]; 
  d_sign_map[idx] = sign;
}

template <typename T_distance, typename T_data, typename T_index> 
__global__ void compensation_idw(char* boundary, T_distance* d_edge, T_index* idx_edge,  T_distance* d_neutral, 
                                char* sign_map, T_data* quantized_data, T_data magnitude, int width, int height, int depth) 
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fastest dimension depth 
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // slowest dimension width
  if (x >= width || y >= height || z >= depth) return;
  size_t stride_x = 1; 
  size_t stride_y = width;
  size_t stride_z = width * height;
  int idx = x*stride_x + y*stride_y + z*stride_z; // fast to slowest 
  char sign = sign_map[idx]; 
  T_distance d1 = d_edge[idx]+0.5;
  T_distance d2 = d_neutral[idx]+0.5;
  double val = (1/d1) / (1/d1 + 1/d2) * sign * magnitude;
  quantized_data[idx] = val + quantized_data[idx];
}


// Optimized compensation: computes distances on-the-fly from packed indices.
// Eliminates the need for distance_edge and distance_neutral float buffers
// (saves 8 bytes/voxel device memory and avoids two full-volume float writes/reads).
// packed_index_edge: packed uint32 nearest-boundary index per voxel (from EDT round 1)
// packed_index_neutral: packed uint32 nearest-neutral-boundary index (from EDT round 2)
// If packed_index_neutral is nullptr, uses a fixed decay: comp = sign * mag * max(0, 1 - d1/R)
// with R = fixed_decay_radius.
__global__ void compensation_idw_nodist(
    char* boundary, const unsigned int* packed_index_edge,
    const unsigned int* packed_index_neutral,
    char* sign_map, float* quantized_data, float magnitude,
    int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = (size_t)x + (size_t)y * width + (size_t)z * width * height;
  char sign = sign_map[idx];
  if (sign == 0) return;

  // Compute d_edge from packed index
  unsigned int pe = packed_index_edge[idx];
  if (pe == 0xFFFFFFFFu) return;
  int ex = pe & 0x3FF;
  int ey = (pe >> 10) & 0x3FF;
  int ez = (pe >> 20) & 0x3FF;
  float dx1 = (float)(ex - x), dy1 = (float)(ey - y), dz1 = (float)(ez - z);
  float d1 = sqrtf(dx1*dx1 + dy1*dy1 + dz1*dz1) + 0.5f;

  float d2;
  if (packed_index_neutral != nullptr) {
    unsigned int pn = packed_index_neutral[idx];
    if (pn == 0xFFFFFFFFu) return;
    int nx = pn & 0x3FF;
    int ny = (pn >> 10) & 0x3FF;
    int nz = (pn >> 20) & 0x3FF;
    float dx2 = (float)(nx - x), dy2 = (float)(ny - y), dz2 = (float)(nz - z);
    d2 = sqrtf(dx2*dx2 + dy2*dy2 + dz2*dz2) + 0.5f;
  } else {
    d2 = 8.0f;  // fixed decay radius fallback
  }

  float val = (1.0f/d1) / (1.0f/d1 + 1.0f/d2) * sign * magnitude;
  quantized_data[idx] += val;
}

// Optimized compensation with downsampled round-2 distance.
// d_edge distance is computed on-the-fly from packed_index_edge.
// d_neutral is read from a DOWNSAMPLED distance buffer (ds_distance_neutral)
// at half resolution in each dimension, with trilinear interpolation.
__global__ void compensation_idw_downsample(
    char* boundary, const unsigned int* packed_index_edge,
    const float* ds_distance_neutral, int ds_w, int ds_h, int ds_d,
    char* sign_map, float* quantized_data, float magnitude,
    int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = (size_t)x + (size_t)y * width + (size_t)z * width * height;
  char sign = sign_map[idx];
  if (sign == 0) return;

  // d_edge from packed index
  unsigned int pe = packed_index_edge[idx];
  if (pe == 0xFFFFFFFFu) return;
  int ex = pe & 0x3FF;
  int ey = (pe >> 10) & 0x3FF;
  int ez = (pe >> 20) & 0x3FF;
  float dx1 = (float)(ex - x), dy1 = (float)(ey - y), dz1 = (float)(ez - z);
  float d1 = sqrtf(dx1*dx1 + dy1*dy1 + dz1*dz1) + 0.5f;

  // Trilinear interpolation of downsampled d_neutral
  // Map full-res coord to downsampled coord (center of 2x2x2 block)
  float fx = (x + 0.5f) * 0.5f - 0.5f;
  float fy = (y + 0.5f) * 0.5f - 0.5f;
  float fz = (z + 0.5f) * 0.5f - 0.5f;
  int x0 = (int)floorf(fx), y0 = (int)floorf(fy), z0 = (int)floorf(fz);
  float wx = fx - x0, wy = fy - y0, wz = fz - z0;

  // Clamp to valid range
  int x1 = min(x0 + 1, ds_w - 1); x0 = max(x0, 0);
  int y1 = min(y0 + 1, ds_h - 1); y0 = max(y0, 0);
  int z1 = min(z0 + 1, ds_d - 1); z0 = max(z0, 0);

  // 8 corners
  #define DS_IDX(a,b,c) ((size_t)(a) + (size_t)(b)*ds_w + (size_t)(c)*ds_w*ds_h)
  float c000 = ds_distance_neutral[DS_IDX(x0,y0,z0)];
  float c100 = ds_distance_neutral[DS_IDX(x1,y0,z0)];
  float c010 = ds_distance_neutral[DS_IDX(x0,y1,z0)];
  float c110 = ds_distance_neutral[DS_IDX(x1,y1,z0)];
  float c001 = ds_distance_neutral[DS_IDX(x0,y0,z1)];
  float c101 = ds_distance_neutral[DS_IDX(x1,y0,z1)];
  float c011 = ds_distance_neutral[DS_IDX(x0,y1,z1)];
  float c111 = ds_distance_neutral[DS_IDX(x1,y1,z1)];
  #undef DS_IDX

  float d2 = c000*(1-wx)*(1-wy)*(1-wz) + c100*wx*(1-wy)*(1-wz)
           + c010*(1-wx)*wy*(1-wz)      + c110*wx*wy*(1-wz)
           + c001*(1-wx)*(1-wy)*wz       + c101*wx*(1-wy)*wz
           + c011*(1-wx)*wy*wz           + c111*wx*wy*wz;

  // Scale back: downsampled distances are in downsampled coords, scale to full-res
  d2 = d2 * 2.0f + 0.5f;

  float val = (1.0f/d1) / (1.0f/d1 + 1.0f/d2) * sign * magnitude;
  quantized_data[idx] += val;
}

#endif
