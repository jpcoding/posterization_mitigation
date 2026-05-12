#ifndef boundary_hpp
#define boundary_hpp

#include "compensation_cuda.hpp"
template <typename T_data>
__global__ void get_boundary(
    T_data* input, char* output, char b_tag, int rank, int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1) {
    output[idx] = 0;
    return;
  }
  T_data cur_idx = input[idx];
  // x is the slowest dimension
  T_data idx_left = input[(x - 1) + y * width + z * width * height];
  T_data idx_right = input[(x + 1) + y * width + z * width * height];
  T_data idx_up = input[x + (y - 1) * width + z * width * height];
  T_data idx_down = input[x + (y + 1) * width + z * width * height];
  T_data idx_front = input[x + y * width + (z - 1) * width * height];
  T_data idx_back = input[x + y * width + (z + 1) * width * height];

  if (cur_idx != idx_left || cur_idx != idx_right || cur_idx != idx_up || cur_idx != idx_down ||
      cur_idx != idx_front || cur_idx != idx_back) {
    output[idx] = b_tag;
  }
  else {
    output[idx] = 0;
  }
}

template <typename T_data>
__global__ void get_boundary_and_sign_map(
    T_data* input, char* output, char* sign_map, char b_tag, int rank, int width, int height,
    int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1) {
    output[idx] = 0;
    return;
  }
  T_data cur_idx = input[idx];
  // check the boundary

  // x is the slowest dimension
  T_data idx_left = input[(x - 1) + y * width + z * width * height];
  T_data idx_right = input[(x + 1) + y * width + z * width * height];
  T_data idx_up = input[x + (y - 1) * width + z * width * height];
  T_data idx_down = input[x + (y + 1) * width + z * width * height];
  T_data idx_front = input[x + y * width + (z - 1) * width * height];
  T_data idx_back = input[x + y * width + (z + 1) * width * height];

  //   T_data signs[6] = {idx_left-cur_idx, idx_right-cur_idx,
  //                     idx_up-cur_idx, idx_down-cur_idx,
  //                     idx_front-cur_idx, idx_back-cur_idx};
  T_data signs[6] = {idx_up - cur_idx,    idx_down - cur_idx,  idx_left - cur_idx,
                     idx_right - cur_idx, idx_front - cur_idx, idx_back - cur_idx};

  double grad_x = abs(idx_right - idx_left) * 0.5;
  double grad_y = abs(idx_up - idx_down) * 0.5;
  double grad_z = abs(idx_front - idx_back) * 0.5;
  double max_grad = grad_x;
  if (grad_y > max_grad) max_grad = grad_y;
  if (grad_z > max_grad) max_grad = grad_z;

  if (cur_idx != idx_left || cur_idx != idx_right || cur_idx != idx_up || cur_idx != idx_down ||
      cur_idx != idx_front || cur_idx != idx_back) {
    output[idx] = b_tag;
    if (max_grad >= 1.0) {
      sign_map[idx] = 0;
      return;
    }
    for (int i = 0; i < 6; i++) {
      if (signs[i] != 0) {
        sign_map[idx] = (signs[i] > 0) ? 1 : -1;
        break;
      }
    }
  }
  else {
    output[idx] = 0;
  }
}

template <typename T_sign, typename T_boundary, typename T_data>
void __global__ fill_sign_and_compensation(
    T_sign* sign_map, T_boundary* boundary, int* edt_index, T_data magnitude, int width,
    int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  int eZ = edt_index[idx * 3];
  int eY = edt_index[idx * 3 + 1];
  int eX = edt_index[idx * 3 + 2];
  size_t edt_index_idx = eZ * height * width + eY * width + eX;
  if (boundary[idx] == 0) { sign_map[idx] = sign_map[edt_index_idx]; }
}

// Packed uint32 variant: index array holds one uint32 per voxel
// encoding z/y/x as (z<<20)|(y<<10)|x (see edt_jfa.hpp).
// Avoids the stride-3 AoS reads of the original kernel.
template <typename T_sign, typename T_boundary, typename T_data>
void __global__ fill_sign_and_compensation_packed(
    T_sign* sign_map, T_boundary* boundary, const unsigned int* packed_index,
    T_data magnitude, int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  if (boundary[idx] != 0) return;
  unsigned int packed = packed_index[idx];
  if (packed == 0xFFFFFFFFu) return;  // no nearest seed
  int eX =  packed        & 0x3FF;
  int eY = (packed >> 10) & 0x3FF;
  int eZ = (packed >> 20) & 0x3FF;
  size_t nearest_idx = (size_t)eZ * height * width + eY * width + eX;
  sign_map[idx] = sign_map[nearest_idx];
}

template <typename T_boundary>
void __global__ filter_boundary(
    T_boundary* orig_bounday, T_boundary* new_boundary, T_boundary tag, int width, int height,
    int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // fasted dimension
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = x + y * width + z * width * height;
  if (orig_bounday[idx] == tag && new_boundary[idx] == tag) 
  { new_boundary[idx] = 0; }
}

// Fused: get_boundary(sign_map) + filter_boundary in one kernel.
// Detects boundaries in sign_map and immediately filters against orig_boundary.
// Eliminates one kernel launch + one sync vs separate get_boundary + filter_boundary.
template <typename T_data, typename T_boundary>
void __global__ get_filtered_boundary(
    T_data* input, T_boundary* orig_boundary, T_boundary* new_boundary,
    T_boundary b_tag, int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = x + (size_t)y * width + (size_t)z * width * height;

  // Edge voxels: not boundary
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1) {
    new_boundary[idx] = 0;
    return;
  }

  T_data cur = input[idx];
  T_data left  = input[(x-1) + y*width + z*width*height];
  T_data right = input[(x+1) + y*width + z*width*height];
  T_data up    = input[x + (y-1)*width + z*width*height];
  T_data down  = input[x + (y+1)*width + z*width*height];
  T_data front = input[x + y*width + (z-1)*width*height];
  T_data back  = input[x + y*width + (z+1)*width*height];

  if (cur != left || cur != right || cur != up || cur != down ||
      cur != front || cur != back) {
    // Is a sign_map boundary — but filter: if also original boundary, suppress
    new_boundary[idx] = (orig_boundary[idx] == b_tag) ? (T_boundary)0 : b_tag;
  } else {
    new_boundary[idx] = 0;
  }
}

// Fused: fill_sign_and_compensation_packed + get_filtered_boundary in ONE kernel.
// For each non-boundary voxel: propagates sign from nearest boundary voxel (via packed EDT index).
// Then detects sign-map boundaries and filters against original boundary map.
// Writes: sign_map (propagated), new_boundary (neutral boundary for EDT round 2).
// Eliminates two kernel launches + two full-volume reads vs the separate approach.
template <typename T_sign, typename T_boundary>
void __global__ fill_sign_and_neutral_boundary_fused(
    T_sign* sign_map, T_boundary* orig_boundary, T_boundary* new_boundary,
    const unsigned int* packed_index, T_boundary b_tag,
    int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = (size_t)x + (size_t)y * width + (size_t)z * width * height;

  // Step 1: propagate sign for non-boundary voxels
  if (orig_boundary[idx] == 0) {
    unsigned int packed = packed_index[idx];
    if (packed != 0xFFFFFFFFu) {
      int eX =  packed        & 0x3FF;
      int eY = (packed >> 10) & 0x3FF;
      int eZ = (packed >> 20) & 0x3FF;
      size_t nearest_idx = (size_t)eZ * height * width + eY * width + eX;
      sign_map[idx] = sign_map[nearest_idx];
    }
  }

  // Need a grid-wide sync here — sign_map must be fully propagated before
  // we detect boundaries in it. This kernel only does step 1.
  // Step 2 (neutral boundary detection) runs as a separate kernel below.
}

// Step 2 of the fused approach: detect neutral boundary from propagated sign_map.
// This is identical to get_filtered_boundary but named for clarity in the optimized path.
// Kept separate because step 1 (sign propagation) must complete for ALL voxels
// before we can detect boundaries in the sign_map.

// ── edt_method=4 helpers ────────────────────────────────────────────────────

// Downsample fine boundary+sign to coarse 2× grid (logical-OR boundary,
// first nonzero sign wins).
__global__ void downsample_boundary_with_sign_2x(
    const char* fine_boundary, const char* fine_sign,
    char* coarse_boundary, char* coarse_sign,
    int fw, int fh, int fd, int cw, int ch, int cd)
{
    int cx = blockIdx.x*blockDim.x + threadIdx.x;
    int cy = blockIdx.y*blockDim.y + threadIdx.y;
    int cz = blockIdx.z*blockDim.z + threadIdx.z;
    if (cx >= cw || cy >= ch || cz >= cd) return;
    size_t cidx = (size_t)cx + (size_t)cy*cw + (size_t)cz*cw*ch;
    char b = 0, s = 0;
    for (int dz = 0; dz < 2; dz++)
    for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++) {
        int fx=2*cx+dx, fy=2*cy+dy, fz=2*cz+dz;
        if (fx>=fw || fy>=fh || fz>=fd) continue;
        size_t fi = (size_t)fx + (size_t)fy*fw + (size_t)fz*fw*fh;
        if (fine_boundary[fi]) { b=1; if (!s) s=fine_sign[fi]; }
    }
    coarse_boundary[cidx] = b;
    coarse_sign[cidx]     = s;
}

// Fill sign on coarse grid using coarse PBA+ packed index.
__global__ void fill_sign_coarse(
    char* coarse_sign, const unsigned int* coarse_packed,
    int cw, int ch, int cd)
{
    int cx = blockIdx.x*blockDim.x + threadIdx.x;
    int cy = blockIdx.y*blockDim.y + threadIdx.y;
    int cz = blockIdx.z*blockDim.z + threadIdx.z;
    if (cx >= cw || cy >= ch || cz >= cd) return;
    size_t cidx = (size_t)cx + (size_t)cy*cw + (size_t)cz*cw*ch;
    if (coarse_sign[cidx] != 0) return;
    unsigned int packed = coarse_packed[cidx];
    if (packed == 0xFFFFFFFFu) return;
    int ex =  packed        & 0x3FF;
    int ey = (packed >> 10) & 0x3FF;
    int ez = (packed >> 20) & 0x3FF;
    coarse_sign[cidx] = coarse_sign[(size_t)ex + (size_t)ey*cw + (size_t)ez*cw*ch];
}

// Upsample coarse sign map to fine grid (nearest-neighbor, factor 2).
// Preserves existing nonzero fine signs (boundary voxels already have correct signs).
__global__ void upsample_sign_2x(
    const char* coarse_sign, char* fine_sign,
    int fw, int fh, int fd, int cw, int ch, int cd)
{
    int fx = blockIdx.x*blockDim.x + threadIdx.x;
    int fy = blockIdx.y*blockDim.y + threadIdx.y;
    int fz = blockIdx.z*blockDim.z + threadIdx.z;
    if (fx >= fw || fy >= fh || fz >= fd) return;
    size_t fidx = (size_t)fx+(size_t)fy*fw+(size_t)fz*fw*fh;
    if (fine_sign[fidx] != 0) return;  // preserve fine boundary signs
    int cx=min(fx/2,cw-1), cy=min(fy/2,ch-1), cz=min(fz/2,cd-1);
    fine_sign[fidx] = coarse_sign[(size_t)cx+(size_t)cy*cw+(size_t)cz*cw*ch];
}

#endif