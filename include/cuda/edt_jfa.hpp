#ifndef EDT_JFA_HPP
#define EDT_JFA_HPP

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// Jump Flooding Algorithm (JFA) for 3D Euclidean Distance Transform
//
// Coordinate encoding: packed uint32
//   bits 20-29 : z  (10 bits, max 1023)
//   bits 10-19 : y  (10 bits, max 1023)
//   bits  0- 9 : x  (10 bits, max 1023)
//   0xFFFFFFFF : invalid / no seed
//
// This gives 1 coalesced load per neighbor (vs 3 stride-3 loads with AoS).
// Max safe dimension: 1023.  GPU VRAM limits us to ~512^3, so this is always fine.
//
// JFA internally allocates two packed uint32 ping-pong buffers.
// At the end it writes the nearest-seed coordinates back into the caller's
// AoS int[N*3] index array (used downstream by fill_sign_and_compensation).

#define JFA_INVALID 0xFFFFFFFFu

__device__ __forceinline__ uint jfa_encode(uint z, uint y, uint x)
{
  return (z << 20) | (y << 10) | x;
}

__device__ __forceinline__ void jfa_decode(uint packed, int& sz, int& sy, int& sx)
{
  sx = (int)(packed & 0x3FFu);
  sy = (int)((packed >> 10) & 0x3FFu);
  sz = (int)((packed >> 20) & 0x3FFu);
}

// ---------------------------------------------------------------------------
// Init: write packed sentinel / seed from boundary map
// ---------------------------------------------------------------------------
__global__ void jfa_init_packed(
    const char* d_boundary, uint* d_packed,
    uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = x + (size_t)y * width + (size_t)z * width * height;
  d_packed[idx] = (d_boundary[idx] == (char)1)
                    ? jfa_encode(z, y, x)
                    : JFA_INVALID;
}

// ---------------------------------------------------------------------------
// JFA step: ping-pong, packed uint32
//   step >= 4 : 6 face neighbors (coarse passes — diagonals never win)
//   step <  4 : all 26 neighbors (fine passes — need full stencil)
// ---------------------------------------------------------------------------
__global__ void jfa_step_3d_packed(
    const uint* src, uint* dst,
    int step, uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;

  size_t idx = x + (size_t)y * width + (size_t)z * width * height;

  uint best = src[idx];
  int best_dist;
  if (best != JFA_INVALID) {
    int bz, by, bx;
    jfa_decode(best, bz, by, bx);
    int dz = bz - (int)z, dy = by - (int)y, dx = bx - (int)x;
    best_dist = dz*dz + dy*dy + dx*dx;
  } else {
    best_dist = 0x7FFFFFFF;
  }

  if (step >= 2) {
    // 6 face neighbors only
    const int offsets[6][3] = {
      {-step,0,0},{step,0,0},{0,-step,0},{0,step,0},{0,0,-step},{0,0,step}
    };
    for (int f = 0; f < 6; ++f) {
      int nz = (int)z + offsets[f][0];
      int ny = (int)y + offsets[f][1];
      int nx = (int)x + offsets[f][2];
      if (nz < 0 || nz >= (int)depth)  continue;
      if (ny < 0 || ny >= (int)height) continue;
      if (nx < 0 || nx >= (int)width)  continue;
      size_t nidx = (size_t)nx + (size_t)ny*width + (size_t)nz*width*height;
      uint nb = src[nidx];
      if (nb == JFA_INVALID) continue;
      int sz, sy, sx;
      jfa_decode(nb, sz, sy, sx);
      int ddz = sz-(int)z, ddy = sy-(int)y, ddx = sx-(int)x;
      int dist = ddz*ddz + ddy*ddy + ddx*ddx;
      if (dist < best_dist) { best_dist = dist; best = nb; }
    }
  } else {
    // full 26-neighbor stencil
    for (int dz = -step; dz <= step; dz += step) {
      int nz = (int)z + dz;
      if (nz < 0 || nz >= (int)depth) continue;
      for (int dy = -step; dy <= step; dy += step) {
        int ny = (int)y + dy;
        if (ny < 0 || ny >= (int)height) continue;
        for (int dx = -step; dx <= step; dx += step) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          int nx = (int)x + dx;
          if (nx < 0 || nx >= (int)width) continue;
          size_t nidx = (size_t)nx + (size_t)ny*width + (size_t)nz*width*height;
          uint nb = src[nidx];
          if (nb == JFA_INVALID) continue;
          int sz, sy, sx;
          jfa_decode(nb, sz, sy, sx);
          int ddz = sz-(int)z, ddy = sy-(int)y, ddx = sx-(int)x;
          int dist = ddz*ddz + ddy*ddy + ddx*ddx;
          if (dist < best_dist) { best_dist = dist; best = nb; }
        }
      }
    }
  }

  dst[idx] = best;
}

// ---------------------------------------------------------------------------
// Shared-memory tiled JFA step — step=1 (26 neighbors at ±1) only.
//
// Block: (8,8,8) = 512 threads
// Shared tile: (10,10,10) = 1000 uint32 = 4000 bytes per block
//   Each thread loads ~2 tile elements cooperatively, then all neighbor reads
//   hit shared memory instead of L2/DRAM.
//
// Launched as: jfa_step_3d_smem<<<grid_smem, block_smem, 4000>>>(...)
// ---------------------------------------------------------------------------
#define SMEM_BX 8
#define SMEM_BY 8
#define SMEM_BZ 8
#define SMEM_TX (SMEM_BX + 2)   // 10
#define SMEM_TY (SMEM_BY + 2)   // 10
#define SMEM_TZ (SMEM_BZ + 2)   // 10
#define SMEM_TILE (SMEM_TX * SMEM_TY * SMEM_TZ)  // 1000

__global__ void jfa_step_3d_smem(
    const uint* src, uint* dst,
    uint width, uint height, uint depth)
{
  // Shared tile: (10,10,10) with 1-voxel halo on each side
  extern __shared__ uint smem[];

  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int tid = tx + ty * SMEM_BX + tz * SMEM_BX * SMEM_BY;

  // Global coords of this thread's voxel
  const int x = blockIdx.x * SMEM_BX + tx;
  const int y = blockIdx.y * SMEM_BY + ty;
  const int z = blockIdx.z * SMEM_BZ + tz;

  // Global coords of tile origin (top-left-front corner, offset -1)
  const int gx0 = blockIdx.x * SMEM_BX - 1;
  const int gy0 = blockIdx.y * SMEM_BY - 1;
  const int gz0 = blockIdx.z * SMEM_BZ - 1;

  // Cooperatively load all 1000 tile elements (2 rounds of 512 threads)
  for (int i = tid; i < SMEM_TILE; i += SMEM_BX * SMEM_BY * SMEM_BZ) {
    int lz = i / (SMEM_TX * SMEM_TY);
    int ly = (i % (SMEM_TX * SMEM_TY)) / SMEM_TX;
    int lx = i % SMEM_TX;
    int gx = gx0 + lx, gy = gy0 + ly, gz = gz0 + lz;
    uint val = JFA_INVALID;
    if (gx >= 0 && gx < (int)width &&
        gy >= 0 && gy < (int)height &&
        gz >= 0 && gz < (int)depth)
      val = src[(size_t)gx + (size_t)gy * width + (size_t)gz * width * height];
    smem[i] = val;
  }
  __syncthreads();

  if (x >= (int)width || y >= (int)height || z >= (int)depth) return;

  // Shared memory coords of this thread (1-indexed offset into tile)
  const int sx = tx + 1, sy = ty + 1, sz = tz + 1;

  uint best = smem[sx + sy * SMEM_TX + sz * SMEM_TX * SMEM_TY];
  int best_dist = 0x7FFFFFFF;
  if (best != JFA_INVALID) {
    int bz, by, bx; jfa_decode(best, bz, by, bx);
    int dz = bz-z, dy = by-y, dx = bx-x;
    best_dist = dz*dz + dy*dy + dx*dx;
  }

  // All 26 neighbors from shared memory (no global reads)
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        uint nb = smem[(sx+dx) + (sy+dy)*SMEM_TX + (sz+dz)*SMEM_TX*SMEM_TY];
        if (nb == JFA_INVALID) continue;
        int nsz, nsy, nsx; jfa_decode(nb, nsz, nsy, nsx);
        int ddz = nsz-z, ddy = nsy-y, ddx = nsx-x;
        int dist = ddz*ddz + ddy*ddy + ddx*ddx;
        if (dist < best_dist) { best_dist = dist; best = nb; }
      }
    }
  }

  dst[(size_t)x + (size_t)y * width + (size_t)z * width * height] = best;
}

// ---------------------------------------------------------------------------
// Shared-memory tiled JFA step — step=2 (6 face neighbors at ±2) only.
// Block: (8,8,8). Shared tile: (12,12,12) = 1728 uint32 = 6912 bytes.
// Halo = 2 on each side; all 6 neighbor reads stay within tile bounds.
// ---------------------------------------------------------------------------
#define SMEM2_BX 8
#define SMEM2_BY 8
#define SMEM2_BZ 8
#define SMEM2_TX (SMEM2_BX + 4)   // 12
#define SMEM2_TY (SMEM2_BY + 4)   // 12
#define SMEM2_TZ (SMEM2_BZ + 4)   // 12
#define SMEM2_TILE (SMEM2_TX * SMEM2_TY * SMEM2_TZ)  // 1728

__global__ void jfa_step_3d_smem2(
    const uint* src, uint* dst,
    uint width, uint height, uint depth)
{
  extern __shared__ uint smem[];

  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int tid = tx + ty * SMEM2_BX + tz * SMEM2_BX * SMEM2_BY;

  const int x = blockIdx.x * SMEM2_BX + tx;
  const int y = blockIdx.y * SMEM2_BY + ty;
  const int z = blockIdx.z * SMEM2_BZ + tz;

  const int gx0 = blockIdx.x * SMEM2_BX - 2;
  const int gy0 = blockIdx.y * SMEM2_BY - 2;
  const int gz0 = blockIdx.z * SMEM2_BZ - 2;

  // Load 1728 tile elements cooperatively (ceil(1728/512)=4 iters max)
  for (int i = tid; i < SMEM2_TILE; i += SMEM2_BX * SMEM2_BY * SMEM2_BZ) {
    int lz = i / (SMEM2_TX * SMEM2_TY);
    int ly = (i % (SMEM2_TX * SMEM2_TY)) / SMEM2_TX;
    int lx = i % SMEM2_TX;
    int gx = gx0 + lx, gy = gy0 + ly, gz = gz0 + lz;
    uint val = JFA_INVALID;
    if (gx >= 0 && gx < (int)width &&
        gy >= 0 && gy < (int)height &&
        gz >= 0 && gz < (int)depth)
      val = src[(size_t)gx + (size_t)gy * width + (size_t)gz * width * height];
    smem[i] = val;
  }
  __syncthreads();

  if (x >= (int)width || y >= (int)height || z >= (int)depth) return;

  const int sx = tx + 2, sy = ty + 2, sz = tz + 2;  // halo offset = 2

  uint best = smem[sx + sy * SMEM2_TX + sz * SMEM2_TX * SMEM2_TY];
  int best_dist = 0x7FFFFFFF;
  if (best != JFA_INVALID) {
    int bz, by, bx; jfa_decode(best, bz, by, bx);
    int dz = bz-z, dy = by-y, dx = bx-x;
    best_dist = dz*dz + dy*dy + dx*dx;
  }

  // 6 face neighbors at ±2; off6 = [dz, dy, dx]
  const int off6[6][3] = {
    {-2,0,0},{2,0,0},{0,-2,0},{0,2,0},{0,0,-2},{0,0,2}
  };
  for (int f = 0; f < 6; ++f) {
    uint nb = smem[(sx + off6[f][2]) +
                  (sy + off6[f][1]) * SMEM2_TX +
                  (sz + off6[f][0]) * SMEM2_TX * SMEM2_TY];
    if (nb == JFA_INVALID) continue;
    int nsz, nsy, nsx; jfa_decode(nb, nsz, nsy, nsx);
    int ddz = nsz-z, ddy = nsy-y, ddx = nsx-x;
    int dist = ddz*ddz + ddy*ddy + ddx*ddx;
    if (dist < best_dist) { best_dist = dist; best = nb; }
  }

  dst[(size_t)x + (size_t)y * width + (size_t)z * width * height] = best;
}

// ---------------------------------------------------------------------------
// Write-back: packed uint32 → AoS int[N*3] for downstream kernels
//   (fill_sign_and_compensation reads index[idx*3+0..2])
// ---------------------------------------------------------------------------
__global__ void jfa_writeback_aos(
    const uint* d_packed, int* index,
    uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = x + (size_t)y*width + (size_t)z*width*height;
  uint packed = d_packed[idx];
  if (packed != JFA_INVALID) {
    int sz, sy, sx;
    jfa_decode(packed, sz, sy, sx);
    index[idx*3]   = sz;
    index[idx*3+1] = sy;
    index[idx*3+2] = sx;
  } else {
    index[idx*3]   = -1;
    index[idx*3+1] = -1;
    index[idx*3+2] = -1;
  }
}

// ---------------------------------------------------------------------------
// Distance: packed uint32 → sqrt(dist²)
// ---------------------------------------------------------------------------
__global__ void jfa_calc_distance_packed(
    const uint* d_packed, float* distance,
    uint width, uint height, uint depth)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) return;
  size_t idx = x + (size_t)y*width + (size_t)z*width*height;
  uint packed = d_packed[idx];
  if (packed != JFA_INVALID) {
    int sz, sy, sx;
    jfa_decode(packed, sz, sy, sx);
    float dz = (float)(sz-(int)z);
    float dy = (float)(sy-(int)y);
    float dx = (float)(sx-(int)x);
    distance[idx] = sqrtf(dz*dz + dy*dy + dx*dx);
  } else {
    distance[idx] = 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline int jfa_initial_step(uint max_dim)
{
  int step = 1;
  while (step * 2 <= (int)max_dim) step *= 2;
  return step;
}

// ---------------------------------------------------------------------------
// Main dispatcher — same external signature as before.
// jfa_level arg is kept for CLI compatibility but no longer changes behavior.
// ---------------------------------------------------------------------------
void edt_3d_jfa_level(
    char* d_boundary, int* index, float* distance,
    uint width, uint height, uint depth, int /*jfa_level*/)
{
  // 64-wide in X for warp-coalesced reads along the fast axis
  dim3 block(64, 4, 2);
  dim3 grid(
      (width  + block.x - 1) / block.x,
      (height + block.y - 1) / block.y,
      (depth  + block.z - 1) / block.z);

  // Separate launch config for step=1 shared-memory kernel
  dim3 block_sm(SMEM_BX, SMEM_BY, SMEM_BZ);
  dim3 grid_sm(
      (width  + SMEM_BX - 1) / SMEM_BX,
      (height + SMEM_BY - 1) / SMEM_BY,
      (depth  + SMEM_BZ - 1) / SMEM_BZ);
  const size_t smem_bytes = SMEM_TILE * sizeof(uint);  // 4000 B

  size_t n_voxels = (size_t)width * height * depth;

  // Allocate two packed uint32 ping-pong buffers.
  // Track original pointers separately so we can always free both.
  uint* d_buf[2] = {nullptr, nullptr};
  cudaError_t e1 = cudaMalloc(&d_buf[0], n_voxels * sizeof(uint));
  cudaError_t e2 = cudaMalloc(&d_buf[1], n_voxels * sizeof(uint));
  if (e1 != cudaSuccess || e2 != cudaSuccess) {
    printf("[JFA] ERROR: failed to allocate packed buffers: %s\n",
           cudaGetErrorString(e1 != cudaSuccess ? e1 : e2));
    cudaFree(d_buf[0]);
    cudaFree(d_buf[1]);
    return;
  }

  uint max_dim = width;
  if (height > max_dim) max_dim = height;
  if (depth  > max_dim) max_dim = depth;

  // src/dst are indices into d_buf[]; swap by XOR without touching the allocs
  int src = 0, dst = 1;

  const size_t smem2_bytes = SMEM2_TILE * sizeof(uint);  // kept for reference, unused
  (void)smem2_bytes;

  // Init seeds into d_buf[src]
  jfa_init_packed<<<grid, block>>>(d_boundary, d_buf[src], width, height, depth);

  // JFA passes: initial_step, .../2, ..., 1
  // step >= 2: 6-face global-memory kernel (tile-load overhead > gain for 6 reads)
  // step == 1: 26-neighbor smem-tiled kernel — loads 1000 elems, reads 26 neighbors
  for (int step = jfa_initial_step(max_dim); step >= 1; step /= 2) {
    if (step == 1)
      jfa_step_3d_smem<<<grid_sm, block_sm, smem_bytes>>>(
          d_buf[src], d_buf[dst], width, height, depth);
    else
      jfa_step_3d_packed<<<grid, block>>>(
          d_buf[src], d_buf[dst], step, width, height, depth);
    src ^= 1; dst ^= 1;
  }
  // +1 JFA refinement pass (step=1, smem kernel)
  jfa_step_3d_smem<<<grid_sm, block_sm, smem_bytes>>>(
      d_buf[src], d_buf[dst], width, height, depth);
  src ^= 1; dst ^= 1;
  // final result is in d_buf[src]

  // Compute float distances from packed coords
  jfa_calc_distance_packed<<<grid, block>>>(
      d_buf[src], distance, width, height, depth);

  // Store packed uint32 result in caller's index array (first N elements, as uint*).
  // fill_sign_and_compensation_packed (boundary_cuda.hpp) reads from this.
  // The index allocation is size*3 ints, so there's always room for size uint32s.
  cudaMemcpy(index, d_buf[src], n_voxels * sizeof(uint), cudaMemcpyDeviceToDevice);

  cudaDeviceSynchronize();

  cudaFree(d_buf[0]);
  cudaFree(d_buf[1]);
}

// Entry point with default level (kept for compatibility)
void edt_3d_jfa(
    char* d_boundary, int* index, float* distance,
    uint width, uint height, uint depth)
{
  edt_3d_jfa_level(d_boundary, index, distance, width, height, depth, 0);
}

#endif // EDT_JFA_HPP
