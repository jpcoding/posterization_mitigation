# JFA EDT Optimization Notes

## Overview

The CUDA EDT (Euclidean Distance Transform) in the compensation pipeline uses the
Jump Flooding Algorithm (JFA). This document summarizes the staged optimization
work, benchmarking results, and profiling findings.

## JFA Optimization Levels

### Current implementation (packed uint32)

`include/cuda/edt_jfa.hpp` now uses a single **packed uint32** encoding for all JFA
passes. The `jfa_level` CLI arg is kept for compatibility but no longer changes behavior.

```
bits 20-29: z  (10 bits, max 1023)
bits 10-19: y  (10 bits, max 1023)
bits  0- 9: x  (10 bits, max 1023)
0xFFFFFFFF  : invalid / no seed
```

1 coalesced load per neighbor (vs 3 stride-3 loads with AoS) → ~3× less memory traffic per read.

Additional optimizations applied:
- **6-face neighbors for step ≥ 4** (first 7 of 11 passes): cuts ~75% of neighbor work
- **Block shape (64, 4, 2)** instead of (8, 8, 8): 64-wide in X for coalesced warp reads
- **Ping-pong uint32 buffers** — allocated/freed internally, AoS index written back at end

### Superseded levels (removed)
| Old Level | Strategy |
|---|---|
| 0 | In-place AoS + sync every pass |
| 1 | In-place AoS, sync at end only |
| 2 | Ping-pong AoS buffers, no per-pass sync |

### CLI Usage

```
test_compensation_cuda <input_file> <rel_eb> <use_chunk> <dim0_fast> <dim1> <dim2_slow> [use_jfa] [jfa_level]
```

Example:
```bash
./test_cuda/test_compensation_cuda /path/to/data.f32 0.01 1 512 512 512 1 2
```

## GPU & Memory

- **GPU:** NVIDIA GeForce RTX 4070 Laptop, 8188 MiB VRAM
- **Memory footprint:** ~43 bytes/voxel on device
  - `int quant_inds` (4B), `float quantized` (4B), `char boundary` (1B),
    `char boundary_neutral` (1B), `float distance_edge` (4B),
    `int index_edge[3]` (12B), `float distance_neutral` (4B),
    `int index_neutral[3]` (12B), `char sign_map` (1B)
- **Max practical volume:** ~160M voxels (~512^3) to stay within 8 GiB VRAM
- **1024^3 would need ~43 GiB** — not feasible on this GPU

## Benchmark Results

### Dataset: NYX 512x512x512 (`velocity_x.f32`, 512 MB), `rel_eb=0.01`

#### Packed uint32 JFA — optimization history (NYX 512³, rel_eb=0.01)

| Optimization applied | edt_total | Total elapsed | PSNR (dB) |
|---|---|---|---|
| AoS JFA baseline (old) | 0.701 s | 0.864 s | 49.27 |
| + Packed uint32 + block(64,4,2) | 0.288 s | 0.469 s | 54.16 |
| + float/sqrtf in distance kernel | 0.272 s | 0.448 s | 54.16 |
| + Eliminate writeback (packed fill_sign) | 0.262 s | 0.434 s | 54.16 |
| + Disable debug host downloads | 0.264 s | 0.299 s | 54.16 |
| + 6-face for step≥2 | 0.230 s | 0.265 s | 54.17 |
| + smem tiling for step=1 **(current)** | **0.200 s** | **0.234 s** | **54.17** |
| (tested: 6-face for step≥1) | 0.185 s | 0.220 s | 53.86 ✗ reverted |

**vs cuSZ (GPU compress only): 21 ms — compensation is ~11× compressor time.**

### Dataset: Hurricane 100x500x500 (`Uf48.bin.f32`), `rel_eb=0.01`

| Method | edt_total | Total elapsed | PSNR (dB) | Padded voxels |
|---|---|---|---|---|
| JFA | 0.037 s | 0.045 s | 50.11 | — |
| **PBA+** | **0.019 s** | **0.027 s** | **50.17** | 26.2M (512×512×100) |

PBA+ non-cubic padding: `z_axis=depth` → 512×512×100 = 26.2M voxels
(vs 134M if padded to 512³ cube — **5× VRAM savings**).

## Profiling (nsys)

`nsys profile` on NYX 512^3 (level 0):

| Kernel | % GPU Time | Invocations | Avg Duration |
|--------|-----------|-------------|--------------|
| `jfa_step_3d` | 88.7% | 22 | ~30 ms each |
| Other kernels | 11.3% | — | — |

`ncu` (NVIDIA Compute Profiler) was blocked by `ERR_NVGPUCTRPERM` — no GPU
hardware counter access without admin reconfiguration.

| Stage | Time (s) | % of Total |
|-------|----------|-----------|
| boundary_detect | 0.009 | 3.4% |
| edt_round1 | 0.117 | 44.1% |
| fill_sign | 0.009 | 3.4% |
| neutral_boundary | 0.005 | 1.9% |
| edt_round2 | 0.113 | 42.6% |
| compensation | 0.012 | 4.5% |
| **edt_total** | **0.230** | **86.7%** |
| **total elapsed** | **0.265** | 100% |

EDT-only throughput (two rounds over 134M voxels):

| Implementation | Mvox/s | edt_total |
|---|---|---|
| Old AoS JFA | 383 | 0.701 s |
| Current JFA (packed uint32, smem step=1) | 1308 | 0.205 s |
| PBA+ (initial port) | 2803 | 0.096 s |
| **PBA+ (pre-alloc + no sync)** | **2913** | **0.092 s** |

### PBA+ vs JFA — side-by-side (NYX 512³, rel_eb=0.01)

| Metric | JFA | PBA+ | Speedup |
|---|---|---|---|
| edt_round1 | 0.103 s | 0.046 s | 2.2× |
| edt_round2 | 0.103 s | 0.046 s | 2.2× |
| **edt_total** | **0.205 s** | **0.092 s** | **2.23×** |
| **Total elapsed** | **0.240 s** | **0.123 s** | **1.95×** |
| PSNR | 54.172 dB | 54.170 dB | identical |

PBA+ uses 5 kernel launches (FloodZ → Maurer → ColorAxis → Maurer → ColorAxis)
vs JFA's 12+ passes. PBA+ computes the *exact* EDT.

Optimizations applied:
- Non-cubic support: (xy_size, z_size) with axis remapping to minimize padding
- Pre-allocated ping-pong buffers (reused across both EDT rounds)
- Removed internal cudaDeviceSynchronize (default stream ordering)

## Why the Optimizations Didn't Help

The sync removal (level 1) and ping-pong buffers (level 2) target launch/sync
overhead, but the `jfa_step_3d` kernel execution time itself is the dominant cost
(~30 ms × 22 invocations = ~660 ms). Sync overhead is negligible compared to
kernel runtime.

## PBA+ Optimized (edt_method=3) — Implemented

Downsampled Round 2 EDT + on-the-fly distance computation. Keeps method=2 (PBA+ baseline) for comparison.

### What was implemented:
1. **Downsampled EDT Round 2**: 2×2×2 logical-OR downsample of neutral_boundary → PBA on 1/8 volume → trilinear interpolation in compensation kernel
2. **On-the-fly distance**: `compensation_idw_downsample` kernel recomputes d_edge from packed index (no float distance_edge buffer needed for compensation)
3. **Fused sign propagation kernel**: `fill_sign_and_neutral_boundary_fused` in boundary_cuda.hpp (still needs separate neutral boundary detection kernel due to grid-wide sync requirement)

### Benchmark: PBA+ baseline vs PBA+ optimized (NYX 512³, rel_eb=0.01)

| Metric | PBA+ (method=2) | PBA+ opt (method=3) | Speedup |
|---|---|---|---|
| edt_round1 | 50 ms | 50 ms | 1.0× |
| fill_sign | ~0 ms | 7 ms | (now visible) |
| neutral_boundary | ~0 ms | 3 ms | (now visible) |
| edt_round2 | **57 ms** | **7.5 ms** | **7.6×** |
| compensation | 9 ms | 10 ms | ~1× |
| **edt_total** | **106 ms** | **57 ms** | **1.85×** |
| **Total elapsed** | **116 ms** | **78 ms** | **1.49×** |
| PSNR | 54.170 dB | 54.104 dB | -0.066 dB |
| vs cuSZ (21 ms) | 5.5× | **3.7×** | — |

### Hurricane 100×500×500 (rel_eb=0.01)

| Metric | PBA+ (method=2) | PBA+ opt (method=3) |
|---|---|---|
| Total elapsed | 22 ms | **15 ms** |
| edt_total | 19 ms | 11 ms |
| PSNR | 50.17 dB | 50.02 dB |

### Remaining CUDA bottleneck
EDT Round 1 (exact PBA+ on full 512³) = 50ms. This is the algorithm floor.
Minor further savings possible (~5-8ms) by skipping distance_edge float write in extract.

## Next Step: CPU Downsampled EDT Round 2

**This is where the next session should start.**

The downsampled Round 2 trick should be ported to the **CPU compensation pipeline**.
The CPU code is in `include/compensation.hpp` (or similar). The approach:

1. Read the CPU EDT and compensation code (likely in `include/`)
2. After CPU EDT round 1 + fill_sign + neutral_boundary, downsample neutral_boundary by 2×2×2
3. Run CPU EDT round 2 on the small volume (1/8 size → ~8× speedup on CPU)
4. In the compensation loop, trilinear-interpolate the downsampled distance_neutral
5. Keep the original full-res path as a flag for comparison

Key files to examine:
- `include/compensation.hpp` — CPU compensation pipeline
- `include/edt.hpp` or `include/edt_cpu.hpp` — CPU EDT implementation
- `test/test_compensation.cpp` — CPU test driver

This is a **paper-worthy algorithmic contribution**: same trick speeds up both CPU and GPU,
PSNR loss is negligible (-0.07 dB), and the CPU relative speedup should be even larger
since CPU EDT is compute-bound (no GPU parallelism to mask the 8× volume reduction).

## CPU Downsampled EDT Round 2 — Implemented

**This is the CPU port of the CUDA downsampled R2 trick.**

### What was implemented:
1. `set_downsample_edt_round2(bool)` setter added to `PM::Compensation` in `include/compensation.hpp`
2. In `get_compensation_map_3d()`: after neutral_boundary is ready, 2×2×2 logical-OR downsample → EDT on 1/8 volume → trilinear interpolation in compensation loop
3. `--downsample_r2 1` CLI flag added to `test/test_quantize_and_edt.cpp`
4. Original full-res path kept as default (flag=false)

### Benchmark: CPU baseline vs CPU downsampled r2 (NYX 512³, rel_eb=0.01, 8 threads)

| Metric | Baseline | Downsampled r2 | Speedup |
|---|---|---|---|
| **Total elapsed** | **5.40 s** | **3.59 s** | **1.51×** |
| PSNR | 54.173 dB | 54.109 dB | -0.064 dB |

PSNR loss matches CUDA (-0.07 dB). Total speedup is 1.51× on the whole pipeline (EDT r1 + fill_sign + boundary2 unchanged).

### CPU Thread Scaling: Full-res vs Downsampled r2 (NYX 512³, rel_eb=0.01)

EDT_OMP_Opt (int16_t coord storage, dist-only r2).

| Threads | Full-res (s) | Speedup | Efficiency | Downsampled r2 (s) | Speedup | Efficiency |
|---------|-------------|---------|------------|--------------------|---------|------------|
| 1       | 56.54       | 1.00×   | 100%       | 33.65              | 1.00×   | 100%       |
| 2       | 22.43       | 2.52×   | 126%       | 14.18              | 2.37×   | 119%       |
| 4       | 10.92       | 5.18×   | 129%       | 5.91               | 5.69×   | 142%       |
| 8       | 5.01        | 11.28×  | 141%       | 3.51               | 9.58×   | 120%       |

Super-linear scaling (>100% efficiency) is expected: single-thread memory bandwidth is
the bottleneck (~1.6 GB features buffer), and multiple threads exploit additional memory
channels. Downsampled r2 scales slightly less well at 8 threads (120% vs 141%) because
the 1/8-volume ds-EDT finishes fast, leaving the serial boundary-downsampling pass and
trilinear interpolation loop as a proportionally larger fraction (Amdahl effect).

## Files Modified

- `include/cuda/edt_jfa.hpp` — Rewrote with packed uint32 kernels:
  `jfa_init_packed`, `jfa_step_3d_packed` (6/26-face), `jfa_writeback_aos`,
  `jfa_calc_distance_packed`, `edt_3d_jfa_level()` dispatcher (block 64×4×2)
- `include/cuda/edt_pba.hpp` — PBA+ port + `edt_3d_pba_downsampled()`,
  `pba_downsample_boundary_2x` kernel, `pba_downsample_dims()` helper
- `include/cuda/compensation_cuda.hpp` — Added `compensation_idw_nodist` (on-the-fly dist)
  and `compensation_idw_downsample` (trilinear interpolated downsampled d_neutral)
- `include/cuda/boundary_cuda.hpp` — Added `fill_sign_and_neutral_boundary_fused` kernel
- `test_cuda/test_compensation.cu` — `edt_method` CLI arg (0=chunk, 1=JFA, 2=PBA+,
  3=PBA+ optimized), per-stage timing, optimized path with downsampled R2

## Saved Artifacts

- nsys profiles: `/tmp/nyx512_l0_nsys.nsys-rep`, `/tmp/nyx512_l1_nsys.nsys-rep`
- Benchmark TSV: `/tmp/nyx512_jfa_stats.tsv`

## 2026-03-07 Session Audit (Codex)

This section records changes made in the March 7, 2026 Codex session so another AI can review/fix safely.

### User-requested goals in that session
1. Optimize method=3 memory/work
2. Add texture-based trilinear path
3. Keep old code path
4. Show performance proof

### Changes that were made during the session

#### `include/cuda/edt_pba.hpp`
- Modified `pba_extract_result` to allow nullable outputs (`d_packed_index`/`d_distance`) so distance-only extraction can skip index writes.
- Modified `edt_3d_pba_downsampled()` to remove `d_ds_packed` temp allocation and call extract with `nullptr` index.

#### `include/cuda/compensation_cuda.hpp`
- Added a new kernel `compensation_idw_downsample_tex(...)` using `tex3D` for hardware trilinear interpolation.

#### `test_cuda/test_compensation.cu`
- Added `edt_method=4` (texture path).
- Added conditional allocations in `run_cuda()` to avoid some buffers in optimized paths.
- Added fallback path for texture-launch failure.
- Added temporary env switch `PM_LEGACY_OPT3` for A/B comparisons.

### What happened in testing
- On this host, method=4 launch failed at runtime with:
  - `the provided PTX was compiled with an unsupported toolchain`
- This prevented valid texture-vs-manual performance conclusions on this machine.

### Restoration status
At user request, these session-introduced edits were reverted in source:
- `include/cuda/edt_pba.hpp` restored to non-null extract behavior and restored `d_ds_packed` temp usage.
- `include/cuda/compensation_cuda.hpp` texture kernel removed.
- `test_cuda/test_compensation.cu` restored to methods `0..3` only and original method=3 behavior.

### Important data-path clarification found during session
- For `/home/jp/data/nyx_512x512x512/velocity_x.f32`, value range is very large
  (`max=31866786`, `min=-50416856`), so `rel_eb=0.01` implies `abs_eb=822836.4`.
- That explains much lower PSNR (~44.76 dB) than prior notes (~54 dB), which likely used different scaling/input.
- **WRONG**: The ~44.76 dB is the quantization-only PSNR. After compensation the PSNR is ~54.17 dB — consistent with prior notes. The Codex session never observed the working compensation because it never rebuilt the binary after reverting source changes.

### 2026-03-07 Fix (Claude Code)

**Root cause:** Codex reverted the source files but did not rebuild the binary. The stale binary still
contained Codex's broken code (conditional allocations, method=4, etc.). Simply running `cmake --build`
from the reverted source produced a working binary.

**Verification after rebuild:**

| Method | Dataset | Total | EDT total | PSNR |
|---|---|---|---|---|
| PBA+ (method=2) | NYX 512³ | 116 ms | 96 ms | 54.170 dB |
| PBA+ opt (method=3) | NYX 512³ | 78 ms | 57 ms | 54.104 dB |
| PBA+ opt (method=3) | Hurricane | 15 ms | 11 ms | 50.02 dB |

All results match the pre-Codex benchmarks in this document. No source changes were needed.
