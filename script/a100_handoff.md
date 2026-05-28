# A100 GPU evaluation — handoff for next agent

This handoff is written for the Claude Code agent running on the A100 machine.
The work below was last validated on an RTX 4070 Laptop (8 GB, ~256 GB/s DRAM).
We need A100 numbers because the laptop's pipeline is HW-memory-bound on most
kernels — see "Why we moved to A100" below.

---

## FIRST THING TO DO: ask the user where the test data is

Test files used on the laptop:

| dataset | dims (fast→slow) | location on laptop | per-file size |
|---|---|---|---|
| NYX 512³ (6 fields: `velocity_x/y/z.f32`, `baryon_density.f32`, `dark_matter_density.f32`, `temperature.f32`) | 512 512 512 | `/home/jp/data/nyx_512x512x512/` | 512 MB each |
| Hurricane (13 raw + 7 log10 fields, `*.bin.f32` and `*.log10.bin.f32`) | 500 500 100 | `/home/jp/data/hurricane_100x500x500/` | 100 MB each |

**Ask the user for the data root paths on the A100 machine.** They will likely
differ (cluster scratch, shared filesystem, etc.). Once known, either:
- pass paths explicitly to the binary on the command line, or
- export `DATA=/path/to/file.f32` and use the helper script (below).

Do not assume any path — wait for the user's answer before running anything that
touches data.

---

## Why we moved to A100

On the RTX 4070 Laptop, the CUDA compensation pipeline runs at **75.7 ms total**
for NYX 512³ velocity_x rel_eb=0.01 with `edt_method=3` (best path).
`ncu` showed:

| Kernel | Duration | Bottleneck | HW utilization |
|---|---|---|---|
| `pba_boundary_sign_init` | 7.14 ms | L2 BW | **94%** |
| `pba_kernelFloodZ` (fine R1) | 9.14 ms | DRAM BW | 85% |
| `get_filtered_boundary` | 3.79 ms | L2 BW | 82% |
| `pba_extract_result` (coarse) | 0.84 ms | DRAM BW | 87% |
| `pba_extract_result` (fine, post-A1) | 6.80 ms | DRAM BW | 61% |
| `fill_sign_and_neutral_boundary_fused` | 7.62 ms | Latency (L1 hit 32%) | 45% DRAM |
| `compensation_idw_downsample` | 10.66 ms | Mixed (L2 hit 53%) | 66% DRAM |

~32 ms of the 76 ms is **at the HW memory ceiling on the 4070 Laptop**. To know
which optimization to chase (smem tile on compensation vs algorithmic refactor
of EDT R1), we need to see whether those kernels remain bandwidth-bound on an
A100. A100 has ~6–8× the DRAM bandwidth (HBM2/HBM2e), 40 MB L2 (vs ~36 MB),
192 KB shared mem per SM (vs 100 KB), and 40/80 GB HBM (vs 8 GB).

Expected outcomes:
- Total elapsed should drop to ~15–25 ms if everything scales with bandwidth.
- If the BW-bound kernels still report 85–94% utilization on A100, then the
  bottleneck is structural and we should pursue **method-5 hybrid (B1)** —
  skip fine PBA R1, use coarse PBA + local exact-EDT band near boundaries.
- If they fall to 30–50%, the kernels are SM-throughput-limited and we should
  pursue **shared-mem tiling (A4)** in `compensation_idw_downsample`.

---

## Build instructions (A100, sm_80)

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_ARCHITECTURES=80
# or "native" if building on the A100 host
cmake --build build --target test_compensation_cuda -j$(nproc)
```

If nvcc complains about the host compiler, add `-DCMAKE_CUDA_HOST_COMPILER=gcc-14`
(or whatever gcc the toolkit accepts). The laptop needed gcc-14 with CUDA 13.0.

---

## Run the baseline (after user provides data path)

Single run, edt_method=3 (best quality + speed):
```bash
./build/test_cuda/test_compensation_cuda <data_path>/velocity_x.f32 0.01 1 512 512 512 3
```

Argument layout: `<file> <rel_eb> <use_chunk> <dim0_fast> <dim1> <dim2_slow> <edt_method>`.

For NYX use `512 512 512`. For Hurricane use `500 500 100` (file is stored fast=500,
mid=500, slow=100). Watch the printed timing block:
```
Elapsed time: ...
StageTime boundary_detect: ...
StageTime edt_round1: ...
StageTime fill_sign: ...
StageTime neutral_boundary: ...
StageTime edt_round2: ...
StageTime compensation: ...
StageTime edt_total: ...
```

Run 5× and report median, plus PSNR (should be ~54.10 dB for NYX velocity_x at
rel_eb=0.01 — unchanged from CPU baseline; deviation indicates a build problem).

---

## Run the GPU sweep across all fields

```bash
bash script/run_gpu_sweep.sh
# produces: script/gpu_results.csv
```

The script iterates the 6 NYX fields × 3 ebs and the 20 Hurricane fields × 3 ebs
(312 runs, ~30 min). Commit `gpu_results.csv` once done.

---

## Run ncu on the top kernels

Performance counters often need elevated permission. Either:

a) Run with sudo (one-off):
```bash
sudo bash script/ncu_top_kernels.sh
```

b) Or set persistently (on a system you administer):
```bash
sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvprof.conf'
# reload nvidia kernel module or reboot
```

On a shared cluster the counters are usually already enabled — just run
`bash script/ncu_top_kernels.sh` directly. Output is `/tmp/ncu_pm_a1.csv`.

The script defaults to NYX velocity_x rel_eb=0.01 edt_method=3. Override via env:
```bash
DATA=/path/to/other.f32 W=512 H=512 D=512 REL_EB=0.005 METHOD=3 \
    sudo bash script/ncu_top_kernels.sh
```

---

## Comparison data

`script/gpu_results.csv` on `main` already contains the RTX 4070 Laptop numbers
(median ~78 ms before A1 / 75.7 ms after A1 for NYX velocity_x rel_eb=0.01).
After the A100 sweep, **append** rather than overwrite — add an extra column
`gpu` with values `rtx4070_laptop` or `a100_40gb` (or whatever the SKU is) so
both are preserved.

CPU reference baselines for PSNR are in
`script/comparison_results.csv` and `script/pareto_results.csv`.

---

## Repo state at handoff

- Branch: `main`, last commit `7886f9f` — "A1: skip unused fine-grid distance write
  in PBA EDT round 1" (validated −2.5 ms on RTX 4070 Laptop, PSNR bit-identical).
- Untracked when handing off: `script/ncu_top_kernels.sh`, `script/a100_handoff.md`
  (this file). Both committed in the same handoff commit.
- The `edt_method=3` path drops the unused fine-grid `distance_edge` float buffer
  (512 MB) — `pba_extract_result` now accepts `nullptr` for distance.

---

## Commit / PR rules (important)

- **Do NOT add `Co-Authored-By: Claude` (or any Claude trailer) in commit messages.**
  The user removed Claude from git contributors and considers it noise.
- Commit messages: imperative, single-purpose, include before/after numbers when
  the change is performance-related (style: see `git log --oneline -5`).
- Don't push to `main` of `origin` without the user's explicit OK if you're doing
  speculative changes; ask first.

---

## Open questions to bring back

After the A100 baseline run, the conversation with the user should resolve:

1. Are the BW-bound kernels still BW-bound on A100? (decides between A4 and B1)
2. Should `gpu_results.csv` add an A100 column or get a separate
   `gpu_results_a100.csv`?
3. Does the user want to attempt 1024³ on the A100 (was infeasible on 8 GB laptop)?
4. Is end-to-end cuSZ integration (compensation kernel called from cuSZ on-device
   data, no H2D/D2H) within scope for this trip?
