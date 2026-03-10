# TPDS Extension Notes

## New Contributions Beyond IPDPS '26

1. **Downsampled EDT round 2** — quality/performance tradeoff (1.51x CPU speedup, -0.064 dB on NYX 512^3)
2. **Full CUDA implementation** — PBA+, JFA, boundary/compensation kernels (3.7x cuSZ throughput)
3. **CPU optimizations** — packed indexes, flat32 indexes, dist-only EDT, coord-type auto-selection
4. **Controlled compensation / deterioration prevention** (new, see below)
5. **Plateau-width attenuation (c_plateau)** — new confidence factor, orthogonal to c_geom (see below)
6. **Weight function ablation** — validates IDW as theoretically and empirically optimal (see below)

### What Would Strengthen the Submission
- Multi-GPU / multi-node GPU scalability study
- Downsample factor Pareto analysis (2x, 4x, etc. vs PSNR vs speedup)
- Formal error bound proof for downsampled variant
- Larger scale datasets (1024^3+)
- End-to-end integration numbers (cuSZ + post-processing total throughput)

---

## Controlled Compensation (Deterioration Prevention)

Current pipeline blindly applies `sign * magnitude * eb` with no mechanism to detect when compensation hurts quality. Four approaches to address this:

### Approach 1: Local smoothness test
If a local region has high quant index variance (many distinct indices in a small neighborhood), the signal has real high-frequency content and compensation risks blurring real features.

```
local_variation = count of distinct quant_index values in k x k x k window
if local_variation > threshold:
    scale down compensation
```

### Approach 2: Neighbor consistency clamping
After computing comp[i], check whether the compensated value creates worse discontinuities than the original:

```
compensated = decompressed[i] + comp[i]
for each neighbor j:
    if |compensated - decompressed[j]| > |quant[i] - quant[j]| * 2 * eb + eb:
        reduce comp[i]
```

### Approach 3: Adaptive eb scaling from d_edge
Use d_edge (already computed) to derive per-point confidence:

```
confidence[i] = min(1.0, d_edge[i] / d_min)
effective_eb[i] = eb * confidence(i)
```

Points near boundaries (small d_edge) get reduced compensation. d_min is tunable (e.g., 2-3 voxels). Large uniform plateaus get full compensation; small fragmented plateaus get reduced.

### Approach 4: Re-quantization self-validation
Re-quantize the compensated result and check consistency:

```
comp_value = decompressed[i] + comp[i]
re_quantized = round(comp_value / (2*eb))
if re_quantized != quant_index[i]:
    clamp comp[i] to stay within the bin
```

Zero additional cost, guarantees error bound is never violated.

### Recommended combination for TPDS
- **Approach 4** as the hard guarantee (zero cost, provably bounded)
- **Approach 3** as soft control (uses existing d_edge, improves average quality)

---

## Weight Function Ablation (IDW Validated as Optimal)

### Background
The IPDPS paper uses IDW weight `w = d2/(d1+d2)` without justification beyond intuition. Three alternatives were implemented and tested to validate this choice:

- **IDW** (p=1): `w = d2/(d1+d2)` — default
- **Power-p IDW** (p=2): `w = d2²/(d1²+d2²)` — classic Shepard weighting; pushes weight toward 0 and 1 more sharply
- **Smoothstep**: `t = d2/(d1+d2); w = t²(3-2t)` — C¹-smooth, no parameters, same boundary conditions as IDW

### Error bound property
All three are naturally bounded in (0,1) and therefore satisfy the error bound `|C_i| ≤ eta*eb` without any clamping. Two-point RBF (inverse multiquadric) can overshoot this range and was rejected for this reason.

### Theoretical justification for IDW
Under a piecewise-linear field model, the quantization error at a plateau point `x` between edge `x_e` and neutral point `x_n` is:
```
error(x) ≈ eb * (x_n - x)/(x_n - x_e) = eb * d2/(d1+d2)
```
IDW with `p=1` is the exact maximum-likelihood estimator under this linear error model. It is not just a heuristic — it is the theoretically correct weight for the piecewise-linear assumption. **Use this as the justification in the paper.**

### Experimental results (NYX 512³, Hurricane)
Summary across all datasets and error bounds:
- **IDW (p=1)**: best or tied best on all configurations
- **Power-p IDW (p=2)**: −0.095 dB on baryon_density eb=1e-2; monotone degradation as p increases
- **Smoothstep**: −0.060 dB on baryon_density eb=1e-2; intermediate between IDW and p=2
- On well-behaved fields (velocity_x, temperature): all three methods nearly identical (Δ < 0.01 dB)

### Paper treatment
One paragraph in the method section + one ablation table. Framing: IDW is theoretically justified under the linear field model; empirical ablation confirms that departing from p=1 degrades quality monotonically. Do not present this as a contribution — present it as validation of the original choice.

---

## Plateau-Width Attenuation (c_plateau) — New Contribution

### Motivation
The existing `c_geom` factor (`r *= min(1, geo_scale/d1)`) reduces compensation when `d1` is large (point is far from the nearest edge on the **same-sign side**). But it does not penalize points where the total plateau width `d1+d2` is very large — situations where boundary geometry is so sparse that the EDT distances are unreliable.

`c_plateau` is orthogonal: it uses `d1+d2` (total EDT distance, both rounds) rather than just `d1`.

### Formula
```
c_plateau = min(1, plateau_cutoff / (d1+d2))
r *= c_plateau
```
Default cutoff: 20 voxels. In the downsampled path, `d1` is taken from the full-res EDT round 1 (same as c_geom).

### Implementation
```cpp
void set_plateau_attenuation(bool v);
void set_plateau_cutoff(double c);  // default 20.0
// CLI: --plateau_attenuation --plateau_cutoff
```

### Experimental results
On NYX 512³ with eb=1e-2, cutoff=10 voxels:
- **dark_matter_density**: harm −0.034 dB → −0.017 dB (halved)
- **baryon_density**: harm reduced similarly
- **velocity_x, temperature**: no change (dense boundaries → d1+d2 always < cutoff)

Cutoff=10 is more aggressive than the default of 20; tune based on Pareto analysis.

### Connection to c_geom
Both belong to the same geometric confidence family:
- `c_geom = min(1, geo_scale/d1)` — penalizes far-from-edge points (same-sign side)
- `c_plateau = min(1, plateau_cutoff/(d1+d2))` — penalizes points on wide plateaus overall

Combined: `r *= c_geom * c_plateau`. The two factors address different failure modes and are multiplicatively composable.

### Paper treatment
Present as part of the "reliability-controlled compensation" section alongside `c_sign` and `c_geom`. The unified confidence decomposition is:
```
C_i = c_sign_i * c_geom_i * c_plateau_i * C0_i
```
where `C0_i = s_i * eta * eb * w(d1_i, d2_i)`.

---

## Why Not Neural Networks

### Core arguments against CNN/NN baselines
1. **No training data needed** — EDT+IDW works on any dataset, any domain, any eb, out of the box. CNNs need domain-specific training data; scientists compress novel simulation data with no training set available.
2. **Guaranteed error bound** — with re-quantization check, EDT+IDW provably stays within eb. No NN can offer this without post-hoc clamping.
3. **CPU-viable** — 3.6s for 512^3 on CPU. An 8-layer CNN on 512^3 volumetric data is impractical on CPU; many HPC workflows lack GPUs at decompression (e.g., post-hoc analysis on login nodes).
4. **Zero amortization cost** — no training, no model storage, no PyTorch/TensorFlow dependency. Header-only C++ library.
5. **Deterministic and reproducible** — same input, same output, every platform. Critical for scientific computing.

### Strategy: do NOT include CNN as a baseline
Including one CNN baseline invites "why not U-Net / transformer / diffusion?" — a never-ending rabbit hole.

Instead:
- **Discuss qualitatively** in related work or discussion section. Cite representative papers, explain why they are fundamentally unsuitable for this problem.
- **Frame the contribution differently** — this method exploits specific knowledge of the quantization process (quant indices, eb, bin structure) for targeted compensation. A CNN treats input as a black box; EDT+IDW treats it as output of a known, invertible process. That is the "quantization-aware" distinction.
- **If a reviewer insists**, address with one CNN number in the rebuttal, not in the paper.

### Recommended baselines for TPDS
- No compensation (raw decompressed output)
- Simple smoothing (Gaussian filter, bilateral filter)
- EDT+IDW full-res EDT round 2
- EDT+IDW + downsampled EDT round 2 (speed/quality tradeoff)
- EDT+IDW + adaptive compensation control (new contribution)

This is a clean, self-contained evaluation. Smoothing baselines show why naive approaches fail (violate eb, blur real features). Variants show the design space explored.

---

## Codex Discussion

### Information model and what can actually be inferred
- The post-processing problem should be framed under the assumption that only the decompressed or quantized data, the error bound `eb`, and the quantization method are available at inference time.
- Under that assumption, exact recovery of the original error is impossible. The method is really choosing a good prior over plausible reconstructions, not inverting the original data exactly.
- This is why a quantization-aware method is defensible: it uses information that is actually available in deployment, instead of assuming access to the original field or true error map.

### TPDS extension: what is likely enough and what is not
- `CPU optimization + fixed downsample + GPU port` by itself is probably not enough if the paper reads like an implementation extension of the IPDPS paper.
- A stronger TPDS story is: `hierarchical or adaptive quantization-aware interpolation` plus `end-to-end high-throughput implementation`.
- The current downsampled second-EDT path is a strong seed contribution because it introduces a real quality/performance tradeoff, especially if it is elevated from a fixed flag into an adaptive method.
- The center of gravity should be:
  - adaptive or hierarchical downsampled `Dist2`
  - quality/performance analysis and ablation
  - integrated GPU pipeline, not just isolated kernels
  - stronger deterioration control

### Stronger deterioration control is a real research contribution
- The current pipeline has only weak safeguards. There is a global `eta`, some sign suppression in fast-varying regions, and a warning for homogeneous quantization, but no real mechanism to control when compensation should be reduced or skipped.
- Without the original data, it is impossible to guarantee "no PSNR or SSIM deterioration" everywhere.
- What can be controlled is deterioration risk through confidence-aware compensation.

### Reliability-controlled compensation
- A practical direction is to compute a local confidence `c_i in [0,1]` and scale compensation as:

```text
C0_i = s_i * eta_max * eb * w(d1_i, d2_i)
C_i  = c_i * C0_i
```

- A useful decomposition is:
  - `c_sign`: sign certainty from all differing neighbors, not just the first differing neighbor
  - `c_geom`: lower confidence when `d1 + d2` is very large, boundaries are sparse, or the plateau is huge
  - `c_ms`: multiscale confidence from disagreement between full-resolution and downsampled compensation
- Then:

```text
c_i = c_sign_i * c_geom_i * c_ms_i
```

- This turns the global `eta` into a local `eta_i = c_i * eta_max`.

### Local accept or reject logic
- After computing a candidate compensation, add a local backtracking rule:

```text
candidate = D'[i] + C_i
if creates_new_extremum(candidate) or violates_local_monotonicity(candidate):
    shrink C_i
```

- Good acceptance tests that do not require the original data:
  - reject updates that create new local extrema
  - reject updates that violate monotonicity implied by the local quantization-index ordering
  - reject updates that increase local roughness too much, such as discrete Laplacian or Hessian energy
- This is a meaningful extension because it directly addresses the negative-tail cases where the current method slightly hurts quality.

### Why downsampling and deterioration control fit together
- Downsampling does not have to be just a speed trick.
- The disagreement between full-resolution and downsampled compensation is itself a useful uncertainty signal.
- If both scales agree, confidence is high. If they disagree, the region is ambiguous and compensation should be reduced.
- This makes the downsampled path part of the quality-control story, not only the performance story.

### Neural networks: do not deny the quality advantage, position the tradeoff
- A modest CNN can plausibly remove these artifacts very well. That does not automatically mean it is the right replacement for the method.
- The strongest argument is not "CNNs are bad"; it is that they occupy a different operating point:
  - better quality may be achievable
  - training cost is high
  - inference cost is high
  - CPU deployment is poor
  - generalization across datasets, variables, compressors, and error bounds is uncertain
  - deterministic relaxed-error control is not natural without post-hoc projection

### Best response to the "why not neural?" reviewer question
- The cleanest answer is one compact comparison table, not a long textual defense.
- If feasible, compare against one small CNN only, under the same information budget at inference time:
  - inputs limited to `D'`, quantization-derived information, and `eb`
  - report SSIM and PSNR
  - report inference throughput and memory
  - report training cost
  - report cross-dataset or cross-error-bound generalization
  - clamp output to the same relaxed bound for fairness
- If the CNN still dominates quality after that, the result is still useful: it shows the learned upper bound while reinforcing the analytical method's advantages in portability, control, and HPC deployment.

### Recommended framing
- Neural methods are a high-quality, high-cost option.
- The proposed method is a training-free, quantization-aware, controllable, HPC-friendly option.
- For TPDS, that distinction is defensible if the paper clearly emphasizes:
  - adaptive multiresolution compensation
  - explicit deterioration control
  - end-to-end GPU and CPU performance
  - deployment without retraining
