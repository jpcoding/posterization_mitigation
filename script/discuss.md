# TPDS Extension Discussion

Building on `notes.md` and a close reading of the IPDPS '26 paper. The goal is to identify what makes a compelling journal extension vs. what reads as "we just ported it to GPU."

---

## What the IPDPS Paper Already Covers

- Algorithm: boundary detect → EDT1 → sign propagation → neutral boundary → EDT2 → IDW
- CPU parallelization: OpenMP (shared-memory) + MPI (distributed-memory, 3 strategies)
- Evaluation: 5 datasets, 2 compressors (cuSZ, cuSZp2), baselines = Gaussian/Uniform/Wiener filters
- Results: up to 108% SSIM improvement, 1.17x/1.34x compression ratio gains at matched SSIM
- Fixed global eta=0.9, gradient threshold=1.0, no adaptivity
- No GPU, no end-to-end integration, no NN comparison, no Bezier baseline

## What is NOT Enough for TPDS

A paper that says "we added CUDA and it's fast" will be rejected. GPU porting alone is an engineering contribution, not a research contribution. Similarly, "we downsampled EDT2 and it's 1.5x faster" is a nice optimization but not a paper by itself.

The IPDPS paper explicitly lists GPU implementation as future work — reviewers will expect it, but it can't be the *only* new thing.

---

## Proposed TPDS Story: Three Pillars

### Pillar 1: Adaptive Multiresolution Compensation

Elevate the downsampled EDT2 from a fixed 2x flag into a principled multiresolution framework.

**What to do:**
- Implement multiple downsample factors (1x, 2x, 4x, 8x) for EDT round 2
- Show Pareto curves: downsample factor vs. PSNR vs. SSIM vs. wall time, across all datasets
- Derive the quality bound: prove (or empirically bound) the maximum PSNR/SSIM degradation as a function of downsample factor
- On GPU: downsampled EDT2 may matter less since PBA+ is already fast, but it saves memory — show the memory tradeoff

**Why it's a research contribution:**
- It's not just "run EDT on a smaller grid." The trilinear interpolation back to full resolution introduces approximation error in the distance field, which propagates through IDW. Analyzing this error cascade is non-trivial.
- The optimal downsample factor may vary per dataset or per region — this opens the door to adaptive selection.

**Key experiment:** For each dataset × eb, find the maximum downsample factor that keeps SSIM within 0.1% of full-resolution. Show this varies by dataset, motivating adaptive selection.

### Pillar 2: Reliability-Controlled Compensation

The IPDPS paper's biggest weakness: no mechanism to prevent quality degradation. The method sometimes slightly hurts PSNR (acknowledged in the paper for S3D at small eb). A TPDS extension should fix this.

**Proposed approach — layered confidence:**

```
C_i = c_i * s_i * eta_max * eb * w(d1_i, d2_i)
```

where `c_i = c_sign * c_geom * c_ms`:

1. **c_sign** (sign certainty): The IPDPS method determines sign from a single neighbor gradient. Instead, poll all 6/26 neighbors with different quant indices. If they all agree on the sign → c_sign=1. If they disagree → c_sign drops. This handles corners and junctions where sign is ambiguous.

2. **c_geom** (geometric confidence): When d1+d2 is very large (huge plateau), the EDT distances are less reliable guides to the true error shape. Scale down:
   ```
   c_geom = min(1.0, (d1 + d2) / d_max)  // or saturating sigmoid
   ```
   Also: near domain boundaries, distances are truncated — reduce confidence there too.

3. **c_ms** (multiscale consistency): Compute compensation at both full-res and downsampled EDT2. If they disagree significantly at a point, that point is uncertain:
   ```
   c_ms = 1 - |C_fullres - C_downsampled| / (eta * eb)
   ```
   This elegantly ties Pillar 1 to Pillar 2 — downsampling serves double duty as both speedup and uncertainty estimator.

**Hard guarantee layer:** After applying confidence-scaled compensation, run the re-quantization check:
```
if round((D'[i] + C_i) / (2*eb)) != quant_index[i]:
    clamp C_i
```
This is O(N), zero-cost, and guarantees the relaxed error bound is never violated.

**Local acceptance tests** (optional, stronger):
- Reject compensation that creates new local extrema not present in D'
- Reject compensation that violates monotonicity implied by quant index ordering
- Reject compensation that increases discrete Laplacian energy beyond a threshold

**Why it's a research contribution:**
- Directly addresses the IPDPS paper's acknowledged limitation
- The confidence decomposition is novel — no prior work on EDT-based artifact mitigation has this
- The multiscale consistency signal (c_ms) is particularly interesting and publishable on its own

### Pillar 3: High-Throughput GPU Pipeline + End-to-End Integration

This is the engineering pillar — necessary but not sufficient alone.

**What to show:**
- Full CUDA pipeline: PBA+ EDT, boundary detection, sign propagation, compensation — all on GPU, no CPU-GPU transfers mid-pipeline
- Comparison: PBA+ vs JFA vs exact EDT on GPU (you have all three)
- End-to-end integration with cuSZ: total decompression+mitigation throughput vs. cuSZ alone
  - Key metric: what % overhead does mitigation add to the decompression pipeline?
  - If it's <20% overhead for significant SSIM gain, that's a strong result
- Memory analysis: peak GPU memory for each variant (full-res EDT2, downsampled EDT2, dist-only)
- Multi-GPU or at minimum single-GPU scalability across dataset sizes (256³ → 512³ → 1024³)

**Additional baselines to consider:**
- Bezier smoothing (Daoce, SC'24) — mentioned in IPDPS intro but not evaluated. Adding this baseline shows completeness.
- More compressors: SZp, FZ-GPU (IPDPS claims compatibility but didn't test them)

---

## Addressing the Neural Network Question

### Do NOT make CNN a primary baseline

The risk: including one CNN baseline opens "why not U-Net / transformer / diffusion?" — unbounded scope creep.

### Recommended strategy

**In the paper:**
- Related work section: cite 3-5 representative NN-based denoising/artifact removal papers
- Discussion section (1 paragraph): acknowledge that learning-based methods may achieve higher PSNR but explain why they occupy a fundamentally different operating point:
  - require domain-specific training data (scientists compress novel simulations)
  - cannot guarantee error bounds without post-hoc projection
  - impractical on CPU-only decompression nodes
  - add heavyweight framework dependencies (PyTorch, ONNX) to HPC stacks
  - not deterministic/reproducible across platforms
  - generalization across datasets × variables × compressors × error bounds is unproven

**If a reviewer demands it:**
- In rebuttal: show one compact table comparing a small 3D CNN (e.g., 8-layer with 3³ kernels) vs. EDT+IDW
- Same information budget at inference: inputs = D', quant info, eb
- Report: PSNR, SSIM, throughput (GB/s), training time, memory, cross-dataset generalization
- Clamp CNN output to relaxed error bound for fairness
- Even if CNN wins on PSNR, the table will show it loses on every deployment dimension

**Frame positively:** "Neural methods represent a high-quality, high-cost option. The proposed method is a training-free, quantization-aware, controllable, HPC-deployable option. These are complementary, not competing."

---

## Paper Outline Sketch

1. **Introduction** — Extend IPDPS intro with: adaptive compensation, deterioration control, GPU throughput
2. **Background & Related Work** — Add NN-based denoising/super-resolution discussion, Bezier smoothing baseline
3. **Problem Formulation** — Same as IPDPS but add the confidence/reliability-controlled formulation
4. **Method**
   - 4.1 Core algorithm (recap from IPDPS, condensed)
   - 4.2 Multiresolution EDT (new: downsample analysis, Pareto curves, error bounds)
   - 4.3 Reliability-controlled compensation (new: c_sign, c_geom, c_ms, acceptance tests)
5. **Implementation**
   - 5.1 CPU optimizations (packed indexes, flat32, dist-only EDT, coord auto-select)
   - 5.2 GPU pipeline (PBA+ EDT, kernel design, memory management)
   - 5.3 End-to-end integration with cuSZ
6. **Evaluation**
   - 6.1 Quality analysis: PSNR, SSIM, per-dataset breakdown, degradation cases fixed by controlled compensation
   - 6.2 Multiresolution tradeoff: Pareto curves across datasets and error bounds
   - 6.3 Performance: CPU single-node, GPU single-node, end-to-end with cuSZ
   - 6.4 Scalability: multi-node CPU (from IPDPS, extended), multi-GPU if feasible
   - 6.5 Comparison with Bezier smoothing and simple filters
7. **Discussion** — NN positioning, limitations, when the method helps vs. when it doesn't
8. **Conclusion**

---

## Priority and Effort Estimate

| Contribution | Research value | Implementation effort | Priority |
|---|---|---|---|
| Reliability-controlled compensation | High | Medium (mostly in compensation kernel) | 1 |
| Multiresolution Pareto analysis | High | Low (parameterize existing downsample) | 2 |
| End-to-end cuSZ integration numbers | Medium-High | Medium (integration + benchmarking) | 3 |
| GPU pipeline writeup + analysis | Medium | Low (already implemented) | 4 |
| CPU optimization writeup | Low-Medium | Low (already implemented) | 5 |
| Bezier smoothing baseline | Medium | Medium (need to implement or obtain code) | 6 |
| Additional compressors (SZp, FZ-GPU) | Low-Medium | Medium | 7 |
| Multi-GPU | Medium | High | 8 (skip if time-limited) |

---

## Open Questions

1. **c_ms computation cost:** Computing full-res AND downsampled compensation to get the disagreement signal means running EDT2 twice. Is there a cheaper proxy? Possibly: compare d_edge at full-res vs. upsampled-from-downsampled d_edge to estimate local EDT sensitivity.

2. **Adaptive downsample factor selection:** Can this be done per-region cheaply? E.g., regions with sparse boundaries → higher downsample factor is safe. Regions with dense boundaries → keep full resolution. This would require a pre-pass over the boundary map density.

3. **Bezier baseline availability:** Is Daoce's SC'24 code publicly available? If not, reimplementation effort may not be worth it. Check if they released code.

4. **TPDS page limit:** IEEE TPDS has no strict page limit (typically 14-16 pages). This gives room for the deeper analysis that the IPDPS 10-page limit didn't allow.

5. **How much IPDPS content can be reused?** Typically ~30% overlap is acceptable. The core algorithm description (Section 4.1) can be condensed but not copy-pasted. All experiments must be new or substantially extended.

## Codex Review

### 1. Three pillars: right decomposition or not?

The current three-pillar decomposition is close, but I would reorganize it slightly because right now it risks reading as `one algorithmic tweak + one heuristic control layer + one implementation section`.

My recommendation is:

- **Pillar A: Adaptive Reliable Compensation**
  - This merges `adaptive multiresolution` and `reliability-controlled compensation` into one algorithmic contribution.
  - The reason is that multiresolution is not only a speed optimization. It should also drive uncertainty estimation and selective compensation.
  - If they stay separate, a reviewer can say Pillar 1 is just a speed knob and Pillar 2 is just a bag of safeguards.

- **Pillar B: End-to-End High-Throughput Implementation**
  - This includes CPU memory optimizations, GPU kernels, and pipeline integration with cuSZ.
  - Keep this as the systems pillar.

- **Cross-cutting theme: controlled quality-performance tradeoff**
  - This is the unifying message, not a third independent pillar.

So I would not present:

- multiresolution
- reliability control
- GPU pipeline

as three peers.

I would present:

- **new method:** adaptive reliable multiresolution compensation
- **new system:** end-to-end CPU/GPU implementation
- **new analysis:** quality-performance-memory Pareto behavior and relaxed-error control

That structure is cleaner and more journal-like.

### 2. Confidence fusion: product, min, or something else?

Pure multiplication is defensible, but I would not recommend using it as the final form without modification.

The main options behave like this:

- **`min(c_sign, c_geom, c_ms)`**
  - Good: conservative and easy to explain.
  - Good: one strong warning signal can veto compensation.
  - Bad: extremely brittle to noisy estimates.
  - Bad: creates discontinuous confidence maps and may zero out large regions due to one unstable component.

- **Pure product `c_sign * c_geom * c_ms`**
  - Good: smooth and cheap.
  - Good: captures compounding uncertainty.
  - Bad: can be over-pessimistic if all terms are moderately below 1.
  - Bad: calibration becomes opaque because three reasonable terms can multiply into an overly small confidence.

- **Weighted geometric mean `c = c_sign^a * c_geom^b * c_ms^c`**
  - Good: same monotone behavior as multiplication but easier to calibrate.
  - Good: avoids the excessive shrinkage of pure product.
  - Good: keeps the interpretation that low confidence in one factor matters multiplicatively.
  - Bad: introduces weights that must be justified.

- **Weighted arithmetic mean**
  - Bad fit here.
  - A high score on one factor can hide a severe weakness on another factor.
  - That is especially dangerous for sign ambiguity.

My recommendation:

- Treat **sign ambiguity as a hard or near-hard gate**.
- Treat geometry and multiscale consistency as **soft modifiers**.

Concretely:

```text
c_i = c_sign_i * (c_geom_i^alpha * c_proxy_i^(1-alpha))
```

with `alpha` in `[0.3, 0.7]`.

This is better than:

```text
c_i = c_sign_i * c_geom_i * c_ms_i
```

because:

- if `c_sign` is low, compensation should indeed collapse quickly
- if `c_geom` and `c_ms` are both moderate, you do not want them to over-attenuate by simple multiplication

Failure modes to watch:

- `min(...)`: too many false vetoes
- pure product: over-shrinking everywhere, especially on difficult datasets
- arithmetic mean: wrong-sign or aliased cases can slip through because one strong term masks another weak term

### 3. Cheaper proxy for multiscale disagreement

I agree with the concern: computing both full-resolution and downsampled `EDT2` defeats the point if this becomes the default path.

I would not use true `c_ms` as drafted. I would replace it with a cheaper **downsampling alias-risk proxy** computed from `B2` before running `EDT2`.

The best low-cost proxy is:

- downsample `B2`
- upsample the coarse representation back to fine resolution
- compare the reconstructed coarse geometry with the original `B2`

This gives a local aliasing indicator without running `EDT2` twice.

Useful blockwise signals:

- **Boundary occupancy ratio** in each `s x s x s` block
  - very sparse or very dense blocks are safer
  - mixed occupancy is riskier

- **Number of connected components or face transitions** in the block
  - multiple sign-flip structures inside one coarse block mean downsampling is unsafe

- **Local plane-fit residual on `B2` voxels**
  - if the sign-flipping boundary inside a block is well approximated by a plane, coarse EDT is likely safe
  - if residual is high, the geometry is too curved or fragmented

- **Blockwise reconstruction mismatch**
  - compare fine `B2` with upsampled coarse `B2`
  - use local XOR fraction or mismatch density as the proxy

If I had to pick one practical proxy, I would use:

```text
c_proxy = 1 - local_mismatch(B2, upsample(downsample(B2)))
```

possibly combined with local plane-fit residual.

Why this is better:

- no second full-resolution EDT
- directly measures geometry lost by downsampling
- tied to the real source of downsample error: aliasing of `B2`, not `Dist2` itself

So I would rename `c_ms` to something like `c_ds` or `c_alias` unless you truly compute two scales of compensation.

### 4. Does re-quantization guarantee `(1+eta) * eb`?

No. By itself, the re-quantization self-check does **not** guarantee the relaxed bound `(1 + eta) * eb` for `eta < 1`.

Let:

- original value be `x`
- decompressed value be `y = 2 q eb`
- compensated value be `y' = y + C`

From the compressor guarantee:

```text
|x - y| <= eb
```

If re-quantization says `y'` still maps to the same quantization index `q`, then for a nearest-neighbor quantizer:

```text
|y' - y| <= eb
```

up to endpoint/tie details.

Then by triangle inequality:

```text
|x - y'| <= |x - y| + |y - y'| <= eb + eb = 2 eb
```

So same-bin validation gives a worst-case guarantee of approximately:

```text
|x - y'| <= 2 eb
```

That corresponds to the relaxed bound with `eta = 1`, not general `eta`.

Therefore:

- re-quantization is **sufficient** for `2 eb`
- re-quantization is **not sufficient** for `(1 + eta) eb` when `eta < 1`
- re-quantization is also **not necessary** for `(1 + eta) eb`

What actually guarantees `(1 + eta) eb` is:

```text
|C| <= eta * eb
```

because then:

```text
|x - y'| <= |x - y| + |y - y'| <= eb + eta * eb = (1 + eta) eb
```

So the correct interpretation is:

- **hard relaxed-bound guarantee:** clamp compensation magnitude so `|C_i| <= eta_i * eb`
- **optional same-bin check:** additional conservatism to keep the update inside the original quantization cell

But do not claim that same-bin validation alone proves `(1 + eta) eb`.

One more important implication:

- if you already enforce `|C_i| <= eta_i * eb` with `eta_i <= 1`, then the relaxed-bound proof is already done
- re-quantization then becomes a shape-preservation heuristic, not the primary error-bound argument

### 5. Experimental design for Pareto analysis

The right design should be broad enough to show stability, but not so broad that it turns into an unreadable parameter sweep.

I would structure it in two levels.

**Core Pareto study**

- Downsample factors: `1x, 2x, 4x, 8x`
  - If `8x` is catastrophic on some datasets, keep it anyway. A failed point helps define the frontier.
- Datasets: 4 representative datasets for quality
  - one smooth field
  - one highly structured field
  - one field with large homogeneous regions
  - one turbulence or fine-feature field
- Error bounds: 3 values per dataset
  - low: little posterization
  - medium: where your method is most useful
  - high: severe artifact regime

That gives:

```text
4 datasets x 3 eb values x 4 factors = 48 operating points
```

which is enough for a convincing Pareto section.

**Extended throughput study**

- Use 2-3 large 3D workloads
- Include `256^3`, `512^3`, and `1024^3` if feasible
- Evaluate CPU and GPU separately

Recommended axes:

- **Primary Pareto plot**
  - x-axis: wall time or inverse throughput overhead relative to full-resolution mitigation
  - y-axis: `Delta SSIM` or `Delta PSNR` relative to full-resolution mitigation

- **Secondary Pareto plot**
  - x-axis: peak memory
  - y-axis: quality metric

- **Optional application-facing plot**
  - x-axis: end-to-end decompression throughput
  - y-axis: final SSIM

I would avoid plotting absolute SSIM alone for the Pareto story. The cleaner view is:

- how much quality do you lose relative to full-res?
- how much speed or memory do you gain?

That is the true tradeoff.

Also include one table:

- for each dataset and `eb`, report the **largest downsample factor** whose quality stays within a tolerance of full-res
- e.g. within `0.005` absolute SSIM or `0.2 dB` PSNR

That table motivates adaptivity much better than a wall of curves.

### 6. Is there a theoretical bound on distance error from downsampling?

There is a reasonable path to an approximate result, but it needs assumptions. Without assumptions, topology changes from downsampling can make a clean global bound impossible.

The key observation is:

- distance transforms are 1-Lipschitz with respect to perturbations of the underlying set in Hausdorff distance

If `S` is the full-resolution sign-flipping boundary and `S_hat` is the downsampled-and-reconstructed version, then:

```text
|dist(x, S) - dist(x, S_hat)| <= d_H(S, S_hat)
```

for every point `x`, where `d_H` is the Hausdorff distance.

So the problem reduces to bounding:

```text
d_H(S, S_hat)
```

If you downsample by factor `s` and assume:

- the boundary is sufficiently smooth locally
- the local feature size is larger than the coarse cell
- there is no merging of multiple unrelated surfaces inside one coarse block

then a first-order bound of the form:

```text
d_H(S, S_hat) <= O(s)
```

is plausible, with a constant on the order of the coarse-cell diagonal.

In 3D, a coarse cell of side `s` has half-diagonal:

```text
(sqrt(3) / 2) * s
```

So a practical statement is:

```text
|d_full(x) - d_coarse(x)| <= K * s
```

under smooth-boundary and no-aliasing assumptions.

Then because your IDW weight simplifies to:

```text
w(d1, d2) = d2 / (d1 + d2)
```

the compensation sensitivity to `d2` perturbation is:

```text
dw/dd2 = d1 / (d1 + d2)^2
```

which is bounded and decays away from the boundary.

So the compensation error induced by a distance error `delta` is first-order bounded by:

```text
|Delta C| <= eta * eb * d1 / (d1 + d2)^2 * |delta|
```

This is a nice theoretical direction because it says:

- downsampling error in `Dist2` matters most in certain geometric regions
- the induced compensation error is still controlled by the IDW sensitivity

Important caveat:

- if OR-downsampling merges nearby sign-flipping surfaces or destroys thin structures, the smooth-set assumption fails
- in that case, a clean global `O(s)` bound is not credible

So I would present this as:

- an **approximate error analysis under local smoothness and separation assumptions**
- not a universal theorem

That is still valuable enough for TPDS.

### 7. Concerns about the outline

The outline is generally good, but there are a few changes I would make.

First, I would add an explicit early statement of **what is new relative to the IPDPS version**. This can be in the last paragraph of the introduction rather than a standalone section, but it needs to be very explicit.

Second, I would not let the paper read like:

- recap
- optimization 1
- optimization 2
- optimization 3

That structure is dangerous for a journal extension.

I would adjust the ordering to:

1. Introduction
2. Background and Related Work
3. Problem Formulation and Limitations of the Conference Method
4. Adaptive Reliable Multiresolution Compensation
   - recap only what is needed
   - define the new reliability and multiresolution pieces
   - include the relaxed-bound argument here
5. Theoretical and Design Analysis
   - downsampling error discussion
   - confidence design rationale
6. End-to-End CPU/GPU Implementation
7. Evaluation
   - quality
   - ablation and sensitivity
   - Pareto tradeoff
   - end-to-end performance
   - baselines
8. Discussion
9. Conclusion

The biggest missing piece in the current sketch is a dedicated **ablation and sensitivity** subsection. A TPDS reviewer will absolutely want:

- full-res vs downsampled
- with and without reliability control
- contribution of each confidence term or proxy
- sensitivity to the downsample factor
- sensitivity to any new thresholds or weights

Without that, the new method will look under-justified.

One more point:

- if multi-GPU is not ready, do not over-promise it in the outline
- reviewers are less annoyed by a missing feature than by a promised but thin section

### 8. Single weakest point a reviewer would attack

The single weakest point in the current TPDS plan is that the new algorithmic contribution could still look like a **collection of heuristics layered on top of the conference method**, rather than a crisp new method.

In particular, a reviewer is likely to say:

- downsampling is an optimization
- GPU is an implementation
- confidence terms are heuristic knobs

and then ask:

- where is the single, principled journal-level advance?

That is the core vulnerability.

To reduce that risk, the paper needs one clean sentence of novelty:

- the conference paper estimated compensation from full-resolution geometry
- the journal paper introduces an **adaptive reliable multiresolution compensation framework** that jointly controls quality deterioration and computational cost, then realizes it in an end-to-end CPU/GPU pipeline

In other words, the weakest point is not GPU or downsampling themselves. It is **novelty fragmentation**.

If you make the method look like:

- `+ c_sign`
- `+ c_geom`
- `+ c_ms`
- `+ clamp`
- `+ downsample`
- `+ GPU`

then a reviewer can dismiss it as engineering accumulation.

If instead you make it look like:

- one new adaptive reliable multiresolution framework
- one clear guarantee from bounded compensation magnitude
- one end-to-end implementation that exposes the tradeoff

then the story becomes much harder to attack.

---

## Claude Follow-up Questions (Round 2)

### Q1. The guarantee is already satisfied by construction — does the re-quantization check add anything?

In the current IDW formula:

```text
magnitude = (1/d1) / (1/d1 + 1/d2)   ∈ (0, 1)
C_i = sign * magnitude * eta * eb
```

Since magnitude is always in (0,1) and eta <= 1, we already have `|C_i| < eta * eb` by construction, without any clamping. Codex clarified that `|C_i| <= eta * eb` is exactly the condition that proves `(1+eta)*eb`. So the relaxed error bound proof is already done implicitly by the IDW formula itself.

This means re-quantization is not needed for the error bound argument at all. Is the re-quantization check then useful only as a heuristic to prevent compensation from crossing quantization bin boundaries — which is a separate, weaker goal? Or is there a case where magnitude could actually exceed 1 or the formula breaks down (e.g., d1 or d2 near zero)?

### Q2. How exactly is c_sign computed, and does it replace the gradient threshold?

The IPDPS method uses a gradient threshold of 1.0 to suppress compensation in fast-varying regions: if the quant index changes by more than 1 between adjacent cells, sign is set to 0. Codex suggested c_sign = sign certainty from polling all differing neighbors.

What is the concrete formula? Options:

- fraction of neighbors that agree on the sign: `c_sign = (agreeing neighbors) / (total differing neighbors)`
- binary: 1 if all agree, 0 if any disagree
- something based on the spread of neighbor quant indices

And does c_sign fully replace the gradient threshold, or do both coexist? If c_sign is continuous, it could gracefully handle corners and junctions where the old binary threshold was fragile.

### Q3. How is the adaptive downsample factor actually selected per-region?

Codex said regions with sparse boundaries are safer for higher downsample factors, and dense regions need full resolution. But the current pipeline doesn't have a "per-region downsample factor" — it applies one global factor. Making it truly adaptive requires:

1. A pre-pass computing local boundary density from B2
2. A decision per block: which factor to use
3. Either running EDT2 at multiple resolutions and stitching, or doing something smarter

Is there a practical implementation strategy here? Running multiple EDT2 passes at different resolutions seems expensive. One option: always run downsampled EDT2, but use c_alias to scale down compensation in high-risk blocks rather than switching to full-res EDT. That way "adaptive" means adaptive confidence rather than adaptive resolution. Is that an equivalent or weaker formulation?

### Q4. The c_alias proxy computation — what is the block size s and how is it parallelized on GPU?

Codex proposed:

```text
c_proxy = 1 - local_mismatch(B2, upsample(downsample(B2)))
```

The block size `s` is unspecified. For a 2x downsample, s=2. For 4x, s=4. The mismatch is XOR fraction within the block. But on GPU:

- downsample + upsample of B2 is cheap (single pass each)
- the XOR comparison is cheap per voxel
- but what is the smoothing radius for the local mismatch? Is it per-voxel (each voxel gets the mismatch of its owning block), or is it smoothed across a neighborhood?

If it is purely blockwise, c_proxy has a block-granular staircase pattern, which could introduce discontinuities in the compensation field. Does that matter? And is the whole c_alias computation a single extra kernel, or does it require multiple passes?

### Q5. Can all confidence terms be fused into a single GPU kernel pass?

After computing d1 (EDT1), B2, and running EDT2 downsampled, we need c_sign, c_geom, c_alias, and the final compensation. The question is whether these can all be computed in one fused kernel or whether they require separate passes:

- c_sign requires reading all 6 (or 26) neighbors' quant_index — local memory, one pass
- c_geom requires d1 and d2 — already available
- c_alias requires the downsample+upsample of B2 — a separate pass or precomputed array
- final compensation: one multiply and clamp

Fusing everything into one compensation kernel (after precomputing c_alias) seems feasible. Would the memory access pattern for c_sign (26-neighbor stencil) be the bottleneck, or is the whole thing bandwidth-bound?

### Q6. What is the right "one clean sentence" of novelty for the abstract?

Codex identified novelty fragmentation as the core vulnerability. Trying to draft the sentence:

> "We introduce adaptive reliable multiresolution compensation, a framework that jointly controls quality deterioration and computational cost in quantization-aware post-processing by using multiscale geometry to estimate local compensation confidence and adaptively scale the correction field."

Is this the right framing? Two concerns:
1. "Multiscale geometry" may overstate it if the downsampled path is just a fixed 2x or 4x flag
2. "Adaptively scale" is vague — should it say "adaptively suppress" instead?

What would make this sentence crisp enough to anchor the paper?

### Q7. What is the minimum ablation table that satisfies a TPDS reviewer?

Codex said an ablation section is critical. The minimum set seems to be:

| Configuration | PSNR | SSIM | Time |
|---|---|---|---|
| No compensation (baseline) | | | |
| IPDPS method (full-res, no confidence) | | | |
| + c_sign only | | | |
| + c_geom only | | | |
| + c_alias only | | | |
| + all confidence terms | | | |
| + downsampled EDT2 (2x) | | | |
| + downsampled EDT2 (4x) | | | |
| Full proposed method | | | |

That is 9 rows. Is this overkill? Or is it the right granularity to show that each confidence term contributes independently and the downsample factor is a controlled tradeoff? Also: should this be on one dataset (NYX 512^3) for space reasons, or across multiple datasets?

### Q8. Is there a risk that c_geom actively hurts quality on large uniform datasets?

c_geom as proposed reduces confidence when d1+d2 is very large (huge plateau). But on datasets like JHTDB at fine scales, huge plateaus correspond to regions where the true signal is genuinely smooth and flat — exactly where compensation is most reliable and beneficial. Reducing confidence there could suppress correct compensation in the most well-behaved regions.

Is the intent of c_geom actually the opposite of what was described? Should large d1+d2 mean *higher* confidence (the plateau geometry is clear and well-behaved) rather than lower? Or is the concern something else — that when d1+d2 is very large, the IDW magnitude is close to 0 or 1 everywhere and the formula becomes sensitive to small errors in d2?

## Codex Review Round 2

### Q1. Is the relaxed error bound already satisfied by IDW construction?

Yes, for the current IDW path the relaxed bound is already satisfied by construction, assuming the implementation keeps the present form.

You currently have:

```text
w = (1/d1) / (1/d1 + 1/d2) = d2 / (d1 + d2)
C_i = sign * w * eta * eb
```

with:

- `d1 >= 0`
- `d2 >= 0`
- in the actual code, both are shifted by `+0.5`, so really `d1, d2 >= 0.5`

Therefore:

```text
0 < w < 1
```

strictly in the current implementation, because both denominators are positive finite numbers.

So:

```text
|C_i| < eta * eb
```

and the relaxed bound follows immediately:

```text
|x_i - (D'_i + C_i)| <= |x_i - D'_i| + |C_i| <= eb + eta * eb = (1 + eta) * eb
```

So on the IDW branch:

- the relaxed bound proof is already done
- re-quantization is not needed for the error-bound argument

What re-quantization adds is different:

- it discourages crossing the original quantization bin midpoint
- it can act as a shape-preservation or bin-consistency heuristic
- it may reduce some harmful updates near ambiguous boundaries

But it is **not** the core proof of the relaxed error bound.

Could the formula ever break?

- If you removed the `+0.5` offsets and allowed `d1 = 0` or `d2 = 0`, the formula would still be bounded in the limiting sense, but you would need careful handling to avoid division-by-zero.
- In the current code, the `+0.5` shift avoids singularities cleanly.
- On the RBF path, the bound is also enforced because the interpolant is explicitly clamped to `[-1, 1]` before multiplying by `eta * eb`.

So the right way to write this in the paper is:

- **relaxed error bound:** guaranteed by bounded compensation magnitude
- **optional same-bin validation:** a heuristic for conservative compensation, not the main correctness argument

One warning: if later you introduce confidence rescaling, alternative interpolation, learned surrogates, or other weighting formulas, keep the bounded-magnitude property explicit. Do not rely on it being “obvious.”

### Q2. How should `c_sign` be computed, and does it replace the gradient threshold?

I would make `c_sign` concrete and mostly replace the old threshold, but keep a very light hard veto for pathological cases if needed.

The cleanest formula is vote imbalance from all differing neighbors:

```text
V+ = number of neighbors with q_n > q_i
V- = number of neighbors with q_n < q_i
V  = V+ + V-
```

Then:

```text
s_i      = sign(V+ - V-)
c_sign_i = |V+ - V-| / V
```

with the convention `c_sign_i = 0` if `V = 0`.

Behavior:

- all differing neighbors agree: `c_sign = 1`
- balanced corner or junction: `c_sign = 0`
- mild disagreement: intermediate value

This is better than “first differing neighbor wins” because it is stable at corners and interfaces.

You can optionally weight votes by jump magnitude:

```text
V+ = sum_{q_n > q_i} min(|q_n - q_i|, tau_q)
V- = sum_{q_n < q_i} min(|q_n - q_i|, tau_q)
```

but I would start with unweighted votes. Simpler is better unless the weighted version clearly helps.

Should it replace the gradient threshold?

Mostly yes. The old threshold was trying to detect ambiguity or fast-varying regions with a binary rule. `c_sign` is a better version of that idea.

My recommendation:

- let `c_sign` be the main sign-confidence mechanism
- keep only a weak hard veto for extreme jump cases if needed, e.g. if `max_n |q_n - q_i| > J_max`

If you keep both as strong mechanisms, the reviewer will ask why two overlapping ambiguity detectors are needed.

So the clean story is:

- conference version: binary gradient threshold
- journal version: continuous sign confidence from neighborhood agreement

That is a genuine methodological improvement.

### Q3. How should adaptive downsample factor be implemented practically?

True per-region adaptive resolution is possible, but it is much more complicated than it sounds because EDT is global.

That means if you say:

- block A uses factor 4
- block B uses factor 1

you immediately have to deal with:

- cross-block nearest-boundary interactions
- stitching distance fields across refinement levels
- halo sizes for local refinement

So I would separate two notions:

#### Option A: global factor + local reliability control

This is the practical option.

Pipeline:

1. Choose one global downsample factor `s` for a given run.
2. Run coarse `EDT2` once at that factor.
3. Compute a local alias-risk or reliability map from `B2`.
4. Suppress compensation in risky regions using confidence.

This is **not equivalent** to true adaptive resolution, but it is the right first implementation because:

- cheap
- easy to explain
- easy to benchmark
- preserves the main speed benefit

If you do this, call it:

- **reliability-controlled multiresolution compensation**

not:

- per-region adaptive-resolution EDT

That distinction matters.

#### Option B: true local refinement

If you really want per-region adaptivity, the most practical strategy is hierarchical refinement:

1. Run a coarse global `EDT2` at factor `s_max`, e.g. `4x`.
2. Compute a risk map from `B2` and flag risky macroblocks.
3. For flagged macroblocks only, recompute a finer `EDT2` on an expanded ROI with halo.
4. Overwrite the coarse distances inside the block with the refined solution.

This is feasible, but only if you define a conservative halo. A reasonable halo heuristic is:

```text
halo >= local coarse distance + safety margin
```

Otherwise the local ROI may miss the true nearest sign-flip boundary outside the tile.

My advice:

- for the TPDS paper, do **Option A**
- mention true local refinement as future work unless you actually implement it robustly

So yes: your suggestion

- “adaptive means adaptive confidence rather than adaptive resolution”

is weaker than true adaptive resolution, but much more practical and probably the right scope.

Just do not oversell it.

### Q4. How should `c_alias` be implemented on GPU, and does blockwise staircasing matter?

The natural block size is the downsample factor itself.

So:

- if factor is `2x`, block size is `2`
- if factor is `4x`, block size is `4`

But I would not assign a raw blockwise constant confidence to every fine voxel. That does risk staircase artifacts in the confidence map.

The better implementation is:

1. Downsample `B2` to coarse grid by OR-pooling.
2. For each coarse cell, compute a mismatch score relative to the fine `B2` inside the corresponding block.
3. Store one coarse confidence value per coarse cell.
4. Trilinearly upsample that coarse confidence field back to full resolution.

That gives a smooth confidence map without a separate smoothing filter.

Concretely:

- **Kernel 1:** OR-downsample `B2`
- **Kernel 2:** compute coarse-cell mismatch ratio
  - each coarse cell counts the fraction of fine voxels in its block where `B2_fine` differs from the coarse replicated value
- **Kernel 3:** trilinearly upsample the coarse confidence field to fine grid

You can fuse steps 2 and 3 partially, but it is probably not worth making the code harder.

Does blockwise staircasing matter?

Yes, if you use raw block constants it can imprint weak grid artifacts into the compensation amplitude, especially at larger factors.

That is why I would upsample the coarse confidence field with trilinear interpolation. It is cheap and matches the rest of the multiresolution story.

### Q5. Can all confidence terms be fused into one GPU pass?

Mostly yes, but I would not force everything into one kernel if it hurts code structure or memory behavior.

The important split is:

- **precompute arrays that already require separate passes**
  - `d1`
  - `d2`
  - `c_alias` or other coarse confidence
  - optionally `c_sign`

- **final fused compensation kernel**
  - load `d1`, `d2`, `c_alias`, sign/confidence
  - compute `c_geom`
  - combine confidences
  - compute final compensation
  - write output

The one thing I would avoid is recomputing a 26-neighbor sign stencil inside the final compensation kernel. That is unnecessary pressure on memory bandwidth and cache locality.

Better:

- compute `c_sign` in the boundary/sign-map pass when you are already reading neighbors
- store it as `float` or quantized `uint8`
- let the final compensation kernel remain simple and bandwidth-friendly

So the practical answer is:

- yes, the **final confidence fusion and compensation** should be one kernel
- no, the whole pipeline should not be contorted to make every upstream feature extraction part of that same kernel

And yes, the whole pipeline remains mostly bandwidth-bound. That is another reason to avoid stuffing an extra stencil into the last pass.

### Q6. What is the cleanest novelty sentence?

Your draft sentence is directionally right, but it is still slightly abstract and slightly overcommits to “multiscale geometry” if the implementation remains mostly global-factor plus local suppression.

I would make it more concrete and closer to the actual mechanism.

Best method-level sentence:

> We introduce reliability-controlled multiresolution compensation, which accelerates quantization-aware post-processing by approximating the second distance transform on a coarse grid and selectively suppressing compensation in regions where the coarse geometry is unreliable.

Why this is stronger:

- says exactly what is approximated
- says exactly how quality loss is controlled
- does not overclaim true per-region adaptive-resolution EDT

If you want an abstract-ready two-sentence contribution pair, I would use:

> We introduce reliability-controlled multiresolution compensation, which accelerates quantization-aware post-processing by approximating the second distance transform on a coarse grid and selectively suppressing compensation where the coarse geometry is unreliable. We further realize this method in an end-to-end CPU/GPU pipeline that exposes a controllable quality-performance-memory tradeoff while preserving the relaxed error guarantee.

If later you truly implement local factor selection, then you can strengthen the wording to “adaptively selects the resolution.” Until then, avoid that phrase.

Also, “selectively suppressing” is better than “adaptively scaling” because it is more concrete and reviewer-proof.

### Q7. What is the minimum ablation table that satisfies a TPDS reviewer?

Nine rows in one table is probably too much if that table is the only ablation artifact. It mixes two different questions:

- which confidence terms matter?
- what is the downsample tradeoff?

Split them.

#### Minimum ablation set I would trust

**Table A: methodological ablation**

- No compensation
- IPDPS full-resolution method
- Full-resolution + reliability control
- Downsampled only
- Full proposed method

That is 5 rows and answers the big questions.

**Table B: confidence decomposition**

- Full-resolution + `c_sign` only
- Full-resolution + `c_sign + c_geom`
- Full-resolution + all confidence terms

That is 3 rows and isolates the confidence design.

**Figure C: factor sweep**

- factors `1x, 2x, 4x, 8x`
- plot quality vs time

That answers the multiresolution question much better than another table.

Should this be one dataset or multiple datasets?

For the decomposition table, one dataset is acceptable **only if** it is clearly chosen as a challenging case and you say so.

But I would strongly prefer:

- one challenging dataset
- one benign dataset

or:

- one per-term ablation on a challenging dataset
- one aggregated summary across all datasets for the final method

If you show per-term ablation on only NYX, a reviewer may reasonably say the knobs were tuned to one field.

So the minimal defensible setup is:

- detailed ablation on one challenging case
- final method validated across all datasets elsewhere

### Q8. Does `c_geom` hurt quality on large uniform plateaus?

Yes. As originally described, that version of `c_geom` is likely backwards.

Large, clean plateaus are often the places where the geometry is *most* reliable:

- boundaries are well separated
- the sign structure is stable
- the IDW sensitivity to distance error is actually lower, not higher

Recall:

```text
w = d2 / (d1 + d2)
dw/dd2 = d1 / (d1 + d2)^2
```

When both `d1` and `d2` are large, this derivative decreases. So the compensation becomes **less** sensitive to `d2` error on broad plateaus.

So penalizing large `d1 + d2` is probably wrong.

The risky cases are more like:

- **small `d1 + d2`**
  - thin structures
  - dense or fragmented sign-flip boundaries
  - unresolved features under downsampling

- **strong asymmetry or truncation**
  - near domain boundaries
  - incomplete support from `B2`

- **poor boundary evidence**
  - almost no `B1`/`B2` support locally
  - very low boundary density globally or blockwise because quantization destroyed structure

That means `c_geom` should probably be redefined.

I would split the ideas:

#### Local geometry confidence

This should **increase** with clean plateau width up to saturation:

```text
c_geom = 1 - exp(-(d1 + d2) / tau_w)
```

or any monotone saturating function.

Interpretation:

- thin ambiguous region: low confidence
- sufficiently wide plateau: high confidence

#### Separate evidence or information-loss metric

If you are worried about extreme homogeneous fields where the quantization index contains little information at all, that is not a local-width issue. That is a different signal:

- blockwise boundary density
- number of unique quant indices
- zero or dominant-index ratio
- coarse/fine `B2` mismatch

That signal should not be conflated with `d1 + d2`.

So the answer to your question is:

- yes, the originally proposed `c_geom` would likely hurt quality on large well-behaved plateaus
- large `d1 + d2` should usually mean **higher**, not lower, local geometry confidence
- the concern about “not enough information left after heavy quantization” should be handled by a separate evidence metric, not by plateau width alone

---

## Claude Follow-up Questions (Round 3)

### Q1. How do you choose tau_w for c_geom without tuning?

Codex redefined c_geom as:

```text
c_geom = 1 - exp(-(d1 + d2) / tau_w)
```

This introduces tau_w as a new hyperparameter. The whole paper's strength is that it requires no tuning — eta=0.9 was fixed globally after an offline sweep, and everything else is derived from quant_index and eb.

Can tau_w be set automatically from something already available? Candidates:
- The median or mean of (d1+d2) across all voxels — no prior needed
- The typical plateau width implied by eb and the data range: if the data range is R, there are roughly R/(2*eb) quantization levels, so typical plateau width ~ N^(1/3) / (R/(2*eb)) in 3D
- Just a fixed constant (e.g., tau_w = 3 or 5 voxels)

If tau_w must be tuned, that is a problem for the paper. If it can be derived from eb and data statistics, it is fine. What is the right choice?

### Q2. How is c_sign propagated to non-boundary points?

In the current pipeline, sign_map is computed at boundary points and then propagated to all non-boundary points using the EDT1 nearest-neighbor index: `sign_map[i] = sign_map[nearest_boundary[i]]`. The same EDT1 result is used.

c_sign is defined at boundary points (from voting among differing neighbors). How should it be propagated to non-boundary points? Options:

- Same EDT1 propagation: `c_sign[i] = c_sign[nearest_boundary[i]]` — trivially cheap, one array lookup
- Decay with distance: `c_sign[i] = c_sign[nearest_boundary[i]] * decay(d1[i])` — penalizes points far from any boundary where the sign assignment is less reliable
- Keep it purely at boundary points and absorb it into the IDW formula differently

Option 1 is simplest and consistent with how sign_map is already propagated. Option 2 has intuitive appeal — a point 50 voxels from the nearest boundary is getting its sign from a very remote source, and that propagation may be unreliable if the sign boundary is not flat. Does distance-decayed c_sign make sense geometrically? Or does d1 already implicitly encode this through the IDW magnitude (which approaches 0 or 1 at large distances anyway)?

### Q3. Trilinear upsampling of c_alias at coarse cell boundaries — is it geometrically meaningful?

Codex recommended trilinearly upsampling the coarse c_alias confidence field to avoid staircase artifacts. But consider two adjacent coarse cells: one with c_alias=0.1 (risky, dense boundary geometry) and one with c_alias=0.9 (safe, clean plateau). The fine voxels near the interface get c_alias ≈ 0.5 from trilinear interpolation — but that blended value has no geometric meaning. The actual geometry of B2 at the fine level near that interface is either risky or safe, not 50% of each.

Is there a better approach than trilinear? Options:
- Nearest-neighbor assignment (blockwise constant) — honest but produces staircase
- Trilinear with a floor (clamp to c_alias > 0.2 to avoid near-zero values from blending) — partially mitigates
- Use the fine B2 directly near coarse cell boundaries: for voxels within s/2 of a coarse cell boundary, compute c_alias at the fine resolution rather than interpolating

Or is this concern overstated — do the staircase artifacts in c_alias actually matter for final compensation quality, or are they washed out by the IDW weighting?

### Q4. What specific limitations of the IPDPS method should Section 3 call out?

Codex recommended adding “Section 3: Problem Formulation and Limitations of the Conference Method.” What goes in it? The IPDPS paper itself acknowledges:
- Slight PSNR decline at small eb on some datasets (S3D)
- Struggles with highly homogeneous quantization regions

Beyond those, what limitations are worth calling out that the new method actually fixes?
- Binary gradient threshold → fixed by continuous c_sign
- Fixed global eta → fixed by per-point eta_i = c_i * eta_max
- Two full-resolution EDT passes → fixed by coarse EDT2
- No deterioration control → fixed by confidence-gated compensation
- Flat sign propagation (ignores distance) → possibly fixed by decay option in Q2

The risk: calling out limitations that the new method does not clearly fix gives reviewers ammunition. Is there a limitation worth calling out that we have NOT addressed, and should we be honest about it or not mention it?

### Q5. Does the “reliability-controlled” framing conflict with the existing “relaxed error bound” claim?

The IPDPS paper already claims a provable relaxed error bound (1+eta)*eb. The TPDS paper now also claims “reliability control” and “selective suppression.” A reviewer could read this as: if the IPDPS method already provably satisfies the error bound, what does the new reliability control actually add? Is the answer simply “quality improvement (better average PSNR/SSIM)” rather than error-bound control? If so, the framing of “reliability-controlled” may be misleading — it sounds like a safety mechanism but the original method was already safe.

How should the paper distinguish between:
- Error bound control (already done in IPDPS, trivially by IDW construction)
- Quality deterioration control (new: preventing cases where compensation hurts PSNR even within the error bound)

These are different things. The second is the real new contribution. Should the terminology change from “reliability-controlled” to something that more precisely captures “preventing quality degradation within the already-safe error bound”?

### Q6. Minimum viable implementation for a first experiment — what to build first?

Given limited development time, what is the smallest implementation that would produce a publishable ablation result? Ranking by effort vs. payoff:

1. **c_sign vote-imbalance** — trivial to add to boundary detection pass, replaces gradient threshold, one extra array
2. **c_geom (saturating plateau width)** — one formula applied per voxel, uses d1+d2 already computed, zero new passes
3. **Coarse EDT2 (already done)** — already implemented, just needs the c_alias proxy on top
4. **c_alias proxy** — needs downsample+upsample+XOR mismatch of B2, ~3 extra kernel passes or loops
5. **Confidence fusion + compensation kernel update** — needs c_sign, c_geom, c_alias combined

Items 1+2+5 can be done without touching the EDT infrastructure at all. Items 3+4 build on existing downsampled EDT2.

Is the right first experiment: implement c_sign and c_geom only (skip c_alias), run the ablation on NYX 512^3, and see whether they improve PSNR on the cases where the IPDPS method currently hurts quality? That would quickly validate whether the confidence framework actually helps before investing in c_alias.

### Q7. What is the title of the TPDS paper?

The IPDPS title is: “Mitigating Artifacts in Pre-quantization Based Scientific Data Compressors with Quantization-aware Interpolation.”

The TPDS extension needs a title that signals new content without being redundant. Options:
- Keep similar framing, add “Adaptive” or “Reliable”: “Adaptive Quantization-aware Interpolation for Artifact Mitigation in Pre-quantization Based Compressors”
- Emphasize the system: “High-Throughput Quantization-aware Artifact Mitigation with Reliability-Controlled Multiresolution Compensation”
- Emphasize the tradeoff: “Quality-Controlled Artifact Mitigation in Scientific Data Compressors: A CPU/GPU Multiresolution Approach”

A TPDS title should signal: (1) it is a journal extension with new content, (2) the central new idea. Which of these works, or what is better?

## Codex Review Round 3

### Q1. Can `tau_w` be set without tuning?

Yes, and the best answer is probably to avoid introducing `tau_w` as a free dataset-level parameter at all.

The cleanest way to do that is to tie the width scale directly to the downsample factor `s`, because the whole point of `c_geom` is to reflect whether the local plateau geometry is sufficiently resolved at that coarse scale.

So instead of:

```text
c_geom = 1 - exp(-(d1 + d2) / tau_w)
```

I would recommend one of these two forms:

```text
c_geom = 1 - exp(-(d1 + d2) / s)
```

or even better, to eliminate the extra symbol entirely:

```text
c_geom = (d1 + d2) / (d1 + d2 + s)
```

where:

- `s = 1` for full-resolution `EDT2`
- `s = 2, 4, 8, ...` for downsampled `EDT2`

Why this is the right scale:

- if the plateau width is small relative to the coarse cell size, downsampling is risky
- if the plateau width spans multiple coarse cells, the geometry is more trustworthy

This gives you:

- no dataset-specific tuning
- no EB-specific tuning
- a parameter derived from the actual approximation being made

That is much stronger than setting `tau_w` from a median or mean of `d1 + d2`, because those statistics vary with the dataset and can quietly become another tuned knob.

So my recommendation is:

- **Do not introduce a tunable `tau_w`.**
- Define `c_geom` directly in terms of the downsample factor `s`.

If you want a sentence for the paper:

> We set the geometry-confidence length scale to the downsampling factor, so confidence increases only when the local plateau width exceeds the coarse-grid resolution used by the approximate second distance transform.

That is clean and reviewer-resistant.

### Q2. How should `c_sign` be propagated to non-boundary points?

Use the same `EDT1` nearest-boundary propagation as the sign itself:

```text
c_sign[i] = c_sign[nearest_boundary[i]]
```

This is the right default.

I would **not** decay `c_sign` with distance.

Why:

- the sign confidence is a property of the boundary evidence that generated the sign
- distance from that boundary is already reflected in the compensation magnitude through `d1` and `d2`
- adding an extra distance decay to `c_sign` would double-penalize broad, well-behaved plateaus, which is exactly what you do not want

In other words:

- `c_sign` should represent **how trustworthy the boundary sign is**
- `d1`/`d2` and `c_geom` already represent **how trustworthy the geometry is away from the boundary**

Mixing those two into one decay would muddy the interpretation.

So I would keep:

- boundary-local computation of `c_sign`
- nearest-boundary propagation to all voxels

and leave distance effects to the other terms.

### Q3. Is trilinear upsampling of `c_alias` meaningful?

Yes, for a confidence field it is meaningful enough, and the concern is somewhat overstated.

The key point is:

- `c_alias` is not a geometric surface
- it is a **soft uncertainty field**

So a voxel with interpolated confidence `0.5` near a boundary between a risky coarse cell and a safe coarse cell does not need to mean “the geometry is literally half risky.” It only needs to mean “this location lies near a transition between low-confidence and high-confidence coarse neighborhoods.”

That is actually a sensible soft gating signal.

I would therefore prefer:

- compute one confidence value per coarse cell
- trilinearly upsample that confidence field

over:

- nearest-neighbor blockwise assignment

because the latter is more likely to imprint coarse-grid artifacts into the compensation amplitude.

If you want an even more conservative version, use:

```text
c_alias_fine = min(trilinear(c_alias_coarse), c_geom)
```

That keeps the interpolated alias confidence from dominating the more local geometry signal.

So my practical answer is:

- trilinear upsampling is fine
- it does not need exact geometric semantics because it is a confidence field, not a boundary reconstruction
- the staircase risk from nearest-neighbor is likely worse than the semantic impurity of interpolation

### Q4. What limitations of the IPDPS method should Section 3 call out?

Only call out limitations that the TPDS paper directly addresses or explicitly revisits.

The safest list is:

- **uniform full-resolution `EDT2` cost**
  - addressed by coarse `EDT2`
- **binary sign suppression in ambiguous boundary regions**
  - addressed by continuous `c_sign`
- **uniform compensation policy across regions with very different geometric reliability**
  - addressed by quality-aware selective compensation
- **lack of a mechanism to reduce harmful updates while remaining within the relaxed error bound**
  - addressed by the new confidence-gated formulation

I would also be honest about one limitation that is only partially addressed:

- **severe information loss under very large error bounds or highly homogeneous quantization**
  - the new method can detect or suppress risky compensation better
  - but it cannot recover information that is no longer present in `Q` and `D'`

That honesty helps rather than hurts, because it shows the paper understands the information-theoretic limit of the problem.

What I would avoid claiming as a fixed limitation unless you actually implement it:

- “flat sign propagation ignores distance”  
  This is not clearly fixed unless you really change the propagation model.

### Q5. Does “reliability-controlled” conflict with the existing relaxed error-bound claim?

Yes, this is a real framing problem.

The conference method was already:

- safe in the relaxed `L_infinity` sense

So if the journal paper says:

- “we add reliability control”

a reviewer may reasonably ask:

- reliable in what sense?
- the old method was already bounded, so what new safety is being added?

The answer is:

- **not error-bound safety**
- **quality-deterioration control within an already safe feasible region**

That distinction needs to be explicit.

So I would strongly consider changing the terminology from:

- `reliability-controlled`

to something like:

- **quality-aware**
- **degradation-aware**
- **selective compensation**
- **quality-deterioration-controlled**

My preference is:

- **quality-aware multiresolution compensation**

because it says exactly what is new:

- the original method already controlled error magnitude
- the new method controls where compensation should be trusted enough to apply aggressively

This also fixes the abstract framing.

The paper should state this explicitly:

> The conference method already guarantees a relaxed error bound through bounded compensation magnitude. The journal extension does not tighten that bound; instead, it introduces quality-aware selective compensation to reduce cases where post-processing degrades PSNR or SSIM despite remaining within the same relaxed bound.

That sentence resolves the ambiguity cleanly.

So yes: I would change the framing.

### Q6. What is the minimum viable first implementation?

The smallest implementation with real scientific value is:

1. `c_sign` vote imbalance
2. simple confidence fusion in the final compensation pass
3. no `c_alias` yet
4. use the existing full-resolution `EDT2`

This is the right first experiment because it tests the central question:

- can confidence-gated compensation reduce the known negative-tail cases of the IPDPS method?

That means your first ablation target should be:

- baseline IPDPS full-resolution method
- full-resolution + `c_sign`
- full-resolution + `c_sign + c_geom`

on:

- one challenging low-EB case where the conference method slightly hurts PSNR
- one moderate-EB case where the conference method already works well

Why this ordering is good:

- it validates the **quality-aware** idea before touching the multiresolution machinery
- if these terms do not help, then there is no point building `c_alias`
- if they do help, then `c_alias` becomes a natural second-stage extension for the coarse `EDT2` path

So yes: your proposed first experiment is the right one.

I would sequence development as:

1. `c_sign`
2. `c_geom`
3. confidence fusion
4. ablation on known failure cases
5. only then add `c_alias` to the downsampled path

### Q7. What should the TPDS title be?

Because of the framing issue in Q5, I would avoid `reliability-controlled` in the title.

I would also avoid titles that foreground GPU too much. GPU belongs in the contributions and abstract, but if it dominates the title the paper risks sounding like a porting paper.

The title should foreground:

- artifact mitigation
- multiresolution or quality-aware selective compensation
- scientific data compressors

My top recommendation is:

> **Quality-Aware Multiresolution Artifact Mitigation for Pre-Quantization-Based Scientific Data Compressors**

Why this works:

- clearly different from the IPDPS title
- foregrounds the new method, not just the implementation
- avoids the confusing “reliability-controlled” phrase
- still reads like a TPDS systems-and-methods paper

If you want to retain the “quantization-aware interpolation” lineage more explicitly, the best alternative is:

> **Quality-Aware Multiresolution Quantization-Aware Interpolation for Artifact Mitigation in Scientific Data Compressors**

This is accurate but a bit long and repetitive.

If you want a title with a slightly stronger systems flavor:

> **Quality-Aware Multiresolution Artifact Mitigation in Scientific Data Compressors**

and then let the abstract carry the CPU/GPU pipeline contribution.

So my ranked preference is:

1. `Quality-Aware Multiresolution Artifact Mitigation for Pre-Quantization-Based Scientific Data Compressors`
2. `Quality-Aware Multiresolution Artifact Mitigation in Scientific Data Compressors`
3. the longer quantization-aware interpolation version only if you feel continuity with the conference paper is strategically important

---

## Claude Decision Memo (Synthesis of Rounds 1–3)

### Final Framing

The TPDS paper introduces **one new method** (quality-aware selective compensation) and **one new system** (end-to-end CPU/GPU pipeline). Do not present them as three separate pillars. The unifying claim is:

> The conference method compensates everywhere within the error bound. The journal method compensates selectively, suppressing updates where local geometry makes compensation likely to degrade quality.

That is the upgrade. Everything else (GPU, downsampling, c_sign, c_geom, c_alias, ablation, Pareto curves) is in service of this claim.

### Final Terminology

- Drop **"reliability-controlled"** — it sounds like an error-bound mechanism, but the error bound was already proven in the IPDPS paper by IDW construction. Using it here confuses reviewers.
- Use **"quality-aware"** — it precisely captures what is new: controlling PSNR/SSIM degradation within an already-safe error bound.
- The compensation field is now **"quality-aware selective compensation"** — selective because it can suppress or reduce the update, quality-aware because the decision is driven by local geometry confidence.
- The downsample contribution is not a speed trick — it is the **coarse-grid EDT2 approximation** whose accuracy is characterized by c_geom and c_alias.

### Abstract Novelty Sentence

> We introduce quality-aware multiresolution compensation, which selectively suppresses post-decompression artifact correction in geometrically unreliable regions while approximating the second distance transform on a coarse grid, jointly controlling quality deterioration and computational cost within a provably bounded error guarantee.

If that is too long for one sentence, split it:

> We introduce quality-aware selective compensation, which reduces harmful updates in geometrically ambiguous regions, and pair it with a coarse-grid EDT2 approximation that cuts runtime while preserving quality in well-behaved plateau regions.

### Title Options

1. **Quality-Aware Multiresolution Artifact Mitigation for Pre-Quantization-Based Scientific Data Compressors** ← recommended
2. Quality-Aware Artifact Mitigation in Scientific Data Compressors: A CPU/GPU Multiresolution Approach
3. Beyond Error Bounds: Quality-Aware Quantization-Aware Interpolation for Scientific Data Compressors (only if you want to make the "beyond error bounds" framing explicit)

Avoid anything with "reliability-controlled" or "high-throughput" in the title. GPU belongs in the abstract, not the title.

### MVP Implementation Plan

In order — stop after each step and run an experiment before proceeding:

1. **c_sign**: Add vote-imbalance computation to the boundary detection pass. Store as float. Propagate via EDT1 nearest-boundary (same mechanism as sign_map). Cost: ~10 lines of code, one extra float array.

2. **c_geom**: In the compensation kernel, compute `c_geom = (d1+d2) / (d1+d2+s)` where s is the downsample factor (s=1 for full-res). Cost: one formula, no extra arrays.

3. **Confidence fusion**: `C_i = c_sign * c_geom * s_i * eta_max * eb * w(d1, d2)`. Replace the current fixed `comepnsation_value` with `eta_max * eb`. Cost: one line change in the compensation loop.

4. **First experiment**: Run on the S3D small-eb case (where IPDPS method hurts PSNR) and NYX 512³ (where it helps). Confirm that c_sign + c_geom reduce the negative-tail cases without hurting the positive cases. If this does not work, reassess the whole confidence framework before touching c_alias or GPU.

5. **c_alias** (only after step 4 validates the approach): Downsample B2 by OR-pooling → compute per-coarse-block mismatch ratio vs. fine B2 → trilinearly upsample to full resolution. Add to confidence fusion: `C_i = c_sign * (c_geom^0.5 * c_alias^0.5) * s_i * eta_max * eb * w(d1, d2)`.

6. **GPU port**: All confidence terms fit naturally into existing GPU kernels. c_sign in boundary kernel, c_geom in compensation kernel, c_alias as a separate pre-pass before compensation.

### Minimum Ablation Plan

**Table A — methodological ablation** (one challenging dataset, e.g. S3D at small eb):

| Config | PSNR | SSIM | Time |
|---|---|---|---|
| No compensation | | | |
| IPDPS full-res | | | |
| Full-res + quality-aware (c_sign + c_geom) | | | |
| Coarse EDT2 only (no confidence) | | | |
| Full proposed (coarse EDT2 + quality-aware) | | | |

**Table B — confidence decomposition** (same dataset):

| Config | PSNR | SSIM |
|---|---|---|
| Full-res + c_sign only | | |
| Full-res + c_sign + c_geom | | |
| Full-res + all confidence (c_sign + c_geom + c_alias) | | |

**Figure C — downsample Pareto** (4 datasets × 3 eb × 4 factors = 48 points):
- x-axis: wall time relative to full-res (or % overhead)
- y-axis: ΔPSNR or ΔSSIM relative to full-res
- Shows the tradeoff curve clearly; one point per (dataset, eb, factor)

**Final comparison table** (all datasets, final proposed method vs. baselines including IPDPS):
- Gaussian filter, IPDPS method, proposed method — PSNR, SSIM, time

This ablation answers every question a TPDS reviewer will ask without ballooning into a 9-row single table.

### Biggest Remaining Technical Risk

**The quality-aware compensation may not help enough on the actual failure cases.**

The IPDPS method only slightly hurts PSNR on a few datasets at small eb. If c_sign and c_geom suppress those cases but also suppress beneficial compensation elsewhere, the net effect on SSIM could be neutral or negative. The whole journal contribution depends on the first experiment (MVP step 4) showing a clear improvement on failure cases with no regression on success cases.

If that experiment fails — if confidence-gated compensation does not reduce the negative-tail without hurting the positive-tail — the current TPDS plan collapses back to "GPU port + downsampled EDT2," which is not enough.

This is the risk to validate first, before writing anything.

### What to Cut If Time Is Short

In order of what to drop:

1. **c_alias** — the c_sign + c_geom confidence is the core idea; c_alias is an add-on for the coarse-EDT2 path. If c_sign + c_geom work, the paper is defensible without c_alias.
2. **Multi-GPU / multi-node GPU** — single-GPU is sufficient. Keep MPI results from IPDPS, don't extend to MPI+GPU.
3. **Bezier smoothing baseline** — unless Daoce's code is public and easy to run, skip it. Gaussian/Uniform/Wiener are sufficient baselines.
4. **Additional compressors** — SZp and FZ-GPU can be "future work" again.
5. **1024³ experiments** — 512³ is enough for the GPU story if you don't have access to larger data or time to run it.

---

## Claude Follow-up Memo: Checkpoint Validation

### Is "validate c_sign + c_geom first" truly the right checkpoint?

Mostly yes, but with one important caveat: the checkpoint as stated is only meaningful if you already know *which* cases the IPDPS method fails on, and whether those failures are caused by sign ambiguity or geometry unreliability — i.e., the exact things c_sign and c_geom are designed to fix. If the negative-tail cases are caused by something else entirely (e.g., the neutral boundary is poorly defined in thin structures, or the plateau decomposition is fundamentally wrong at small eb), then c_sign and c_geom will not help and the failure will be misread as "confidence framework doesn't work" when the real issue is upstream.

So before running the experiment, one diagnostic question must be answered: on the known failure cases (S3D small eb, CESM at specific eb), is the compensation sign correct but the magnitude wrong, or is the sign itself wrong at some boundary points?

- If sign is wrong → c_sign is the right fix, checkpoint is valid
- If sign is right but magnitude is wrong → c_geom or the IDW formula shape is the issue
- If both are fine but boundary detection is the issue → the whole confidence framework is the wrong level of intervention

### Strongest Argument Against This Plan

The IPDPS method's negative-tail cases (slight PSNR degradation at small eb on S3D) are small in absolute magnitude — fractions of a dB. If c_sign and c_geom reduce those cases by, say, 0.1 dB, a TPDS reviewer may reasonably say the improvement is within noise and not a meaningful contribution. The checkpoint could "succeed" in the sense that quality-aware compensation helps slightly, but fail to produce a compelling result table. You could invest weeks of implementation and experimentation and end up with a result that is technically correct but not publishable on its own.

### Strongest Argument For This Plan

Every other component of the TPDS paper — GPU pipeline, Pareto curves, ablation tables — is expensive to produce but does not fundamentally change the story. The quality-aware compensation is the only genuinely new algorithmic idea. Validating it cheaply first (it really is ~15 lines of code) before committing to paper writing and full experimental campaigns is classically correct engineering discipline. If it fails, you save weeks. If it succeeds even modestly, you have the experimental anchor around which everything else is organized.

### If the Checkpoint Succeeds: Next 3 Discussion Questions

1. **How much of the quality improvement comes from c_sign vs. c_geom independently?** If one dominates, the other may be cut from the paper, simplifying the story significantly.

2. **Does quality-aware compensation also help the GPU pipeline, or only the CPU path?** The GPU compensation kernel is structurally simpler — if c_sign and c_geom are cheap to fuse there, the GPU pipeline gains the same quality improvement essentially for free, which strengthens the end-to-end story.

3. **At what eb range does quality-aware compensation help vs. hurt?** The IPDPS paper's negative-tail is concentrated at small eb. If quality-aware compensation helps at small eb but slightly hurts at medium eb (where the IPDPS method is already strong), the paper needs to discuss adaptive eta_max selection by eb range — which opens a new design question.

### If the Checkpoint Fails: Fallback TPDS Story

If c_sign + c_geom do not meaningfully reduce the negative-tail cases, the TPDS story becomes purely a systems paper:

- **Contribution 1**: Full CPU/GPU pipeline with end-to-end integration into cuSZ, including packed-index, flat32, and dist-only EDT optimizations
- **Contribution 2**: Coarse-grid EDT2 as a principled quality-performance-memory tradeoff, with Pareto analysis across datasets and error bounds
- **Contribution 3**: Formal error analysis of the downsampling approximation (the Hausdorff-distance + IDW-sensitivity bound from Round 2)

This is weaker than the quality-aware story but still publishable at TPDS as a systems-and-analysis paper. The key reframe: the paper's claim shifts from "we prevent quality degradation" to "we characterize and control the quality-performance tradeoff of multiresolution compensation." That is a different but defensible contribution, closer to what the IPDPS paper's parallelization section established.

### Refined Paper Framing (3 Sentences)

Pre-quantization-based compressors trade decompressed data quality for throughput, producing structured artifacts that can be mitigated by quantization-aware post-processing — but the existing approach applies compensation uniformly, occasionally degrading quality in geometrically ambiguous regions. We introduce quality-aware multiresolution compensation, which selectively suppresses correction where local geometry evidence is weak and approximates the second distance transform on a coarse grid, jointly reducing harmful updates and improving throughput. We realize this in an end-to-end CPU/GPU pipeline and characterize the quality-performance-memory tradeoff through Pareto analysis across five scientific datasets and two leading compressors.

### Refined Abstract Novelty Sentence (1 Sentence)

> We introduce quality-aware multiresolution compensation, which selectively suppresses artifact correction in geometrically ambiguous regions and approximates the second distance transform at coarse resolution, reducing quality-degrading updates while preserving the relaxed error guarantee and improving throughput by up to Xx on GPU.

(Fill Xx once GPU numbers are confirmed.)

### Top 5 Unresolved Discussion Decisions

1. **Does c_sign + c_geom actually reduce negative-tail cases on real data?** — blocks everything; must be answered by experiment before paper writing begins

2. **What is the right fusion formula for confidence terms?** — Codex recommended `c_sign * (c_geom^0.5 * c_alias^0.5)` but the exponents are unvalidated; this needs empirical sensitivity analysis

3. **Is c_alias worth the implementation cost?** — depends on (1) and on whether the coarse-EDT2 path introduces enough aliasing in practice to need it; currently justified in theory but not validated

4. **What is the paper's primary metric — PSNR or SSIM?** — the IPDPS paper leads with SSIM, but quality-aware compensation is more naturally argued in PSNR terms (preventing degradation); the two metrics may tell different stories on the same data

5. **How to handle the homogeneous-field failure mode honestly?** — the IPDPS paper acknowledged this; the TPDS paper should show that quality-aware compensation at least does not make it worse, and ideally shows that blockwise evidence metrics (part of c_alias) partially detect it

## Codex Memo Response

### 1. Can the sign-vs-magnitude diagnostic be inferred without new experiments?

Partly, yes. From the existing algorithm and IPDPS behavior, the more likely root cause of the known small-EB failures is **magnitude**, not catastrophic sign error.

Why:

- The conference method already suppresses some ambiguous boundary signs with the gradient threshold, so gross sign mistakes are somewhat filtered.
- If sign were frequently wrong, I would expect visibly wrong local corrections and larger quality regressions, not just slight PSNR dips in a few low-EB cases.
- The reported failures occur exactly where the quantized data already has weak artifacts. That is the regime where any nonzero correction can become unnecessary or slightly harmful even if the sign is correct.

So the prior is:

- `c_geom` or general gating is more likely to matter than `c_sign`
- `c_sign` still matters at corners/junctions, but is less likely to be the dominant explanation for the known low-EB negative tail

I would state this as an inference, not a fact. Existing results support “overcorrection on already-good reconstructions” more strongly than “systematic sign inversion.”

### 2. Is the fallback systems story publishable at TPDS?

Only marginally, unless it gains a stronger algorithmic hook.

`GPU + downsampled EDT2 + Pareto + approximate error analysis` is better than a pure porting paper, but its weakest point is still obvious:

- the method is basically the conference method
- the new content is mostly approximation and implementation

That can be publishable only if the systems results are unusually strong:

- true end-to-end cuSZ integration
- clear throughput and memory wins
- large-scale GPU results
- a clean multiresolution approximation analysis

Even then, the weakest part of the fallback is **method novelty**. A reviewer can still say the paper is a very good optimization study of an existing algorithm rather than a clear new algorithmic contribution.

So the fallback is defensible, but weaker and riskier than the quality-aware story.

### 3. Critique of the refined 3-sentence framing

The structure is good, but `geometrically ambiguous regions` is still too vague on its own. A reviewer will ask what that means operationally.

It should be grounded in explicit signals, for example:

- low sign consensus at boundaries
- unresolved or poorly supported coarse-grid `B2` geometry
- thin or fragmented plateau regions relative to the coarse-grid scale

So I would replace `geometrically ambiguous regions` with something like:

- `regions with low boundary-sign consensus or weak coarse-grid support`

That is more formal and immediately tied to the proposed confidence terms.

### 4. Which metric should anchor the paper: PSNR or SSIM?

The paper’s main claim should still anchor on **SSIM**, not PSNR.

Reason:

- the application is artifact mitigation, especially posterization and banding
- that is a structural/perceptual quality problem, which SSIM captures better than PSNR
- the IPDPS paper already established that framing, and abandoning it would create an unnecessary reviewer question

But the new quality-aware idea should be justified with **PSNR no-regression / reduced-negative-tail analysis**.

So the clean split is:

- **main paper claim:** better artifact mitigation quality, anchored by SSIM
- **new journal-method justification:** fewer harmful updates, demonstrated through PSNR robustness and failure-case reduction

If you try to anchor the whole journal paper on PSNR, you risk weakening the core motivation for posterization mitigation.

### 5. Is there a more important unresolved decision Claude missed?

Yes: **what is the pass/fail criterion for the checkpoint experiment?**

That is more important than several of the listed top-5 items, because without it the checkpoint is not actually a checkpoint.

You should decide in advance:

- how much PSNR recovery on the known negative-tail case counts as a real win
- what non-regression on SSIM or positive-tail cases is required
- whether “small improvement on S3D but neutral elsewhere” is enough to continue

Without that, the experiment can be interpreted after the fact in whatever direction is convenient, which weakens the whole planning process.

---

## Claude Decision Memo (Checkpoint Spec)

### Checkpoint Pass/Fail Rule

**Pass requires both conditions:**
- Negative-tail cases: PSNR improvement of ≥ 0.15 dB on at least 2 of the known failure configurations (e.g. S3D at eb=0.001, CESM at its worst eb). This threshold is chosen because it is above measurement noise and visually meaningful, but does not demand a dramatic result.
- Positive-tail cases: SSIM regression ≤ 0.002 absolute on any dataset/eb where IPDPS already improves quality. This is approximately 0.3% of a typical SSIM score of 0.7 — within acceptable tolerance.

**Partial pass (not fail, not full pass):** PSNR improvement exists but < 0.15 dB, or improvement is clear on 1 case only. In this scenario do not abandon the framework — instead run c_geom in isolation (without c_sign) to check whether the magnitude issue dominates the sign issue as Codex inferred.

**Fail:** No measurable PSNR improvement on any known failure case, or SSIM regression > 0.002 on any previously-improving case. This triggers the fallback assessment.

### Framing Sentence

**Preferred:**
> We introduce quality-aware multiresolution compensation, which scales down artifact correction at boundaries with low sign consensus or insufficient plateau width relative to the coarse-grid scale, and approximates the second distance transform at reduced resolution, jointly cutting runtime and preventing degradation cases within the established relaxed error bound.

**Backup (shorter):**
> We introduce quality-aware multiresolution compensation, which suppresses correction where boundary-sign evidence is weak or the local plateau is poorly resolved at the coarse grid scale, and approximates the second distance transform at reduced resolution to control the quality-performance tradeoff.

Both avoid "geometrically ambiguous" and ground the claim in operational signals (sign consensus, plateau width vs. grid scale) that are directly tied to c_sign and c_geom.

### Fallback Verdict

**No.** A partial-success fallback (GPU + Pareto + error analysis, no strong quality-aware headline) is not strong enough for TPDS on its own.

Reason: the Pareto analysis and error bound approximation are both supporting analysis, not independent contributions. They explain and characterize an approximation you already made; they do not propose something new. TPDS requires a method contribution, not only a systems study of an existing method. The GPU pipeline and memory analysis would fill the implementation section of a strong paper but cannot carry the paper alone.

The right response to a partial checkpoint success is not to fall back to the systems story — it is to reframe the contribution more narrowly around the cases where quality-aware compensation provably helps and quantify those precisely, even if the effect is modest. A modest but real and theoretically grounded effect is publishable; a systems extension without new method is risky.

### Guarantee-vs-Quality Sentence

> The conference method guarantees that compensation never exceeds the relaxed error bound by construction; the journal extension addresses a separate goal — reducing the subset of cases within that safe region where compensation degrades PSNR or SSIM despite remaining formally correct.

### Title Terminology Decision

Keep **"quality-aware"**. Do not switch to "selective compensation."

Reason: "selective compensation" describes a mechanism (applying compensation selectively), not a goal. It would draw reviewer attention to *how* the selection is done and invite questions about whether the selection criterion is principled. "Quality-aware" describes the *objective* — the system is aware of quality consequences and responds to them. That framing is harder to attack and matches the paper's main contribution claim (reducing quality degradation). It also avoids sounding like a minor engineering tweak.

Final recommended title: **Quality-Aware Multiresolution Artifact Mitigation for Pre-Quantization-Based Scientific Data Compressors**

## Codex Checkpoint Response

### 1. Is `0.15 dB PSNR` the right pass/fail threshold, and should the rule use PSNR, SSIM, or both?

It should use **both**, but asymmetrically.

- **PSNR** should define success on the known negative-tail cases, because that is the actual failure mode being fixed.
- **SSIM** should define non-regression on the cases where the IPDPS method already works well, because SSIM is still the paper’s main artifact-mitigation quality metric.

I would soften the `0.15 dB` threshold slightly. For a first checkpoint, `0.10 dB` is a better minimum bar, because the known failures are already small. Requiring `0.15 dB` may create a false fail on a real but modest effect.

Recommended checkpoint rule:

- **Pass:** at least `+0.10 dB` PSNR recovery on a canonical failure case, and no more than `0.002` absolute SSIM regression on positive-tail cases.
- **Strong pass:** `+0.15 dB` or more on at least two failure cases, with the same SSIM non-regression bound.
- **Fail:** less than `+0.05 dB` PSNR recovery or clear SSIM regression beyond `0.002`.

So the rule should not be PSNR-only. It should be:

- PSNR for fixing the failure
- SSIM for preserving the paper’s main quality story

### 2. Is there a cheaper diagnostic than “run `c_geom` alone” before implementing it?

Yes. Before implementing anything new, the cheapest diagnostic is to inspect **existing IPDPS outputs** on the known failure case and ask where the compensated field differs harmfully from the decompressed field.

Specifically, using existing saved outputs, figures, or line profiles:

- If the harmful changes occur mostly on broad smooth plateaus where the quantized field already looks good, that points to **magnitude / overcorrection**.
- If the harmful changes cluster around corners, thin structures, or sign-flip junctions, that points more to **sign ambiguity**.

So the cheapest pre-implementation diagnostic is not another method variant. It is:

- compare `D'' - D'` spatially on the failure cases using existing outputs
- classify whether the damage is plateau overcorrection or boundary-sign error

That can be reasoned from existing evaluation artifacts without adding `c_geom` first.

### 3. If the checkpoint completely fails, is there any TPDS reframe left?

Only a weak one.

I still agree with the “no” verdict for the current fallback as stated. If the checkpoint completely fails, then the paper loses its main new method hook. What remains is:

- GPU realization
- downsampled `EDT2`
- Pareto and error analysis

That is useful, but still vulnerable to the “optimization study of an existing method” criticism.

The only viable reframe would be:

- make **multiresolution `EDT2` itself** the main new method, not just a speed trick
- strengthen the theory and system story substantially
- show unusually strong end-to-end gains and memory savings

Even then, it is a riskier TPDS submission than the quality-aware story. So if the checkpoint completely fails, the TPDS plan is still possible, but only with a much stronger multiresolution-analysis paper than currently sketched.

### 4. Is “insufficient plateau width relative to the coarse-grid scale” precise enough?

Yes, **if it is defined immediately**. On its own, it is too vague.

One-line formal definition:

> We define the local plateau-width proxy at voxel `i` as `w_i = d1_i + d2_i`, and call a region poorly resolved at coarse scale `s` when `w_i < s`, i.e., when the estimated plateau width is smaller than one coarse-grid cell.

That is precise enough to defend, and it ties directly to the proposed `c_geom` form.

### 5. Final title decision

**Final title:**

1. **Quality-Aware Multiresolution Artifact Mitigation for Pre-Quantization-Based Scientific Data Compressors**
   - Best balance of novelty, continuity with the IPDPS paper, and emphasis on the new method rather than just the GPU implementation.

**Backups:**

2. **Quality-Aware Artifact Mitigation in Scientific Data Compressors with Multiresolution Quantization-Aware Interpolation**
   - Keeps stronger continuity with the conference method, but is longer and slightly more awkward.

3. **Multiresolution Artifact Mitigation for Pre-Quantization-Based Scientific Data Compressors**
   - Cleaner and shorter, but weaker because it drops the central “quality-aware” hook.
