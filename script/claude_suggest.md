# Next Steps: Suggested Priorities

Based on the IPDPS paper, `discuss.md`, and `notes.md`. Organized by what to do first, not what's most interesting.

---

## Immediate Priority: Lock Down the TPDS Story

Before writing code, settle on the paper structure. The discuss.md "Codex Review" section has the right instinct: don't present three independent pillars. Present:

1. **New method:** Adaptive reliable compensation — multiresolution downsampling + `c_geom` + `c_plateau` + `c_sign` confidence decomposition, with IDW as theoretically justified interpolant
2. **New system:** End-to-end CPU/GPU implementation
3. **New analysis:** Quality-performance-memory Pareto behavior + weight function ablation validating IDW

This avoids the "three incremental tweaks" reading that kills journal extensions.

**Action:** Write a 1-page outline with section headers and 2-3 sentence summaries per section. Get co-author buy-in before implementing anything new.

---

## Step 1: Reliability-Controlled Compensation (Highest Research Value, Moderate Effort)

This is the single most important new contribution. The IPDPS paper acknowledges quality degradation on S3D at small eb — fixing this is the clearest delta over the conference version.

### What to implement first

Start simple. Don't build the full `c_sign * c_geom * c_proxy` framework from day one.

**Phase 1 — Baseline controls (1-2 weeks):**
- Implement `|C_i| <= eta * eb` clamping as the hard guarantee. This is the real error bound argument, not re-quantization (see discuss.md Section 4 on why same-bin check alone gives 2*eb, not (1+eta)*eb).
- Implement local extremum rejection: if the compensated value creates a new local min/max not present in the decompressed data, shrink compensation.
- Run quality evaluation on all 5 datasets. Measure: how many degradation cases (PSNR worse than no-compensation) are eliminated?

**Phase 2 — Confidence scaling (DONE for c_geom and c_plateau):**
- `c_geom`: already implemented — scales down when `d1` is large.
- `c_plateau`: **NEW, implemented** — `min(1, cutoff/(d1+d2))`, orthogonal to c_geom. Default cutoff=20. Halves harm on dark_matter_density at eb=1e-2. See notes.md for full results.
- `c_sign`: still TODO — poll all 6 face-neighbors for sign agreement, not just the first differing neighbor.
- Fused form: `C_i = c_sign * c_geom * c_plateau * C0_i`. c_sign as hard gate; c_geom and c_plateau as soft multipliers.

**Phase 3 — Multiscale proxy (1-2 weeks):**
- Don't compute full-res AND downsampled EDT2 (defeats the purpose). Instead, use the cheap proxy: `c_proxy = 1 - local_mismatch(B2, upsample(downsample(B2)))`. This measures geometry aliasing from downsampling without running EDT twice.

**Completed (no longer TODO):**
- Weight function ablation: IDW (p=1), Power-p IDW (p=2), Smoothstep all implemented and tested. IDW is optimal — theoretically justified as MLE under piecewise-linear field model. Power-p and Smoothstep degrade quality monotonically. Publish as validation/ablation, not as a new contribution. See notes.md for details.

### Key result to target

A table showing: for each dataset x eb, the percentage of points where IPDPS compensation hurts quality (PSNR decreases) vs. the new controlled compensation. If you can take that from ~5-10% down to <1%, that's a strong result.

---

## Step 2: Multiresolution Pareto Analysis (High Research Value, Low Effort)

Most of this is already implemented (downsampled EDT2 exists). The work is experimental, not implementation.

### What to run

- Downsample factors: 1x, 2x, 4x, 8x for EDT round 2
- 4 datasets (pick one smooth, one structured, one with large plateaus, one turbulent)
- 3 error bounds per dataset (small/medium/large)
- That's 48 operating points — manageable

### What to plot

1. **Primary Pareto:** x = wall time overhead relative to full-res, y = delta-SSIM relative to full-res. One curve per dataset.
2. **Memory Pareto:** x = peak memory, y = quality. Matters for GPU where memory is constrained.
3. **The key finding to look for:** What is the maximum downsample factor per dataset that keeps SSIM within 0.1% of full-res? If this varies by dataset (it almost certainly does), that motivates adaptive selection.

### Connection to Step 1

The disagreement between full-res and downsampled compensation should feed into the confidence proxy (c_proxy). This makes downsampling both a speed trick AND a quality signal — that's the unified story.

---

## Step 3: End-to-End Integration Numbers (Medium-High Value, Medium Effort)

The GPU pipeline is already implemented (PBA+, JFA, kernels). What's missing is the integration story.

### Critical benchmarks to produce

1. **Overhead percentage:** cuSZ decompression alone vs. cuSZ + mitigation. If mitigation adds <20% overhead for significant SSIM gain, that's the headline number.
2. **Throughput comparison:** GB/s for the full pipeline on 256^3, 512^3, and 1024^3 if feasible.
3. **GPU EDT comparison:** PBA+ vs JFA vs exact. You have all three — show when each is best.
4. **Memory breakdown:** Peak GPU memory for each variant (full-res EDT2, downsampled EDT2, dist-only).

### What to write up from existing CPU work

The CPU optimizations (packed indexes, flat32, dist-only EDT, coord-type auto-selection) should be described but don't need extensive new experiments. A single table showing speedup from each optimization is sufficient.

---

## Step 4: Baselines and Comparisons

### Must-have baselines
- No compensation (raw decompressed)
- Gaussian filter
- Bilateral filter
- Full-res EDT+IDW
- Downsampled EDT+IDW (various factors)
- Controlled-compensation EDT+IDW (new)

### Nice-to-have
- **Bezier smoothing (Daoce, SC'24):** Check if code is public. If not, skip — reimplementation cost is too high for a baseline.
- **Additional compressors (SZp, FZ-GPU):** Worth doing if integration is easy. The IPDPS paper claims generality but only tested cuSZ and cuSZp2.

### Neural networks: Do NOT implement as a baseline

The discuss.md and notes.md are aligned on this and I agree. Including one CNN baseline invites scope creep. Instead:

- **In related work:** Cite 3-5 representative papers, acknowledge they may achieve higher PSNR.
- **In discussion:** One paragraph framing the tradeoff — "neural methods are high-quality, high-cost; this method is training-free, quantization-aware, HPC-deployable."
- **If a reviewer insists:** Have one compact comparison table ready for the rebuttal (small 3D CNN, same information budget, report quality + throughput + training cost + generalization).

---

## Step 5: Writing Strategy

### Content reuse from IPDPS
- Core algorithm description: condense to ~60% of original length. Don't copy-paste — rewrite with the adaptive/controlled formulation integrated from the start.
- MPI results: keep as-is or condense. They're valid but not the new contribution.
- All quality experiments: must be new or substantially extended (new baselines, new metrics, controlled compensation results).

### New sections needed
- Multiresolution framework with Pareto analysis
- Reliability-controlled compensation (formulation + ablation)
- GPU pipeline description and benchmarks
- End-to-end integration results
- Discussion section positioning vs. neural methods

### Page budget (TPDS ~14-16 pages)
- Introduction: 1.5 pages
- Background & Related Work: 2 pages (add NN discussion)
- Method: 4 pages (core algorithm condensed + new multiresolution + reliability control)
- Implementation: 2 pages (CPU optimizations + GPU pipeline)
- Evaluation: 4 pages (quality + Pareto + performance + end-to-end)
- Discussion: 0.5 pages
- Conclusion: 0.5 pages

---

## What I Would NOT Spend Time On

1. **Multi-GPU:** High effort, medium value. Skip unless you have easy access to multi-GPU nodes and the framework is straightforward to extend. Single-GPU is enough for TPDS.
2. **Formal error bound proof for downsampled EDT:** The empirical Pareto curves are more convincing and much less effort than a formal proof of the interpolation error cascade.
3. **Adaptive per-region downsample factor:** Interesting but complex to implement and evaluate. Save for future work. The uniform downsample factor with Pareto analysis is sufficient.
4. **More than 5 datasets:** Diminishing returns. Better to have deeper analysis on the existing datasets than breadth.

---

## Suggested Timeline (Rough)

| Phase | What | Deliverable |
|---|---|---|
| Weeks 1-2 | Paper outline + co-author alignment | 1-page outline approved |
| Weeks 2-4 | Reliability-controlled compensation (Phase 1-2) | Code + quality improvement table |
| Weeks 4-5 | Multiresolution Pareto experiments | 48-point Pareto dataset + plots |
| Weeks 5-6 | Multiscale proxy (Phase 3) + connect to Pareto | Unified confidence framework |
| Weeks 6-8 | End-to-end integration benchmarks | Throughput + memory tables |
| Weeks 8-10 | Additional baselines (Bezier if available, extra compressors) | Comparison tables |
| Weeks 10-14 | Writing + revision | Draft ready for internal review |

---

## Summary: The One-Sentence TPDS Pitch

"We extend EDT+IDW posterization mitigation with adaptive multiresolution compensation, reliability-controlled quality guarantees, and an end-to-end GPU pipeline, achieving [X]% fewer degradation cases, [Y] GB/s throughput, and [Z]% overhead when integrated with cuSZ."

Fill in X, Y, Z after Steps 1-3.
