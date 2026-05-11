#!/usr/bin/env python3
"""
Profile where false corrections (harms) cluster in a single compensation run.

Expects sidecar files written by `test_quantize_and_edt --profile_harm <prefix>`:
  <prefix>_d_edge.f32      : per-voxel d1 (EDT round 1)
  <prefix>_d_neutral.f32   : per-voxel d2 (EDT round 2)
  <prefix>_quant.i32       : per-voxel quantization index
  <prefix>_comp.f32        : per-voxel compensation map
  <prefix>_dec.f32         : per-voxel decompressed (pre-comp) data
And the original file separately via --original.

Reports the harm rate and net MSE change as a function of each of:
  - d1 (distance to nearest edge)
  - IDW magnitude d2/(d1+d2)
  - |comp[i]|
  - sign-agreement fraction (recomputed from quant_index, 6 face neighbors)
  - local distinct-quant count in 3x3x3 window
And a 2-D (d1, idw_mag) heat-map.
"""

import argparse
import sys
import numpy as np


def load_raw(path, dtype, shape):
    arr = np.fromfile(path, dtype=dtype)
    if arr.size != np.prod(shape):
        sys.exit(f"size mismatch: {path} has {arr.size}, expected {np.prod(shape)}")
    return arr.reshape(shape)


def sign_agree_fraction(quant, comp_sign):
    """For each voxel, fraction of face-neighbors with differing quant whose
    implied direction (sign(neighbor_q - cur_q) along positive axis,
    sign(cur_q - neighbor_q) along negative axis) matches comp_sign.

    Matches the c_sign logic in compensation.hpp.
    Returns array of same shape; voxels with no differing neighbor get NaN.
    """
    n_differ = np.zeros_like(quant, dtype=np.int8)
    n_agree = np.zeros_like(quant, dtype=np.int8)
    # 6 face neighbors. For axis a, shift -1 = "negative-step" direction, +1 = "positive-step".
    # Following the boundary-detect convention: at a neg-step neighbor, implied sign = sign(cur - nb).
    # At a pos-step neighbor, implied = sign(nb - cur). Together they correctly flag the high side.
    for axis in range(3):
        for shift in (-1, +1):
            nb = np.roll(quant, -shift, axis=axis)  # nb = neighbor in +shift direction relative to cur
            diff = nb != quant
            if shift > 0:
                implied = np.sign(nb.astype(np.int32) - quant.astype(np.int32))
            else:
                implied = np.sign(quant.astype(np.int32) - nb.astype(np.int32))
            n_differ += diff.astype(np.int8)
            n_agree += (diff & (implied.astype(np.int8) == comp_sign)).astype(np.int8)
    frac = np.full(quant.shape, np.nan, dtype=np.float32)
    valid = n_differ > 0
    frac[valid] = n_agree[valid] / n_differ[valid]
    return frac


def local_distinct_3x3x3(quant):
    """Approximation: count of distinct values among the 27 voxels in a 3x3x3 box centered on each voxel.
    Exact count requires sort/unique per window which is expensive; we approximate via std() (cheap)
    and also via a coarser distinct count using comparisons against the center.
    Returns float32 with std-dev of indices in the 3x3x3 window — a reliable proxy for "roughness".
    """
    from scipy.ndimage import uniform_filter
    q = quant.astype(np.float32)
    mean = uniform_filter(q, size=3, mode='nearest')
    mean_sq = uniform_filter(q * q, size=3, mode='nearest')
    var = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(var, dtype=np.float32)


def bin_stats(feature, pre_err, post_err, edges, label):
    n_total = np.zeros(len(edges) - 1, dtype=np.int64)
    n_harm = np.zeros_like(n_total)
    n_help = np.zeros_like(n_total)
    sum_dsq = np.zeros(len(edges) - 1, dtype=np.float64)
    delta_sq = post_err.astype(np.float64) ** 2 - pre_err.astype(np.float64) ** 2
    eps = 1e-30
    is_harm = post_err > pre_err + eps
    is_help = post_err < pre_err - eps
    # Digitize feature into bins
    idx = np.digitize(feature.ravel(), edges) - 1
    flat_delta = delta_sq.ravel()
    flat_harm = is_harm.ravel()
    flat_help = is_help.ravel()
    valid = (idx >= 0) & (idx < len(n_total))
    # Mask NaN feature values
    valid &= np.isfinite(feature.ravel())
    idx = idx[valid]
    flat_delta = flat_delta[valid]
    flat_harm = flat_harm[valid]
    flat_help = flat_help[valid]
    np.add.at(n_total, idx, 1)
    np.add.at(n_harm, idx, flat_harm)
    np.add.at(n_help, idx, flat_help)
    np.add.at(sum_dsq, idx, flat_delta)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if n_total[i] == 0:
            continue
        harm_rate = n_harm[i] / n_total[i]
        net_dsq = sum_dsq[i] / n_total[i]
        rows.append((lo, hi, n_total[i], harm_rate, net_dsq))
    print(f"\n### Harm vs {label}")
    print(f"{'bin_low':>10}  {'bin_high':>10}  {'count':>12}  {'harm_rate':>10}  {'net_dsq':>14}")
    for lo, hi, n, hr, nd in rows:
        print(f"{lo:>10.4g}  {hi:>10.4g}  {n:>12d}  {hr:>10.4f}  {nd:>14.4e}")


def heatmap_2d(f1, f2, pre_err, post_err, edges1, edges2, label1, label2):
    delta_sq = post_err.astype(np.float64) ** 2 - pre_err.astype(np.float64) ** 2
    eps = 1e-30
    is_harm = post_err > pre_err + eps
    nb1, nb2 = len(edges1) - 1, len(edges2) - 1
    n_total = np.zeros((nb1, nb2), dtype=np.int64)
    n_harm = np.zeros_like(n_total)
    sum_dsq = np.zeros((nb1, nb2), dtype=np.float64)
    i1 = np.digitize(f1.ravel(), edges1) - 1
    i2 = np.digitize(f2.ravel(), edges2) - 1
    valid = (i1 >= 0) & (i1 < nb1) & (i2 >= 0) & (i2 < nb2)
    valid &= np.isfinite(f1.ravel()) & np.isfinite(f2.ravel())
    i1, i2 = i1[valid], i2[valid]
    flat_harm = is_harm.ravel()[valid]
    flat_delta = delta_sq.ravel()[valid]
    np.add.at(n_total, (i1, i2), 1)
    np.add.at(n_harm, (i1, i2), flat_harm)
    np.add.at(sum_dsq, (i1, i2), flat_delta)
    print(f"\n### 2-D heatmap: harm-rate by ({label1} × {label2})")
    # Header
    header = f"{label1+'/'+label2:>12}  " + "  ".join(f"{(edges2[j]+edges2[j+1])/2:>8.3g}" for j in range(nb2))
    print(header)
    for i in range(nb1):
        mid = (edges1[i] + edges1[i + 1]) / 2.0
        cells = []
        for j in range(nb2):
            if n_total[i, j] == 0:
                cells.append(f"{'-':>8}")
            else:
                rate = n_harm[i, j] / n_total[i, j]
                cells.append(f"{rate:>8.3f}")
        print(f"{mid:>12.3g}  " + "  ".join(cells))
    # Also print net dsq heatmap (signed): negative = beneficial, positive = harmful
    print(f"\n### 2-D heatmap: net per-voxel Δ(err²) by ({label1} × {label2}) — negative = beneficial")
    print(header)
    for i in range(nb1):
        mid = (edges1[i] + edges1[i + 1]) / 2.0
        cells = []
        for j in range(nb2):
            if n_total[i, j] == 0:
                cells.append(f"{'-':>8}")
            else:
                nd = sum_dsq[i, j] / n_total[i, j]
                cells.append(f"{nd:>8.1e}")
        print(f"{mid:>12.3g}  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="sidecar prefix passed to --profile_harm")
    ap.add_argument("--original", required=True, help="path to original .f32")
    ap.add_argument("--dims", nargs=3, type=int, required=True)
    ap.add_argument("--no-local-distinct", action="store_true",
                    help="skip the local-distinct (scipy) breakdown")
    args = ap.parse_args()

    shape = tuple(args.dims)
    p = args.prefix
    print(f"Loading {p}_*.{{f32,i32}} and {args.original}, dims={shape} ...")
    original = load_raw(args.original, np.float32, shape)
    dec = load_raw(f"{p}_dec.f32", np.float32, shape)
    comp = load_raw(f"{p}_comp.f32", np.float32, shape)
    quant = load_raw(f"{p}_quant.i32", np.int32, shape)
    d1 = load_raw(f"{p}_d_edge.f32", np.float32, shape)
    d2 = load_raw(f"{p}_d_neutral.f32", np.float32, shape)

    pre_err = np.abs(dec - original)
    post_err = np.abs(dec + comp - original)

    eps = 1e-30
    is_harm = post_err > pre_err + eps
    is_help = post_err < pre_err - eps
    n_total = original.size
    n_harm = int(is_harm.sum())
    n_help = int(is_help.sum())
    delta_sq = post_err.astype(np.float64) ** 2 - pre_err.astype(np.float64) ** 2
    net_mse_delta = delta_sq.mean()
    rms_harm = np.sqrt(delta_sq[is_harm].mean()) if n_harm > 0 else 0.0
    rms_help = np.sqrt((-delta_sq[is_help]).mean()) if n_help > 0 else 0.0
    print("\n## Summary")
    print(f"voxels:        {n_total}")
    print(f"harm rate:     {n_harm/n_total:.6f}  ({n_harm} voxels)")
    print(f"benefit rate:  {n_help/n_total:.6f}  ({n_help} voxels)")
    print(f"net mean Δ(err²): {net_mse_delta:.6e}  (negative = beneficial)")
    print(f"harm   RMS Δerr: {rms_harm:.6e}")
    print(f"benefit RMS Δerr: {rms_help:.6e}")

    # ===== Feature breakdowns =====
    idw_mag = np.where((d1 + d2) > 0, d2 / (d1 + d2), 0.0)
    comp_mag = np.abs(comp)

    bin_stats(d1, pre_err, post_err,
              edges=np.array([0, 0.5, 1, 2, 3, 5, 8, 13, 20, 35, 60, 1e9]),
              label="d1 (dist to nearest edge)")

    bin_stats(idw_mag, pre_err, post_err,
              edges=np.linspace(0, 1.0, 11),
              label="IDW magnitude d2/(d1+d2)")

    bin_stats(comp_mag, pre_err, post_err,
              edges=np.linspace(0, comp_mag.max() + 1e-12, 11),
              label="|comp|")

    comp_sign = np.sign(comp).astype(np.int8)
    sign_frac = sign_agree_fraction(quant, comp_sign)
    bin_stats(sign_frac, pre_err, post_err,
              edges=np.array([-0.01, 0.0, 0.25, 0.5, 0.75, 1.01]),
              label="sign-agreement fraction (NaN means no differing neighbor)")

    if not args.no_local_distinct:
        try:
            roughness = local_distinct_3x3x3(quant)
            bin_stats(roughness, pre_err, post_err,
                      edges=np.array([0, 0.1, 0.3, 0.7, 1.5, 3, 1e9]),
                      label="local quant-index std (3×3×3 window)")
        except ImportError:
            print("(scipy not available, skipping local-distinct breakdown)")

    # 2-D heatmap (most informative single view)
    heatmap_2d(d1, idw_mag, pre_err, post_err,
               edges1=np.array([0, 0.5, 1, 2, 3, 5, 8, 13, 20, 35, 60, 1e9]),
               edges2=np.linspace(0, 1.0, 6),
               label1="d1", label2="IDW")


if __name__ == "__main__":
    main()
