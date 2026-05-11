#!/usr/bin/env python3
"""Compare original pareto_results.csv (c_geom OFF) against pareto_results_cgeom_p80f1.csv."""

import csv
import os
import sys
from collections import defaultdict

def f(x):
    try: return float(x)
    except: return None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "script", "pareto_results.csv")
NEW  = os.path.join(ROOT, "script", "pareto_results_cgeom_p80f1.csv")

# Map: (dataset, field, rel_eb, factor) -> row
def load(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k = (r["dataset"], r["field"], r["rel_eb"], r["downsample_factor"])
            out[k] = r
    return out

base = load(BASE)
new  = load(NEW)

# Per-(field, eb) summary aggregated across factors
agg = defaultdict(lambda: defaultdict(lambda: {
    "n": 0, "n_better": 0, "n_worse": 0, "n_disaster": 0, "sum_dpsnr": 0.0,
    "max_neg": 0.0, "max_pos": 0.0,
}))

per_factor = []
catastrophic = []
big_wins = []

for k in sorted(base.keys()):
    if k not in new: continue
    b, n = base[k], new[k]
    p_b, p_n = f(b["final_psnr"]), f(n["final_psnr"])
    if p_b is None or p_n is None: continue
    s_b, s_n = f(b["final_ssim"]), f(n["final_ssim"])
    dpsnr = p_n - p_b
    dssim = (s_n - s_b) if (s_b is not None and s_n is not None) else None
    per_factor.append((k, p_b, p_n, dpsnr, dssim, n.get("geo_scale", "")))
    if dpsnr < -0.1:
        catastrophic.append((k, p_b, p_n, dpsnr))
    if dpsnr > 0.5:
        big_wins.append((k, p_b, p_n, dpsnr))
    a = agg[k[0]][k[1]]  # by dataset
    a["n"] += 1
    if dpsnr > 0.005:    a["n_better"] += 1
    elif dpsnr < -0.005: a["n_worse"] += 1
    if dpsnr < -0.1:     a["n_disaster"] += 1
    a["sum_dpsnr"] += dpsnr
    a["max_neg"] = min(a["max_neg"], dpsnr)
    a["max_pos"] = max(a["max_pos"], dpsnr)

print(f"\n## Per-dataset summary (averaged over {sum(a['n'] for d in agg.values() for a in d.values())} aligned rows)")
print(f"{'dataset':<12} {'n':>5} {'n_better':>9} {'n_worse':>9} {'n_disaster<-0.1':>16} {'mean ΔPSNR':>12} {'max +ΔPSNR':>12} {'max -ΔPSNR':>12}")
for ds in sorted(agg.keys()):
    nA = sum(a["n"] for a in agg[ds].values())
    nB = sum(a["n_better"] for a in agg[ds].values())
    nW = sum(a["n_worse"] for a in agg[ds].values())
    nD = sum(a["n_disaster"] for a in agg[ds].values())
    s = sum(a["sum_dpsnr"] for a in agg[ds].values())
    mx = max(a["max_pos"] for a in agg[ds].values())
    mn = min(a["max_neg"] for a in agg[ds].values())
    print(f"{ds:<12} {nA:>5} {nB:>9} {nW:>9} {nD:>16} {s/nA if nA else 0:>+12.4f} {mx:>+12.4f} {mn:>+12.4f}")

print("\n## Catastrophic regressions (ΔPSNR < -0.1 dB)")
if not catastrophic:
    print("  (none)")
for k, p_b, p_n, d in sorted(catastrophic, key=lambda x: x[3]):
    print(f"  {k[0]:<10} {k[1]:<22} eb={k[2]:<6} f={k[3]:<2}  {p_b:>8.3f} → {p_n:>8.3f}  ΔPSNR={d:+.3f}")

print("\n## Big wins (ΔPSNR > 0.5 dB)")
if not big_wins:
    print("  (none)")
for k, p_b, p_n, d in sorted(big_wins, key=lambda x: -x[3])[:20]:
    print(f"  {k[0]:<10} {k[1]:<22} eb={k[2]:<6} f={k[3]:<2}  {p_b:>8.3f} → {p_n:>8.3f}  ΔPSNR={d:+.3f}")

# Per-factor breakdown
print("\n## ΔPSNR aggregated by downsample factor")
fagg = defaultdict(lambda: {"n": 0, "sum": 0.0, "n_disaster": 0})
for (k, p_b, p_n, d, ds, gs) in per_factor:
    fagg[k[3]]["n"] += 1
    fagg[k[3]]["sum"] += d
    if d < -0.1: fagg[k[3]]["n_disaster"] += 1
for fac in sorted(fagg.keys(), key=lambda s: int(s)):
    a = fagg[fac]
    print(f"  factor={fac:<2}  n={a['n']:>3}  mean ΔPSNR={a['sum']/a['n']:>+.4f}  disasters={a['n_disaster']}")
