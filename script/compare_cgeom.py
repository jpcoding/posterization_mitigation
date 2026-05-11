#!/usr/bin/env python3
"""Compare cgeom=off vs cgeom=on rows in pareto_cgeom_results.csv."""

import csv
import sys
from collections import defaultdict

def f(x):
    try: return float(x)
    except: return None

path = sys.argv[1] if len(sys.argv) > 1 else "script/pareto_cgeom_results.csv"

# Index: (dataset, field, rel_eb) -> {cgeom -> row}
rows = defaultdict(dict)
with open(path) as fh:
    rd = csv.DictReader(fh)
    for r in rd:
        key = (r["dataset"], r["field"], r["rel_eb"])
        rows[key][r["cgeom"]] = r

print("\n## Per-field deltas (cgeom_on @ p80) − (cgeom_off baseline)")
print(f"{'dataset':<10} {'field':<22} {'rel_eb':<7} {'gs':>7} {'psnr_off':>9} {'psnr_on':>9} {'Δpsnr':>8} {'Δssim':>8} {'harm_off':>9} {'harm_on':>9} {'Δharm':>8}")
print("-" * 130)

agg = defaultdict(lambda: {"n": 0, "n_better": 0, "n_worse": 0, "sum_dpsnr": 0.0, "sum_dharm": 0.0})

for key, dct in sorted(rows.items()):
    if "off" not in dct or "on" not in dct:
        continue
    off, on = dct["off"], dct["on"]
    dataset, field, eb = key
    gs = on.get("geo_scale", "")
    p_off, p_on = f(off["final_psnr"]), f(on["final_psnr"])
    s_off, s_on = f(off["final_ssim"]), f(on["final_ssim"])
    h_off, h_on = f(off["harm_rate"]), f(on["harm_rate"])
    dpsnr = (p_on - p_off) if (p_on is not None and p_off is not None) else None
    dssim = (s_on - s_off) if (s_on is not None and s_off is not None) else None
    dharm = (h_on - h_off) if (h_on is not None and h_off is not None) else None
    print(f"{dataset:<10} {field:<22} {eb:<7} {gs[:7]:>7} "
          f"{p_off if p_off is not None else 'na':>9.3f} {p_on if p_on is not None else 'na':>9.3f} "
          f"{dpsnr if dpsnr is not None else 0:>+8.3f} "
          f"{dssim if dssim is not None else 0:>+8.4f} "
          f"{h_off if h_off is not None else 0:>9.4f} {h_on if h_on is not None else 0:>9.4f} "
          f"{dharm if dharm is not None else 0:>+8.4f}")
    if dpsnr is not None:
        a = agg[dataset]
        a["n"] += 1
        if dpsnr > 0.001:
            a["n_better"] += 1
        elif dpsnr < -0.001:
            a["n_worse"] += 1
        a["sum_dpsnr"] += dpsnr
        if dharm is not None:
            a["sum_dharm"] += dharm

print("\n## Per-dataset summary")
print(f"{'dataset':<12} {'n':>5} {'n_better':>10} {'n_worse':>10} {'mean ΔPSNR':>12} {'mean Δharm':>12}")
for dataset, a in agg.items():
    n = a["n"] or 1
    print(f"{dataset:<12} {a['n']:>5} {a['n_better']:>10} {a['n_worse']:>10} "
          f"{a['sum_dpsnr']/n:>+12.4f} {a['sum_dharm']/n:>+12.4f}")
