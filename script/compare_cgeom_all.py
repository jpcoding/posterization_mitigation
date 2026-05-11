#!/usr/bin/env python3
"""Join cgeom sweeps (off, p80, p80f1, gs5) and compare against baseline."""

import csv
import os
import sys
from collections import defaultdict

def f(x):
    try: return float(x)
    except: return None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "script", "pareto_cgeom_results.csv")
V2 = os.path.join(ROOT, "script", "pareto_cgeom_v2_results.csv")

# Map: (dataset, field, rel_eb) -> {cfg -> row}
rows = defaultdict(dict)

def load(path):
    if not os.path.exists(path): return
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r["dataset"], r["field"], r["rel_eb"])
            cfg = r["cgeom"]
            # Normalize: v1 used "off"/"on" (on = p80, no floor)
            if cfg == "on": cfg = "p80"
            rows[key][cfg] = r

load(V1)
load(V2)

CONFIGS = ["p80", "p80f1", "gs5"]

print(f"\n## Per-field ΔPSNR vs cgeom=off baseline")
print(f"{'dataset':<10} {'field':<22} {'eb':<7}  {'PSNR_off':>9}  "
      + "  ".join(f"{c+'_dPSNR':>9}" for c in CONFIGS) + "   "
      + "  ".join(f"{c+'_gs':>8}" for c in CONFIGS))
print("-" * 140)

# Aggregates
agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "n_better": 0, "n_worse": 0,
                                                "n_catastrophic": 0, "sum_dpsnr": 0.0,
                                                "best_psnr_count": 0}))

best_psnr_winners = defaultdict(int)

for key in sorted(rows.keys()):
    dataset, field, eb = key
    d = rows[key]
    if "off" not in d:
        continue
    p_off = f(d["off"]["final_psnr"])
    if p_off is None: continue
    line = f"{dataset:<10} {field:<22} {eb:<7}  {p_off:>9.3f}  "
    dps = {}
    gss = {}
    for cfg in CONFIGS:
        if cfg in d:
            p_c = f(d[cfg]["final_psnr"])
            gs = d[cfg].get("geo_scale", "")
            if p_c is not None:
                dpsnr = p_c - p_off
                dps[cfg] = dpsnr
                line += f"{dpsnr:>+9.3f}  "
                gss[cfg] = gs
                # Aggregate
                a = agg[dataset][cfg]
                a["n"] += 1
                if dpsnr > 0.005:    a["n_better"] += 1
                elif dpsnr < -0.005: a["n_worse"]  += 1
                if dpsnr < -0.5:     a["n_catastrophic"] += 1
                a["sum_dpsnr"] += dpsnr
            else:
                line += f"{'na':>9}  "
                gss[cfg] = ""
        else:
            line += f"{'?':>9}  "
            gss[cfg] = ""
    line += "  ".join(f"{gss.get(c,'')[:8]:>8}" for c in CONFIGS)
    # Mark the winning config
    if dps:
        best_cfg = max(dps, key=dps.get)
        if dps[best_cfg] > 0.005:
            best_psnr_winners[best_cfg] += 1
            line += f"  ← {best_cfg}"
        elif dps[best_cfg] < -0.005:
            best_psnr_winners["none"] += 1
            line += "  (all hurt)"
        else:
            best_psnr_winners["tie"] += 1
    print(line)

print("\n## Per-dataset, per-config summary")
header = f"{'dataset':<12} {'cfg':<7} {'n':>4} {'n_better':>9} {'n_worse':>9} {'n_disaster<-0.5':>16} {'mean ΔPSNR':>12}"
print(header)
print("-" * len(header))
for ds in sorted(agg.keys()):
    for cfg in CONFIGS:
        a = agg[ds][cfg]
        if a["n"] == 0: continue
        mean = a["sum_dpsnr"] / a["n"]
        print(f"{ds:<12} {cfg:<7} {a['n']:>4} {a['n_better']:>9} {a['n_worse']:>9} "
              f"{a['n_catastrophic']:>16} {mean:>+12.4f}")

print("\n## Winner count (PSNR-maximizing config per (field, eb), Δ > 0.005)")
for k, v in sorted(best_psnr_winners.items(), key=lambda x: -x[1]):
    print(f"  {k:<12} {v}")
