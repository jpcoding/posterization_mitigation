#!/usr/bin/env python3
"""
Plot Pareto curves from pareto_results.csv.

Two figure types:
  1. pareto_psnr.pdf  — grid: rows=field, cols=eb  |  x=speedup, y=PSNR gain vs factor=0
  2. pareto_ssim.pdf  — same layout but y=SSIM gain vs factor=0

Usage:
  python3 plot_pareto.py [pareto_results.csv]
"""

import sys
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------- config ----------
SCRIPT_DIR = pathlib.Path(__file__).parent
CSV_PATH   = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "pareto_results.csv"
OUT_DIR    = SCRIPT_DIR / "pareto_plots"
OUT_DIR.mkdir(exist_ok=True)

FACTOR_LABELS = {0: "full-res", 2: "2×", 4: "4×", 8: "8×"}
FACTOR_MARKERS = {0: "o", 2: "s", 4: "^", 8: "D"}
EB_COLORS = {"1e-3": "#1f77b4", "5e-3": "#ff7f0e", "1e-2": "#2ca02c",
             "0.001": "#1f77b4", "0.005": "#ff7f0e", "0.01": "#2ca02c"}

def eb_label(eb: str) -> str:
    fmap = {"1e-3": "1×10⁻³", "5e-3": "5×10⁻³", "1e-2": "1×10⁻²",
            "0.001": "1×10⁻³", "0.005": "5×10⁻³", "0.01": "1×10⁻²"}
    return fmap.get(str(eb), str(eb))

# ---------- load ----------
df = pd.read_csv(CSV_PATH)
df["rel_eb"] = df["rel_eb"].astype(str)
df["downsample_factor"] = df["downsample_factor"].astype(int)
df["psnr_gain"] = df["final_psnr"] - df["initial_psnr"]
df["ssim_gain"] = df["final_ssim"] - df["initial_ssim"]

# speedup relative to factor=0 within same (dataset, field, rel_eb)
baseline = df[df["downsample_factor"] == 0][["dataset","field","rel_eb","edt_total_s","psnr_gain","ssim_gain","final_psnr","final_ssim"]].copy()
baseline = baseline.rename(columns={
    "edt_total_s": "base_time",
    "psnr_gain":   "base_psnr_gain",
    "ssim_gain":   "base_ssim_gain",
    "final_psnr":  "base_psnr",
    "final_ssim":  "base_ssim",
})
df = df.merge(baseline, on=["dataset","field","rel_eb"])
df["speedup"]       = df["base_time"] / df["edt_total_s"]
df["delta_psnr"]    = df["final_psnr"] - df["base_psnr"]    # vs factor=0
df["delta_ssim"]    = df["final_ssim"] - df["base_ssim"]    # vs factor=0
df["delta_psnr_gain"] = df["psnr_gain"] - df["base_psnr_gain"]
df["delta_ssim_gain"] = df["ssim_gain"] - df["base_ssim_gain"]

datasets = df["dataset"].unique()
ebs      = sorted(df["rel_eb"].unique(), key=float)
factors  = sorted(df["downsample_factor"].unique())

# ---------- helper: draw one ax ----------
def draw_ax(ax, sub, metric_col, factor_anchor=0):
    """sub: rows for one (dataset, field, eb). metric_col: 'delta_psnr' or 'delta_ssim'."""
    sub = sub.sort_values("speedup")
    color = EB_COLORS.get(sub["rel_eb"].iloc[0], "gray")
    xs = sub["speedup"].tolist()
    ys = sub[metric_col].tolist()
    ax.plot(xs, ys, color=color, linewidth=1.2, zorder=2)
    for _, row in sub.iterrows():
        f = int(row["downsample_factor"])
        ax.scatter(row["speedup"], row[metric_col],
                   marker=FACTOR_MARKERS[f], color=color, s=50, zorder=3,
                   label=f"factor={FACTOR_LABELS[f]}" if row["rel_eb"] == ebs[0] else None)

# ---------- figure 1: per-field, x=speedup, y=delta_psnr_gain ----------
# Layout: one row per field, one col per dataset
def make_grid_fig(metric_col, ylabel, suffix, ylim=None):
    fields_by_ds = {ds: sorted(df[df["dataset"]==ds]["field"].unique()) for ds in datasets}
    max_fields = max(len(v) for v in fields_by_ds.values())
    ncols = len(datasets)
    nrows = max_fields

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.0 * nrows),
                              squeeze=False)

    for col_i, ds in enumerate(sorted(datasets)):
        fields = sorted(fields_by_ds[ds])
        for row_i, field in enumerate(fields):
            ax = axes[row_i][col_i]
            for eb in ebs:
                sub = df[(df["dataset"]==ds) & (df["field"]==field) & (df["rel_eb"]==eb)]
                if sub.empty:
                    continue
                draw_ax(ax, sub, metric_col)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.axvline(1, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
            ax.set_title(f"{ds}/{field}", fontsize=8, pad=3)
            ax.set_xlabel("Speedup vs full-res EDT2", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, linewidth=0.4, alpha=0.5)
        # hide unused rows
        for row_i in range(len(fields), nrows):
            axes[row_i][col_i].set_visible(False)

    # legend: eb colors
    from matplotlib.lines import Line2D
    eb_handles = [Line2D([0],[0], color=EB_COLORS.get(e,"gray"), linewidth=1.5,
                          label=f"eb={eb_label(e)}") for e in ebs]
    factor_handles = [plt.scatter([],[], marker=FACTOR_MARKERS[f], color="gray", s=40,
                                   label=f"{FACTOR_LABELS[f]}") for f in factors if f != 0]
    fig.legend(handles=eb_handles + factor_handles,
               loc="lower center", ncol=len(ebs)+len(factors)-1,
               fontsize=7, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Pareto: speedup vs {ylabel.split('(')[0].strip()}", fontsize=10, y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out = OUT_DIR / f"pareto_{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    out_png = OUT_DIR / f"pareto_{suffix}.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out_png}")


# ---------- figure 2: absolute quality vs absolute time per field ----------
def make_abs_fig(metric_col, ylabel, suffix):
    fields_by_ds = {ds: sorted(df[df["dataset"]==ds]["field"].unique()) for ds in datasets}
    max_fields = max(len(v) for v in fields_by_ds.values())
    ncols = len(datasets)
    nrows = max_fields

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.0 * nrows),
                              squeeze=False)

    for col_i, ds in enumerate(sorted(datasets)):
        fields = sorted(fields_by_ds[ds])
        for row_i, field in enumerate(fields):
            ax = axes[row_i][col_i]
            for eb in ebs:
                sub = df[(df["dataset"]==ds) & (df["field"]==field) & (df["rel_eb"]==eb)]
                if sub.empty:
                    continue
                sub = sub.sort_values("edt_total_s")
                color = EB_COLORS.get(eb, "gray")
                xs = sub["edt_total_s"].tolist()
                ys = sub[metric_col].tolist()
                ax.plot(xs, ys, color=color, linewidth=1.2, zorder=2)
                for _, row in sub.iterrows():
                    f = int(row["downsample_factor"])
                    ax.scatter(row["edt_total_s"], row[metric_col],
                               marker=FACTOR_MARKERS[f], color=color, s=50, zorder=3)
                    ax.annotate(FACTOR_LABELS[f], (row["edt_total_s"], row[metric_col]),
                                textcoords="offset points", xytext=(4, 2), fontsize=5)
            ax.set_title(f"{ds}/{field}", fontsize=8, pad=3)
            ax.set_xlabel("edt_total (s)", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, linewidth=0.4, alpha=0.5)
        for row_i in range(len(fields), nrows):
            axes[row_i][col_i].set_visible(False)

    from matplotlib.lines import Line2D
    eb_handles = [Line2D([0],[0], color=EB_COLORS.get(e,"gray"), linewidth=1.5,
                          label=f"eb={eb_label(e)}") for e in ebs]
    fig.legend(handles=eb_handles, loc="lower center", ncol=len(ebs),
               fontsize=7, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Time vs {ylabel.split('(')[0].strip()}", fontsize=10, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out = OUT_DIR / f"pareto_{suffix}_abs.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    out_png = OUT_DIR / f"pareto_{suffix}_abs.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out_png}")


# ---------- summary table ----------
def print_summary():
    print("\n=== Summary: NYX velocity_x ===")
    sub = df[(df["dataset"]=="nyx") & (df["field"]=="velocity_x")]
    if sub.empty:
        return
    print(f"{'eb':<8} {'factor':<8} {'speedup':>8} {'delta_psnr':>12} {'delta_ssim':>12} {'edt_total_s':>12}")
    for _, row in sub.sort_values(["rel_eb","downsample_factor"]).iterrows():
        print(f"{row['rel_eb']:<8} {row['downsample_factor']:<8} "
              f"{row['speedup']:>8.2f} {row['delta_psnr']:>+12.4f} "
              f"{row['delta_ssim']:>+12.6f} {row['edt_total_s']:>12.4f}")


# ---------- run ----------
print(f"Loaded {len(df)} rows from {CSV_PATH}")
print(f"Datasets: {list(df['dataset'].unique())}")
print(f"Fields:   {list(df['field'].unique())}")
print(f"EBs:      {ebs}")
print(f"Factors:  {factors}")

make_grid_fig("delta_psnr",      "ΔPSNR vs full-res (dB)",  "psnr_delta")
make_grid_fig("delta_ssim",      "ΔSSIM vs full-res",       "ssim_delta")
make_abs_fig( "final_psnr",      "Final PSNR (dB)",         "psnr")
make_abs_fig( "final_ssim",      "Final SSIM",              "ssim")

print_summary()
print(f"\nPlots in: {OUT_DIR}/")
