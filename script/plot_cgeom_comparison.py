#!/usr/bin/env python3
"""
Compare c_geom (p80+floor=1) vs baseline across all 312 sweep configs.
Input:  pareto_results.csv           (baseline)
        pareto_results_cgeom_p80f1.csv (c_geom)
Output: pareto_plots/cgeom_*.pdf/png
"""

import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

SCRIPT_DIR = pathlib.Path(__file__).parent
OUT_DIR    = SCRIPT_DIR / "pareto_plots"
OUT_DIR.mkdir(exist_ok=True)

df_base  = pd.read_csv(SCRIPT_DIR / "pareto_results.csv").dropna(subset=['field'])
df_cgeom = pd.read_csv(SCRIPT_DIR / "pareto_results_cgeom_p80f1.csv").dropna(subset=['field'])

merged = df_base.merge(
    df_cgeom[['field','rel_eb','downsample_factor','final_psnr','final_ssim',
              'harm_rate','benefit_rate','geo_scale','sparsity']],
    on=['field','rel_eb','downsample_factor'],
    suffixes=('_base','_cgeom')
)
merged['psnr_delta']    = merged['final_psnr_cgeom'] - merged['final_psnr_base']
merged['ssim_delta']    = merged['final_ssim_cgeom'] - merged['final_ssim_base']
merged['psnr_gain_base'] = merged['final_psnr_base'] - merged['initial_psnr']
merged['psnr_gain_cgeom']= merged['final_psnr_cgeom'] - merged['initial_psnr']
merged['rel_eb'] = merged['rel_eb'].astype(str)
merged['downsample_factor'] = merged['downsample_factor'].astype(int)

EB_COLORS = {"0.001": "#1f77b4", "0.005": "#ff7f0e", "0.01": "#2ca02c"}
FACTOR_MARKERS = {0: "o", 2: "s", 4: "^", 8: "D"}
ebs     = sorted(merged['rel_eb'].unique(), key=float)
factors = sorted(merged['downsample_factor'].unique())
fields  = sorted(merged['field'].unique())

# ── Figure 1: PSNR delta (c_geom − baseline) per field, grouped by eb ──────
def plot_delta_summary():
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(fields))
    width = 0.25
    for i, eb in enumerate(ebs):
        sub = merged[merged['rel_eb'] == eb].groupby('field')['psnr_delta'].mean().reindex(fields, fill_value=0)
        bars = ax.bar(x + (i - 1) * width, sub.values, width, label=f"eb={eb}",
                      color=list(EB_COLORS.values())[i], alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Mean ΔPSNR (c_geom − baseline), dB", fontsize=9)
    ax.set_title("c_geom (p80 + floor=1.0) vs baseline: mean PSNR delta per field\n"
                 "(averaged over downsample factors; skipped fields shown as 0)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = OUT_DIR / f"cgeom_delta_per_field.{ext}"
        fig.savefig(p, bbox_inches='tight', dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)

# ── Figure 2: scatter PSNR_base vs PSNR_cgeom, coloured by eb ───────────────
def plot_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (col_b, col_c, label) in zip(axes, [
        ('final_psnr_base', 'final_psnr_cgeom', 'PSNR (dB)'),
        ('final_ssim_base', 'final_ssim_cgeom', 'SSIM'),
    ]):
        for eb in ebs:
            sub = merged[merged['rel_eb'] == eb]
            ax.scatter(sub[col_b], sub[col_c], alpha=0.4, s=15,
                       color=EB_COLORS[eb], label=f"eb={eb}")
        lims = [merged[[col_b, col_c]].min().min() * 0.999,
                merged[[col_b, col_c]].max().max() * 1.001]
        ax.plot(lims, lims, 'k--', linewidth=0.8, label='y=x')
        ax.set_xlabel(f"Baseline {label}", fontsize=9)
        ax.set_ylabel(f"c_geom {label}", fontsize=9)
        ax.set_title(f"Baseline vs c_geom: {label}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = OUT_DIR / f"cgeom_scatter.{ext}"
        fig.savefig(p, bbox_inches='tight', dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)

# ── Figure 3: PSNR delta vs downsample factor for active (non-zero) fields ──
def plot_by_factor():
    active = merged[merged['psnr_delta'].abs() > 0.001]['field'].unique()
    active = sorted(active)
    ncols = 4
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for idx, field in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        sub = merged[merged['field'] == field]
        for eb in ebs:
            s = sub[sub['rel_eb'] == eb].sort_values('downsample_factor')
            if s.empty: continue
            ax.plot(s['downsample_factor'], s['psnr_delta'],
                    marker='o', color=EB_COLORS[eb], linewidth=1.2, label=f"eb={eb}")
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_title(field, fontsize=7, pad=2)
        ax.set_xlabel("Downsample factor", fontsize=7)
        ax.set_ylabel("ΔPSNR (dB)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xticks([0, 2, 4, 8])
        ax.grid(linewidth=0.4, alpha=0.5)
    # hide unused
    for idx in range(len(active), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    eb_handles = [Line2D([0],[0], color=EB_COLORS[e], linewidth=1.5, label=f"eb={e}") for e in ebs]
    fig.legend(handles=eb_handles, loc='lower center', ncol=len(ebs), fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("c_geom ΔPSNR by downsample factor (active fields only)", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ('pdf', 'png'):
        p = OUT_DIR / f"cgeom_by_factor.{ext}"
        fig.savefig(p, bbox_inches='tight', dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)

# ── Print summary table ──────────────────────────────────────────────────────
def print_summary():
    print("\n=== c_geom (p80+floor=1) vs baseline: summary ===")
    print(f"Total configs: {len(merged)}")
    print(f"  Improved (>+0.01 dB):  {(merged['psnr_delta'] >  0.01).sum()}")
    print(f"  Neutral (|Δ|≤0.01 dB): {(merged['psnr_delta'].abs() <= 0.01).sum()}")
    print(f"  Hurt   (<-0.01 dB):   {(merged['psnr_delta'] < -0.01).sum()}")
    print(f"Mean ΔPSNR: {merged['psnr_delta'].mean():.4f} dB")
    print(f"Max improvement: {merged['psnr_delta'].max():.3f} dB  "
          f"({merged.loc[merged['psnr_delta'].idxmax(), 'field']}, "
          f"eb={merged.loc[merged['psnr_delta'].idxmax(), 'rel_eb']})")
    print(f"Max regression: {merged['psnr_delta'].min():.3f} dB  "
          f"({merged.loc[merged['psnr_delta'].idxmin(), 'field']}, "
          f"eb={merged.loc[merged['psnr_delta'].idxmin(), 'rel_eb']})")
    print("\nWorst regressions:")
    print(merged.nsmallest(5, 'psnr_delta')[
        ['field','rel_eb','downsample_factor','final_psnr_base','final_psnr_cgeom','psnr_delta']
    ].to_string(index=False))

# ── run ──────────────────────────────────────────────────────────────────────
print_summary()
plot_delta_summary()
plot_scatter()
plot_by_factor()
print(f"\nAll plots saved to {OUT_DIR}/")
