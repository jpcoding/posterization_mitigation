#!/usr/bin/env python3
"""
Parse MPI output files — trimmed mean version.
Drops the largest and smallest iteration (by avg exchange time),
averages over the remaining 3.

Usage:
  python parse_exchange_time_trimmed.py --dir blcoking
  python parse_exchange_time_trimmed.py --dir non_blocking
"""

import re
import glob
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True, help='Subfolder containing opt_*.out files')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, args.dir)
LABEL = args.dir

EXPECTED_ITERS = 5

# Map case name to (output file glob, expected num_ranks)
cases = {
    "4x4x4 (64 ranks)":  ("opt_4x4x4*.out", 64),
    "8x4x4 (128 ranks)": ("opt_8x4x4*.out", 128),
    "8x8x4 (256 ranks)": ("opt_8x8x4*.out", 256),
}


def parse_output_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    iter_blocks = re.split(r'Running with rel_eb=[\d.]+, iteration \d+', content)
    iter_blocks = [b for b in iter_blocks if b.strip()]

    results = []
    for block in iter_blocks:
        total_time_match = re.search(r'Rank 0, time:\s+([\d.]+)', block)
        if not total_time_match:
            continue
        total_time = float(total_time_match.group(1))

        exchange_times = re.findall(r'Rank \d+, total exchange time:\s+([\d.]+)', block)
        if not exchange_times:
            continue

        exchange_times = [float(t) for t in exchange_times]
        avg_exchange = sum(exchange_times) / len(exchange_times)

        results.append({
            'total_time': total_time,
            'avg_exchange_time': avg_exchange,
            'num_ranks': len(exchange_times),
            'proportion': avg_exchange / total_time * 100,
        })

    return results


def trimmed_mean(iterations):
    """Drop min and max by total_time, return remaining."""
    if len(iterations) < 3:
        return iterations  # not enough to trim
    sorted_iters = sorted(iterations, key=lambda x: x['total_time'])
    return sorted_iters[1:-1]  # drop first (min) and last (max)


# ==========================================================
# Parse all cases
# ==========================================================
print("=" * 70)
print(f"MPI Communication Time Analysis — {LABEL} Trimmed Mean (drop min/max)")
print("=" * 70)

summary = {}
all_iterations = {}  # for variation plot

for case_name, (pattern, expected_ranks) in cases.items():
    files = sorted(glob.glob(os.path.join(BASE_DIR, pattern)))
    files = [f for f in files if f.endswith('.out')]

    if not files:
        print(f"\n--- {case_name}: NO OUTPUT FILE FOUND ---")
        summary[case_name] = None
        all_iterations[case_name] = []
        continue

    filepath = files[0]
    print(f"\n--- {case_name} ---")
    print(f"File: {os.path.basename(filepath)}")

    iterations = parse_output_file(filepath)
    all_iterations[case_name] = iterations
    complete = len(iterations) >= EXPECTED_ITERS

    if not iterations:
        print(f"  ** JOB STILL RUNNING / NO COMPLETE ITERATIONS YET **")
        summary[case_name] = None
        continue

    status = "COMPLETE" if complete else f"INCOMPLETE ({len(iterations)}/{EXPECTED_ITERS})"
    print(f"Status: {status}")

    # Show all iterations
    for i, it in enumerate(iterations):
        print(f"  Iter {i+1}: total={it['total_time']:.2f}s, "
              f"exchange={it['avg_exchange_time']:.2f}s, "
              f"proportion={it['proportion']:.2f}%")

    # Trimmed mean
    trimmed = trimmed_mean(iterations)
    dropped_indices = [i+1 for i, it in enumerate(iterations) if it not in trimmed]
    print(f"  Dropped iterations (min/max): {dropped_indices}")

    avg_total_time = sum(it['total_time'] for it in trimmed) / len(trimmed)
    avg_exchange_time = sum(it['avg_exchange_time'] for it in trimmed) / len(trimmed)
    avg_proportion = avg_exchange_time / avg_total_time * 100

    print(f"\n  TRIMMED AVERAGE over {len(trimmed)} iterations:")
    print(f"    Total time (Rank 0):       {avg_total_time:.4f} s")
    print(f"    Avg exchange time (ranks):  {avg_exchange_time:.4f} s")
    print(f"    Exchange time proportion:   {avg_proportion:.2f}%")

    summary[case_name] = {
        'avg_total_time': avg_total_time,
        'avg_exchange_time': avg_exchange_time,
        'proportion': avg_proportion,
        'n_iters': len(trimmed),
        'n_total_iters': len(iterations),
        'complete': complete,
    }

# ==========================================================
# Summary Table
# ==========================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE (Trimmed Mean)")
print("=" * 70)
print(f"{'Case':<22} {'Ranks':>6} {'Total(s)':>10} {'Exchange(s)':>12} {'Proportion':>12} {'Status':>12}")
print("-" * 76)
for case_name, (_, expected_ranks) in cases.items():
    s = summary.get(case_name)
    if s is None:
        print(f"{case_name:<22} {expected_ranks:>6} {'--':>10} {'--':>12} {'--':>12} {'PENDING':>12}")
    else:
        status = "OK" if s['complete'] else f"{s['n_total_iters']}/{EXPECTED_ITERS}"
        print(f"{case_name:<22} {expected_ranks:>6} {s['avg_total_time']:>10.4f} "
              f"{s['avg_exchange_time']:>12.4f} {s['proportion']:>11.2f}% {status:>12}")
print("=" * 76)

# ==========================================================
# Filter cases with data
# ==========================================================
plot_cases = [(name, summary[name]) for name in cases if summary.get(name) is not None]

if not plot_cases:
    print("\nNo completed cases to plot.")
else:
    labels = []
    total_times = []
    exchange_times = []
    compute_times = []
    proportions = []
    ranks_list = []

    for name, s in plot_cases:
        labels.append(name.split(" ")[0])
        total_times.append(s['avg_total_time'])
        exchange_times.append(s['avg_exchange_time'])
        compute_times.append(s['avg_total_time'] - s['avg_exchange_time'])
        proportions.append(s['proportion'])
        _, (_, nr) = name, cases[name]
        ranks_list.append(nr)

    # ==========================================================
    # Bar Chart (trimmed mean)
    # ==========================================================
    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(len(labels))
    width = 0.2

    bars_compute = ax.bar(x, compute_times, width, label='Computation', color='#2b6cb0')
    bars_exchange = ax.bar(x, exchange_times, width, bottom=compute_times,
                           label='Communication', color='#e53e3e')

    for i, (total, prop) in enumerate(zip(total_times, proportions)):
        ax.text(x[i], total + 0.5, f'{prop:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('MPI Grid Configuration', fontsize=13)
    ax.set_ylabel('Time (s)', fontsize=13)
    # ax.set_title(f'{LABEL}: Communication Proportion (Trimmed Mean)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n({r} ranks)' for l, r in zip(labels, ranks_list)], fontsize=11)
    ax.legend(fontsize=11, loc='upper left', ncol=2)
    ax.set_ylim(0, max(total_times) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    chart_png = os.path.join(BASE_DIR, 'exchange_time_proportion_trimmed.png')
    chart_pdf = os.path.join(BASE_DIR, 'exchange_time_proportion_trimmed.pdf')
    fig.savefig(chart_png, dpi=200, bbox_inches='tight')
    fig.savefig(chart_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\nTrimmed bar chart saved to:\n  {chart_png}\n  {chart_pdf}")

    # ==========================================================
    # Communication Variation Plot — grouped bars, all 5 runs
    # ==========================================================
    fig2, ax2 = plt.subplots(figsize=(9, 5))

    n_cases = len(plot_cases)
    n_iters = max(len(all_iterations[name]) for name, _ in plot_cases)
    iter_indices = np.arange(1, n_iters + 1)
    bar_width = 0.8 / n_cases  # width per case within a group
    colors = ['#2b6cb0', '#e53e3e', '#38a169']

    for ci, (name, _) in enumerate(plot_cases):
        iters = all_iterations[name]
        vals = [it['avg_exchange_time'] for it in iters]
        label = name.split(" ")[0]
        offset = (ci - (n_cases - 1) / 2) * bar_width
        positions = iter_indices[:len(vals)] + offset
        ax2.bar(positions, vals, bar_width * 0.9, label=f'{label}',
                color=colors[ci % len(colors)], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax2.set_xlabel('Run', fontsize=13)
    ax2.set_ylabel('Avg Exchange Time (s)', fontsize=13)
    ax2.set_title(f'{LABEL}: Communication Variation Across 5 Runs', fontsize=14, fontweight='bold')
    ax2.set_xticks(iter_indices)
    ax2.set_xticklabels([f'Run {i}' for i in iter_indices], fontsize=11)
    ax2.legend(fontsize=11, title='Configuration')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    var_png = os.path.join(BASE_DIR, 'exchange_time_variation.png')
    var_pdf = os.path.join(BASE_DIR, 'exchange_time_variation.pdf')
    fig2.savefig(var_png, dpi=200)
    fig2.savefig(var_pdf)
    plt.close(fig2)
    print(f"\nVariation plot saved to:\n  {var_png}\n  {var_pdf}")

    # ==========================================================
    # LaTeX Table (Trimmed Mean)
    # ==========================================================
    latex = r"""\begin{table}[htbp]
\centering
\caption{""" + LABEL + r""" communication: weak scaling overhead (trimmed mean).}
\label{tab:exchange_time_trimmed_""" + LABEL + r"""}
\begin{tabular}{lrrrr}
\toprule
Configuration & Ranks & Total Time (s) & Exchange Time (s) & Proportion (\%) \\
\midrule
"""
    for name, s in plot_cases:
        config = name.split(" ")[0]
        _, (_, nr) = name, cases[name]
        latex += f"{config} & {nr} & {s['avg_total_time']:.2f} & {s['avg_exchange_time']:.2f} & {s['proportion']:.2f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    tex_file = os.path.join(BASE_DIR, 'exchange_time_table_trimmed.tex')
    with open(tex_file, 'w') as f:
        f.write(latex)

    print(f"\nLaTeX table saved to:\n  {tex_file}")
    print("\n--- LaTeX Table (Trimmed Mean) ---")
    print(latex)
