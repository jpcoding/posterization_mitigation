#!/usr/bin/env python3
"""Re-parse existing pareto log files to regenerate CSV with harm_rate and guard columns."""
import re
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "pareto_logs"
CSV_IN  = SCRIPT_DIR / "pareto_results.csv"
CSV_OUT = SCRIPT_DIR / "pareto_results.csv"  # overwrite in place

def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    def first_match(pattern):
        for l in lines:
            m = re.search(pattern, l)
            if m:
                return m.group(1)
        return ""

    def last_match(pattern):
        result = ""
        for l in lines:
            m = re.search(pattern, l)
            if m:
                result = m.group(1)
        return result

    # Stage times
    def stage(name):
        return first_match(rf"StageTime {re.escape(name)}:\s+([\d.]+)")

    # PSNR/SSIM pairs: first pair = initial, last pair = final
    psnr_vals, ssim_vals = [], []
    got_psnr = None
    for l in lines:
        mp = re.match(r"PSNR\s*=\s*([\d.]+)", l)
        ms = re.match(r"SSIM\s*=\s*([\d.]+)", l)
        if mp:
            got_psnr = mp.group(1)
        if ms and got_psnr is not None:
            psnr_vals.append(got_psnr)
            ssim_vals.append(ms.group(1))
            got_psnr = None

    initial_psnr = psnr_vals[0]  if psnr_vals else ""
    initial_ssim = ssim_vals[0]  if ssim_vals else ""
    final_psnr   = psnr_vals[-1] if psnr_vals else ""
    final_ssim   = ssim_vals[-1] if ssim_vals else ""

    # Guards
    sparsity     = first_match(r"^Sparsity\s+([\d.eE+\-]+)$")
    edge_density = first_match(r"^EdgeDensity\s+([\d.eE+\-]+)$")
    skipped = "1" if "skipping compensation" in text else "0"
    skip_reason = ""
    if "Sparsity" in text and "skipping compensation" in text:
        skip_reason = "sparsity"
    elif "EdgeDensity" in text and "skipping compensation" in text:
        skip_reason = "edge_density"
    elif "too many zeros" in text or "sparsity" in text and skipped == "1":
        skip_reason = "pre_check"

    # Harm / benefit
    harm_rate    = first_match(r"harm_rate\s*=\s*([\d.eE+\-]+)")
    benefit_rate = first_match(r"benefit_rate\s*=\s*([\d.eE+\-]+)")

    return dict(
        edt_total_s          = stage("edt_total"),
        edt_round1_s         = stage("edt_round1"),
        edt_round2_s         = stage("edt_round2"),
        fill_sign_s          = stage("fill_sign"),
        neutral_boundary_s   = stage("neutral_boundary"),
        downsample_boundary_s= stage("downsample_boundary"),
        compensation_stage_s = stage("compensation"),
        initial_psnr=initial_psnr, initial_ssim=initial_ssim,
        final_psnr=final_psnr,     final_ssim=final_ssim,
        sparsity=sparsity,   edge_density=edge_density,
        skipped=skipped,     skip_reason=skip_reason,
        harm_rate=harm_rate, benefit_rate=benefit_rate,
    )


def main():
    rows_in = list(csv.DictReader(CSV_IN.open()))
    print(f"Re-parsing {len(rows_in)} rows ...")

    new_cols = ["sparsity", "edge_density", "skipped", "skip_reason",
                "harm_rate", "benefit_rate"]
    fieldnames = [f for f in rows_in[0].keys() if f not in new_cols] + new_cols

    out_rows = []
    for row in rows_in:
        log_path = Path(row["log_file"])
        if not log_path.exists():
            print(f"  MISSING log: {log_path}", file=sys.stderr)
            for c in new_cols:
                row[c] = ""
            out_rows.append(row)
            continue

        parsed = parse_log(log_path)
        # overwrite timing/quality cols with freshly parsed values (more reliable)
        for k, v in parsed.items():
            row[k] = v
        out_rows.append(row)

    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Written {len(out_rows)} rows → {CSV_OUT}")

    # Quick summary
    skipped = sum(1 for r in out_rows if r["skipped"] == "1")
    print(f"\nSkipped (guards): {skipped} / {len(out_rows)} configs")
    for reason in ["sparsity", "edge_density", "pre_check"]:
        n = sum(1 for r in out_rows if r["skip_reason"] == reason)
        if n:
            print(f"  {reason}: {n}")

    # Harm rate summary for compensated configs
    compensated = [r for r in out_rows if r["skipped"] == "0" and r["harm_rate"]]
    if compensated:
        harm_vals = [float(r["harm_rate"]) for r in compensated]
        print(f"\nHarm rate across {len(compensated)} compensated configs:")
        print(f"  mean={sum(harm_vals)/len(harm_vals):.3f}  "
              f"min={min(harm_vals):.3f}  max={max(harm_vals):.3f}")
        degraded = [r for r in compensated
                    if float(r["final_psnr"] or 0) < float(r["initial_psnr"] or 0)]
        print(f"  Net PSNR degraded: {len(degraded)} configs")
        for r in degraded:
            delta = float(r["final_psnr"]) - float(r["initial_psnr"])
            print(f"    {r['dataset']}/{r['field']} eb={r['rel_eb']} "
                  f"f={r['downsample_factor']}  Δ={delta:+.2f} dB")


if __name__ == "__main__":
    main()
