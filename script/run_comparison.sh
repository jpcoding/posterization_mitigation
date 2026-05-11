#!/usr/bin/env bash
# Compare IPDPS baseline vs current config on all NYX + Hurricane fields.
# IPDPS baseline: ipdps26 branch binary, eta=0.9, no guards, full-res EDT.
# Current config: main branch, ds=2, geo_auto p80+floor=1, sparsity/edge guards.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_CURR="$ROOT_DIR/build/test/test_quantize_and_edt"
BIN_IPDPS="/tmp/ipdps26_wt/build/test/test_quantize_and_edt"
THREADS="${THREADS:-8}"
EBS=(1e-3 5e-3 1e-2)
NYX_DIR="/home/jp/data/nyx_512x512x512"
HUR_DIR="/home/jp/data/hurricane_100x500x500"
OUT_CSV="$ROOT_DIR/script/comparison_results.csv"

NYX_FIELDS=(velocity_x velocity_y velocity_z baryon_density dark_matter_density temperature)
HUR_FIELDS=(
  CLOUDf48.bin CLOUDf48.log10.bin
  Pf48.bin PRECIPf48.bin PRECIPf48.log10.bin
  QCLOUDf48.bin QCLOUDf48.log10.bin
  QGRAUPf48.bin QGRAUPf48.log10.bin
  QICEf48.bin QICEf48.log10.bin
  QRAINf48.bin QRAINf48.log10.bin
  QSNOWf48.bin QSNOWf48.log10.bin
  QVAPORf48.bin TCf48.bin Uf48.bin Vf48.bin Wf48.bin
)

for bin in "$BIN_CURR" "$BIN_IPDPS"; do
  if [[ ! -x "$bin" ]]; then
    echo "ERROR: binary not found: $bin" >&2; exit 1
  fi
done

echo "dataset,field,rel_eb,initial_psnr,initial_ssim,ipdps_psnr,ipdps_ssim,ipdps_delta,curr_psnr,curr_ssim,curr_delta,curr_skipped" \
  > "$OUT_CSV"

extract_psnr() { grep '^PSNR' "$1" 2>/dev/null | tail -1 | awk '{print $3}' | tr -d ',' || echo ""; }
extract_ssim() { grep '^SSIM' "$1" 2>/dev/null | tail -1 | awk '{print $3}' || echo ""; }
extract_init_psnr() { grep '^PSNR' "$1" 2>/dev/null | head -1 | awk '{print $3}' | tr -d ',' || echo ""; }
extract_init_ssim() { grep '^SSIM' "$1" 2>/dev/null | head -1 | awk '{print $3}' || echo ""; }
delta() {
  local a="$1" b="$2"
  [[ -n "$a" && -n "$b" ]] && python3 -c "print(f'{float(\"$b\")-float(\"$a\"):.4f}')" 2>/dev/null || echo ""
}

run_one() {
  local dataset="$1" field="$2" dims="$3" input_file="$4" eb="$5"
  local log_i="/tmp/log_ipdps_${dataset}_${field}_${eb}.txt"
  local log_c="/tmp/log_curr_${dataset}_${field}_${eb}.txt"

  # --- IPDPS baseline ---
  "$BIN_IPDPS" \
    -N 3 -d $dims \
    -i "$input_file" \
    -m rel -e "$eb" \
    -q /tmp/q_sweep.bin -c /tmp/c_sweep.bin \
    -t "$THREADS" \
    --eta 0.9 \
    > "$log_i" 2>&1 || true

  # --- Current config ---
  "$BIN_CURR" \
    -N 3 -d $dims \
    -i "$input_file" \
    -m rel -e "$eb" \
    -q /tmp/q_sweep.bin -c /tmp/c_sweep.bin \
    -t "$THREADS" \
    --cpu_index_mode flat32 \
    --downsample_r2 2 \
    --geo_auto 1 --geo_percentile 80 --geo_scale_min 1.0 \
    > "$log_c" 2>&1 || true

  local init_psnr init_ssim
  init_psnr=$(extract_init_psnr "$log_c")
  init_ssim=$(extract_init_ssim "$log_c")

  local ipdps_psnr ipdps_ssim ipdps_delta
  ipdps_psnr=$(extract_psnr "$log_i")
  ipdps_ssim=$(extract_ssim "$log_i")
  ipdps_delta=$(delta "$init_psnr" "$ipdps_psnr")

  local curr_psnr curr_ssim curr_delta curr_skip
  curr_psnr=$(extract_psnr "$log_c")
  curr_ssim=$(extract_ssim "$log_c")
  curr_delta=$(delta "$init_psnr" "$curr_psnr")
  curr_skip=0
  grep -q 'skipping compensation' "$log_c" 2>/dev/null && curr_skip=1 || true

  echo "$dataset,$field,$eb,${init_psnr:-NA},${init_ssim:-NA},${ipdps_psnr:-NA},${ipdps_ssim:-NA},${ipdps_delta:-NA},${curr_psnr:-NA},${curr_ssim:-NA},${curr_delta:-NA},$curr_skip" \
    >> "$OUT_CSV"

  local skip_tag=""
  [[ $curr_skip -eq 1 ]] && skip_tag=" [skip]"
  printf "  %-12s %-26s eb=%-5s  init=%6s  ipdps=%6s (%+6s)  curr=%6s (%+6s)%s\n" \
    "$dataset" "$field" "$eb" \
    "${init_psnr:-?}" "${ipdps_psnr:-?}" "${ipdps_delta:-?}" \
    "${curr_psnr:-?}" "${curr_delta:-?}" "$skip_tag"
}

printf "  %-12s %-26s %-9s  %-6s  %-20s  %-20s\n" \
  "Dataset" "Field" "eb" "Init" "IPDPS (Δ)" "Current (Δ)"
echo "  $(printf '%0.s-' {1..100})"

echo "=== NYX 512x512x512 ==="
for field in "${NYX_FIELDS[@]}"; do
  for eb in "${EBS[@]}"; do
    run_one nyx "$field" "512 512 512" "$NYX_DIR/${field}.f32" "$eb"
  done
done

echo ""
echo "=== Hurricane 100x500x500 ==="
for field in "${HUR_FIELDS[@]}"; do
  for eb in "${EBS[@]}"; do
    run_one hurricane "$field" "100 500 500" "$HUR_DIR/${field}.f32" "$eb"
  done
done

echo ""
echo "Results saved to: $OUT_CSV"
