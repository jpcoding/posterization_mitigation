#!/usr/bin/env bash
# Run recommended config on all NYX + Hurricane fields, ds=2, geo_auto p80+floor=1.
# Output: CSV with initial_psnr, final_psnr, psnr_delta, harm_rate, benefit_rate, skipped.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/build/test/test_quantize_and_edt"
THREADS="${THREADS:-8}"
EBS=(1e-3 5e-3 1e-2)
NYX_DIR="/home/jp/data/nyx_512x512x512"
HUR_DIR="/home/jp/data/hurricane_100x500x500"
OUT_CSV="$ROOT_DIR/script/current_config_results.csv"

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

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN" >&2; exit 1
fi

echo "dataset,field,rel_eb,initial_psnr,initial_ssim,final_psnr,final_ssim,psnr_delta,harm_rate,benefit_rate,skipped,skip_reason" \
  > "$OUT_CSV"

run_one() {
  local dataset="$1" field="$2" dims="$3" input_file="$4" eb="$5"
  local qfile="/tmp/q_sweep.bin"
  local cfile="/tmp/c_sweep.bin"
  local log="/tmp/log_${dataset}_${field}_${eb}.txt"

  "$BIN" \
    -N 3 -d $dims \
    -i "$input_file" \
    -m rel -e "$eb" \
    -q "$qfile" -c "$cfile" \
    -t "$THREADS" \
    --cpu_index_mode flat32 \
    --downsample_r2 2 \
    --geo_auto 1 --geo_percentile 80 --geo_scale_min 1.0 \
    > "$log" 2>&1 || true

  local init_psnr init_ssim final_psnr final_ssim harm benefit
  init_psnr=$(  grep -m1 '^PSNR'    "$log" 2>/dev/null | awk '{print $3}' | tr -d ',' || echo "")
  init_ssim=$(  grep -m1 '^SSIM'    "$log" 2>/dev/null | awk '{print $3}' || echo "")
  final_psnr=$( grep    '^PSNR'    "$log" 2>/dev/null | tail -1 | awk '{print $3}' | tr -d ',' || echo "")
  final_ssim=$( grep    '^SSIM'    "$log" 2>/dev/null | tail -1 | awk '{print $3}' || echo "")
  harm=$(       grep 'harm_rate'   "$log" 2>/dev/null | awk '{print $3}' | head -1 || echo "0")
  benefit=$(    grep 'benefit_rate' "$log" 2>/dev/null | awk '{print $3}' | head -1 || echo "0")
  local skip_count=0
  if grep -q 'skipping compensation' "$log" 2>/dev/null; then skip_count=1; fi
  skip_reason=$(grep 'skipping compensation' "$log" 2>/dev/null | head -1 | sed 's/.*skipping compensation/skipping compensation/' || echo "")

  local delta=""
  if [[ -n "${final_psnr:-}" && -n "${init_psnr:-}" ]]; then
    delta=$(python3 -c "print(f'{float(\"${final_psnr}\")-float(\"${init_psnr}\"):.4f}')" 2>/dev/null || echo "")
  fi

  echo "$dataset,$field,$eb,${init_psnr:-NA},${init_ssim:-NA},${final_psnr:-NA},${final_ssim:-NA},${delta:-NA},${harm:-NA},${benefit:-NA},${skip_count},\"${skip_reason}\"" \
    >> "$OUT_CSV"

  local skip_tag=""
  if [[ $skip_count -gt 0 ]]; then skip_tag="  [SKIPPED]"; fi
  printf "  %-12s %-28s eb=%-5s  init=%s  final=%s  Δ=%s%s\n" \
    "$dataset" "$field" "$eb" "${init_psnr:-?}" "${final_psnr:-?}" "${delta:-?}" "$skip_tag"
}

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
