#!/usr/bin/env bash
# c_geom Pareto sweep: NYX + Hurricane, all 3 rel ebs, factor=2,
# comparing cgeom_off (baseline) vs cgeom_on @ --geo_auto --geo_percentile 80.
# Output: script/pareto_cgeom_results.csv  +  script/pareto_cgeom_logs/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/script"
LOG_DIR="$SCRIPT_DIR/pareto_cgeom_logs"
CSV_PATH="$SCRIPT_DIR/pareto_cgeom_results.csv"
CPU_BIN="$ROOT_DIR/build/test/test_quantize_and_edt"

THREADS="${THREADS:-16}"
EBS=(1e-3 5e-3 1e-2)
FACTOR=2
GEO_PERCENTILE="${GEO_PERCENTILE:-80}"

NYX_DIR="/home/jp/data/nyx_512x512x512"
NYX_FIELDS=(velocity_x velocity_y velocity_z baryon_density dark_matter_density temperature)
NYX_DIMS="512 512 512"

HUR_DIR="/home/jp/data/hurricane_100x500x500"
HUR_FIELDS=(
  CLOUDf48.bin Pf48.bin PRECIPf48.bin QCLOUDf48.bin QGRAUPf48.bin QICEf48.bin
  QRAINf48.bin QSNOWf48.bin QVAPORf48.bin TCf48.bin Uf48.bin Vf48.bin Wf48.bin
  CLOUDf48.log10.bin PRECIPf48.log10.bin QCLOUDf48.log10.bin QGRAUPf48.log10.bin
  QICEf48.log10.bin QRAINf48.log10.bin QSNOWf48.log10.bin
)
HUR_DIMS="100 500 500"

[[ ! -x "$CPU_BIN" ]] && { echo "ERROR: binary not found: $CPU_BIN" >&2; exit 1; }
mkdir -p "$LOG_DIR"

cat > "$CSV_PATH" <<'EOF'
dataset,field,dims,rel_eb,cgeom,geo_scale,harm_rate,benefit_rate,harm_rms,benefit_rms,initial_psnr,initial_ssim,final_psnr,final_ssim,log_file
EOF

extract() { awk -F': ' -v p="$1" '$0~p{print $2}' "$2" | tail -1; }
extract_kv() { awk -v k="$1" '$1==k && $2=="="{print $3}' "$2" | tail -1; }

extract_psnr_ssim() {
  awk -v which="$1" '
    /^PSNR = / { psnr=$3; gsub(",","",psnr); got_psnr=1 }
    /^SSIM = / && got_psnr {
      ssim=$3; got_psnr=0
      if (which=="first" && seen==0) { result=psnr","ssim; seen=1 }
      if (which=="last")             { result=psnr","ssim }
    }
    END { print result }
  ' "$2"
}

run_one() {
  local dataset="$1" field="$2" dims="$3" eb="$4" cgeom="$5" input_file="$6"
  local safe_field="${field//./_}"
  local log="$LOG_DIR/${dataset}_${safe_field}_eb${eb}_cgeom${cgeom}.log"
  local qfile="$LOG_DIR/q_${dataset}_${safe_field}_eb${eb}_cgeom${cgeom}.bin"
  local cfile="$LOG_DIR/c_${dataset}_${safe_field}_eb${eb}_cgeom${cgeom}.bin"
  local d0 d1 d2; read -r d0 d1 d2 <<< "$dims"

  printf "  %-30s eb=%-6s cgeom=%s ... " "${dataset}/${field}" "$eb" "$cgeom"

  local extra=()
  if [[ "$cgeom" == "on" ]]; then
    extra=(--geo_attenuation 1 --geo_auto 1 --geo_percentile "$GEO_PERCENTILE")
  fi

  "$CPU_BIN" \
    -N 3 -d "$d0" "$d1" "$d2" \
    -i "$input_file" \
    -m rel -e "$eb" \
    -q "$qfile" -c "$cfile" \
    -t "$THREADS" \
    --cpu_index_mode flat32 \
    --downsample_r2 "$FACTOR" \
    --eta 0.9 \
    "${extra[@]}" \
    > "$log" 2>&1

  local geo_scale harm_rate benefit_rate harm_rms benefit_rms
  geo_scale="$(awk '/^AdaptiveGeoScale/{v=$NF} END{print v}' "$log")"
  harm_rate="$(extract_kv 'harm_rate' "$log")"
  benefit_rate="$(extract_kv 'benefit_rate' "$log")"
  harm_rms="$(awk -F'= *' '/^harm_rms_err_increase/{print $2}' "$log" | tail -1)"
  benefit_rms="$(awk -F'= *' '/^benefit_rms_err_decrease/{print $2}' "$log" | tail -1)"

  local initial_pair final_pair
  initial_pair="$(extract_psnr_ssim first "$log")"
  final_pair="$(extract_psnr_ssim   last  "$log")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$dataset" "$field" "${d0}x${d1}x${d2}" "$eb" "$cgeom" \
    "${geo_scale:-}" "${harm_rate:-}" "${benefit_rate:-}" "${harm_rms:-}" "${benefit_rms:-}" \
    "$initial_pair" "$final_pair" "$log" \
    >> "$CSV_PATH"

  local final_psnr; final_psnr="${final_pair%%,*}"
  printf "done (gs=%-6s harm=%-7s final_psnr=%s)\n" "${geo_scale:-na}" "${harm_rate:-na}" "$final_psnr"
  rm -f "$qfile" "$cfile"
}

total=$(( (${#NYX_FIELDS[@]} + ${#HUR_FIELDS[@]}) * ${#EBS[@]} * 2 ))
echo "=== c_geom sweep: $total runs (geo_percentile=$GEO_PERCENTILE) ==="

echo "--- NYX 512^3 ---"
for field in "${NYX_FIELDS[@]}"; do
  input="$NYX_DIR/${field}.f32"
  [[ ! -f "$input" ]] && { echo "SKIP missing: $input"; continue; }
  for eb in "${EBS[@]}"; do
    for cgeom in off on; do
      run_one "nyx" "$field" "$NYX_DIMS" "$eb" "$cgeom" "$input"
    done
  done
done

echo ""
echo "--- Hurricane 100x500x500 ---"
for field in "${HUR_FIELDS[@]}"; do
  input="$HUR_DIR/${field}.f32"
  [[ ! -f "$input" ]] && { echo "SKIP missing: $input"; continue; }
  for eb in "${EBS[@]}"; do
    for cgeom in off on; do
      run_one "hurricane" "$field" "$HUR_DIMS" "$eb" "$cgeom" "$input"
    done
  done
done

echo "=== Done. Results: $CSV_PATH ==="
