#!/usr/bin/env bash
# Mirror of pareto_sweep.sh, but every run has c_geom on @ --geo_auto --geo_percentile 80 --geo_scale_min 1.0.
# Compare result against pareto_results.csv (baseline, c_geom off).
# Output: script/pareto_results_cgeom_p80f1.csv

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/script"
LOG_DIR="$SCRIPT_DIR/pareto_logs_cgeom_p80f1"
CSV_PATH="$SCRIPT_DIR/pareto_results_cgeom_p80f1.csv"
CPU_BIN="$ROOT_DIR/build/test/test_quantize_and_edt"

THREADS="${THREADS:-16}"
EBS=(1e-3 5e-3 1e-2)
FACTORS=(0 2 4 8)

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

# Match the original CSV header for easy join
cat > "$CSV_PATH" <<'EOF'
dataset,field,dims,rel_eb,downsample_factor,geo_scale,edt_total_s,edt_round1_s,edt_round2_s,fill_sign_s,neutral_boundary_s,downsample_boundary_s,compensation_stage_s,initial_psnr,initial_ssim,final_psnr,final_ssim,log_file,sparsity,edge_density,skipped,skip_reason,harm_rate,benefit_rate
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
  local dataset="$1" field="$2" dims="$3" eb="$4" factor="$5" input_file="$6"
  local safe_field="${field//./_}"
  local log="$LOG_DIR/${dataset}_${safe_field}_eb${eb}_f${factor}.log"
  local qfile="$LOG_DIR/q_${dataset}_${safe_field}_eb${eb}_f${factor}.bin"
  local cfile="$LOG_DIR/c_${dataset}_${safe_field}_eb${eb}_f${factor}.bin"
  local d0 d1 d2; read -r d0 d1 d2 <<< "$dims"

  printf "  %-30s eb=%-6s f=%s ... " "${dataset}/${field}" "$eb" "$factor"

  "$CPU_BIN" \
    -N 3 -d "$d0" "$d1" "$d2" \
    -i "$input_file" \
    -m rel -e "$eb" \
    -q "$qfile" -c "$cfile" \
    -t "$THREADS" \
    --cpu_index_mode flat32 \
    --downsample_r2 "$factor" \
    --eta 0.9 \
    --geo_attenuation 1 --geo_auto 1 --geo_percentile 80 --geo_scale_min 1.0 \
    > "$log" 2>&1

  local edt_total edt_round1 edt_round2 fill_sign neutral_bnd ds_bnd comp_stage
  edt_total="$(extract   'StageTime edt_total:'           "$log")"
  edt_round1="$(extract  'StageTime edt_round1:'          "$log")"
  edt_round2="$(extract  'StageTime edt_round2:'          "$log")"
  fill_sign="$(extract   'StageTime fill_sign:'           "$log")"
  neutral_bnd="$(extract 'StageTime neutral_boundary:'    "$log")"
  ds_bnd="$(extract      'StageTime downsample_boundary:' "$log")"
  comp_stage="$(extract  'StageTime compensation:'        "$log")"

  local geo_scale sparsity edge_density skipped skip_reason harm_rate benefit_rate
  geo_scale="$(awk '/^AdaptiveGeoScale/{
    for(i=1;i<=NF;i++) if($i=="floored"){v=$(i+2)}
    if(v=="") v=$NF
  } END{print v}' "$log")"
  sparsity="$(extract_kv 'Sparsity' "$log" 2>/dev/null || true)"
  if [[ -z "$sparsity" ]]; then sparsity="$(awk '/^Sparsity /{print $2}' "$log" | tail -1)"; fi
  edge_density="$(awk '/^EdgeDensity /{print $2}' "$log" | tail -1)"
  if grep -q "skipping compensation" "$log"; then skipped=1; skip_reason="$(grep -m1 "skipping" "$log" | tr ',' ';')"; else skipped=0; skip_reason=""; fi
  harm_rate="$(extract_kv 'harm_rate' "$log")"
  benefit_rate="$(extract_kv 'benefit_rate' "$log")"

  local initial_pair final_pair
  initial_pair="$(extract_psnr_ssim first "$log")"
  final_pair="$(extract_psnr_ssim   last  "$log")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$dataset" "$field" "${d0}x${d1}x${d2}" "$eb" "$factor" \
    "${geo_scale:-}" \
    "$edt_total" "$edt_round1" "$edt_round2" \
    "$fill_sign" "$neutral_bnd" "$ds_bnd" "$comp_stage" \
    "$initial_pair" "$final_pair" \
    "$log" \
    "${sparsity:-}" "${edge_density:-}" "${skipped:-0}" "${skip_reason:-}" "${harm_rate:-}" "${benefit_rate:-}" \
    >> "$CSV_PATH"

  local final_psnr; final_psnr="${final_pair%%,*}"
  printf "done (gs=%-8s final_psnr=%s)\n" "${geo_scale:-na}" "$final_psnr"
  rm -f "$qfile" "$cfile"
}

total=$(( ${#NYX_FIELDS[@]} * ${#EBS[@]} * ${#FACTORS[@]} + ${#HUR_FIELDS[@]} * ${#EBS[@]} * ${#FACTORS[@]} ))
echo "=== Pareto sweep (c_geom on @ p80+floor=1.0): $total runs ==="

echo "--- NYX 512^3 ---"
for field in "${NYX_FIELDS[@]}"; do
  input="$NYX_DIR/${field}.f32"
  [[ ! -f "$input" ]] && { echo "SKIP missing: $input"; continue; }
  for eb in "${EBS[@]}"; do
    for f in "${FACTORS[@]}"; do
      run_one "nyx" "$field" "$NYX_DIMS" "$eb" "$f" "$input"
    done
  done
done

echo ""
echo "--- Hurricane 100x500x500 ---"
for field in "${HUR_FIELDS[@]}"; do
  input="$HUR_DIR/${field}.f32"
  [[ ! -f "$input" ]] && { echo "SKIP missing: $input"; continue; }
  for eb in "${EBS[@]}"; do
    for f in "${FACTORS[@]}"; do
      run_one "hurricane" "$field" "$HUR_DIMS" "$eb" "$f" "$input"
    done
  done
done

echo "=== Done. Results: $CSV_PATH ==="
