#!/usr/bin/env bash
# c_geom v2 sweep: two candidate fixes for the auto-percentile failure:
#   p80f1 : --geo_auto 1 --geo_percentile 80 --geo_scale_min 1.0
#   gs5   : --geo_auto 0 --geo_scale 5
# Output: script/pareto_cgeom_v2_results.csv (appends to existing if present)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/script"
LOG_DIR="$SCRIPT_DIR/pareto_cgeom_v2_logs"
CSV_PATH="$SCRIPT_DIR/pareto_cgeom_v2_results.csv"
CPU_BIN="$ROOT_DIR/build/test/test_quantize_and_edt"

THREADS="${THREADS:-16}"
EBS=(1e-3 5e-3 1e-2)
FACTOR=2

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
dataset,field,dims,rel_eb,cgeom,geo_scale,harm_rate,benefit_rate,initial_psnr,initial_ssim,final_psnr,final_ssim,log_file
EOF

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
  local dataset="$1" field="$2" dims="$3" eb="$4" cfg="$5" input_file="$6"
  local safe_field="${field//./_}"
  local log="$LOG_DIR/${dataset}_${safe_field}_eb${eb}_${cfg}.log"
  local qfile="$LOG_DIR/q_${dataset}_${safe_field}_eb${eb}_${cfg}.bin"
  local cfile="$LOG_DIR/c_${dataset}_${safe_field}_eb${eb}_${cfg}.bin"
  local d0 d1 d2; read -r d0 d1 d2 <<< "$dims"

  printf "  %-30s eb=%-6s %-8s ... " "${dataset}/${field}" "$eb" "$cfg"

  local extra=()
  case "$cfg" in
    p80f1)
      extra=(--geo_attenuation 1 --geo_auto 1 --geo_percentile 80 --geo_scale_min 1.0)
      ;;
    gs5)
      extra=(--geo_attenuation 1 --geo_auto 0 --geo_scale 5)
      ;;
    *)
      echo "unknown cfg: $cfg" >&2; exit 1
      ;;
  esac

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

  local geo_scale harm_rate benefit_rate
  # Floored line uses different format; grab effective gs from either line
  geo_scale="$(awk '/^AdaptiveGeoScale/{for(i=1;i<=NF;i++) if($i=="floored"){v=$(i+2)}else if($i~/^[0-9.+-]+e?[0-9+-]*$/) v=$i} END{print v}' "$log")"
  if [[ -z "$geo_scale" && "$cfg" == "gs5" ]]; then geo_scale="5.0"; fi
  harm_rate="$(extract_kv 'harm_rate' "$log")"
  benefit_rate="$(extract_kv 'benefit_rate' "$log")"

  local initial_pair final_pair
  initial_pair="$(extract_psnr_ssim first "$log")"
  final_pair="$(extract_psnr_ssim   last  "$log")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$dataset" "$field" "${d0}x${d1}x${d2}" "$eb" "$cfg" \
    "${geo_scale:-}" "${harm_rate:-}" "${benefit_rate:-}" \
    "$initial_pair" "$final_pair" "$log" \
    >> "$CSV_PATH"

  local final_psnr; final_psnr="${final_pair%%,*}"
  printf "done (gs=%-8s harm=%-7s final_psnr=%s)\n" "${geo_scale:-na}" "${harm_rate:-na}" "$final_psnr"
  rm -f "$qfile" "$cfile"
}

total=$(( (${#NYX_FIELDS[@]} + ${#HUR_FIELDS[@]}) * ${#EBS[@]} * 2 ))
echo "=== c_geom v2 sweep: $total runs ==="

for cfg in p80f1 gs5; do
  echo "--- config: $cfg ---"
  for field in "${NYX_FIELDS[@]}"; do
    input="$NYX_DIR/${field}.f32"
    [[ ! -f "$input" ]] && { echo "SKIP missing: $input"; continue; }
    for eb in "${EBS[@]}"; do
      run_one "nyx" "$field" "$NYX_DIMS" "$eb" "$cfg" "$input"
    done
  done
  for field in "${HUR_FIELDS[@]}"; do
    input="$HUR_DIR/${field}.f32"
    [[ ! -f "$input" ]] && { echo "SKIP missing: $input"; continue; }
    for eb in "${EBS[@]}"; do
      run_one "hurricane" "$field" "$HUR_DIMS" "$eb" "$cfg" "$input"
    done
  done
done

echo "=== Done. Results: $CSV_PATH ==="
