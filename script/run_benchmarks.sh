#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/script"
LOG_DIR="$SCRIPT_DIR/logs"
CSV_PATH="$SCRIPT_DIR/benchmark_results.csv"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/benchmark_config.sh}"

CPU_BIN="$ROOT_DIR/build/test/test_quantize_and_edt"
GPU_BIN="$ROOT_DIR/build/test_cuda/test_compensation_cuda"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing benchmark config: $CONFIG_PATH" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_PATH"

CPU_THREADS="${CPU_THREADS:-8}"
CPU_INDEX_MODE="${CPU_INDEX_MODE:-packed}"
BENCHMARK_ENABLE_GPU="${BENCHMARK_ENABLE_GPU:-auto}"
GPU_JFA_LEVEL="${GPU_JFA_LEVEL:-1}"

mkdir -p "$LOG_DIR"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

require_exec() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "Missing executable: $path" >&2
    exit 1
  fi
}

require_exec "$CPU_BIN"

GPU_AVAILABLE=0
case "$BENCHMARK_ENABLE_GPU" in
  1|true|TRUE|yes|YES)
    require_exec "$GPU_BIN"
    GPU_AVAILABLE=1
    ;;
  0|false|FALSE|no|NO)
    GPU_AVAILABLE=0
    ;;
  auto|AUTO)
    if [[ -x "$GPU_BIN" ]]; then
      GPU_AVAILABLE=1
    fi
    ;;
  *)
    echo "Invalid BENCHMARK_ENABLE_GPU value: $BENCHMARK_ENABLE_GPU" >&2
    exit 1
    ;;
esac

extract_metric() {
  local pattern="$1"
  local file="$2"
  awk -F': ' -v pat="$pattern" '$0 ~ pat {print $2}' "$file" | tail -n 1
}

extract_cpu_comp_time() {
  local file="$1"
  awk -F'= ' '/^compensation time = / {print $2}' "$file" | tail -n 1
}

extract_wall_time() {
  local file="$1"
  sed -n 's/^wall=\([0-9.]*\) s$/\1/p' "$file" | tail -n 1
}

extract_psnr_nrmse() {
  local which="$1"
  local file="$2"
  awk -v which="$which" '
    /^PSNR = / {
      psnr=$3
      gsub(",", "", psnr)
      nrmse=$5
      if (which == "first" && !seen) {
        print psnr "," nrmse
        seen=1
      }
      if (which == "last") {
        last=psnr "," nrmse
      }
    }
    END {
      if (which == "last") {
        print last
      }
    }
  ' "$file"
}

write_csv_header() {
  cat > "$CSV_PATH" <<'EOF'
dataset,backend,mode,cpu_index_mode,input_file,dims,rel_eb,use_chunk,threads,jfa_level,downsample_r2,wall_s,elapsed_s,compensation_time_s,edt_total_s,edt_round1_s,fill_sign_s,neutral_boundary_s,downsample_boundary_s,edt_round2_s,compensation_stage_s,initial_psnr,initial_nrmse,final_psnr,final_nrmse,log_file
EOF
}

append_cpu_row() {
  local dataset="$1"
  local input_file="$2"
  local dims="$3"
  local rel_eb="$4"
  local mode="$5"
  local downsample_r2="$6"
  local log_file="$7"

  local wall_s
  local comp_time
  local edt_total_s
  local edt_round1_s
  local fill_sign_s
  local neutral_boundary_s
  local downsample_boundary_s
  local edt_round2_s
  local compensation_stage_s
  local initial_pair
  local final_pair

  wall_s="$(extract_wall_time "$log_file")"
  comp_time="$(extract_cpu_comp_time "$log_file")"
  edt_total_s="$(extract_metric '^StageTime edt_total:' "$log_file")"
  edt_round1_s="$(extract_metric '^StageTime edt_round1:' "$log_file")"
  fill_sign_s="$(extract_metric '^StageTime fill_sign:' "$log_file")"
  neutral_boundary_s="$(extract_metric '^StageTime neutral_boundary:' "$log_file")"
  downsample_boundary_s="$(extract_metric '^StageTime downsample_boundary:' "$log_file")"
  edt_round2_s="$(extract_metric '^StageTime edt_round2:' "$log_file")"
  compensation_stage_s="$(extract_metric '^StageTime compensation:' "$log_file")"
  initial_pair="$(extract_psnr_nrmse first "$log_file")"
  final_pair="$(extract_psnr_nrmse last "$log_file")"

  {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$dataset" "cpu" "$mode" "$CPU_INDEX_MODE" "$input_file" "$dims" "$rel_eb" "" "$CPU_THREADS" "" "$downsample_r2"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$wall_s" "" "$comp_time" "$edt_total_s" "$edt_round1_s" "$fill_sign_s" "$neutral_boundary_s" "$downsample_boundary_s" "$edt_round2_s"
    printf '%s,%s,%s,%s\n' \
      "$compensation_stage_s" "$initial_pair" "$final_pair" "$log_file"
  } >> "$CSV_PATH"
}

append_gpu_row() {
  local dataset="$1"
  local input_file="$2"
  local dims="$3"
  local rel_eb="$4"
  local mode="$5"
  local edt_method="$6"
  local log_file="$7"

  local wall_s
  local elapsed_s
  local edt_total_s
  local edt_round1_s
  local fill_sign_s
  local neutral_boundary_s
  local edt_round2_s
  local compensation_stage_s
  local initial_pair
  local final_pair

  wall_s="$(extract_wall_time "$log_file")"
  elapsed_s="$(extract_metric '^Elapsed time:' "$log_file")"
  edt_total_s="$(extract_metric '^StageTime edt_total:' "$log_file")"
  edt_round1_s="$(extract_metric '^StageTime edt_round1:' "$log_file")"
  fill_sign_s="$(extract_metric '^StageTime fill_sign:' "$log_file")"
  neutral_boundary_s="$(extract_metric '^StageTime neutral_boundary:' "$log_file")"
  edt_round2_s="$(extract_metric '^StageTime edt_round2:' "$log_file")"
  compensation_stage_s="$(extract_metric '^StageTime compensation:' "$log_file")"
  initial_pair="$(extract_psnr_nrmse first "$log_file")"
  final_pair="$(extract_psnr_nrmse last "$log_file")"

  {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$dataset" "gpu" "$mode" "" "$input_file" "$dims" "$rel_eb" "1" "" "$GPU_JFA_LEVEL" ""
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$wall_s" "$elapsed_s" "" "$edt_total_s" "$edt_round1_s" "$fill_sign_s" "$neutral_boundary_s" "" "$edt_round2_s"
    printf '%s,%s,%s,%s\n' \
      "$compensation_stage_s" "$initial_pair" "$final_pair" "$log_file"
  } >> "$CSV_PATH"
}

run_cpu_case() {
  local dataset="$1"
  local input_file="$2"
  local rel_eb="$3"
  local d0="$4"
  local d1="$5"
  local d2="$6"
  local downsample_r2="$7"
  local mode="$8"
  local log_file="$LOG_DIR/${dataset}_cpu_${mode}.log"
  local quant_file="$LOG_DIR/${dataset}_cpu_${mode}.q"
  local comp_file="$LOG_DIR/${dataset}_cpu_${mode}.c"

  echo "Running CPU $dataset $mode"
  /usr/bin/time -f 'wall=%e s' \
    "$CPU_BIN" \
    -N 3 -d "$d0" "$d1" "$d2" \
    -i "$input_file" \
    -m rel -e "$rel_eb" \
    -q "$quant_file" \
    -c "$comp_file" \
    -t "$CPU_THREADS" \
    --cpu_index_mode "$CPU_INDEX_MODE" \
    --no_ssim 1 \
    --downsample_r2 "$downsample_r2" \
    >"$log_file" 2>&1

  append_cpu_row "$dataset" "$input_file" "${d0}x${d1}x${d2}" "$rel_eb" "$mode" "$downsample_r2" "$log_file"
}

run_gpu_case() {
  local dataset="$1"
  local input_file="$2"
  local rel_eb="$3"
  local d0="$4"
  local d1="$5"
  local d2="$6"
  local edt_method="$7"
  local mode="$8"
  local log_file="$LOG_DIR/${dataset}_gpu_${mode}.log"

  echo "Running GPU $dataset $mode"
  /usr/bin/time -f 'wall=%e s' \
    "$GPU_BIN" \
    "$input_file" "$rel_eb" 1 "$d0" "$d1" "$d2" "$edt_method" "$GPU_JFA_LEVEL" \
    >"$log_file" 2>&1

  append_gpu_row "$dataset" "$input_file" "${d0}x${d1}x${d2}" "$rel_eb" "$mode" "$edt_method" "$log_file"
}

main() {
  local nyx="${NYX_INPUT:?NYX_INPUT is not set in $CONFIG_PATH}"
  local hurricane="${HURRICANE_INPUT:?HURRICANE_INPUT is not set in $CONFIG_PATH}"
  local nyx_rel_eb="${NYX_REL_EB:-0.01}"
  local hurricane_rel_eb="${HURRICANE_REL_EB:-0.01}"
  local nyx_cpu_dims="${NYX_CPU_DIMS:-512 512 512}"
  local nyx_gpu_dims="${NYX_GPU_DIMS:-512 512 512}"
  local hurricane_cpu_dims="${HURRICANE_CPU_DIMS:-100 500 500}"
  local hurricane_gpu_dims="${HURRICANE_GPU_DIMS:-500 500 100}"

  local nyx_cpu_d0 nyx_cpu_d1 nyx_cpu_d2
  local nyx_gpu_d0 nyx_gpu_d1 nyx_gpu_d2
  local hurricane_cpu_d0 hurricane_cpu_d1 hurricane_cpu_d2
  local hurricane_gpu_d0 hurricane_gpu_d1 hurricane_gpu_d2

  read -r nyx_cpu_d0 nyx_cpu_d1 nyx_cpu_d2 <<< "$nyx_cpu_dims"
  read -r nyx_gpu_d0 nyx_gpu_d1 nyx_gpu_d2 <<< "$nyx_gpu_dims"
  read -r hurricane_cpu_d0 hurricane_cpu_d1 hurricane_cpu_d2 <<< "$hurricane_cpu_dims"
  read -r hurricane_gpu_d0 hurricane_gpu_d1 hurricane_gpu_d2 <<< "$hurricane_gpu_dims"

  require_file "$nyx"
  require_file "$hurricane"

  write_csv_header

  if [[ "$GPU_AVAILABLE" -eq 1 ]]; then
    echo "GPU benchmarks enabled"
  else
    echo "GPU benchmarks skipped"
  fi

  run_cpu_case nyx "$nyx" "$nyx_rel_eb" "$nyx_cpu_d0" "$nyx_cpu_d1" "$nyx_cpu_d2" 0 baseline
  run_cpu_case nyx "$nyx" "$nyx_rel_eb" "$nyx_cpu_d0" "$nyx_cpu_d1" "$nyx_cpu_d2" 1 downsample_r2
  if [[ "$GPU_AVAILABLE" -eq 1 ]]; then
    run_gpu_case nyx "$nyx" "$nyx_rel_eb" "$nyx_gpu_d0" "$nyx_gpu_d1" "$nyx_gpu_d2" 0 chunk
    run_gpu_case nyx "$nyx" "$nyx_rel_eb" "$nyx_gpu_d0" "$nyx_gpu_d1" "$nyx_gpu_d2" 1 jfa
    run_gpu_case nyx "$nyx" "$nyx_rel_eb" "$nyx_gpu_d0" "$nyx_gpu_d1" "$nyx_gpu_d2" 2 pba
    run_gpu_case nyx "$nyx" "$nyx_rel_eb" "$nyx_gpu_d0" "$nyx_gpu_d1" "$nyx_gpu_d2" 3 pba_opt
  fi

  run_cpu_case hurricane "$hurricane" "$hurricane_rel_eb" "$hurricane_cpu_d0" "$hurricane_cpu_d1" "$hurricane_cpu_d2" 0 baseline
  run_cpu_case hurricane "$hurricane" "$hurricane_rel_eb" "$hurricane_cpu_d0" "$hurricane_cpu_d1" "$hurricane_cpu_d2" 1 downsample_r2
  if [[ "$GPU_AVAILABLE" -eq 1 ]]; then
    run_gpu_case hurricane "$hurricane" "$hurricane_rel_eb" "$hurricane_gpu_d0" "$hurricane_gpu_d1" "$hurricane_gpu_d2" 0 chunk
    run_gpu_case hurricane "$hurricane" "$hurricane_rel_eb" "$hurricane_gpu_d0" "$hurricane_gpu_d1" "$hurricane_gpu_d2" 1 jfa
    run_gpu_case hurricane "$hurricane" "$hurricane_rel_eb" "$hurricane_gpu_d0" "$hurricane_gpu_d1" "$hurricane_gpu_d2" 2 pba
    run_gpu_case hurricane "$hurricane" "$hurricane_rel_eb" "$hurricane_gpu_d0" "$hurricane_gpu_d1" "$hurricane_gpu_d2" 3 pba_opt
  fi

  echo "Wrote CSV to $CSV_PATH"
}

main "$@"
