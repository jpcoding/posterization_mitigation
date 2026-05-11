#!/usr/bin/env bash
# GPU benchmark sweep: all NYX + Hurricane fields, edt_method=3 (PBA+ optimized).
# Run on a GPU node. Output: script/gpu_results.csv
# Usage: bash script/run_gpu_sweep.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/build/test_cuda/test_compensation_cuda"
EDT_METHOD="${EDT_METHOD:-3}"   # 3 = PBA+ optimized (recommended)
OUT_CSV="$ROOT_DIR/script/gpu_results.csv"
LOG_DIR="$ROOT_DIR/script/gpu_logs"
mkdir -p "$LOG_DIR"

NYX_DIR="/home/jp/data/nyx_512x512x512"
HUR_DIR="/home/jp/data/hurricane_100x500x500"
EBS=(1e-3 5e-3 1e-2)

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
  echo "ERROR: CUDA binary not found: $BIN" >&2
  echo "Build with: cmake --build build --target test_compensation_cuda" >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true

echo "dataset,field,rel_eb,dims,initial_psnr,final_psnr,psnr_delta,elapsed_ms,boundary_ms,edt1_ms,fill_sign_ms,neutral_ms,edt2_ms,comp_ms,edt_total_ms,edt_method" \
  > "$OUT_CSV"

run_one() {
  local dataset="$1" field="$2" w="$3" h="$4" d="$5" input_file="$6" eb="$7"
  local log="$LOG_DIR/${dataset}_${field}_eb${eb}.log"

  # CUDA binary arg order: <file> <rel_eb> <use_chunk=0> <dim_fast> <dim_mid> <dim_slow> [edt_method]
  "$BIN" "$input_file" "$eb" 0 "$w" "$h" "$d" "$EDT_METHOD" \
    > "$log" 2>&1 || true

  local init_psnr final_psnr elapsed boundary edt1 fill_sign neutral edt2 comp edt_total delta dims
  dims="${w}x${h}x${d}"
  init_psnr=$( grep '^PSNR'                  "$log" | head -1 | awk '{print $3}' | tr -d ',' || echo "")
  final_psnr=$(grep '^PSNR'                  "$log" | tail -1 | awk '{print $3}' | tr -d ',' || echo "")
  elapsed=$(   grep '^Elapsed time:'          "$log" | tail -1 | awk '{printf "%.1f", $3*1000}' || echo "")
  boundary=$(  grep 'StageTime boundary'      "$log" | awk '{printf "%.1f", $2*1000}' || echo "")
  edt1=$(      grep 'StageTime edt_round1'    "$log" | awk '{printf "%.1f", $2*1000}' || echo "")
  fill_sign=$( grep 'StageTime fill_sign'     "$log" | awk '{printf "%.1f", $2*1000}' || echo "")
  neutral=$(   grep 'StageTime neutral'       "$log" | awk '{printf "%.1f", $2*1000}' || echo "")
  edt2=$(      grep 'StageTime edt_round2'    "$log" | awk '{printf "%.1f", $2*1000}' || echo "")
  comp=$(      grep 'StageTime compensation'  "$log" | awk '{printf "%.1f", $2*1000}' || echo "")
  edt_total=$( grep 'StageTime edt_total'     "$log" | awk '{printf "%.1f", $2*1000}' || echo "")
  delta=""
  [[ -n "$final_psnr" && -n "$init_psnr" ]] && \
    delta=$(python3 -c "print(f'{float(\"$final_psnr\")-float(\"$init_psnr\"):.4f}')" 2>/dev/null || echo "")

  echo "$dataset,$field,$eb,$dims,${init_psnr:-NA},${final_psnr:-NA},${delta:-NA},${elapsed:-NA},${boundary:-NA},${edt1:-NA},${fill_sign:-NA},${neutral:-NA},${edt2:-NA},${comp:-NA},${edt_total:-NA},$EDT_METHOD" \
    >> "$OUT_CSV"

  printf "  %-12s %-26s eb=%-5s  init=%s  final=%s  Δ=%s  total=%sms  edt1=%sms  edt2=%sms\n" \
    "$dataset" "$field" "$eb" \
    "${init_psnr:-?}" "${final_psnr:-?}" "${delta:-?}" \
    "${elapsed:-?}" "${edt1:-?}" "${edt2:-?}"
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
echo "EDT method: $EDT_METHOD  (3=PBA+ optimized)"
echo ""

echo "=== NYX 512x512x512 ==="
for field in "${NYX_FIELDS[@]}"; do
  for eb in "${EBS[@]}"; do
    # CUDA binary takes dims as: width(fast) height depth(slow)
    # NYX is 512x512x512 so all same
    run_one nyx "$field" 512 512 512 "$NYX_DIR/${field}.f32" "$eb"
  done
done

echo ""
echo "=== Hurricane 100x500x500 ==="
for field in "${HUR_FIELDS[@]}"; do
  for eb in "${EBS[@]}"; do
    # Hurricane: 100x500x500, fastest dim = 500
    run_one hurricane "$field" 500 500 100 "$HUR_DIR/${field}.f32" "$eb"
  done
done

echo ""
echo "Results saved to: $OUT_CSV"
