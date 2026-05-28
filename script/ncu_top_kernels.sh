#!/usr/bin/env bash
# ncu profile of the top kernels in the CUDA compensation pipeline.
# Run with sudo (GPU performance counters require it unless
# /etc/modprobe.d/nvprof.conf sets NVreg_RestrictProfilingToAdminUsers=0).
#
#   sudo bash script/ncu_top_kernels.sh
#
# Output: /tmp/ncu_pm_a1.csv (machine-readable per-metric per-kernel)

set -euo pipefail

NCU=${NCU:-/usr/local/cuda/bin/ncu}
REPO=${REPO:-/home/jp/git/posterization_mitigation}
BIN=${BIN:-$REPO/build/test_cuda/test_compensation_cuda}
DATA=${DATA:-/home/jp/data/nyx_512x512x512/velocity_x.f32}
REL_EB=${REL_EB:-0.01}
W=${W:-512}; H=${H:-512}; D=${D:-512}
METHOD=${METHOD:-3}
OUT=${OUT:-/tmp/ncu_pm_a1.csv}

KERNELS='regex:compensation_idw_downsample|fill_sign_and_neutral_boundary_fused|pba_kernelFloodZ|pba_extract_result|pba_boundary_sign_init|get_filtered_boundary'

echo "[ncu] binary : $BIN"
echo "[ncu] data   : $DATA"
echo "[ncu] dims   : ${W}x${H}x${D}  rel_eb=${REL_EB}  method=${METHOD}"
echo "[ncu] kernels: $KERNELS"
echo "[ncu] output : $OUT"

"$NCU" \
    --kernel-name "$KERNELS" \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --launch-count 32 \
    --csv --log-file "$OUT" \
    "$BIN" "$DATA" "$REL_EB" 1 "$W" "$H" "$D" "$METHOD"

echo "[ncu] done   : $OUT ($(wc -l <"$OUT") lines)"
