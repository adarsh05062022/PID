#!/usr/bin/env bash
# Wait out the gain sweep's generation, score both artists, render the table.
#
# Assumes run_gain_sweep.sh is already running (or has finished). Each stage is
# resumable and skips completed work, so this is safe to re-run at any point.
#
#   bash scripts/style/run_sweep_end_to_end.sh <lpips_gpu> <acc_gpu>
#
# The two scoring runs go on separate GPUs so Van Gogh and Kelly score in
# parallel; each loads its models once for all twelve of its arms.
#
# NOTE on process checks: patterns are written as [g]enerate_artist so this
# script's own command line does not match them. Without that, pgrep -f finds
# this very shell and every liveness check reports a phantom worker -- which is
# exactly how a dead driver went unnoticed once already.
#
# Superseded by run_overnight.sh, which does this plus the TECA-vector sweep.
# Kept for reference; only ever covers the default (CASteer) SV_SET.

set -u

GPU_VG=${1:-0}
GPU_KL=${2:-1}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python}
OUT=results/sd14/style/casteer_vectors
LOGS=logs/style
mkdir -p "$LOGS"

count_complete() {
    local c=0
    for d in "$OUT"/casteer_kp*/ "$OUT"/adaptive_kg_kp*/; do
        [ -d "$d/all" ] || continue
        [ "$(ls "$d/all"/*.png 2>/dev/null | wc -l)" -eq 100 ] && c=$((c + 1))
    done
    echo "$c"
}

echo "[$(date +%H:%M:%S)] waiting for generation ($(count_complete)/24 arms complete)"
while true; do
    c=$(count_complete)
    [ "$c" -ge 24 ] && { echo "[$(date +%H:%M:%S)] generation complete: 24/24"; break; }
    workers=$(pgrep -fc '[g]enerate_artist\.py' || true)
    driver=$(pgrep -fc '[r]un_gain_sweep\.sh' || true)
    if [ "${workers:-0}" -eq 0 ] && [ "${driver:-0}" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] generation stopped with only ${c}/24 arms complete."
        echo "Re-run: bash scripts/style/run_gain_sweep.sh both <gpus> 2"
        exit 1
    fi
    sleep 60
done

echo "[$(date +%H:%M:%S)] scoring both artists in parallel (LPIPS then Acc, batched per artist)"
bash scripts/style/score_sweep.sh vangogh "$GPU_VG" > "$LOGS/score_sweep_vangogh.log" 2>&1 &
p1=$!
bash scripts/style/score_sweep.sh kelly "$GPU_KL" > "$LOGS/score_sweep_kelly.log" 2>&1 &
p2=$!

fail=0
wait $p1 || fail=$((fail + 1))
wait $p2 || fail=$((fail + 1))
[ "$fail" -gt 0 ] && echo "WARNING: ${fail} scoring run(s) returned non-zero — check $LOGS/score_sweep_*.log"

echo "[$(date +%H:%M:%S)] rendering table"
PYTHONPATH="$REPO_ROOT" "$PY" scripts/style/make_results_table.py \
    --comparisons "$OUT/comparison_sweep_vangogh.json" "$OUT/comparison_sweep_kelly.json" \
    --output "$OUT/results_table.png" \
    --title "Artist-style erasure: CASteer vs Adaptive Kg gain sweep (SD-v1.4)"

echo "[$(date +%H:%M:%S)] done -> $OUT/results_table.png"
