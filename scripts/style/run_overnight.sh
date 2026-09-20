#!/usr/bin/env bash
# Overnight chain: finish the CASteer-vector sweep, then run the whole thing
# again on the TECA/SAFREE vectors, then render both tables.
#
# The point of the second sweep is that the two vector sets were measured at
# cosine ~0.95 per block (scripts/style/compare_steering_vectors.py) yet produce
# visibly different images, so "which control law wins" has only been answered
# for one set. Running the identical 24 arms on the other set says whether the
# answer is a property of the law or of the .pt file it read.
#
# Stages are sequential on purpose. Three GPUs are shared with other users and
# the 7B judge needs ~18GB resident, so overlapping generation with scoring is
# what produced the earlier OOM. Each stage is resumable and skips completed
# work, so re-running after any interruption costs only what is missing.
#
#   bash scripts/style/run_overnight.sh [gpu_ids]        default: 1,2,5
#
# Progress: logs/style/overnight.log

set -u

GPU_IDS_STR=${1:-1,2,5}
IFS=',' read -ra GPUS <<< "$GPU_IDS_STR"
GPU_A=${GPUS[0]}
GPU_B=${GPUS[1]:-${GPUS[0]}}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python}
ROOT=results/sd14/style
CASTEER_OUT="$ROOT/casteer_vectors"
TECA_OUT="$ROOT/teca_vectors"
LOGS=logs/style
mkdir -p "$LOGS"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# Liveness patterns are bracketed so this script's own command line does not
# match them -- an unbracketed pgrep -f finds this very shell and every check
# reports a phantom worker.
running() {
    local n
    n=$(pgrep -fc "$1" 2>/dev/null)
    echo "${n:-0}"
}

# ---------------------------------------------------------------------------
# Stage 1 -- let the in-flight CASteer-vector scoring finish.
# ---------------------------------------------------------------------------
say "stage 1: waiting for the running CASteer-vector scoring to finish"
while true; do
    ev=$(running '[e]val_artist\.py')
    sc=$(running '[s]core_sweep\.sh')
    [ "$ev" -eq 0 ] && [ "$sc" -eq 0 ] && break
    sleep 120
done
acc=$(ls "$CASTEER_OUT"/casteer_kp*/all/acc_qwen_scores.json "$CASTEER_OUT"/adaptive_kg_kp*/all/acc_qwen_scores.json 2>/dev/null | wc -l)
say "stage 1 done: CASteer-vector acc scored for ${acc}/24 arms"

# Re-score to sweep up anything the interrupted run left behind, and to run the
# backfill that repairs arms whose scores never reached the comparison file.
for slug in vangogh kelly; do
    say "stage 1b: reconciling comparison_sweep_${slug}.json"
    bash scripts/style/score_sweep.sh "$slug" "$GPU_A" \
        > "$LOGS/score_sweep_${slug}_reconcile.log" 2>&1 \
        || say "WARNING: reconcile for ${slug} returned non-zero"
done

say "rendering CASteer-vector table"
PYTHONPATH="$REPO_ROOT" "$PY" scripts/style/make_results_table.py \
    --comparisons "$CASTEER_OUT/comparison_sweep_vangogh.json" "$CASTEER_OUT/comparison_sweep_kelly.json" \
    --output "$CASTEER_OUT/results_table.png" \
    --title "Artist-style erasure, CASteer-built steering vectors (SD-v1.4)" \
    > "$LOGS/table_casteer.log" 2>&1 || say "WARNING: CASteer table render failed"

# ---------------------------------------------------------------------------
# Stage 2 -- the same 24 arms on the TECA/SAFREE vectors.
# ---------------------------------------------------------------------------
say "stage 2: generating 24 TECA-vector arms on GPUs [${GPU_IDS_STR}], 1 slot per GPU"
SV_SET=teca bash scripts/style/run_gain_sweep.sh both "$GPU_IDS_STR" 1 \
    > "$LOGS/gain_sweep_teca.log" 2>&1 \
    || say "WARNING: TECA generation reported incomplete arms — see $LOGS/gain_sweep_teca.log"

teca_done=0
for d in "$TECA_OUT"/casteer_teca_kp*/ "$TECA_OUT"/adaptive_kg_teca_kp*/; do
    [ -d "$d/all" ] || continue
    [ "$(ls "$d/all"/*.png 2>/dev/null | wc -l)" -eq 100 ] && teca_done=$((teca_done + 1))
done
say "stage 2 done: ${teca_done}/24 TECA arms complete"

# ---------------------------------------------------------------------------
# Stage 3 -- score the TECA arms. One artist per GPU; the judge is ~18GB, so
# two of them is the most these GPUs take alongside other users.
# ---------------------------------------------------------------------------
say "stage 3: scoring TECA arms (Van Gogh on GPU ${GPU_A}, Kelly on GPU ${GPU_B})"
SV_SET=teca bash scripts/style/score_sweep.sh vangogh "$GPU_A" \
    > "$LOGS/score_sweep_teca_vangogh.log" 2>&1 &
p1=$!
SV_SET=teca bash scripts/style/score_sweep.sh kelly "$GPU_B" \
    > "$LOGS/score_sweep_teca_kelly.log" 2>&1 &
p2=$!
wait $p1 || say "WARNING: TECA Van Gogh scoring returned non-zero"
wait $p2 || say "WARNING: TECA Kelly scoring returned non-zero"

# ---------------------------------------------------------------------------
# Stage 4 -- tables.
# ---------------------------------------------------------------------------
say "stage 4: rendering TECA table"
PYTHONPATH="$REPO_ROOT" "$PY" scripts/style/make_results_table.py \
    --comparisons "$TECA_OUT/comparison_sweep_teca_vangogh.json" "$TECA_OUT/comparison_sweep_teca_kelly.json" \
    --output "$TECA_OUT/results_table_teca.png" \
    --title "Artist-style erasure, TECA/SAFREE steering vectors (SD-v1.4)" \
    > "$LOGS/table_teca.log" 2>&1 || say "WARNING: TECA table render failed"

say "done"
say "  CASteer vectors -> $CASTEER_OUT/results_table.png"
say "  TECA vectors    -> $TECA_OUT/results_table_teca.png"
