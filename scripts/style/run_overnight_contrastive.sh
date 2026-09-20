#!/usr/bin/env bash
# Overnight chain for the contrastive-negative vector sets: build Kelly
# McKernan's heldout/retain steering vectors, then run the identical 24-arm
# gain sweep (see run_gain_sweep.sh) on each, scored the same way as the
# casteer/teca tables in run_overnight.sh.
#
# estimate_steering_vectors_style.py builds a vector two ways: `retain` fits
# its negatives to the eval set's own other four artists, which optimizes
# Acc_u/LPIPS_u directly; `heldout` uses artists absent from both eval CSVs,
# so any gain there is real generalization rather than having been shown the
# answer. Running the identical 24 arms on both says whether `retain`'s edge
# over the original CASteer vector survives on a vector that never saw the
# eval artists, or whether it was just fitting the test.
#
# Van Gogh already has both .pt files (built by hand -- see
# results/sd14/steering_vectors_{heldout,retain}/Van Gogh.pt). This script's
# first stage builds Kelly McKernan's so both artists run, matching the
# casteer/teca tables.
#
# Stages are sequential on purpose, same reasoning as run_overnight.sh: the
# 7B judge needs ~18GB resident and generation is kept off the scoring GPU
# while it runs. Every stage is resumable and skips completed work, so
# re-running after any interruption costs only what is missing.
#
#   bash scripts/style/run_overnight_contrastive.sh [gpu_ids]   default: 1,2,5
#
# Progress: redirect to a log yourself, e.g.
#   nohup bash scripts/style/run_overnight_contrastive.sh 1,2,5 \
#       > logs/style/overnight_contrastive.log 2>&1 &

set -u

GPU_IDS_STR=${1:-1,2,5}
IFS=',' read -ra GPUS <<< "$GPU_IDS_STR"
GPU_A=${GPUS[0]}
GPU_B=${GPUS[1]:-${GPUS[0]}}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python}
OUT=results/sd14/style/contrastive_vectors
LOGS=logs/style
mkdir -p "$LOGS" "$OUT"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Stage 0 -- build Kelly McKernan's heldout + retain vectors if missing.
# estimate_steering_vectors_style.py itself no-ops and returns if the output
# .pt already exists, so this is safe to re-run; the explicit check here is
# just so the log says which vectors were reused.
# ---------------------------------------------------------------------------
say "stage 0: Kelly McKernan heldout/retain steering vectors"
for neg in heldout retain; do
    out_dir="results/sd14/steering_vectors_${neg}"
    if [ -f "$out_dir/Kelly McKernan.pt" ]; then
        say "  ${neg}: already built"
        continue
    fi
    say "  ${neg}: building on GPU ${GPU_A} (~15 min)"
    PYTHONPATH="$REPO_ROOT" "$PY" scripts/diffusion/estimate_steering_vectors_style.py \
        --concept "Kelly McKernan" --negatives "$neg" --num_prompts 50 --gpu "$GPU_A" \
        --output_dir "$out_dir" \
        > "$LOGS/sv_${neg}_kelly.log" 2>&1
    if [ ! -f "$out_dir/Kelly McKernan.pt" ]; then
        say "ERROR: ${neg} vector build failed for Kelly McKernan — see $LOGS/sv_${neg}_kelly.log"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Stage 1 -- heldout: 24-arm sweep over both artists, then score + table.
# ---------------------------------------------------------------------------
say "stage 1: generating 24 heldout-vector arms on GPUs [${GPU_IDS_STR}], 1 slot per GPU"
SV_SET=heldout bash scripts/style/run_gain_sweep.sh both "$GPU_IDS_STR" 1 \
    > "$LOGS/gain_sweep_heldout.log" 2>&1 \
    || say "WARNING: heldout generation reported incomplete arms — see $LOGS/gain_sweep_heldout.log"

say "stage 1b: scoring heldout arms (Van Gogh on GPU ${GPU_A}, Kelly on GPU ${GPU_B})"
SV_SET=heldout bash scripts/style/score_sweep.sh vangogh "$GPU_A" \
    > "$LOGS/score_sweep_heldout_vangogh.log" 2>&1 &
p1=$!
SV_SET=heldout bash scripts/style/score_sweep.sh kelly "$GPU_B" \
    > "$LOGS/score_sweep_heldout_kelly.log" 2>&1 &
p2=$!
wait $p1 || say "WARNING: heldout Van Gogh scoring returned non-zero"
wait $p2 || say "WARNING: heldout Kelly scoring returned non-zero"

say "rendering heldout table"
PYTHONPATH="$REPO_ROOT" "$PY" scripts/style/make_results_table.py \
    --comparisons "$OUT/comparison_sweep_heldout_vangogh.json" "$OUT/comparison_sweep_heldout_kelly.json" \
    --output "$OUT/results_table_heldout.png" \
    --title "Artist-style erasure, held-out-negative steering vectors (SD-v1.4)" \
    > "$LOGS/table_heldout.log" 2>&1 || say "WARNING: heldout table render failed"

# ---------------------------------------------------------------------------
# Stage 2 -- retain: the same 24 arms on the retain vectors.
# ---------------------------------------------------------------------------
say "stage 2: generating 24 retain-vector arms on GPUs [${GPU_IDS_STR}], 1 slot per GPU"
SV_SET=retain bash scripts/style/run_gain_sweep.sh both "$GPU_IDS_STR" 1 \
    > "$LOGS/gain_sweep_retain.log" 2>&1 \
    || say "WARNING: retain generation reported incomplete arms — see $LOGS/gain_sweep_retain.log"

say "stage 2b: scoring retain arms (Van Gogh on GPU ${GPU_A}, Kelly on GPU ${GPU_B})"
SV_SET=retain bash scripts/style/score_sweep.sh vangogh "$GPU_A" \
    > "$LOGS/score_sweep_retain_vangogh.log" 2>&1 &
p1=$!
SV_SET=retain bash scripts/style/score_sweep.sh kelly "$GPU_B" \
    > "$LOGS/score_sweep_retain_kelly.log" 2>&1 &
p2=$!
wait $p1 || say "WARNING: retain Van Gogh scoring returned non-zero"
wait $p2 || say "WARNING: retain Kelly scoring returned non-zero"

say "rendering retain table"
PYTHONPATH="$REPO_ROOT" "$PY" scripts/style/make_results_table.py \
    --comparisons "$OUT/comparison_sweep_retain_vangogh.json" "$OUT/comparison_sweep_retain_kelly.json" \
    --output "$OUT/results_table_retain.png" \
    --title "Artist-style erasure, retain-negative steering vectors (SD-v1.4)" \
    > "$LOGS/table_retain.log" 2>&1 || say "WARNING: retain table render failed"

say "done"
say "  heldout vectors -> $OUT/results_table_heldout.png"
say "  retain vectors  -> $OUT/results_table_retain.png"
