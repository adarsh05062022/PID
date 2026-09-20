#!/usr/bin/env bash
# Generate all four arms of the artist-erasure comparison for one artist.
#
# Every arm goes through scripts/style/generate_artist.py, on the same steering
# vectors and the same cross-attention hooks, so the only thing that differs
# between the output directories is the steering law:
#
#   baseline     no steering    -- the LPIPS reference every other arm is measured against
#   casteer      beta = 2.0     -- Eq. 6, the published control
#   pid          flat Kg        -- one global scalar applied identically to every block
#   adaptive_kg  reweighted Kg  -- the method under test
#
# `pid` is the arm that isolates what "adaptive" buys: it shares kp/ki/kg/kd with
# adaptive_kg and differs only in how the global term is split across blocks.
#
# Usage:
#   bash scripts/style/run_artist.sh <vangogh|kelly> <gpu_a> <gpu_b> <gpu_c> <gpu_d>
#   e.g.  bash scripts/style/run_artist.sh vangogh 1 6 7 1
#
# Four GPU ids, one per arm; repeat an id to co-locate arms (SD-v1.4 fp16 is
# ~6GB per worker). Re-running skips images that already exist.

set -u

SLUG=${1:?"Usage: bash $0 <vangogh|kelly> <gpu_a> <gpu_b> <gpu_c> <gpu_d>"}
GPU_A=${2:?"need 4 gpu ids"}; GPU_B=${3:?}; GPU_C=${4:?}; GPU_D=${5:?}

case "$SLUG" in
    vangogh) ARTIST="Van Gogh";       SV="results/sd14/steering_vectors/Van Gogh.pt" ;;
    kelly)   ARTIST="Kelly McKernan"; SV="results/sd14/steering_vectors/Kelly McKernan.pt" ;;
    *) echo "unknown artist slug: $SLUG (expected vangogh or kelly)"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python}
GEN=scripts/style/generate_artist.py
ROOT=results/sd14/style
# The steered arms cluster under casteer_vectors/ (see scripts/style/README.md);
# baseline is vector-set-independent and stays at the shared top level so
# score_sweep.sh's CASteer AND TECA runs read the same reference images.
OUT="$ROOT/casteer_vectors"
LOGS=logs/style
mkdir -p "$LOGS" "$OUT"

if [ ! -f "$SV" ]; then
    echo "Missing steering vectors: $SV"
    echo "Build them with:"
    echo "  PYTHONPATH=. $PY scripts/diffusion/estimate_steering_vectors.py \\"
    echo "      --model_name sd14 --concept \"$ARTIST\" --mode style --num_prompts 50 \\"
    echo "      --output_dir ./results/sd14/steering_vectors"
    exit 1
fi

# kp is CASteer's beta, so the casteer and PID arms apply an identical
# proportional term and differ only in what the integrators add on top.
# ki/kg ~ kp/50 (the number of denoising steps) per the gain-scale note in
# core/pid_steering_step_block.py; kd = 0 matches the earlier sd14 style runs.
KP=2.0; KI=0.01; KG=0.01; KD=0.0

# baseline has no vector set, so it is not "in" casteer_vectors/ with the other
# three arms -- it lives at the shared top level (see OUT/ROOT split above).
base_of() { [ "$1" = baseline ] && echo "$ROOT" || echo "$OUT"; }

PIDS=()
launch() {  # launch <gpu> <arm> <extra args...>
    local gpu=$1 arm=$2; shift 2
    local dir; dir="$(base_of "$arm")/${arm}_${SLUG}"
    echo "GPU ${gpu} -> ${dir#$ROOT/}"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO_ROOT" setsid nohup "$PY" "$GEN" \
        --artist "$ARTIST" --save_dir "$dir" --gpus 0 "$@" \
        > "$LOGS/${arm}_${SLUG}.log" 2>&1 < /dev/null &
    PIDS+=($!)
}

launch "$GPU_A" adaptive_kg --controller adaptive_kg --concept_path "$SV" \
       --kp $KP --ki $KI --kg $KG --kd $KD --intermediate_clipping
launch "$GPU_B" casteer --controller casteer --concept_path "$SV" \
       --steering_strength $KP --intermediate_clipping
launch "$GPU_C" pid --controller pid --concept_path "$SV" \
       --kp $KP --ki $KI --kg $KG --kd $KD --intermediate_clipping
launch "$GPU_D" baseline --no_steer

echo "Launched ${#PIDS[@]} arms for ${ARTIST} — waiting..."
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done

for arm in baseline casteer pid adaptive_kg; do
    n=$(ls "$(base_of "$arm")/${arm}_${SLUG}/all/"*.png 2>/dev/null | wc -l)
    echo "  ${arm}_${SLUG}: ${n}/100 images"
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED arm(s) failed — see $LOGS/*_${SLUG}.log"
    exit 1
fi
echo "Done. Score with:  bash scripts/style/score_artist.sh $SLUG <gpu>"
