#!/usr/bin/env bash
# Steering-vector ablation: hold the control law fixed, swap the vectors.
#
# The main sweep (run_artist.sh) compares control laws on ONE set of steering
# vectors -- the ones scripts/diffusion/estimate_steering_vectors.py builds from
# "{ImageNet class}, {artist} style" prompt pairs. This script re-runs the same
# three laws on the TECA/SAFREE vectors instead, which were built from a
# hand-written prompt list at fp32 on a different scheduler.
#
# The two families agree to cosine ~0.95 per block (see
# scripts/style/compare_steering_vectors.py), so the expectation is that the
# images barely move. This is the experiment that checks that expectation
# instead of assuming it -- if the images DO move, then every control-law
# conclusion from the main sweep is really a statement about vector extraction.
#
# Usage:
#   bash scripts/style/run_vector_ablation.sh <vangogh|kelly> <gpu_a> <gpu_b> <gpu_c> [full]
#   e.g.  bash scripts/style/run_vector_ablation.sh vangogh 1 3 6
#         bash scripts/style/run_vector_ablation.sh vangogh 1 3 6 full
#
# Without "full" only the 10 grid cases are generated (5 erased + 5 preserved),
# which is what the side-by-side grid needs. With "full" all 100 run, so the
# arms can also be scored with score_artist.sh. Images already present are
# skipped, so the 10-case run is a free down payment on the full one.

set -u

SLUG=${1:?"Usage: bash $0 <vangogh|kelly> <gpu_a> <gpu_b> <gpu_c> [full]"}
GPU_A=${2:?"need 3 gpu ids"}; GPU_B=${3:?}; GPU_C=${4:?}
MODE=${5:-grid}

SAFREE_STYLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/SAFREE/style"

case "$SLUG" in
    vangogh) ARTIST="Van Gogh";       SV="$SAFREE_STYLE/sv_vangogh_final_beta1.pt"
             GRID_CSV=big_artist_prompts_grid10.csv ;;
    kelly)   ARTIST="Kelly McKernan"; SV="$SAFREE_STYLE/sv_kellymckernan_final_beta1.pt"
             GRID_CSV=short_niche_art_prompts_grid10.csv ;;
    *) echo "unknown artist slug: $SLUG (expected vangogh or kelly)"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

[ -f "$SV" ] || { echo "Missing TECA steering vectors: $SV"; exit 1; }

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python}
GEN=scripts/style/generate_artist.py
OUT=results/sd14/style/teca_vectors
LOGS=logs/style
mkdir -p "$LOGS" "$OUT"

if [ "$MODE" = "full" ]; then
    CSV_ARG=()
    echo "Mode: full 100-prompt set"
else
    CSV_ARG=(--prompts_csv "exp/datasets/eval/artists/grid/$GRID_CSV")
    echo "Mode: 10 grid cases only"
fi

# Identical to the main sweep, so the only variable is the vector file.
KP=2.0; KI=0.01; KG=0.01; KD=0.0

PIDS=()
launch() {  # launch <gpu> <arm> <extra args...>
    local gpu=$1 arm=$2; shift 2
    echo "GPU ${gpu} -> ${arm}_teca_${SLUG}"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO_ROOT" setsid nohup "$PY" "$GEN" \
        --artist "$ARTIST" --save_dir "$OUT/${arm}_teca_${SLUG}" --gpus 0 \
        --concept_path "$SV" "${CSV_ARG[@]}" "$@" \
        > "$LOGS/${arm}_teca_${SLUG}.log" 2>&1 < /dev/null &
    PIDS+=($!)
}

launch "$GPU_A" adaptive_kg --controller adaptive_kg \
       --kp $KP --ki $KI --kg $KG --kd $KD --intermediate_clipping
launch "$GPU_B" casteer --controller casteer \
       --steering_strength $KP --intermediate_clipping
launch "$GPU_C" pid --controller pid \
       --kp $KP --ki $KI --kg $KG --kd $KD --intermediate_clipping

echo "Launched ${#PIDS[@]} TECA-vector arms for ${ARTIST} — waiting..."
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done

for arm in casteer pid adaptive_kg; do
    n=$(ls "$OUT/${arm}_teca_${SLUG}/all/"*.png 2>/dev/null | wc -l)
    echo "  ${arm}_teca_${SLUG}: ${n} images"
done

[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED arm(s) failed — see $LOGS/*_teca_${SLUG}.log"; exit 1; }
echo "Done. Build the grid with scripts/style/make_vector_grid.py"
