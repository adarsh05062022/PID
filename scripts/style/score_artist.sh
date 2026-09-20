#!/usr/bin/env bash
# Score every generated arm for one artist into a single comparison JSON.
#
# LPIPS for all arms first (cheap, one AlexNet), then the VLM accuracies (one
# 7B load for the whole sweep), then the table.
#
# Usage:
#   bash scripts/style/score_artist.sh <vangogh|kelly> [gpu]
#   e.g.  bash scripts/style/score_artist.sh vangogh 3

set -u

SLUG=${1:?"Usage: bash $0 <vangogh|kelly> [gpu]"}
GPU=${2:-0}

case "$SLUG" in
    vangogh) ARTIST="Van Gogh" ;;
    kelly)   ARTIST="Kelly McKernan" ;;
    *) echo "unknown artist slug: $SLUG (expected vangogh or kelly)"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python}
# The VLM judge needs transformers >= 4.49 for Qwen2.5-VL; the generation env is
# pinned older, so the two halves of the scorer run in different environments.
VLM_PY=${VLM_PYTHON:-/storage/s25017/miniconda3/envs/vlmjudge/bin/python}
EVAL=scripts/style/eval_artist.py
ROOT=results/sd14/style
# casteer/pid/adaptive_kg cluster under casteer_vectors/ (run_artist.sh writes
# them there); baseline has no vector set and stays at the shared top level.
OUT="$ROOT/casteer_vectors"
COMPARISON="$OUT/comparison_${SLUG}.json"
LOGS=logs/style
mkdir -p "$LOGS"

BASELINE="$ROOT/baseline_${SLUG}/all"
if [ ! -d "$BASELINE" ]; then
    echo "Missing baseline images: $BASELINE — run scripts/style/run_artist.sh $SLUG first"
    exit 1
fi

declare -A TAGS=( [baseline]=Baseline [casteer]=CASteer [pid]=PID [adaptive_kg]=AdaptiveKg )

# The VLM decodes greedily, so repeat passes over the same image return the same
# answer and N_RUNS>1 only multiplies the cost. Kept configurable because a
# sampling judge would need it.
N_RUNS=${N_RUNS:-1}
# Already-scored arms are skipped so an interrupted sweep can be resumed;
# FORCE=1 rescores everything.
FORCE=${FORCE:-0}

scored() { [ "$FORCE" != "1" ] && [ -f "$1" ]; }

# baseline lives at $ROOT (no vector set); the other three at $OUT (casteer_vectors/).
dir_of() { [ "$1" = baseline ] && echo "$BASELINE" || echo "$OUT/${1}_${SLUG}/all"; }

for arm in baseline casteer pid adaptive_kg; do
    DIR="$(dir_of "$arm")"
    [ -d "$DIR" ] || { echo "skip ${arm}: no images"; continue; }

    # The baseline scored against itself is LPIPS 0 by construction, so only the
    # steered arms get an LPIPS row; its Acc row is the un-erased reference the
    # other Acc numbers are read against.
    [ "$arm" = "baseline" ] && continue
    if scored "$DIR/lpips_scores.json"; then echo "skip LPIPS ${arm}: already scored"; continue; fi

    echo "=== LPIPS ${arm}_${SLUG} ==="
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO_ROOT" "$PY" "$EVAL" lpips \
        --artist "$ARTIST" --baseline_dir "$BASELINE" --method_dir "$DIR" \
        --tag "${TAGS[$arm]}" --append_to "$COMPARISON" \
        2>&1 | tee "$LOGS/lpips_${arm}_${SLUG}.log" | grep -E "^LPIPS|^Appended"
done

for arm in baseline casteer pid adaptive_kg; do
    DIR="$(dir_of "$arm")"
    [ -d "$DIR" ] || continue
    if scored "$DIR/acc_qwen_scores.json"; then echo "skip Acc ${arm}: already scored"; continue; fi

    echo "=== Acc ${arm}_${SLUG} ==="
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO_ROOT" "$VLM_PY" "$EVAL" acc-qwen \
        --artist "$ARTIST" --method_dir "$DIR" --n_runs "$N_RUNS" \
        --tag "${TAGS[$arm]}" --append_to "$COMPARISON" \
        2>&1 | tee "$LOGS/acc_${arm}_${SLUG}.log" | grep -E "^Acc_|^Appended"
done

echo
PYTHONPATH="$REPO_ROOT" "$PY" "$EVAL" report --comparison "$COMPARISON"
echo
echo "Comparison JSON: $COMPARISON"
