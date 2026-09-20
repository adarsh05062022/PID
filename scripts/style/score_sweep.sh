#!/usr/bin/env bash
# Score every gain-sweep arm for one artist into a single comparison JSON.
#
# score_artist.sh addresses four fixed arm names; the sweep has twelve whose
# gains live in the directory name, so this discovers them by glob and derives
# the display label from the name. Both metrics run batched -- one process for
# all LPIPS arms, one for all Acc arms -- because on this NFS mount loading
# AlexNet and the 7B judge costs more than scoring an arm does.
#
# SV_SET selects which steering-vector set's arms to score, and keeps every
# set in its own comparison file:
#   casteer (default)  casteer_kp*_<slug>             -> comparison_sweep_<slug>.json
#   teca               casteer_teca_kp*_<slug>        -> comparison_sweep_teca_<slug>.json
#   heldout            casteer_heldoutsv_kp*_<slug>   -> comparison_sweep_heldout_<slug>.json
#   retain             casteer_retainsv_kp*_<slug>    -> comparison_sweep_retain_<slug>.json
#
# Usage:
#   bash scripts/style/score_sweep.sh <vangogh|kelly> [gpu]
#   SV_SET=teca bash scripts/style/score_sweep.sh vangogh 1
#   FORCE=1 rescores arms that already have score JSONs.

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
# Qwen2.5-VL needs transformers >= 4.49; the generation env is pinned older, so
# the two halves of the scorer run in different environments.
VLM_PY=${VLM_PYTHON:-/storage/s25017/miniconda3/envs/vlmjudge/bin/python}
EVAL=scripts/style/eval_artist.py
ROOT=results/sd14/style
LOGS=logs/style
mkdir -p "$LOGS"

SV_SET=${SV_SET:-casteer}
# Arms and the comparison JSON cluster under one directory per vector set
# (heldout/retain share contrastive_vectors/, see run_gain_sweep.sh); the
# baseline is vector-set-independent (no steering applied) and stays at the
# shared top level so every SV_SET run reads the exact same reference.
case "$SV_SET" in
    casteer) OUT="$ROOT/casteer_vectors"; ARM_TAG="";
             COMPARISON="$OUT/comparison_sweep_${SLUG}.json" ;;
    teca)    OUT="$ROOT/teca_vectors"; ARM_TAG="teca_";
             COMPARISON="$OUT/comparison_sweep_teca_${SLUG}.json" ;;
    heldout) OUT="$ROOT/contrastive_vectors"; ARM_TAG="heldoutsv_";
             COMPARISON="$OUT/comparison_sweep_heldout_${SLUG}.json" ;;
    retain)  OUT="$ROOT/contrastive_vectors"; ARM_TAG="retainsv_";
             COMPARISON="$OUT/comparison_sweep_retain_${SLUG}.json" ;;
    *) echo "unknown SV_SET: $SV_SET (expected casteer, teca, heldout or retain)"; exit 1 ;;
esac
LOGTAG="${SV_SET}_${SLUG}"

BASELINE="$ROOT/baseline_${SLUG}/all"
[ -d "$BASELINE" ] || { echo "Missing baseline images: $BASELINE"; exit 1; }

N_RUNS=${N_RUNS:-1}   # greedy decoding, so repeat passes are identical
FORCE=${FORCE:-0}

# Turn a directory name into the label the table shows. The SV_SET tag is not
# part of the label: the two sets live in separate comparison files, so inside
# one table "CASteer kp=1.5" is unambiguous and the tag would just be noise
# repeated on every row.
#   casteer_kp1.5_vangogh                   -> "CASteer kp=1.5"
#   adaptive_kg_teca_kp0.5_kd0.1_vangogh    -> "AdaptiveKg kp=0.5 kd=0.1"
label_of() {
    local name=$1
    local kp kd
    kp=$(sed -n 's/.*_kp\([0-9.]*\).*/\1/p' <<< "$name")
    kd=$(sed -n 's/.*_kd\([0-9.]*\).*/\1/p' <<< "$name")
    if [[ "$name" == casteer_* ]]; then
        echo "CASteer kp=${kp}"
    else
        echo "AdaptiveKg kp=${kp} kd=${kd}"
    fi
}

# Collect arms that are fully generated, CASteer first then Adaptive Kg, each
# ascending in kp (sort -V so 1.5 sorts before 2.0, not after 10).
ARMS=()
while IFS= read -r d; do
    [ -d "$d/all" ] || continue
    [ "$(ls "$d/all"/*.png 2>/dev/null | wc -l)" -eq 100 ] || {
        echo "skip $(basename "$d"): only $(ls "$d/all"/*.png 2>/dev/null | wc -l)/100 images"; continue; }
    ARMS+=("$(basename "$d")")
done < <( { ls -d "$OUT"/casteer_${ARM_TAG}kp*_"${SLUG}" 2>/dev/null | sort -V
            ls -d "$OUT"/adaptive_kg_${ARM_TAG}kp*_"${SLUG}" 2>/dev/null | sort -V; } )

[ ${#ARMS[@]} -gt 0 ] || { echo "No ${SV_SET} sweep arms found for ${SLUG} — run run_gain_sweep.sh first"; exit 1; }
echo "Found ${#ARMS[@]} ${SV_SET} arms for ${ARTIST}"

# Build the batch lists, skipping arms already scored.
build_batch() {  # build_batch <score_json_name>
    local jsonname=$1
    local dirs=() labels=()
    for arm in "${ARMS[@]}"; do
        if [ "$FORCE" != "1" ] && [ -f "$OUT/${arm}/all/${jsonname}" ]; then
            echo "  skip ${arm}: already scored" >&2
            continue
        fi
        dirs+=("$OUT/${arm}/all")
        labels+=("$(label_of "$arm")")
    done
    ( IFS=,; echo "${dirs[*]:-}" )
    ( IFS=,; echo "${labels[*]:-}" )
}

echo
echo "=== LPIPS (${ARTIST}, ${SV_SET}) ==="
mapfile -t LP < <(build_batch lpips_scores.json)
if [ -n "${LP[0]}" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO_ROOT" "$PY" "$EVAL" lpips \
        --artist "$ARTIST" --baseline_dir "$BASELINE" \
        --method_dirs "${LP[0]}" --tags "${LP[1]}" \
        --append_to "$COMPARISON" > "$LOGS/lpips_sweep_${LOGTAG}.log" 2>&1
    grep -E "^LPIPS|^Appended" "$LOGS/lpips_sweep_${LOGTAG}.log" | tail -40
else
    echo "  all arms already scored"
fi

echo
echo "=== Acc (${ARTIST}, ${SV_SET}) ==="
mapfile -t AC < <(build_batch acc_qwen_scores.json)
if [ -n "${AC[0]}" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO_ROOT" "$VLM_PY" "$EVAL" acc-qwen \
        --artist "$ARTIST" --method_dirs "${AC[0]}" --tags "${AC[1]}" --n_runs "$N_RUNS" \
        --append_to "$COMPARISON" > "$LOGS/acc_sweep_${LOGTAG}.log" 2>&1
    grep -E "^Acc_|^Appended" "$LOGS/acc_sweep_${LOGTAG}.log" | tail -40
else
    echo "  all arms already scored"
fi

# ---------------------------------------------------------------------------
# Backfill arms whose scores exist on disk but never reached the comparison.
#
# build_batch skips an arm that already has its score JSON, which is what makes
# an interrupted sweep cheap to resume -- but the skip happens BEFORE the append,
# so a pre-scored arm never lands in the comparison file and silently vanishes
# from the table. That is how "CASteer kp=2.0" (adopted from the earlier
# single-point run, scores carried over) went missing from the first render.
#
# Reading the per-arm JSONs back is free, so do it for every arm every time:
# idempotent, and it repairs comparisons written before this fix existed.
# ---------------------------------------------------------------------------
BACKFILL_ARGS=()
for arm in "${ARMS[@]}"; do
    BACKFILL_ARGS+=("$(label_of "$arm")" "$OUT/${arm}/all")
done
PYTHONPATH="$REPO_ROOT" "$PY" - "$COMPARISON" "$ARTIST" "${BACKFILL_ARGS[@]}" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join('scripts', 'style'))
from eval_artist import append_to_master

comparison, artist = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
existing = {}
if os.path.exists(comparison):
    with open(comparison) as f:
        existing = json.load(f).get(artist, {})

added = 0
for i in range(0, len(rest), 2):
    tag, d = rest[i], rest[i + 1]
    for fname, section in (('lpips_scores.json', 'lpips'), ('acc_qwen_scores.json', 'acc')):
        if section in existing.get(tag, {}):
            continue
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            scores = json.load(f)
        append_to_master(comparison, artist, tag, section, {**scores, 'method_dir': d})
        added += 1
print(f'backfilled {added} missing section(s) into {comparison}')
PY

# The unsteered row: already scored by the earlier sweep, just carried into this
# comparison so the table has its reference. LPIPS against itself is 0 by
# construction, so only Acc is recorded. Read from BASELINE (top-level, shared
# across both vector sets), not OUT (this SV_SET's own arm folder).
if [ -f "$BASELINE/acc_qwen_scores.json" ]; then
    PYTHONPATH="$REPO_ROOT" "$PY" - "$COMPARISON" "$ARTIST" \
        "$BASELINE/acc_qwen_scores.json" "$BASELINE" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join('scripts', 'style'))
from eval_artist import append_to_master
comparison, artist, scores_path, method_dir = sys.argv[1:5]
with open(scores_path) as f:
    scores = json.load(f)
append_to_master(comparison, artist, 'Baseline', 'acc', {**scores, 'method_dir': method_dir})
PY
fi

echo
PYTHONPATH="$REPO_ROOT" "$PY" "$EVAL" report --comparison "$COMPARISON"
echo
echo "Comparison JSON: $COMPARISON"
