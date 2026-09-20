#!/usr/bin/env bash
# Gain sweep: CASteer vs Adaptive Kg across kp, on the artist-erasure protocol.
#
#   CASteer      kp in {0.5, 1.0, 1.5, 2.0}          (ki = kg = kd = 0 by law)
#   Adaptive Kg  kp in {0.5, 1.0, 1.5, 2.0}
#                x kd in {0.1, 0.5},  ki = kg = 0.01
#
# 12 arms per artist. Directory names carry the gains so score_sweep.sh can
# derive labels without a lookup table.
#
# SV_SET picks which steering vectors every arm reads. The control laws and the
# protocol are identical across sets; only the .pt file changes, which is what
# makes the sets comparable arm for arm:
#
#   casteer (default)  results/sd14/steering_vectors/<Artist>.pt
#                      built by scripts/diffusion/estimate_steering_vectors.py
#                      from "{ImageNet class}, {artist} style" prompt pairs
#   teca               SAFREE/style/sv_<artist>_final_beta1.pt
#                      built by style_erasure_TECA_final.py from hand-written
#                      per-concept prompt pairs, fp32, DPMSolver
#   heldout            results/sd14/steering_vectors_heldout/<Artist>.pt
#                      built by estimate_steering_vectors_style.py, negatives =
#                      artists absent from both eval CSVs -- any gain here is
#                      genuine generalization, not fit to the eval set
#   retain             results/sd14/steering_vectors_retain/<Artist>.pt
#                      same script, negatives = the eval set's OTHER four
#                      artists -- optimizes Acc_u/LPIPS_u directly, so it is
#                      the upper bound `heldout` is checked against
#
# The tag goes BEFORE kp in the directory name (casteer_teca_kp1.5_vangogh), so
# score_sweep.sh's casteer_kp*_<slug> glob cannot pick up the other set's arms.
#
# Usage:
#   bash scripts/style/run_gain_sweep.sh <vangogh|kelly|both> <gpu_ids> [slots_per_gpu]
#   SV_SET=teca bash scripts/style/run_gain_sweep.sh both 1,2,5 2
#
# Resumable: an arm already holding 100 images is skipped, and generate_artist.py
# independently skips images that exist, so an interrupted arm resumes mid-way.

set -u

WHICH=${1:?"Usage: bash $0 <vangogh|kelly|both> <gpu_ids> [slots_per_gpu]"}
GPU_IDS_STR=${2:?"need gpu ids, e.g. 1,2,5"}
SLOTS_PER_GPU=${3:-2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python}
GEN=scripts/style/generate_artist.py
LOGS=logs/style

SV_SET=${SV_SET:-casteer}
SAFREE_STYLE="$(dirname "$REPO_ROOT")/SAFREE/style"
case "$SV_SET" in
    casteer) ARM_TAG=""; OUT="results/sd14/style/casteer_vectors" ;;
    teca)    ARM_TAG="teca_"; OUT="results/sd14/style/teca_vectors" ;;
    # heldout and retain are both built by estimate_steering_vectors_style.py
    # from the same contrastive-negative recipe -- only which negatives differ
    # -- so they cluster together under contrastive_vectors/ and are keyed
    # apart by the heldoutsv_/retainsv_ arm tag instead of by directory.
    heldout) ARM_TAG="heldoutsv_"; OUT="results/sd14/style/contrastive_vectors" ;;
    retain)  ARM_TAG="retainsv_"; OUT="results/sd14/style/contrastive_vectors" ;;
    *) echo "unknown SV_SET: $SV_SET (expected casteer, teca, heldout or retain)"; exit 1 ;;
esac
mkdir -p "$LOGS" "$OUT"

KP_VALUES=(0.5 1.0 1.5 2.0)
KD_VALUES=(0.1 0.5)
KI=0.01
KG=0.01

IFS=',' read -ra GPUS <<< "$GPU_IDS_STR"
MAX_CONCURRENT=$(( ${#GPUS[@]} * SLOTS_PER_GPU ))

case "$WHICH" in
    vangogh) SLUGS=(vangogh) ;;
    kelly)   SLUGS=(kelly) ;;
    both)    SLUGS=(vangogh kelly) ;;
    *) echo "unknown target: $WHICH (expected vangogh, kelly or both)"; exit 1 ;;
esac

artist_of() { [ "$1" = "vangogh" ] && echo "Van Gogh" || echo "Kelly McKernan"; }
sv_of() {
    local artist; artist="$(artist_of "$1")"
    case "$SV_SET" in
        teca)
            if [ "$1" = "vangogh" ]; then echo "$SAFREE_STYLE/sv_vangogh_final_beta1.pt"
            else echo "$SAFREE_STYLE/sv_kellymckernan_final_beta1.pt"; fi ;;
        heldout) echo "results/sd14/steering_vectors_heldout/${artist}.pt" ;;
        retain)  echo "results/sd14/steering_vectors_retain/${artist}.pt" ;;
        *)       echo "results/sd14/steering_vectors/${artist}.pt" ;;
    esac
}

for slug in "${SLUGS[@]}"; do
    sv="$(sv_of "$slug")"
    [ -f "$sv" ] || { echo "Missing steering vectors: $sv"; exit 1; }
done

# ---------------------------------------------------------------------------
# Adopt the kp=2.0 CASteer cell from the existing single-point run.
#
# `casteer_<slug>` is already exactly this configuration -- same vectors, same
# protocol constants, steering_strength 2.0 with intermediate clipping -- and it
# is already scored. Hardlinking it (same filesystem) costs nothing and keeps
# the sweep's kp=2.0 row identical to the number already reported, rather than
# a fresh sample of fp16 nondeterminism.
#
# Only valid for the default vector set: casteer_<slug> was generated from
# results/sd14/steering_vectors, so adopting it into a teca-tagged sweep would
# silently mix vector sets in one table.
# ---------------------------------------------------------------------------
adopt_kp2() {
    local slug=$1
    [ -z "$ARM_TAG" ] || return 1
    local src="$OUT/casteer_${slug}"
    local dst="$OUT/casteer_kp2.0_${slug}"

    [ -d "$dst/all" ] && [ "$(ls "$dst/all"/*.png 2>/dev/null | wc -l)" -eq 100 ] && return 0
    [ -d "$src/all" ] || return 1

    local ok
    ok=$("$PY" - "$src/config.json" <<'PY'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except Exception:
    print("no"); raise SystemExit
print("yes" if (c.get("controller") == "casteer"
                and float(c.get("steering_strength", -1)) == 2.0
                and c.get("intermediate_clipping") is True
                and int(c.get("n_prompts", 0)) == 100) else "no")
PY
)
    [ "$ok" = "yes" ] || { echo "  kp2.0 adopt: $src config does not match, will generate"; return 1; }
    [ "$(ls "$src/all"/*.png 2>/dev/null | wc -l)" -eq 100 ] || return 1

    mkdir -p "$dst"
    cp -al "$src/all" "$dst/all" 2>/dev/null || cp -a "$src/all" "$dst/all"
    cp -a "$src/config.json" "$dst/config.json"
    for j in lpips_scores.json acc_qwen_scores.json acc_qwen_scores_log.json; do
        [ -f "$src/all/$j" ] && cp -a "$src/all/$j" "$dst/all/$j"
    done
    echo "  adopted casteer_kp2.0_${slug} from casteer_${slug} (hardlinked, scores carried over)"
    return 0
}

# ---------------------------------------------------------------------------
# Job list
# ---------------------------------------------------------------------------
JOBS=()   # "slug|arm_dir_name|controller|kp|kd"
for slug in "${SLUGS[@]}"; do
    adopt_kp2 "$slug"
    for kp in "${KP_VALUES[@]}"; do
        JOBS+=("${slug}|casteer_${ARM_TAG}kp${kp}_${slug}|casteer|${kp}|0")
    done
    for kp in "${KP_VALUES[@]}"; do
        for kd in "${KD_VALUES[@]}"; do
            JOBS+=("${slug}|adaptive_kg_${ARM_TAG}kp${kp}_kd${kd}_${slug}|adaptive_kg|${kp}|${kd}")
        done
    done
done

echo "Sweep [SV_SET=${SV_SET}]: ${#JOBS[@]} arms over GPUs [${GPU_IDS_STR}] x ${SLOTS_PER_GPU} slots = ${MAX_CONCURRENT} concurrent"

launch() {  # launch <gpu> <slug> <arm> <controller> <kp> <kd>
    local gpu=$1 slug=$2 arm=$3 ctrl=$4 kp=$5 kd=$6
    local artist; artist="$(artist_of "$slug")"
    local sv; sv="$(sv_of "$slug")"
    local args=(--artist "$artist" --save_dir "$OUT/${arm}" --gpus 0
                --concept_path "$sv" --intermediate_clipping)
    if [ "$ctrl" = "casteer" ]; then
        args+=(--controller casteer --steering_strength "$kp")
    else
        args+=(--controller adaptive_kg --kp "$kp" --ki $KI --kg $KG --kd "$kd")
    fi
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO_ROOT" setsid nohup \
        "$PY" "$GEN" "${args[@]}" > "$LOGS/${arm}.log" 2>&1 < /dev/null &
}

idx=0
launched=0
skipped=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r slug arm ctrl kp kd <<< "$job"

    n=$(ls "$OUT/${arm}/all"/*.png 2>/dev/null | wc -l)
    if [ "$n" -eq 100 ]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Cap concurrency. Round-robin by job index keeps the GPUs balanced, since
    # every arm is the same 100 images of work.
    while (( $(jobs -pr | wc -l) >= MAX_CONCURRENT )); do sleep 10; done

    gpu=${GPUS[$(( idx % ${#GPUS[@]} ))]}
    echo "[$(date +%H:%M:%S)] GPU ${gpu} -> ${arm}  (${ctrl} kp=${kp} kd=${kd})"
    launch "$gpu" "$slug" "$arm" "$ctrl" "$kp" "$kd"
    idx=$((idx + 1))
    launched=$((launched + 1))
    sleep 3   # stagger model loads so they do not all hit the NFS cache at once
done

echo "Launched ${launched} arms, skipped ${skipped} already complete — waiting..."
wait

echo
echo "=== sweep complete [SV_SET=${SV_SET}] ==="
incomplete=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r slug arm ctrl kp kd <<< "$job"
    n=$(ls "$OUT/${arm}/all"/*.png 2>/dev/null | wc -l)
    printf "  %-48s %3d/100\n" "$arm" "$n"
    [ "$n" -eq 100 ] || incomplete=$((incomplete + 1))
done

if [ "$incomplete" -gt 0 ]; then
    echo "WARNING: ${incomplete} arm(s) incomplete — re-run this script to resume them."
    exit 1
fi
echo "All arms complete. Score with:  SV_SET=${SV_SET} bash scripts/style/score_sweep.sh <vangogh|kelly> <gpu>"
