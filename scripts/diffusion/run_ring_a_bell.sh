#!/usr/bin/env bash
# Ring-A-Bell nudity attack (datasets/nudity-ring-a-bell.csv, 79 prompts) on
# SD-1.4 with two controllers:
#
#   p_only      Kp = 2.0, Ki = Kg = Kd = 0        (P term only == CASteer beta=2)
#   adaptive_kg Kp = 2.0, Ki = Kg = 0.01, Kd = 0.02 (PID with adaptive global term)
#
# Both arms go through --controller adaptive_kg; at zero I/G/D gains it is
# bit-exact to CrossAttentionOutputSteering / PIDSteering (see
# run_kp_sweep.sh and logs/kp_only_compare.log) (images are identical either way).
# One image per prompt, seed 42, same vectors/hooks/blocks as the
# unsafe_plus_safe runs. Resumable (existing images are skipped).
#
# Usage:
#   bash scripts/diffusion/run_ring_a_bell.sh <gpu_ids>      # arms round-robin over GPUs
#   ARMS_SEL="p_only" bash scripts/diffusion/run_ring_a_bell.sh 5
#
# Score afterwards with:
#   python scripts/diffusion/eval_nudenet.py results/sd14/ring_a_bell/<arm>

set -u

GPU_IDS_STR=${1:?"Usage: bash $0 <gpu_ids>, e.g. 5,2"}
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python3}
PROMPTS=datasets/nudity-ring-a-bell.csv
VECTORS=results/sd14/steering_vectors/nudity.pt
OUT=results/sd14/ring_a_bell
LOGS=logs/ring_a_bell
mkdir -p "$OUT" "$LOGS"

KP=2.0; KI=0.01; KG=0.01; KD=0.02
read -r -a ARMS_SEL <<< "${ARMS_SEL:-p_only adaptive_kg}"

# arm spec: "<kp> <ki> <kg> <kd> <dirname>"
ARMS=()
for a in "${ARMS_SEL[@]}"; do
    case "$a" in
        p_only)      ARMS+=("$KP 0.0 0.0 0.0 p_only_kp${KP}") ;;
        adaptive_kg) ARMS+=("$KP $KI $KG $KD adaptive_kg_kp${KP}_ki${KI}_kg${KG}_kd${KD}") ;;
        *) echo "unknown arm '$a' (p_only|adaptive_kg)"; exit 1 ;;
    esac
done

run_arm() {
    local gpu=$1 kp=$2 ki=$3 kg=$4 kd=$5 name=$6
    local log="$LOGS/${name}.log"
    echo "=========== adaptive_kg kp=$kp ki=$ki kg=$kg kd=$kd on GPU $gpu -> $OUT/$name ===========" | tee "$log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO_ROOT" "$PY" scripts/diffusion/run_with_steering.py \
        --model_name sd14 \
        --prompts_csv "$PROMPTS" \
        --generate_concept ring_a_bell \
        --output_dir "$OUT/$name" \
        --controller adaptive_kg --kp "$kp" --ki "$ki" --kg "$kg" --kd "$kd" \
        --intermediate_clipping \
        --num_images_per_prompt 1 --seed 42 \
        erase --concept_path "$VECTORS" >> "$log" 2>&1
    local rc=$?
    echo "RESULT $name exit=$rc" | tee -a "$log"
    return $rc
}

i=0
pids=()
for arm in "${ARMS[@]}"; do
    gpu=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}
    # shellcheck disable=SC2086
    run_arm "$gpu" $arm &
    pids+=($!)
    i=$((i + 1))
    sleep 60   # stagger: model load memory shows up late on the shared box
done

fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
echo "RING-A-BELL DONE (fail=$fail)"
exit $fail
