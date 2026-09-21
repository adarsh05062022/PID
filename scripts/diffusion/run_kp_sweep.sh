#!/usr/bin/env bash
# Kp sweep on unsafe_plus_safe: CASteer (Kp only) vs our method (adaptive_kg),
# with the --diag per-step/per-block PID trace written for every arm.
#
#   CASteer      kp in {2.0, 1.5, 1.0, 0.5}     ki = kg = kd = 0
#   Our method   kp in {2.0, 1.5, 1.0, 0.5}     ki = kg = 0.01, kd = 0.02
#
# The CASteer arm is run through --controller adaptive_kg at Ki=Kg=Kd=0, which
# is bit-exact to CrossAttentionOutputSteering (Eq. 6, beta = Kp; see
# logs/kp_only_compare.log) and, unlike the CASteer class itself, supports
# --diag -- so its pid_records/ IS CASteer's e/u trace. adaptive_kg rather than
# pid because only adaptive_kg logs mean_u_raw / mean_u_out / mean_e_signed,
# which the step-x-block analysis is built on. Everything else (vectors,
# hooks, blocks, prompts, seed) matches the existing
# results/sd14/unsafe_plus_safe runs.
#
# Layout (gains in the directory name, as elsewhere under results/sd14):
#   results/sd14/unsafe_plus_safe_kp_sweep/
#       casteer_kp1.5/{0..16}/42-0.png + pid_records/000NN-42.csv
#       adaptive_kg_kp1.5_ki0.01_kg0.01_kd0.02/...
#       ...
#
# Usage:
#   bash scripts/diffusion/run_kp_sweep.sh <gpu_ids>
#   bash scripts/diffusion/run_kp_sweep.sh 2,5,3,1,7,6     # one arm per GPU
#   KPS="2.0" bash scripts/diffusion/run_kp_sweep.sh 2,5   # only the Kp=2.0 arms
#   METHODS="casteer" bash scripts/diffusion/run_kp_sweep.sh 2,5,3,1   # one controller only
#
# Arms are assigned round-robin over the GPU list and run concurrently.
# Resumable: run_with_steering.py skips images that already exist (note the
# --diag trace is only written when an image is generated, so to regenerate
# pid_records/ the arm directory has to be removed first).

set -u

GPU_IDS_STR=${1:?"Usage: bash $0 <gpu_ids>, e.g. 2,5,3"}
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python3}
PROMPTS="$(dirname "$REPO_ROOT")/SAFREE/datasets/unsafe_plus_safe.csv"
VECTORS=results/sd14/steering_vectors/nudity.pt
OUT=results/sd14/unsafe_plus_safe_kp_sweep
LOGS=logs/kp_sweep
mkdir -p "$OUT" "$LOGS"

KI=0.01; KG=0.01; KD=0.02
# Space-separated; override KPS="..." / METHODS="..." to run a subset of arms.
read -r -a KPS <<< "${KPS:-2.0 1.5 1.0 0.5}"
read -r -a METHODS <<< "${METHODS:-casteer ours}"

# arm spec: "<controller> <kp> <ki> <kg> <kd> <dirname>"
ARMS=()
for kp in "${KPS[@]}"; do
    for method in "${METHODS[@]}"; do
        case "$method" in
            casteer) ARMS+=("adaptive_kg $kp 0.0 0.0 0.0 casteer_kp${kp}") ;;
            ours)    ARMS+=("adaptive_kg $kp $KI $KG $KD adaptive_kg_kp${kp}_ki${KI}_kg${KG}_kd${KD}") ;;
            *) echo "unknown method '$method' (casteer|ours)"; exit 1 ;;
        esac
    done
done

run_arm() {
    local gpu=$1 controller=$2 kp=$3 ki=$4 kg=$5 kd=$6 name=$7
    local log="$LOGS/${name}.log"
    echo "=========== $controller kp=$kp ki=$ki kg=$kg kd=$kd on GPU $gpu -> $OUT/$name ===========" | tee "$log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO_ROOT" "$PY" scripts/diffusion/run_with_steering.py \
        --model_name sd14 \
        --prompts_csv "$PROMPTS" \
        --generate_concept unsafe_plus_safe \
        --output_dir "$OUT/$name" \
        --controller "$controller" --kp "$kp" --ki "$ki" --kg "$kg" --kd "$kd" \
        --intermediate_clipping --diag \
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
done

fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
echo "SWEEP DONE (fail=$fail)"
exit $fail
