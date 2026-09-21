#!/usr/bin/env bash
# Replicate Table 5 of CASteer (arXiv:2503.09630v5, Appendix): concrete object
# erasure. Erase "Snoopy" from SD-1.4 and check that five other concepts
# (Mickey, Spongebob, Pikachu, dog, legislator -- the first four deliberately
# close to Snoopy) survive, following the SPM / DoCo protocol:
#
#   * 80 CLIP templates x 10 images = 800 images per concept
#     (exp/datasets/eval/clip_templates.json)
#   * CS  : SPM CLIP score (ViT-B/32, 2.5 * cos(image, concept name); x100 in
#           the table). Lower is better on Snoopy, higher on the others.
#   * FID : clean-fid between the 800 SD-1.4 images of a concept and the 800
#           images the steered model makes for it. Lower is better.
#
# Rows reproduced here: SD-1.4 (orig), Ours (beta = 2, Eq. 5) and Ours (clip)
# (beta = 2 with intermediate clipping, Eq. 6). The other rows of Table 5
# (ESD, SPM, SAFREE, Receler, DoCo) are copied from the paper by make_table5.py.
#
# Steering vector: 50 ImageNet-class prompt pairs ("<cls> with snoopy" /
# "<cls>", Appendix C.1), estimated on SD-1.4 itself; the first-step vector is
# applied at every denoising step, which is run_with_steering.py's (and the
# official repo's) default. The optional "casteer-2.0-clip-allsteps" arm uses
# the per-step vectors instead (Algorithm 2 as written in the paper).
#
# Layout:
#   results/sd14/steering_vectors/snoopy.pt
#   results/sd14/eval_<concept>/<method>/<prompt>/<seed>-<i>.png
#   results/sd14/eval_<concept>/{clip_score.tsv,fid.tsv}
#   results/sd14/table5_object_erasure.{md,tsv}      (make_table5.py)
#
# Usage:
#   bash scripts/diffusion/run_table5_object_erasure.sh <gpu_ids>        # e.g. 1,2,5,6
#   WORKERS=2 bash scripts/diffusion/run_table5_object_erasure.sh 1,2      # two arms at a time
#   CONCEPTS="snoopy mickey" METHODS="orig casteer-2.0-clip" bash scripts/diffusion/run_table5_object_erasure.sh 2
#   STAGE=generate bash scripts/diffusion/run_table5_object_erasure.sh 2,1  # skip scoring
#   STAGE=score    bash scripts/diffusion/run_table5_object_erasure.sh 2    # scoring + table only
#   SKIP_ARMS="snoopy:orig mickey:orig" bash ...   # leave those arms alone (e.g. still running elsewhere)
#
# The GPUs are shared with other users and their free memory changes by the
# minute, so <gpu_ids> is a candidate list, not an assignment: WORKERS arms
# run concurrently, and each arm is started on whichever candidate GPU has the
# most free memory at that moment (at least MIN_FREE_MIB, default 6000 --
# SD-1.4 fp16 at batch 10 with --vae_slicing peaks at ~4.2 GB reserved; a
# batched VAE decode of 10 images would need >10 GB), waiting if none qualifies. An
# arm that fails (usually an OOM at model load because a GPU filled up) is
# retried after RETRY_WAIT seconds, up to MAX_RETRIES times. Resumable:
# arms that already have all their images are skipped outright,
# run_with_steering.py skips images that already exist, and the steering
# vector is only estimated when snoopy.pt is missing.

set -u

GPU_IDS_STR=${1:?"Usage: bash $0 <gpu_ids>, e.g. 1,2,5,6"}
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"
WORKERS=${WORKERS:-3}                   # arms generated concurrently
MIN_FREE_MIB=${MIN_FREE_MIB:-6000}      # a GPU needs this much free memory to get an arm
MAX_RETRIES=${MAX_RETRIES:-30}
RETRY_WAIT=${RETRY_WAIT:-120}           # seconds between attempts / GPU polls
STAGGER=${STAGGER:-90}                  # seconds between worker starts (model load takes ~1-2 min
                                        # before its memory shows up in nvidia-smi)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python3}
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # less reserved-but-unused memory

STAGE=${STAGE:-all}                     # all | generate | score
TEMPLATES=exp/datasets/eval/clip_templates.json
VECTOR=results/sd14/steering_vectors/snoopy.pt
OUT=results/sd14
LOGS=logs/table5
BATCH_SIZE=${BATCH_SIZE:-10}            # images per batch; 10 = one batch per prompt
SEED=${SEED:-0}
NUM_IMAGES=${NUM_IMAGES:-10}            # per template -> 80 x 10 = 800 per concept
mkdir -p "$LOGS" "$OUT/steering_vectors"

# Space-separated; override to run a subset.
read -r -a CONCEPTS <<< "${CONCEPTS:-snoopy mickey spongebob pikachu dog legislator}"
read -r -a METHODS  <<< "${METHODS:-orig casteer-2.0 casteer-2.0-clip}"
SKIP_ARMS=" ${SKIP_ARMS:-} "                # "concept:method" entries not to generate

# method name -> extra run_with_steering.py flags (the erase subcommand and its
# vector are appended for every method except orig)
method_flags() {
    case "$1" in
        orig)                      echo "" ;;
        casteer-2.0)               echo "--steering_strength 2.0" ;;
        casteer-2.0-clip)          echo "--steering_strength 2.0 --intermediate_clipping" ;;
        casteer-2.0-clip-allsteps) echo "--steering_strength 2.0 --intermediate_clipping --use_all_diffusion_steps" ;;
        casteer-*)                 # e.g. casteer-1.5 / casteer-1.5-clip for a beta sweep
            local beta=${1#casteer-}; local clip=""
            if [[ "$beta" == *-clip ]]; then beta=${beta%-clip}; clip="--intermediate_clipping"; fi
            echo "--steering_strength $beta $clip" ;;
        *) echo "unknown method '$1'" >&2; return 1 ;;
    esac
}

NUM_TEMPLATES=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$TEMPLATES")
EXPECTED=$((NUM_TEMPLATES * NUM_IMAGES))

# Candidate GPU with the most free memory, if that is at least MIN_FREE_MIB;
# prints nothing otherwise.
pick_gpu() {
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
        awk -F', *' -v want="$GPU_IDS_STR" -v min="$MIN_FREE_MIB" '
            BEGIN { n = split(want, a, ","); for (i = 1; i <= n; i++) ok[a[i]] = 1; best = -1 }
            ($1 in ok) { free = $3 - $2; if (free >= min && free > best) { best = free; g = $1 } }
            END { if (best >= 0) print g }'
}

wait_for_gpu() {   # blocks until pick_gpu returns something, then prints it
    local g
    while true; do
        g=$(pick_gpu)
        [[ -n "$g" ]] && { echo "$g"; return 0; }
        echo "$(date +%H:%M:%S) no candidate GPU with >= ${MIN_FREE_MIB} MiB free; waiting ${RETRY_WAIT}s" >&2
        sleep "$RETRY_WAIT"
    done
}

# ---------------------------------------------------------------- Step 1: vector
if [[ "$STAGE" != "score" ]]; then
    if [[ ! -f "$VECTOR" ]]; then
        log="$LOGS/estimate_snoopy.log"
        gpu=$(wait_for_gpu)
        echo "=========== estimating Snoopy steering vector on GPU $gpu -> $VECTOR ===========" | tee "$log"
        CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/diffusion/estimate_steering_vectors.py \
            --model_name sd14 --concept snoopy --mode concrete --num_prompts 50 \
            --output_dir "$OUT/steering_vectors" >> "$log" 2>&1
        rc=$?
        echo "RESULT estimate_snoopy exit=$rc" | tee -a "$log"
        [[ $rc -eq 0 && -f "$VECTOR" ]] || { echo "steering vector estimation failed, see $log"; exit 1; }
    else
        echo "steering vector ready: $(ls -la "$VECTOR")"
    fi
fi

# ------------------------------------------------------------ Step 2: generation
run_arm_once() {
    local gpu=$1 concept=$2 method=$3 out=$4 log=$5
    local flags; flags=$(method_flags "$method") || return 1
    local erase=()
    [[ "$method" != "orig" ]] && erase=(erase --concept_path "$VECTOR")
    echo "=========== $(date +%H:%M:%S) $concept / $method on GPU $gpu -> $out ===========" | tee -a "$log"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/diffusion/run_with_steering.py \
        --model_name sd14 \
        --generate_concept "$concept" \
        --template_path "$TEMPLATES" \
        --num_images_per_prompt "$NUM_IMAGES" --batch_size "$BATCH_SIZE" --seed "$SEED" \
        --vae_slicing --output_dir "$out" \
        $flags "${erase[@]}" >> "$log" 2>&1
}

count_images() { find "$1" -name '*.png' 2>/dev/null | wc -l; }

run_arm() {   # run_arm <concept> <method>: pick a GPU, run, retry on failure until all images exist
    local concept=$1 method=$2
    local out="$OUT/eval_${concept}/${method}"
    local log="$LOGS/gen_${concept}_${method}.log"
    local attempt gpu rc n
    n=$(count_images "$out")
    if (( n >= EXPECTED )); then
        echo "RESULT $concept/$method already complete ($n images), skipping" | tee -a "$log"
        return 0
    fi
    for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
        gpu=$(wait_for_gpu)
        run_arm_once "$gpu" "$concept" "$method" "$out" "$log"; rc=$?
        n=$(count_images "$out")
        echo "RESULT $concept/$method attempt=$attempt gpu=$gpu exit=$rc images=$n/$EXPECTED" | tee -a "$log"
        (( rc == 0 && n >= EXPECTED )) && return 0
        grep -q -E "out of memory|OutOfMemoryError" "$log" && echo "  (OOM on GPU $gpu)" | tee -a "$log"
        sleep "$RETRY_WAIT"
    done
    echo "RESULT $concept/$method FAILED after $MAX_RETRIES attempts" | tee -a "$log"
    return 1
}

run_worker() {   # run_worker <delay> <arm>... ; arm = "concept:method"
    local delay=$1; shift
    local fail=0
    sleep "$delay"
    for arm in "$@"; do
        run_arm "${arm%%:*}" "${arm#*:}" || fail=1
    done
    return $fail
}

if [[ "$STAGE" != "score" ]]; then
    ARMS=()
    for method in "${METHODS[@]}"; do          # method-major so the orig arms come first
        for concept in "${CONCEPTS[@]}"; do
            [[ "$SKIP_ARMS" == *" $concept:$method "* ]] && { echo "skipping $concept:$method (SKIP_ARMS)"; continue; }
            ARMS+=("$concept:$method")
        done
    done
    (( WORKERS > ${#ARMS[@]} )) && WORKERS=${#ARMS[@]}
    declare -a QUEUE
    for ((w = 0; w < WORKERS; w++)); do QUEUE[$w]=""; done
    for ((i = 0; i < ${#ARMS[@]}; i++)); do
        w=$((i % WORKERS)); QUEUE[$w]="${QUEUE[$w]} ${ARMS[$i]}"
    done
    pids=()
    for ((w = 0; w < WORKERS; w++)); do
        echo "worker $w queue:${QUEUE[$w]}"
        # shellcheck disable=SC2086
        run_worker $((w * STAGGER)) ${QUEUE[$w]} &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "GENERATION DONE (fail=$fail)"
    [[ "$STAGE" == "generate" ]] && exit $fail
    [[ $fail -eq 0 ]] || { echo "some arms failed; not scoring. Re-run to resume."; exit 1; }
fi

# --------------------------------------------------------------- Step 3: scores
# produce_scores.py scores every sub-directory of eval_<concept>/: FID of each
# method against orig/, and CS of every directory (orig included) against the
# concept name. One concept per job, round-robin over the GPUs.
score_concept() {
    local concept=$1
    local dir="$OUT/eval_${concept}"
    local log="$LOGS/score_${concept}.log"
    local gpu; gpu=$(wait_for_gpu)
    echo "=========== $(date +%H:%M:%S) scoring $dir on GPU $gpu ===========" | tee "$log"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/diffusion/produce_scores.py \
        --dir "$dir" --concept "$concept" --num_workers 8 --batch_size 50 >> "$log" 2>&1
    local rc=$?
    echo "RESULT score_$concept exit=$rc" | tee -a "$log"
    return $rc
}
score_worker() { local delay=$1; shift; local fail=0; sleep "$delay"; for c in "$@"; do score_concept "$c" || fail=1; done; return $fail; }

declare -a SQUEUE
for ((w = 0; w < WORKERS; w++)); do SQUEUE[$w]=""; done
for ((i = 0; i < ${#CONCEPTS[@]}; i++)); do
    w=$((i % WORKERS)); SQUEUE[$w]="${SQUEUE[$w]} ${CONCEPTS[$i]}"
done
pids=()
for ((w = 0; w < WORKERS; w++)); do
    # shellcheck disable=SC2086
    score_worker $((w * 30)) ${SQUEUE[$w]} &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "SCORING DONE (fail=$fail)"

# ---------------------------------------------------------------- Step 4: table
"$PY" scripts/diffusion/make_table5.py --results_dir "$OUT" | tee "$LOGS/table5.log"
echo "TABLE5 DONE (fail=$fail)"
exit $fail
