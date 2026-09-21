#!/usr/bin/env bash
# Analysis for the Van Gogh Kp sweep produced by
# scripts/diffusion/run_vangogh_kp_sweep.sh. Style counterpart of
# scripts/diffusion/analyze_kp_sweep.sh: same two analysis scripts, with the
# prompt groups being forget (Van Gogh, 0-4) / retain (other artists, 5-12)
# instead of unsafe / safe.
# CPU only; reads each arm's pid_records/ and never touches the images.
#
#   analysis/kp<kp>/          compare_step_block_error.py: CASteer (Kp only) vs
#                             our method at that Kp -- per-group and pooled
#                             u/e heatmaps, marginals, per_prompt.csv, summary.md
#   analysis/kp_sweep_*.csv   summarize_kp_sweep.py: every arm stacked into one
#                             table, plus u-vs-step and mean-u-vs-Kp plots
#
# Usage:
#   bash scripts/diffusion/analyze_vangogh_kp_sweep.sh [sweep_dir]

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY=${PYTHON:-/storage/s25017/miniconda3/envs/munba3/bin/python3}
SWEEP=${1:-results/sd14/vangogh_kp_sweep}
ANALYSIS="$SWEEP/analysis"
# Not GROUPS: that is bash's read-only group-id array, and assigning it is
# silently ignored.
PROMPT_GROUPS="forget:0-4,retain:5-12"
KI=0.01; KG=0.01; KD=0.02
read -r -a KPS <<< "${KPS:-2.0 1.5 1.0 0.5}"

fail=0
for kp in "${KPS[@]}"; do
    base="$SWEEP/casteer_kp${kp}/pid_records"
    ours="$SWEEP/adaptive_kg_kp${kp}_ki${KI}_kg${KG}_kd${KD}/pid_records"
    if [ ! -d "$base" ] || [ ! -d "$ours" ]; then
        echo "SKIP kp=$kp: missing $base or $ours"
        fail=1
        continue
    fi
    echo "=========== compare kp=$kp -> $ANALYSIS/kp${kp} ==========="
    PYTHONPATH="$REPO_ROOT" "$PY" scripts/diffusion/compare_step_block_error.py \
        --baseline_dir "$base" --baseline_label "CASteer (Kp=${kp})" \
        --method_dir "$ours" --method_label "Our method (Kp=${kp}, ki=${KI}, kg=${KG}, kd=${KD})" \
        --groups "$PROMPT_GROUPS" \
        --output_dir "$ANALYSIS/kp${kp}" || fail=1
done

echo "=========== cross-Kp summary -> $ANALYSIS ==========="
PYTHONPATH="$REPO_ROOT" "$PY" scripts/diffusion/summarize_kp_sweep.py \
    --sweep_dir "$SWEEP" --groups "$PROMPT_GROUPS" --output_dir "$ANALYSIS" || fail=1

echo "ANALYSIS DONE (fail=$fail)"
exit $fail
