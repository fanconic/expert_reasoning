#!/usr/bin/env bash
set -u

export GPU_NUM="2"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

# Keep V0 for compatibility with this project's guidance usage.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# Efficiency-first defaults (override at launch)
: "${N_SAMPLES:=10}"
: "${CHUNK_STEP_SIZE:=15}"
: "${CHUNK_CANDIDATES:=2}"

# This vLLM build does not support per-request logits processors,
# so we use chunk guidance only.
METHOD="chunk"
VARIANTS=("sparse")
REWARD_FLAGS=("model.reward_lb=-5.0" "model.reward_ub=5.0" "model.clip_reward_model=true")
ASSIGNED_PAIRS=(
    #"math qwen4b"
    "medicine qwen4b"
    "mmlu qwen4b"
)

FAILED_RUNS=()
LAUNCHED=0

run_cmd() {
    local label="$1"
    shift
    echo -e "\n▶ ${label}\n  $*"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        FAILED_RUNS+=("${label} (exit=${rc})")
        echo "  ✗ FAILED"
    else
        echo "  ✓ OK"
    fi
}

run_eval() {
    local DATASET="$1"
    local MODEL="$2"
    local VARIANT="$3"

    local DENSE_ARG=""
    case "$VARIANT" in
        sparse) DENSE_ARG="model.dense_rewards=false" ;;
        partial | partial_fixed | full) DENSE_ARG="model.dense_rewards=${VARIANT}" ;;
        *)
            echo "Unknown variant: ${VARIANT}"
            FAILED_RUNS+=("${DATASET}_${MODEL}_${METHOD}_${VARIANT} (unknown_variant)")
            return
            ;;
    esac

    local WNAME="${MODEL}_${VARIANT}"
    local LABEL="${DATASET}_${MODEL}_${METHOD}_${VARIANT}"
    local OUTFILE="\${model.name}/eval_results_${DATASET}_${MODEL}_${METHOD}_${VARIANT}_guided.jsonl"

    local CMD=(
        bash "$RUNNER" evaluate.py
        --config-path="configs/${DATASET}/${MODEL}"
        --config-name="irl_eval"
        "wandb.run_name=${WNAME}"
        "sampling.n_samples=${N_SAMPLES}"
        "${DENSE_ARG}"
        "${REWARD_FLAGS[@]}"
        "sampling.temperature=0.5"
        "++guidance.method=${METHOD}"
        "++guidance.step_size=${CHUNK_STEP_SIZE}"
        "++guidance.n_candidates=${CHUNK_CANDIDATES}"
        "++eval.compute_policy_log_probs=false"
        "++eval.compute_reward_model_scores=true"
        "++eval.output_file=${OUTFILE}"
        "++eval.mode=generate"
    )

    run_cmd "${LABEL}" "${CMD[@]}"
}

for PAIR in "${ASSIGNED_PAIRS[@]}"; do
    read -r DATASET MODEL <<< "$PAIR"
    for VARIANT in "${VARIANTS[@]}"; do
        run_eval "$DATASET" "$MODEL" "$VARIANT"
        ((LAUNCHED++))
    done
done

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
echo "Launched evaluations: ${LAUNCHED}"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All assigned guided evaluations succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
