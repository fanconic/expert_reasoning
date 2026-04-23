#!/usr/bin/env bash
set -u

export GPU_NUM="2"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

VARIANTS=("grpo" "sft" "sparse" "partial" "partial_fixed" "full")
ASSIGNED_PAIRS=(
    "math llama3b"
    "math qwen3b"
    "medicine qwen4b"
    "mmlu llama8b"
)

REWARD_FLAGS=("model.reward_lb=-5.0" "model.reward_ub=5.0" "model.clip_reward_model=true")
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

    local CONFIG_NAME=""
    local DENSE_ARG=""
    case "$VARIANT" in
        sft) CONFIG_NAME="sft_eval" ;;
        grpo) CONFIG_NAME="grpo_eval" ;;
        sparse)
            CONFIG_NAME="irl_eval"
            DENSE_ARG="model.dense_rewards=false"
            ;;
        partial | partial_fixed | full)
            CONFIG_NAME="irl_eval"
            DENSE_ARG="model.dense_rewards=${VARIANT}"
            ;;
        *)
            echo "Unknown variant: ${VARIANT}"
            FAILED_RUNS+=("${DATASET}_${MODEL}_${VARIANT} (unknown_variant)")
            return
            ;;
    esac

    local WNAME="${MODEL}_${VARIANT}"
    local OUTFILE="\${model.name}/eval_results_${DATASET}_${MODEL}_${VARIANT}_t0p5.jsonl"
    local LABEL="${DATASET}_${MODEL}_${VARIANT}"

    local CMD=(
        bash "$RUNNER" evaluate.py
        --config-path="configs/${DATASET}/${MODEL}"
        --config-name="${CONFIG_NAME}"
        "wandb.run_name=${WNAME}"
        "sampling.temperature=0.5"
        "++eval.compute_policy_log_probs=false"
        "++eval.output_file=${OUTFILE}"
    )

    if [[ "${CONFIG_NAME}" == "irl_eval" ]]; then
        CMD+=(
            "${DENSE_ARG}"
            "${REWARD_FLAGS[@]}"
            "++eval.compute_reward_model_scores=true"
        )
    fi

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
    echo "All assigned evaluations succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
