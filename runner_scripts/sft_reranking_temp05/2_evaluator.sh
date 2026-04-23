#!/usr/bin/env bash
set -u

export GPU_NUM="2"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

VARIANTS=("sparse" "partial" "partial_fixed" "full")
ASSIGNED_PAIRS=(
    "math llama3b"
    "math qwen3b"
    "medicine qwen4b"
    "mmlu llama8b"
)

REWARD_FLAGS=("model.reward_lb=-5.0" "model.reward_ub=5.0" "model.clip_reward_model=true")
FAILED_RUNS=()
LAUNCHED=0

dataset_outputs_root() {
    local dataset="$1"
    case "$dataset" in
        math) echo "/mnt/pdata/caf83/icml_math/outputs" ;;
        medicine) echo "/mnt/pdata/caf83/icml_medicine/outputs" ;;
        mmlu) echo "/mnt/pdata/caf83/icml_mmlu/outputs" ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            return 1
            ;;
    esac
}

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
            FAILED_RUNS+=("${DATASET}_${MODEL}_${VARIANT} (unknown_variant)")
            return
            ;;
    esac

    local ROOT_DIR
    ROOT_DIR="$(dataset_outputs_root "${DATASET}")" || {
        FAILED_RUNS+=("${DATASET}_${MODEL}_${VARIANT} (unknown_dataset)")
        return
    }

    local SFT_POLICY_DIR="${ROOT_DIR}/${MODEL}_sft/best_model"
    local TRACE_FILE="${SFT_POLICY_DIR}/eval_results_${DATASET}_${MODEL}_sft_t0p5.jsonl"

    local WNAME="${MODEL}_${VARIANT}"
    local OUTFILE="\${model.name}/eval_results_${DATASET}_${MODEL}_${VARIANT}_on_sft_t0p5.jsonl"
    local LABEL="${DATASET}_${MODEL}_${VARIANT}_on_sft_t0p5"

    local CMD=(
        bash "$RUNNER" evaluate.py
        --config-path="configs/${DATASET}/${MODEL}"
        --config-name="irl_eval"
        "wandb.run_name=${WNAME}"
        "${DENSE_ARG}"
        "${REWARD_FLAGS[@]}"
        "sampling.temperature=0.5"
        "model.policy_name=${SFT_POLICY_DIR}"
        "++eval.mode=pregenerated_policy_and_reward"
        "++eval.pregenerated_jsonl_path=${TRACE_FILE}"
        "++eval.compute_policy_log_probs=false"
        "++eval.compute_reward_model_scores=true"
        "++eval.output_file=${OUTFILE}"
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
    echo "All assigned SFT-trace evaluations succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi

bash runner_scripts/sft_reranking_temp05/2_logprobs.sh