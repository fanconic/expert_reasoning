#!/usr/bin/env bash
set -u

export GPU_NUM="3"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

VARIANTS=("sparse" "partial" "partial_fixed" "full")
ASSIGNED_PAIRS=(
    #"math qwen7b"
    "medicine llama3b"
    #"medicine qwen3b"
    "medicine qwen7b"
)

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

run_logprobs() {
    local DATASET="$1"
    local MODEL="$2"
    local VARIANT="$3"

    local ROOT_DIR
    ROOT_DIR="$(dataset_outputs_root "${DATASET}")" || {
        FAILED_RUNS+=("${DATASET}_${MODEL}_${VARIANT} (unknown_dataset)")
        return
    }

    local SFT_POLICY_DIR="${ROOT_DIR}/${MODEL}_sft/best_model"
    local TRACE_FILE="${SFT_POLICY_DIR}/eval_results_${DATASET}_${MODEL}_sft_t0p5.jsonl"
    if [[ ! -f "${TRACE_FILE}" ]]; then
        FAILED_RUNS+=("${DATASET}_${MODEL}_${VARIANT} (missing_trace)")
        echo "  ✗ MISSING TRACE: ${TRACE_FILE}"
        return
    fi

    local WNAME="${MODEL}_${VARIANT}"
    local OUTFILE="\${model.name}/eval_results_logprobs_${DATASET}_${MODEL}_${VARIANT}_on_sft_t0p5.jsonl"
    local LABEL="${DATASET}_${MODEL}_${VARIANT}_logprobs_on_sft_t0p5"

    local CMD=(
        bash "$RUNNER" evaluate.py
        --config-path="configs/${DATASET}/${MODEL}"
        --config-name="irl_eval"
        "wandb.run_name=${WNAME}"
        "sampling.temperature=0.5"
        "eval.per_device_eval_batch_size=4"
        "model.policy_name=${SFT_POLICY_DIR}"
        "++eval.mode=pregenerated_policy"
        "++eval.pregenerated_jsonl_path=${TRACE_FILE}"
        "++eval.compute_policy_log_probs=true"
        "++eval.compute_reward_model_scores=false"
        "++eval.output_file=${OUTFILE}"
    )

    run_cmd "${LABEL}" "${CMD[@]}"
}

for PAIR in "${ASSIGNED_PAIRS[@]}"; do
    read -r DATASET MODEL <<< "$PAIR"
    for VARIANT in "${VARIANTS[@]}"; do
        run_logprobs "$DATASET" "$MODEL" "$VARIANT"
        ((LAUNCHED++))
    done
done

echo -e "\n======================\nGPU ${GPU_NUM} LOGPROBS SUMMARY\n======================"
echo "Launched logprobs evaluations: ${LAUNCHED}"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All assigned logprobs evaluations succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi


bash runner_scripts/transferability_temp05/3_runner.sh
