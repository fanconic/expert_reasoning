#!/usr/bin/env bash
# Rebuttal restart: GSM8K Qwen2.5-7B full dense AIRL.
set -u

GPU_NUM="${GPU_NUM:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

if [[ ! -f "${RUNNER}" ]]; then
    echo "Missing GPU runner: ${RUNNER}"
    exit 1
fi

DATASET="math"
MODEL="qwen7b"
VARIANT="full"
DENSE_VAL="full"
RUN_NAME="${MODEL}_${VARIANT}_rebuttal_restart"
WB_PROJECT="neurips_airl_rebuttal_${DATASET}"
OUTPUT_DIR="/mnt/pdata/caf83/neurips2026/${DATASET}/outputs/${RUN_NAME}"
EVAL_TRACE_FILE="${OUTPUT_DIR}/best_model/eval_results_${DATASET}_${MODEL}_${VARIANT}_t0p5.jsonl"

# Retake overrides.
IRL_TRAIN_PARAMS=(
    model.reward_updates_per_policy_step=3
    training.beta=0.1
    training.buffer_size=50
    training.max_steps=400
    eval.eval_steps=100
    model.policy_learning_rate=5e-6
    model.reward_learning_rate=1e-5
    training.gradient_accumulation_steps=8
    model.max_prompt_length=300
    model.max_completion_length=824
    training.reward_warmup_steps=250
)
COMMON_REWARD_FLAGS=(
    model.clip_reward_model=true
    model.reward_lb=-5.0
    model.reward_ub=5.0
)
EVAL_FLAGS=(
    sampling.temperature=0.5
    model.max_prompt_length=300
    model.max_completion_length=824
)
IRL_LORA_FLAGS=(
    model.lora_rank=256
    model.policy_lora_rank=256
    model.reward_lora_rank=256
)

FAILED_RUNS=()

run_cmd() {
    local label="$1"
    shift
    echo -e "\n[RUN] ${label}\n  $*"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        FAILED_RUNS+=("${label} (exit=${rc})")
        echo "  FAILED"
    else
        echo "  OK"
    fi
}

run_qwen7b_gsm8k_dense() {
    run_cmd "${RUN_NAME}_TRAIN" \
        bash "${RUNNER}" train_irl.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="irl_train" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            training.output_dir="${OUTPUT_DIR}" \
            model.dense_rewards="${DENSE_VAL}" \
            "${IRL_LORA_FLAGS[@]}" \
            "${COMMON_REWARD_FLAGS[@]}" \
            "${IRL_TRAIN_PARAMS[@]}"

    run_cmd "${RUN_NAME}_EVAL" \
        bash "${RUNNER}" evaluate.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="irl_eval" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            model.name="${OUTPUT_DIR}/best_model" \
            model.dense_rewards="${DENSE_VAL}" \
            "${IRL_LORA_FLAGS[@]}" \
            "${COMMON_REWARD_FLAGS[@]}" \
            "${EVAL_FLAGS[@]}" \
            ++eval.output_file="${EVAL_TRACE_FILE}"
}

run_qwen7b_gsm8k_dense

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
    echo -e "FAILURES: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
echo "GSM8K Qwen2.5-7B dense restart succeeded on GPU ${GPU_NUM}."
