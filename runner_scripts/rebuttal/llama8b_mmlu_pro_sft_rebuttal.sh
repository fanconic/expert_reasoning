#!/usr/bin/env bash
# Rebuttal: MMLU-Pro Llama-3.1-8B SFT training and eval.
set -u

GPU_NUM="${GPU_NUM:-1}"
SEED="${SEED:-42}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${EXPERT_REASONING_UV_BOOTSTRAPPED:-0}" != "1" \
    && -z "${VIRTUAL_ENV:-}" \
    && ( -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == "base" ) \
    && -d "${REPO_ROOT}/.venv" ]] \
    && command -v uv >/dev/null 2>&1; then
    echo "No project Conda/virtualenv detected; re-running under uv with the repo .venv."
    export EXPERT_REASONING_UV_BOOTSTRAPPED=1
    exec uv run bash "${SCRIPT_PATH}" "$@"
fi

RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
if [[ ! -f "${RUNNER}" ]]; then
    echo "Missing GPU runner: ${RUNNER}"
    exit 1
fi

DATASET="mmlu"
MODEL="llama8b"
RUN_NAME="${RUN_NAME:-llama8b_sft_rebuttal}"
WB_PROJECT="${WB_PROJECT:-neurips_airl_rebuttal_${DATASET}}"
OUTPUT_DIR="/mnt/pdata/caf83/neurips2026/${DATASET}/outputs/${RUN_NAME}"
EVAL_TRACE_FILE="${OUTPUT_DIR}/best_model/eval_results_${DATASET}_${MODEL}_sft_rebuttal_t0p5.jsonl"

SFT_TRAIN_PARAMS=(
    training.max_steps="${SFT_MAX_STEPS:-500}"
    eval.eval_steps="${SFT_EVAL_STEPS:-100}"
    training.learning_rate="${SFT_LR:-1e-4}"
    training.per_device_train_batch_size="${SFT_TRAIN_BATCH_SIZE:-6}"
    training.gradient_accumulation_steps="${SFT_GRAD_ACCUM:-6}"
    model.max_prompt_length="${SFT_MAX_PROMPT_LENGTH:-300}"
    model.max_completion_length="${SFT_MAX_COMPLETION_LENGTH:-824}"
    model.lora_rank="${SFT_LORA_RANK:-256}"
    model.gpu_memory_utilization="${SFT_GPU_MEMORY_UTILIZATION:-0.6}"
)
EVAL_FLAGS=(
    sampling.temperature="${SFT_EVAL_TEMPERATURE:-0.5}"
    model.max_prompt_length="${SFT_MAX_PROMPT_LENGTH:-300}"
    model.max_completion_length="${SFT_MAX_COMPLETION_LENGTH:-824}"
    model.lora_rank="${SFT_LORA_RANK:-256}"
    model.gpu_memory_utilization="${SFT_GPU_MEMORY_UTILIZATION:-0.6}"
    eval.per_device_eval_batch_size="${SFT_EVAL_BATCH_SIZE:-7}"
)

FAILED_RUNS=()

run_cmd() {
    local label="$1"
    shift
    echo -e "\n[RUN] ${label}\n  $*"
    "$@"
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        FAILED_RUNS+=("${label} (exit=${rc})")
        echo "  FAILED"
    else
        echo "  OK"
    fi
    return "${rc}"
}

TRAIN_OK=1
if [[ "${RUN_TRAIN}" == "1" ]]; then
    if ! run_cmd "${RUN_NAME}_TRAIN" \
        bash "${RUNNER}" train_sft.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="sft_train" \
            seed="${SEED}" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            training.output_dir="${OUTPUT_DIR}" \
            "${SFT_TRAIN_PARAMS[@]}"; then
        TRAIN_OK=0
    fi
fi

if [[ "${RUN_EVAL}" == "1" && "${TRAIN_OK}" == "1" ]]; then
    POLICY_ADAPTER="${OUTPUT_DIR}/best_model/adapter_config.json"
    if [[ ! -f "${POLICY_ADAPTER}" ]]; then
        echo "Skipping ${RUN_NAME}_EVAL because saved model artifacts are incomplete:"
        echo "  expected policy adapter: ${POLICY_ADAPTER}"
        FAILED_RUNS+=("${RUN_NAME}_EVAL_PRECHECK (missing saved model artifacts)")
    else
        run_cmd "${RUN_NAME}_EVAL" \
            bash "${RUNNER}" evaluate.py \
                --config-path="configs/${DATASET}/${MODEL}" \
                --config-name="sft_eval" \
                seed="${SEED}" \
                wandb.run_name="${RUN_NAME}" \
                wandb.project="${WB_PROJECT}" \
                model.name="${OUTPUT_DIR}/best_model" \
                "${EVAL_FLAGS[@]}" \
                ++eval.output_file="${EVAL_TRACE_FILE}"
    fi
elif [[ "${RUN_EVAL}" == "1" ]]; then
    echo "Skipping ${RUN_NAME}_EVAL because training did not finish."
fi

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
    echo -e "FAILURES: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi

if [[ "${RUN_TRAIN}" != "1" && "${RUN_EVAL}" != "1" ]]; then
    echo "No train/eval commands were requested."
else
    echo "MMLU-Pro Llama-3.1-8B SFT rebuttal run succeeded on GPU ${GPU_NUM}."
fi
