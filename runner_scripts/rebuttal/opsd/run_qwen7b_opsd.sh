#!/usr/bin/env bash
# Shared launcher for Qwen2.5-7B OPSD rebuttal baselines.
set -u

GPU_NUM="${GPU_NUM:-1}"
SEED="${SEED:-42}"
DATASET="${DATASET:-math}"
MODEL="${MODEL:-qwen7b}"
OPSD_FAST="${OPSD_FAST:-0}"
RUN_NAME_WAS_SET="${RUN_NAME+x}"
RUN_NAME="${RUN_NAME:-${MODEL}_opsd_${DATASET}}"
if [[ "${OPSD_FAST}" == "1" && -z "${RUN_NAME_WAS_SET}" ]]; then
    RUN_NAME="${MODEL}_opsd_${DATASET}_fast"
fi
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

case "${DATASET}" in
    math) DATASET_NAME="gsm8k_kd" ;;
    mmlu) DATASET_NAME="mmlu_kd" ;;
    medicine) DATASET_NAME="medical_kd" ;;
    *)
        echo "Unknown OPSD dataset '${DATASET}'. Expected math, mmlu, or medicine."
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

WB_PROJECT="neurips_airl_rebuttal_${DATASET}"
OUTPUT_DIR="/mnt/pdata/caf83/neurips2026/${DATASET}/outputs/${RUN_NAME}"
EVAL_TRACE_FILE="${OUTPUT_DIR}/best_model/eval_results_${DATASET}_${MODEL}_opsd_t0p5.jsonl"

if [[ "${OPSD_FAST}" == "1" ]]; then
    OPSD_MAX_STEPS="${OPSD_MAX_STEPS:-400}"
    OPSD_EVAL_STEPS="${OPSD_EVAL_STEPS:-100}"
    OPSD_GRAD_ACCUM="${OPSD_GRAD_ACCUM:-1}"
    OPSD_NUM_GENERATIONS="${OPSD_NUM_GENERATIONS:-8}"
    OPSD_PER_DEVICE_BATCH="${OPSD_PER_DEVICE_BATCH:-16}"
    OPSD_MAX_COMPLETION_LENGTH="${OPSD_MAX_COMPLETION_LENGTH:-824}"
    OPSD_MAX_MICRO_BATCH="${OPSD_MAX_MICRO_BATCH:-4}"
else
    OPSD_MAX_STEPS="${OPSD_MAX_STEPS:-400}"
    OPSD_EVAL_STEPS="${OPSD_EVAL_STEPS:-100}"
    OPSD_GRAD_ACCUM="${OPSD_GRAD_ACCUM:-16}"
    OPSD_NUM_GENERATIONS="${OPSD_NUM_GENERATIONS:-8}"
    OPSD_PER_DEVICE_BATCH="${OPSD_PER_DEVICE_BATCH:-16}"
    OPSD_MAX_COMPLETION_LENGTH="${OPSD_MAX_COMPLETION_LENGTH:-824}"
    OPSD_MAX_MICRO_BATCH="${OPSD_MAX_MICRO_BATCH:-16}"
fi
OPSD_MAX_PROMPT_LENGTH="${OPSD_MAX_PROMPT_LENGTH:-300}"
OPSD_LR="${OPSD_LR:-5e-6}"
OPSD_LORA_RANK="${OPSD_LORA_RANK:-256}"
OPSD_EVAL_MAX_COMPLETION_LENGTH="${OPSD_EVAL_MAX_COMPLETION_LENGTH:-${OPSD_MAX_COMPLETION_LENGTH}}"

OPSD_TRAIN_PARAMS=(
    training.max_steps="${OPSD_MAX_STEPS}"
    eval.eval_steps="${OPSD_EVAL_STEPS}"
    training.gradient_accumulation_steps="${OPSD_GRAD_ACCUM}"
    training.num_generations="${OPSD_NUM_GENERATIONS}"
    training.per_device_train_batch_size="${OPSD_PER_DEVICE_BATCH}"
    training.learning_rate="${OPSD_LR}"
    model.max_prompt_length="${OPSD_MAX_PROMPT_LENGTH}"
    model.max_completion_length="${OPSD_MAX_COMPLETION_LENGTH}"
    model.fast_inference=false
)
OPSD_FLAGS=(
    dataset.name="${DATASET_NAME}"
    model.lora_rank="${OPSD_LORA_RANK}"
    ++opsd.mode=direct
    ++opsd.reward_temperature=1.0
    ++opsd.reward_lb=-5.0
    ++opsd.reward_ub=5.0
    ++opsd.max_micro_batch="${OPSD_MAX_MICRO_BATCH}"
    ++opsd.normalize_weights=false
    ++opsd.log_first_batch=true
)
EVAL_FLAGS=(
    dataset.name="${DATASET_NAME}"
    sampling.temperature=0.5
    model.max_prompt_length="${OPSD_MAX_PROMPT_LENGTH}"
    model.max_completion_length="${OPSD_EVAL_MAX_COMPLETION_LENGTH}"
    model.lora_rank="${OPSD_LORA_RANK}"
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

echo "OPSD settings: fast=${OPSD_FAST} steps=${OPSD_MAX_STEPS} batch=${OPSD_PER_DEVICE_BATCH} grad_accum=${OPSD_GRAD_ACCUM} generations=${OPSD_NUM_GENERATIONS} completion=${OPSD_MAX_COMPLETION_LENGTH} micro_batch=${OPSD_MAX_MICRO_BATCH}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
    run_cmd "${RUN_NAME}_TRAIN" \
        bash "${RUNNER}" train_opsd.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="grpo_train" \
            seed="${SEED}" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            training.output_dir="${OUTPUT_DIR}" \
            "${OPSD_FLAGS[@]}" \
            "${OPSD_TRAIN_PARAMS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
    run_cmd "${RUN_NAME}_EVAL" \
        bash "${RUNNER}" evaluate.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="grpo_eval" \
            seed="${SEED}" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            model.name="${OUTPUT_DIR}/best_model" \
            "${EVAL_FLAGS[@]}" \
            ++eval.compute_policy_log_probs=false \
            ++eval.output_file="${EVAL_TRACE_FILE}"
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
    echo "${DATASET} Qwen2.5-7B OPSD run succeeded on GPU ${GPU_NUM}."
fi
