#!/usr/bin/env bash
# Shared launcher for sparse fixed-critic AIRL/RLHF rebuttal runs.
set -u

: "${DATASET:?Set DATASET to math, mmlu, or medicine.}"
: "${MODEL:?Set MODEL to qwen7b, qwen4b, or llama8b.}"

GPU_NUM="${GPU_NUM:-1}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-${MODEL}_rlhf_sparse_fixed_critic_${DATASET}}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
STRICT_WARMUP="${STRICT_WARMUP:-0}"

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

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
if [[ ! -f "${RUNNER}" ]]; then
    echo "Missing GPU runner: ${RUNNER}"
    exit 1
fi

case "${DATASET}" in
    math)
        PROMPT_LENGTH="${RLHF_MAX_PROMPT_LENGTH:-300}"
        REWARD_LB="${RLHF_REWARD_LB:--5.0}"
        REWARD_UB="${RLHF_REWARD_UB:-5.0}"
        ;;
    mmlu)
        PROMPT_LENGTH="${RLHF_MAX_PROMPT_LENGTH:-300}"
        REWARD_LB="${RLHF_REWARD_LB:--2.0}"
        REWARD_UB="${RLHF_REWARD_UB:-2.0}"
        ;;
    medicine)
        PROMPT_LENGTH="${RLHF_MAX_PROMPT_LENGTH:-350}"
        REWARD_LB="${RLHF_REWARD_LB:--5.0}"
        REWARD_UB="${RLHF_REWARD_UB:-5.0}"
        ;;
    *)
        echo "Unknown DATASET='${DATASET}'. Expected math, mmlu, or medicine."
        exit 1
        ;;
esac

WB_PROJECT="${WB_PROJECT:-neurips_airl_rebuttal_rlhf_${DATASET}}"
OUTPUT_DIR="/mnt/pdata/caf83/neurips2026/${DATASET}/outputs/${RUN_NAME}"
EVAL_TRACE_FILE="${OUTPUT_DIR}/best_model/eval_results_${DATASET}_${MODEL}_rlhf_sparse_fixed_critic_t0p5.jsonl"

WARMUP_REWARD_DIR="${WARMUP_REWARD_DIR:-${DEFAULT_WARMUP_REWARD_DIR:-}}"
WARMUP_FLAGS=()
if [[ -n "${WARMUP_REWARD_DIR}" && "${WARMUP_REWARD_DIR}" != "none" ]]; then
    if [[ -d "${WARMUP_REWARD_DIR}" ]]; then
        echo "Using sparse reward warmup checkpoint: ${WARMUP_REWARD_DIR}"
        WARMUP_FLAGS=(
            model.warmup_reward_dir="${WARMUP_REWARD_DIR}"
            training.reward_warmup_steps=0
        )
    elif [[ "${STRICT_WARMUP}" == "1" ]]; then
        echo "Requested warmup checkpoint does not exist: ${WARMUP_REWARD_DIR}"
        exit 1
    else
        echo "Warmup checkpoint not found at ${WARMUP_REWARD_DIR}; running sparse warmup from scratch."
        WARMUP_FLAGS=(
            training.reward_warmup_steps="${RLHF_REWARD_WARMUP_STEPS:-250}"
        )
    fi
else
    echo "No sparse warmup checkpoint supplied; running sparse warmup from scratch."
    WARMUP_FLAGS=(
        training.reward_warmup_steps="${RLHF_REWARD_WARMUP_STEPS:-250}"
    )
fi

RLHF_MAX_STEPS="${RLHF_MAX_STEPS:-400}"
RLHF_EVAL_STEPS="${RLHF_EVAL_STEPS:-100}"
RLHF_NUM_GENERATIONS="${RLHF_NUM_GENERATIONS:-8}"
RLHF_MAX_COMPLETION_LENGTH="${RLHF_MAX_COMPLETION_LENGTH:-824}"
RLHF_TRAIN_BATCH_SIZE="${RLHF_TRAIN_BATCH_SIZE:-16}"
RLHF_GRAD_ACCUM="${RLHF_GRAD_ACCUM:-8}"
RLHF_MAX_MICRO_BATCH="${RLHF_MAX_MICRO_BATCH:-4}"
RLHF_REWARD_SCORE_MICRO_BATCH="${RLHF_REWARD_SCORE_MICRO_BATCH:-8}"
RLHF_POLICY_LR="${RLHF_POLICY_LR:-5e-6}"
RLHF_REWARD_LR="${RLHF_REWARD_LR:-1e-5}"
RLHF_BETA="${RLHF_BETA:-0.1}"
RLHF_POLICY_GPU_MEMORY_UTILIZATION="${RLHF_POLICY_GPU_MEMORY_UTILIZATION:-0.4}"
RLHF_REWARD_GPU_MEMORY_UTILIZATION="${RLHF_REWARD_GPU_MEMORY_UTILIZATION:-0.4}"
RLHF_EVAL_BATCH_SIZE="${RLHF_EVAL_BATCH_SIZE:-16}"

TRAIN_PARAMS=(
    training.max_steps="${RLHF_MAX_STEPS}"
    eval.eval_steps="${RLHF_EVAL_STEPS}"
    training.per_device_train_batch_size="${RLHF_TRAIN_BATCH_SIZE}"
    training.gradient_accumulation_steps="${RLHF_GRAD_ACCUM}"
    training.max_micro_batch="${RLHF_MAX_MICRO_BATCH}"
    ++training.reward_score_micro_batch="${RLHF_REWARD_SCORE_MICRO_BATCH}"
    ++training.freeze_reward_after_warmup=true
    training.beta="${RLHF_BETA}"
    training.buffer_size=0
    model.policy_learning_rate="${RLHF_POLICY_LR}"
    model.reward_learning_rate="${RLHF_REWARD_LR}"
    model.policy_gpu_memory_utilization="${RLHF_POLICY_GPU_MEMORY_UTILIZATION}"
    model.reward_gpu_memory_utilization="${RLHF_REWARD_GPU_MEMORY_UTILIZATION}"
    model.max_prompt_length="${PROMPT_LENGTH}"
    model.max_completion_length="${RLHF_MAX_COMPLETION_LENGTH}"
    sampling.num_generations="${RLHF_NUM_GENERATIONS}"
)

RLHF_FLAGS=(
    model.dense_rewards=false
    model.advantage_calculation=grpo
    ++model.use_outcome_rewards=false
    ++model.reward_updates_per_policy_step=0
    ++model.classifier_loss=bce
    model.clip_reward_model=true
    model.reward_lb="${REWARD_LB}"
    model.reward_ub="${REWARD_UB}"
)

LORA_FLAGS=(
    model.lora_rank=256
    model.policy_lora_rank=256
    model.reward_lora_rank=256
)

EVAL_FLAGS=(
    sampling.temperature=0.5
    model.dense_rewards=false
    model.max_prompt_length="${PROMPT_LENGTH}"
    model.max_completion_length="${RLHF_MAX_COMPLETION_LENGTH}"
    eval.per_device_eval_batch_size="${RLHF_EVAL_BATCH_SIZE}"
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

echo "Sparse fixed-critic RLHF settings: dataset=${DATASET} model=${MODEL} steps=${RLHF_MAX_STEPS} warmup=${WARMUP_REWARD_DIR:-fresh} reward_updates=0 freeze_after_warmup=true"

TRAIN_OK=1
if [[ "${RUN_TRAIN}" == "1" ]]; then
    if ! run_cmd "${RUN_NAME}_TRAIN" \
        bash "${RUNNER}" train_irl.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="irl_train" \
            seed="${SEED}" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            training.output_dir="${OUTPUT_DIR}" \
            "${RLHF_FLAGS[@]}" \
            "${LORA_FLAGS[@]}" \
            "${WARMUP_FLAGS[@]}" \
            "${TRAIN_PARAMS[@]}"; then
        TRAIN_OK=0
    fi
fi

if [[ "${RUN_EVAL}" == "1" && "${TRAIN_OK}" == "1" ]]; then
    POLICY_ADAPTER="${OUTPUT_DIR}/best_model/adapter_config.json"
    REWARD_ADAPTER="${OUTPUT_DIR}/best_model/reward_model/adapter_config.json"
    if [[ ! -f "${POLICY_ADAPTER}" || ! -f "${REWARD_ADAPTER}" ]]; then
        echo "Skipping ${RUN_NAME}_EVAL because saved model artifacts are incomplete:"
        echo "  expected policy adapter: ${POLICY_ADAPTER}"
        echo "  expected reward adapter: ${REWARD_ADAPTER}"
        FAILED_RUNS+=("${RUN_NAME}_EVAL_PRECHECK (missing saved model artifacts)")
    else
        run_cmd "${RUN_NAME}_EVAL" \
            bash "${RUNNER}" evaluate.py \
                --config-path="configs/${DATASET}/${MODEL}" \
                --config-name="irl_eval" \
                seed="${SEED}" \
                wandb.run_name="${RUN_NAME}" \
                wandb.project="${WB_PROJECT}" \
                model.name="${OUTPUT_DIR}/best_model" \
                "${RLHF_FLAGS[@]}" \
                "${LORA_FLAGS[@]}" \
                "${EVAL_FLAGS[@]}" \
                ++eval.compute_policy_log_probs=false \
                ++eval.compute_reward_model_scores=true \
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
    echo "${DATASET} ${MODEL} sparse fixed-critic RLHF run succeeded on GPU ${GPU_NUM}."
fi
