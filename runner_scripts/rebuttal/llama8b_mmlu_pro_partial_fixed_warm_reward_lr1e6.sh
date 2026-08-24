#!/usr/bin/env bash
# Rebuttal restart: MMLU-Pro Llama-3.1-8B partial-fixed AIRL, reusing warmed reward.
set -u

GPU_NUM="${GPU_NUM:-1}"
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
VARIANT="partial_fixed"
DENSE_VAL="partial_fixed"
PARTIAL_FIXED_N="${PARTIAL_FIXED_N:-15}"
RUN_NAME="${MODEL}_${VARIANT}_rebuttal_warm_reward_lr1e6"
WB_PROJECT="neurips_airl_rebuttal_${DATASET}"
OUTPUT_DIR="/mnt/pdata/caf83/neurips2026/${DATASET}/outputs/${RUN_NAME}"
EVAL_TRACE_FILE="${OUTPUT_DIR}/best_model/eval_results_${DATASET}_${MODEL}_${VARIANT}_warm_reward_lr1e6_t0p5.jsonl"
WARMUP_REWARD_DIR="/mnt/pdata/caf83/neurips2026/${DATASET}/outputs/llama8b_partial_fixed_rebuttal_restart/reward_model_warmup"

if [[ ! -f "${WARMUP_REWARD_DIR}/adapter_model.safetensors" ]]; then
    echo "Missing warmed reward adapter: ${WARMUP_REWARD_DIR}/adapter_model.safetensors"
    exit 1
fi

IRL_TRAIN_PARAMS=(
    model.reward_updates_per_policy_step=1
    training.beta=0.1
    training.buffer_size=50
    training.max_steps=400
    eval.eval_steps=100
    model.policy_learning_rate=5e-6
    model.reward_learning_rate=1e-6
    training.gradient_accumulation_steps=8
    model.max_prompt_length=300
    model.max_completion_length=824
    training.reward_warmup_steps=1
    +training.continue_reward_warmup_after_load=true
    model.warmup_reward_dir="${WARMUP_REWARD_DIR}"
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
INTERVAL_FLAGS=(
    model.dense_rewards="${DENSE_VAL}"
    model.dense_partial_fixed_n="${PARTIAL_FIXED_N}"
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
    return "${rc}"
}

run_llama8b_mmlu_pro_partial_fixed_warm_reward_lr1e6() {
    if ! run_cmd "${RUN_NAME}_TRAIN" \
        bash "${RUNNER}" train_irl.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="irl_train" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            training.output_dir="${OUTPUT_DIR}" \
            "${INTERVAL_FLAGS[@]}" \
            "${IRL_LORA_FLAGS[@]}" \
            "${COMMON_REWARD_FLAGS[@]}" \
            "${IRL_TRAIN_PARAMS[@]}"; then
        echo "Skipping ${RUN_NAME}_EVAL because training did not finish."
        return
    fi

    run_cmd "${RUN_NAME}_EVAL" \
        bash "${RUNNER}" evaluate.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="irl_eval" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            model.name="${OUTPUT_DIR}/best_model" \
            "${INTERVAL_FLAGS[@]}" \
            "${IRL_LORA_FLAGS[@]}" \
            "${COMMON_REWARD_FLAGS[@]}" \
            "${EVAL_FLAGS[@]}" \
            ++eval.output_file="${EVAL_TRACE_FILE}"
}

run_llama8b_mmlu_pro_partial_fixed_warm_reward_lr1e6

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
    echo -e "FAILURES: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
echo "MMLU-Pro Llama-3.1-8B partial-fixed warm-reward LR 1e-6 run succeeded on GPU ${GPU_NUM}."
