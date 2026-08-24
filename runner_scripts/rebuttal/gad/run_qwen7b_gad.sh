#!/usr/bin/env bash
# Shared launcher for GAD rebuttal baselines.
set -u

GPU_NUM="${GPU_NUM:-1}"
SEED="${SEED:-42}"
DATASET="${DATASET:-math}"
MODEL="${MODEL:-qwen7b}"
RUN_NAME="${RUN_NAME:-${MODEL}_gad_${DATASET}}"

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

WB_PROJECT="neurips_airl_rebuttal_gad_${DATASET}"
OUTPUT_DIR="/mnt/pdata/caf83/neurips2026/${DATASET}/outputs/${RUN_NAME}"
WARMUP_REWARD_DIR="${WARMUP_REWARD_DIR:-${OUTPUT_DIR}/reward_model_warmup}"
GAD_RESUME_FROM_CHECKPOINT="${GAD_RESUME_FROM_CHECKPOINT:-}"
if [[ -z "${GAD_RESUME_FROM_CHECKPOINT}" && "${GAD_AUTO_RESUME:-0}" == "1" ]]; then
    GAD_RESUME_FROM_CHECKPOINT="$(
        find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
            | sort -V \
            | tail -n 1
    )"
fi

WARMUP_FLAGS=()
RESUME_FLAGS=()
prepare_policy_resume_checkpoint() {
    local source_checkpoint="$1"
    local prepared_checkpoint="${OUTPUT_DIR}/resume_policy_$(basename "${source_checkpoint}")"

    mkdir -p "${prepared_checkpoint}"
    find "${prepared_checkpoint}" -maxdepth 1 -type l -delete

    local item
    for item in "${source_checkpoint}"/*; do
        if [[ -f "${item}" ]]; then
            ln -sfn "${item}" "${prepared_checkpoint}/$(basename "${item}")"
        fi
    done

    echo "${prepared_checkpoint}"
}

if [[ -n "${GAD_RESUME_FROM_CHECKPOINT}" ]]; then
    if [[ ! -d "${GAD_RESUME_FROM_CHECKPOINT}" ]]; then
        echo "Requested resume checkpoint does not exist: ${GAD_RESUME_FROM_CHECKPOINT}"
        exit 1
    fi
    TRAIN_RESUME_CHECKPOINT="${GAD_RESUME_FROM_CHECKPOINT}"
    echo "Resuming trainer from checkpoint: ${GAD_RESUME_FROM_CHECKPOINT}"
    if [[ -d "${GAD_RESUME_FROM_CHECKPOINT}/reward_model" ]]; then
        echo "Using reward model from resume checkpoint: ${GAD_RESUME_FROM_CHECKPOINT}/reward_model"
        TRAIN_RESUME_CHECKPOINT="$(prepare_policy_resume_checkpoint "${GAD_RESUME_FROM_CHECKPOINT}")"
        echo "Using policy-only trainer checkpoint view: ${TRAIN_RESUME_CHECKPOINT}"
        WARMUP_FLAGS=(
            model.warmup_reward_dir="${GAD_RESUME_FROM_CHECKPOINT}/reward_model"
            training.reward_warmup_steps=0
        )
    else
        echo "Resume checkpoint has no reward_model directory; falling back to warmup checkpoint handling."
    fi
    RESUME_FLAGS=(
        ++training.resume_from_checkpoint="${TRAIN_RESUME_CHECKPOINT}"
    )
fi

if [[ ${#WARMUP_FLAGS[@]} -eq 0 && -d "${WARMUP_REWARD_DIR}" ]]; then
    echo "Using reward warmup checkpoint: ${WARMUP_REWARD_DIR}"
    WARMUP_FLAGS=(
        model.warmup_reward_dir="${WARMUP_REWARD_DIR}"
        training.reward_warmup_steps=0
    )
elif [[ ${#WARMUP_FLAGS[@]} -eq 0 ]]; then
    echo "No reward warmup checkpoint found at ${WARMUP_REWARD_DIR}; running warmup from scratch."
    WARMUP_FLAGS=(
        training.reward_warmup_steps=250
    )
fi

WANDB_RESUME_FLAGS=()
if [[ -n "${GAD_WANDB_RUN_ID:-}" ]]; then
    WANDB_RESUME_FLAGS=(
        ++wandb.id="${GAD_WANDB_RUN_ID}"
        ++wandb.resume="${GAD_WANDB_RESUME:-allow}"
    )
fi

GAD_MAX_STEPS="${GAD_MAX_STEPS:-400}"
GAD_EVAL_STEPS="${GAD_EVAL_STEPS:-100}"
GAD_NUM_GENERATIONS="${GAD_NUM_GENERATIONS:-8}"
GAD_MAX_COMPLETION_LENGTH="${GAD_MAX_COMPLETION_LENGTH:-824}"
GAD_REWARD_UPDATES="${GAD_REWARD_UPDATES:-2}"
GAD_REWARD_SCORE_MICRO_BATCH="${GAD_REWARD_SCORE_MICRO_BATCH:-8}"
GAD_POLICY_GPU_MEMORY_UTILIZATION="${GAD_POLICY_GPU_MEMORY_UTILIZATION:-0.3}"
GAD_EVAL_BATCH_SIZE="${GAD_EVAL_BATCH_SIZE:-16}"

GAD_TRAIN_PARAMS=(
    training.max_steps="${GAD_MAX_STEPS}"
    eval.eval_steps="${GAD_EVAL_STEPS}"
    training.per_device_train_batch_size=16
    training.gradient_accumulation_steps=8
    training.max_micro_batch=4
    ++training.reward_score_micro_batch="${GAD_REWARD_SCORE_MICRO_BATCH}"
    eval.per_device_eval_batch_size=16
    eval.eval_accumulation_steps=4
    sampling.num_generations="${GAD_NUM_GENERATIONS}"
    model.reward_updates_per_policy_step="${GAD_REWARD_UPDATES}"
    model.policy_learning_rate=5e-6
    model.reward_learning_rate=1e-5
    model.policy_gpu_memory_utilization="${GAD_POLICY_GPU_MEMORY_UTILIZATION}"
    training.beta=0.1
    training.buffer_size=0
    model.max_prompt_length=300
    model.max_completion_length="${GAD_MAX_COMPLETION_LENGTH}"
)
GAD_FLAGS=(
    model.classifier_loss=gad_pairwise
    model.dense_rewards=false
    model.use_outcome_rewards=false
    model.num_neg_perturbations_per_expert=0
    "model.neg_perturb_fns=[]"
    model.switch_label_if_correct=false
    model.add_expert_to_policy_optim=false
    model.add_expert_to_policy_balanced=false
    model.disc_pairwise_margin=0.0
    ++model.disc_pairwise_negatives_per_prompt="${GAD_PAIRS_PER_PROMPT:-1}"
)
COMMON_REWARD_FLAGS=(
    model.clip_reward_model=true
    model.reward_lb=-5.0
    model.reward_ub=5.0
)
LORA_FLAGS=(
    model.lora_rank=256
    model.policy_lora_rank=256
    model.reward_lora_rank=256
)
EVAL_FLAGS=(
    sampling.temperature=0.5
    model.dense_rewards=false
    model.max_prompt_length=300
    model.max_completion_length="${GAD_MAX_COMPLETION_LENGTH}"
    eval.per_device_eval_batch_size="${GAD_EVAL_BATCH_SIZE}"
    eval.max_micro_batch=4
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

resolve_eval_model_dir() {
    if [[ -n "${GAD_EVAL_MODEL_DIR:-}" ]]; then
        echo "${GAD_EVAL_MODEL_DIR}"
        return 0
    fi

    if [[ -f "${OUTPUT_DIR}/best_model/adapter_config.json" ]]; then
        echo "${OUTPUT_DIR}/best_model"
        return 0
    fi

    find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | sort -V \
        | tail -n 1
}

run_gad_eval() {
    local eval_model_dir
    eval_model_dir="$(resolve_eval_model_dir)"
    if [[ -z "${eval_model_dir}" || ! -d "${eval_model_dir}" ]]; then
        echo "Skipping ${RUN_NAME}_EVAL because no eval model directory was found."
        echo "  checked output dir: ${OUTPUT_DIR}"
        echo "  optionally set GAD_EVAL_MODEL_DIR=/path/to/checkpoint-or-best_model"
        FAILED_RUNS+=("${RUN_NAME}_EVAL_PRECHECK (missing eval model directory)")
        return 1
    fi

    local eval_trace_file
    eval_trace_file="${GAD_EVAL_TRACE_FILE:-${eval_model_dir}/eval_results_${DATASET}_${MODEL}_gad_t0p5.jsonl}"

    POLICY_ADAPTER="${eval_model_dir}/adapter_config.json"
    REWARD_ADAPTER="${eval_model_dir}/reward_model/adapter_config.json"
    if [[ ! -f "${POLICY_ADAPTER}" || ! -f "${REWARD_ADAPTER}" ]]; then
        echo "Skipping ${RUN_NAME}_EVAL because saved model artifacts are incomplete:"
        echo "  eval model dir: ${eval_model_dir}"
        echo "  expected policy adapter: ${POLICY_ADAPTER}"
        echo "  expected reward adapter: ${REWARD_ADAPTER}"
        FAILED_RUNS+=("${RUN_NAME}_EVAL_PRECHECK (missing saved model artifacts)")
        return 1
    fi

    run_cmd "${RUN_NAME}_EVAL" \
        bash "${RUNNER}" evaluate.py \
            --config-path="configs/${DATASET}/${MODEL}" \
            --config-name="irl_eval" \
            seed="${SEED}" \
            wandb.run_name="${RUN_NAME}" \
            wandb.project="${WB_PROJECT}" \
            model.name="${eval_model_dir}" \
            "${COMMON_REWARD_FLAGS[@]}" \
            "${LORA_FLAGS[@]}" \
            "${EVAL_FLAGS[@]}" \
            ++eval.compute_policy_log_probs=false \
            ++eval.compute_reward_model_scores=true \
            ++eval.output_file="${eval_trace_file}"
}

if [[ "${GAD_EVAL_ONLY:-0}" == "1" ]]; then
    echo "GAD_EVAL_ONLY=1; skipping training and evaluating ${RUN_NAME}."
    run_gad_eval
elif ! run_cmd "${RUN_NAME}_TRAIN" \
    bash "${RUNNER}" train_irl.py \
        --config-path="configs/${DATASET}/${MODEL}" \
        --config-name="irl_train" \
        seed="${SEED}" \
        wandb.run_name="${RUN_NAME}" \
        wandb.project="${WB_PROJECT}" \
        "${WANDB_RESUME_FLAGS[@]}" \
        training.output_dir="${OUTPUT_DIR}" \
        "${GAD_FLAGS[@]}" \
        "${COMMON_REWARD_FLAGS[@]}" \
        "${LORA_FLAGS[@]}" \
        "${WARMUP_FLAGS[@]}" \
        "${RESUME_FLAGS[@]}" \
        "${GAD_TRAIN_PARAMS[@]}"; then
    echo "Skipping ${RUN_NAME}_EVAL because training did not finish."
else
    run_gad_eval
fi

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
    echo -e "FAILURES: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
echo "${DATASET} ${MODEL} GAD run succeeded on GPU ${GPU_NUM}."
