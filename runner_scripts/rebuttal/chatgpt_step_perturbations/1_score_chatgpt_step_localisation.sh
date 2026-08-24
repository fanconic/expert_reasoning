#!/usr/bin/env bash
set -u

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

export GPU_NUM="1"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

: "${PAIRS_JSONL:=localisation/chatgpt_step_perturbations/gsm8k_qwen7b_sft_step_perturbations_full.jsonl}"
: "${OUTPUT_ROOT:=localisation/chatgpt_step_perturbations/scores}"
: "${LOG_DIR:=${OUTPUT_ROOT}/logs}"
: "${WINDOWS:=1 7}"
: "${MAX_EXAMPLES:=0}"
: "${START_INDEX:=0}"
: "${MAX_LENGTH:=1124}"
: "${POLICY_MICRO_BATCH:=4}"
: "${REWARD_MICRO_BATCH:=8}"
: "${ENTROPY_TOKEN_CHUNK_SIZE:=32}"
: "${PARTIAL_FIXED_N:=15}"
: "${REWARD_GPU_MEMORY_UTILIZATION:=0.4}"
: "${TARGET_POSITION_SOURCE:=step_first_diff}"
: "${SKIP_EXISTING:=1}"
: "${FORCE:=0}"
: "${INCLUDE_TEXT:=0}"
: "${LIVE_LOG:=1}"
: "${LOG_TAIL_LINES:=8}"
: "${QWEN7B_FULL_REWARD_CHECKPOINT:=/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_full_rebuttal_restart/checkpoint-100}"

IFS=' ' read -r -a WINDOW_ARR <<< "${WINDOWS}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

ASSIGNED_TASKS=(
    "policy llama8b"
    "policy llama8b_base"
    "reward qwen7b partial_fixed"
    "reward qwen4b full"
    "reward llama8b partial_fixed"
)

FAILED_RUNS=()
LAUNCHED=0
SKIPPED=0

policy_model_path() {
    local model="$1"
    case "${model}" in
        qwen7b) echo "/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model" ;;
        qwen4b) echo "/mnt/pdata/caf83/icml_math/outputs/qwen4b_sft/best_model" ;;
        llama8b) echo "/mnt/pdata/caf83/icml_math/outputs/llama8b_sft/best_model" ;;
        qwen7b_base) echo "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit" ;;
        qwen4b_base) echo "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit" ;;
        llama8b_base) echo "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit" ;;
        *)
            echo "Unknown policy model: ${model}" >&2
            return 1
            ;;
    esac
}

reward_config_path() {
    local model="$1"
    case "${model}" in
        qwen7b) echo "configs/math/qwen7b/irl_eval.yaml" ;;
        qwen4b) echo "configs/math/qwen4b/irl_eval.yaml" ;;
        llama8b) echo "configs/math/llama8b/irl_eval.yaml" ;;
        *)
            echo "Unknown reward model: ${model}" >&2
            return 1
            ;;
    esac
}

reward_checkpoint_dir() {
    local model="$1"
    local density="$2"
    if [[ "${model}" == "qwen7b" && "${density}" == "full" ]]; then
        echo "${QWEN7B_FULL_REWARD_CHECKPOINT}"
        return
    fi
    echo "/mnt/pdata/caf83/icml_math/outputs/${model}_${density}/best_model"
}

run_cmd() {
    local label="$1"
    shift
    local log_file="${LOG_DIR}/${label}.log"
    echo -e "\n▶ ${label}"
    echo "  GPU: ${GPU_NUM}"
    echo "  log: ${log_file}"
    echo "  cmd: $*"
    local rc
    if [[ "${LIVE_LOG}" == "1" ]]; then
        stdbuf -oL -eL "$@" 2>&1 | tee "${log_file}"
        rc=${PIPESTATUS[0]}
    else
        "$@" > "${log_file}" 2>&1
        rc=$?
    fi
    if [[ ${rc} -ne 0 ]]; then
        FAILED_RUNS+=("${label} (exit=${rc})")
        echo "  ✗ FAILED; last log lines:"
        tail -n 40 "${log_file}" || true
    else
        echo "  ✓ OK; last log lines:"
        tail -n "${LOG_TAIL_LINES}" "${log_file}" || true
    fi
    return "${rc}"
}

run_policy() {
    local model="$1"
    local policy_name="${model}_sft"
    if [[ "${model}" == *_base ]]; then
        policy_name="${model}"
    fi
    local model_path
    model_path="$(policy_model_path "${model}")" || return
    local output_dir="${OUTPUT_ROOT}/${policy_name}_policy_token_baselines"
    local summary_file="${output_dir}/policy_token_baselines_summary.json"

    echo -e "\nPreparing policy-token scoring: model=${model}, target=${TARGET_POSITION_SOURCE}"
    echo "  model_path=${model_path}"
    echo "  output_dir=${output_dir}"

    if [[ "${model_path}" == /* && ! -e "${model_path}" ]]; then
        FAILED_RUNS+=("${policy_name} (missing_model=${model_path})")
        echo "Skipping ${policy_name}: missing policy model ${model_path}"
        return
    fi
    if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" && -f "${summary_file}" ]]; then
        if grep -q '"prob_largest_drop"' "${summary_file}"; then
            echo "Skipping ${policy_name}: existing ${summary_file}"
            ((SKIPPED++))
            return
        fi
        echo "Recomputing ${policy_name}: existing summary lacks token-probability metrics."
    fi

    local cmd=(
        bash "${RUNNER}" src/eval/localisation_policy_token_baselines.py
        --pair-details "${PAIRS_JSONL}"
        --output-dir "${output_dir}"
        --policy-model "${model_path}"
        --max-length "${MAX_LENGTH}"
        --max-examples "${MAX_EXAMPLES}"
        --start-index "${START_INDEX}"
        --micro-batch "${POLICY_MICRO_BATCH}"
        --entropy-token-chunk-size "${ENTROPY_TOKEN_CHUNK_SIZE}"
        --all-severities
        --no-require-table-valid
        --target-position-source "${TARGET_POSITION_SOURCE}"
        --windows "${WINDOW_ARR[@]}"
    )
    if [[ "${INCLUDE_TEXT}" == "1" ]]; then
        cmd+=(--include-text)
    fi

    run_cmd "${policy_name}_policy_token_baselines_gpu${GPU_NUM}" "${cmd[@]}"
    ((LAUNCHED++))
}

run_reward() {
    local model="$1"
    local density="$2"
    local config_path
    config_path="$(reward_config_path "${model}")" || return
    local checkpoint_dir
    checkpoint_dir="$(reward_checkpoint_dir "${model}" "${density}")"
    local output_dir="${OUTPUT_ROOT}/${model}_${density}_reward_localisation"
    local summary_file="${output_dir}/summary.json"

    echo -e "\nPreparing reward scoring: model=${model}, density=${density}, target=${TARGET_POSITION_SOURCE}"
    echo "  config=${config_path}"
    echo "  checkpoint=${checkpoint_dir}"
    echo "  output_dir=${output_dir}"

    if [[ ! -f "${checkpoint_dir}/reward_model/adapter_config.json" ]]; then
        FAILED_RUNS+=("${model}_${density} (missing_checkpoint=${checkpoint_dir})")
        echo "Skipping ${model}_${density}: missing reward checkpoint ${checkpoint_dir}"
        return
    fi
    if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" && -f "${summary_file}" ]]; then
        if [[ "${checkpoint_dir}" == "${QWEN7B_FULL_REWARD_CHECKPOINT}" ]] \
            && ! grep -q "\"checkpoint_dir\": \"${checkpoint_dir}\"" "${summary_file}"; then
            echo "Recomputing ${model}_${density}: existing summary used a different checkpoint."
        else
            echo "Skipping ${model}_${density}: existing ${summary_file}"
            ((SKIPPED++))
            return
        fi
    fi

    local cmd=(
        bash "${RUNNER}" src/eval/localisation_reward_on_pairs.py
        --pairs-jsonl "${PAIRS_JSONL}"
        --output-dir "${output_dir}"
        --config "${config_path}"
        --checkpoint-dir "${checkpoint_dir}"
        --dense-reward-mode "${density}"
        --dense-partial-fixed-n "${PARTIAL_FIXED_N}"
        --reward-gpu-memory-utilization "${REWARD_GPU_MEMORY_UTILIZATION}"
        --max-examples "${MAX_EXAMPLES}"
        --start-index "${START_INDEX}"
        --max-micro-batch "${REWARD_MICRO_BATCH}"
        --target-position-source "${TARGET_POSITION_SOURCE}"
        --windows "${WINDOW_ARR[@]}"
    )
    if [[ "${INCLUDE_TEXT}" == "1" ]]; then
        cmd+=(--include-text)
    fi

    run_cmd "${model}_${density}_reward_localisation_gpu${GPU_NUM}" "${cmd[@]}"
    ((LAUNCHED++))
}

if [[ ! -f "${PAIRS_JSONL}" ]]; then
    echo "Missing pairs JSONL: ${PAIRS_JSONL}"
    exit 1
fi

echo "======================"
echo "GPU ${GPU_NUM} CHATGPT-STEP LOCALISATION SCORING"
echo "======================"
echo "Pairs: ${PAIRS_JSONL}"
echo "Outputs: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Windows: ${WINDOWS}"
echo "Max examples: ${MAX_EXAMPLES}"
echo "Target position source: ${TARGET_POSITION_SOURCE}"
echo "Assigned tasks:"
printf "  %s\n" "${ASSIGNED_TASKS[@]}"

for task in "${ASSIGNED_TASKS[@]}"; do
    read -r kind model density <<< "${task}"
    case "${kind}" in
        policy) run_policy "${model}" ;;
        reward) run_reward "${model}" "${density}" ;;
        *)
            FAILED_RUNS+=("${task} (unknown_task_kind)")
            echo "Unknown task kind: ${task}"
            ;;
    esac
done

echo -e "\n======================"
echo "GPU ${GPU_NUM} SCORING SUMMARY"
echo "======================"
echo "Launched runs: ${LAUNCHED}"
echo "Skipped existing runs: ${SKIPPED}"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All assigned scoring runs succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
