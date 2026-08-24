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

: "${GPU_A:=0}"
: "${GPU_B:=1}"
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
: "${SKIP_EXISTING:=1}"
: "${FORCE:=0}"
: "${INCLUDE_TEXT:=0}"
: "${TARGET_POSITION_SOURCE:=step_first_diff}"
: "${QWEN7B_FULL_REWARD_CHECKPOINT:=/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_full_rebuttal_restart/checkpoint-100}"

IFS=' ' read -r -a WINDOW_ARR <<< "${WINDOWS}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
FAILURE_FILE="${LOG_DIR}/failures.$$.txt"
rm -f "${FAILURE_FILE}"

gpu_runner() {
    local gpu="$1"
    echo "runner_scripts/${gpu}_run_gpu_node.sh"
}

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

record_failure() {
    local label="$1"
    local rc="$2"
    printf "%s (exit=%s)\n" "${label}" "${rc}" >> "${FAILURE_FILE}"
}

run_logged() {
    local label="$1"
    shift
    local log_file="${LOG_DIR}/${label}.log"
    echo -e "\n▶ ${label}"
    echo "  log: ${log_file}"
    echo "  cmd: $*"
    "$@" > "${log_file}" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "  ✗ FAILED (${rc}); tail follows:"
        tail -n 40 "${log_file}" || true
        record_failure "${label}" "${rc}"
        return "${rc}"
    fi
    echo "  ✓ OK"
    tail -n 8 "${log_file}" || true
    return 0
}

run_policy() {
    local gpu="$1"
    local model="$2"
    local policy_name="${model}_sft"
    if [[ "${model}" == *_base ]]; then
        policy_name="${model}"
    fi
    local model_path
    model_path="$(policy_model_path "${model}")" || return 1
    local output_dir="${OUTPUT_ROOT}/${policy_name}_policy_token_baselines"
    local summary_file="${output_dir}/policy_token_baselines_summary.json"

    if [[ "${model_path}" == /* && ! -e "${model_path}" ]]; then
        echo "Missing policy model: ${model_path}"
        record_failure "${policy_name}_policy" "missing_model"
        return 1
    fi
    if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" && -f "${summary_file}" ]]; then
        if grep -q '"prob_largest_drop"' "${summary_file}"; then
            echo "Skipping ${policy_name}: existing ${summary_file}"
            return 0
        fi
        echo "Recomputing ${policy_name}: existing summary lacks token-probability metrics."
    fi

    local runner
    runner="$(gpu_runner "${gpu}")"
    local cmd=(
        bash "${runner}" src/eval/localisation_policy_token_baselines.py
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
    run_logged "${policy_name}_policy_token_baselines_gpu${gpu}" "${cmd[@]}"
}

run_reward() {
    local gpu="$1"
    local model="$2"
    local density="$3"
    local config_path
    config_path="$(reward_config_path "${model}")" || return 1
    local checkpoint_dir
    checkpoint_dir="$(reward_checkpoint_dir "${model}" "${density}")"
    local output_dir="${OUTPUT_ROOT}/${model}_${density}_reward_localisation"
    local summary_file="${output_dir}/summary.json"

    if [[ ! -f "${checkpoint_dir}/reward_model/adapter_config.json" ]]; then
        echo "Missing reward checkpoint: ${checkpoint_dir}"
        record_failure "${model}_${density}_reward" "missing_checkpoint"
        return 1
    fi
    if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" && -f "${summary_file}" ]]; then
        if [[ "${checkpoint_dir}" == "${QWEN7B_FULL_REWARD_CHECKPOINT}" ]] \
            && ! grep -q "\"checkpoint_dir\": \"${checkpoint_dir}\"" "${summary_file}"; then
            echo "Recomputing ${model}_${density}: existing summary used a different checkpoint."
        else
            echo "Skipping ${model}_${density}: existing ${summary_file}"
            return 0
        fi
    fi

    local runner
    runner="$(gpu_runner "${gpu}")"
    local cmd=(
        bash "${runner}" src/eval/localisation_reward_on_pairs.py
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
    run_logged "${model}_${density}_reward_localisation_gpu${gpu}" "${cmd[@]}"
}

if [[ ! -f "${PAIRS_JSONL}" ]]; then
    echo "Missing pairs JSONL: ${PAIRS_JSONL}"
    exit 1
fi

worker_a() {
    run_policy "${GPU_A}" qwen7b || return 1
    run_policy "${GPU_A}" qwen4b || return 1
    run_policy "${GPU_A}" qwen7b_base || return 1
    run_policy "${GPU_A}" qwen4b_base || return 1
    run_reward "${GPU_A}" qwen7b full || return 1
    run_reward "${GPU_A}" qwen4b partial_fixed || return 1
    run_reward "${GPU_A}" llama8b full || return 1
}

worker_b() {
    run_policy "${GPU_B}" llama8b || return 1
    run_policy "${GPU_B}" llama8b_base || return 1
    run_reward "${GPU_B}" qwen7b partial_fixed || return 1
    run_reward "${GPU_B}" qwen4b full || return 1
    run_reward "${GPU_B}" llama8b partial_fixed || return 1
}

echo "Scoring ChatGPT-step localisation pairs with two GPU queues."
echo "GPU_A=${GPU_A}, GPU_B=${GPU_B}"
echo "Pairs: ${PAIRS_JSONL}"
echo "Outputs: ${OUTPUT_ROOT}"

worker_a &
pid_a=$!
worker_b &
pid_b=$!

wait "${pid_a}"
rc_a=$?
wait "${pid_b}"
rc_b=$?

echo -e "\n======================"
echo "CHATGPT-STEP LOCALISATION SCORING SUMMARY"
echo "======================"
if [[ -f "${FAILURE_FILE}" ]]; then
    echo "Failures:"
    sed 's/^/  /' "${FAILURE_FILE}"
else
    echo "No recorded per-run failures."
fi
echo "GPU_A worker exit: ${rc_a}"
echo "GPU_B worker exit: ${rc_b}"
echo "Logs: ${LOG_DIR}"

if [[ ${rc_a} -ne 0 || ${rc_b} -ne 0 || -f "${FAILURE_FILE}" ]]; then
    exit 1
fi
