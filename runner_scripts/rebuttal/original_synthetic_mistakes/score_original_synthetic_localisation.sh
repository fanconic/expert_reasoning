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

: "${GPU_NUM:=0}"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

: "${PAIRS_JSONL:=outputs/gsm8k_process_sensitivity_pregen/pair_details.jsonl}"
: "${OUTPUT_ROOT:=outputs/gsm8k_process_sensitivity_pregen/rebuttal_scores}"
: "${LOG_DIR:=${OUTPUT_ROOT}/logs}"
if [[ -z "${POLICY_KEYS+x}" ]]; then
    POLICY_KEYS="qwen7b_base qwen4b_base llama8b_base"
fi
: "${RUN_REWARD:=1}"
: "${WINDOWS:=1 7}"
: "${MAX_EXAMPLES:=0}"
: "${START_INDEX:=0}"
: "${MAX_LENGTH:=1124}"
: "${POLICY_MICRO_BATCH:=4}"
: "${REWARD_MICRO_BATCH:=8}"
: "${ENTROPY_TOKEN_CHUNK_SIZE:=32}"
: "${REWARD_GPU_MEMORY_UTILIZATION:=0.4}"
: "${TARGET_POSITION_SOURCE:=diff}"
: "${SKIP_EXISTING:=1}"
: "${FORCE:=0}"
: "${INCLUDE_TEXT:=0}"
: "${LIVE_LOG:=1}"
: "${LOG_TAIL_LINES:=8}"
: "${QWEN7B_FULL_REWARD_CHECKPOINT:=/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_full_rebuttal_restart/checkpoint-100}"

IFS=' ' read -r -a WINDOW_ARR <<< "${WINDOWS}"
IFS=' ' read -r -a POLICY_KEY_ARR <<< "${POLICY_KEYS}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

FAILED_RUNS=()
LAUNCHED=0
SKIPPED=0

policy_model_path() {
    local model="$1"
    case "${model}" in
        qwen7b_base) echo "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit" ;;
        qwen4b_base) echo "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit" ;;
        llama8b_base) echo "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit" ;;
        *)
            echo "Unknown policy model: ${model}" >&2
            return 1
            ;;
    esac
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
    local policy_name="$1"
    local model_path
    model_path="$(policy_model_path "${policy_name}")" || return
    local output_dir="${OUTPUT_ROOT}/${policy_name}_policy_token_baselines"
    local summary_file="${output_dir}/policy_token_baselines_summary.json"

    echo -e "\nPreparing original-synthetic policy scoring: ${policy_name}"
    echo "  pairs=${PAIRS_JSONL}"
    echo "  model_path=${model_path}"
    echo "  output_dir=${output_dir}"

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

run_qwen7b_full_rebuttal_reward() {
    local output_dir="${OUTPUT_ROOT}/qwen7b_full_rebuttal_reward_localisation"
    local summary_file="${output_dir}/summary.json"

    echo -e "\nPreparing original-synthetic reward scoring: qwen7b_full_rebuttal"
    echo "  pairs=${PAIRS_JSONL}"
    echo "  checkpoint=${QWEN7B_FULL_REWARD_CHECKPOINT}"
    echo "  output_dir=${output_dir}"

    if [[ ! -f "${QWEN7B_FULL_REWARD_CHECKPOINT}/reward_model/adapter_config.json" ]]; then
        FAILED_RUNS+=("qwen7b_full_rebuttal (missing_checkpoint=${QWEN7B_FULL_REWARD_CHECKPOINT})")
        echo "Skipping qwen7b_full_rebuttal: missing reward checkpoint ${QWEN7B_FULL_REWARD_CHECKPOINT}"
        return
    fi
    if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" && -f "${summary_file}" ]]; then
        if grep -q "\"checkpoint_dir\": \"${QWEN7B_FULL_REWARD_CHECKPOINT}\"" "${summary_file}"; then
            echo "Skipping qwen7b_full_rebuttal: existing ${summary_file}"
            ((SKIPPED++))
            return
        fi
        echo "Recomputing qwen7b_full_rebuttal: existing summary used a different checkpoint."
    fi

    local cmd=(
        bash "${RUNNER}" src/eval/localisation_reward_on_pairs.py
        --pairs-jsonl "${PAIRS_JSONL}"
        --output-dir "${output_dir}"
        --config "configs/math/qwen7b/irl_eval.yaml"
        --checkpoint-dir "${QWEN7B_FULL_REWARD_CHECKPOINT}"
        --dense-reward-mode "full"
        --dense-partial-fixed-n 15
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

    run_cmd "qwen7b_full_rebuttal_reward_localisation_gpu${GPU_NUM}" "${cmd[@]}"
    ((LAUNCHED++))
}

if [[ ! -f "${PAIRS_JSONL}" ]]; then
    echo "Missing original synthetic pair details: ${PAIRS_JSONL}"
    exit 1
fi

echo "======================"
echo "GPU ${GPU_NUM} ORIGINAL SYNTHETIC LOCALISATION SCORING"
echo "======================"
echo "Pairs: ${PAIRS_JSONL}"
echo "Outputs: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Policy keys: ${POLICY_KEYS}"
echo "Run reward: ${RUN_REWARD}"
echo "Windows: ${WINDOWS}"
echo "Max examples: ${MAX_EXAMPLES}"
echo "Target position source: ${TARGET_POSITION_SOURCE}"

for policy_key in "${POLICY_KEY_ARR[@]}"; do
    if [[ -z "${policy_key}" ]]; then
        continue
    fi
    run_policy "${policy_key}"
done

if [[ "${RUN_REWARD}" == "1" ]]; then
    run_qwen7b_full_rebuttal_reward
fi

echo -e "\n======================"
echo "GPU ${GPU_NUM} ORIGINAL SYNTHETIC SUMMARY"
echo "======================"
echo "Launched runs: ${LAUNCHED}"
echo "Skipped existing runs: ${SKIPPED}"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All original-synthetic scoring runs succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
