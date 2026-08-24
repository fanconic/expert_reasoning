#!/usr/bin/env bash
# Evaluate the trained GSM8K Qwen2.5-7B OPSD policy on the test split.
set -u
set -o pipefail

GPU_NUM="${GPU_NUM:-0}"
SEED="${SEED:-42}"
MODEL="${MODEL:-qwen7b}"
DATASET="math"
DATASET_NAME="gsm8k_kd"
RUN_NAME="${RUN_NAME:-${MODEL}_opsd_${DATASET}}"
RUN_EVAL="${RUN_EVAL:-1}"

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

WB_PROJECT="${WB_PROJECT:-neurips_airl_rebuttal_opsd_${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/pdata/caf83/neurips2026/${DATASET}/outputs/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/best_model}"
EVAL_TRACE_FILE="${EVAL_TRACE_FILE:-${CHECKPOINT_DIR}/eval_results_${DATASET}_${MODEL}_opsd_t0p5.jsonl}"
LOG_FILE="${LOG_FILE:-${CHECKPOINT_DIR}/eval_results_${DATASET}_${MODEL}_opsd_t0p5.log}"

SAMPLING_N_SAMPLES="${SAMPLING_N_SAMPLES:-16}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.5}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.95}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-7}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-300}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-824}"
LORA_RANK="${LORA_RANK:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

if [[ ! -f "${CHECKPOINT_DIR}/adapter_config.json" ]]; then
    echo "Missing OPSD checkpoint: ${CHECKPOINT_DIR}/adapter_config.json"
    echo "Set CHECKPOINT_DIR=/path/to/best_model if the run name differs."
    exit 1
fi

mkdir -p "$(dirname "${EVAL_TRACE_FILE}")" "$(dirname "${LOG_FILE}")"

CMD=(
    bash "${RUNNER}" evaluate.py
    --config-path="configs/${DATASET}/${MODEL}"
    --config-name="grpo_eval"
    seed="${SEED}"
    wandb.run_name="${RUN_NAME}"
    wandb.project="${WB_PROJECT}"
    model.name="${CHECKPOINT_DIR}"
    dataset.name="${DATASET_NAME}"
    dataset.split=test
    sampling.n_samples="${SAMPLING_N_SAMPLES}"
    sampling.temperature="${SAMPLING_TEMPERATURE}"
    sampling.top_p="${SAMPLING_TOP_P}"
    eval.ks=[1,3,5,10]
    eval.per_device_eval_batch_size="${EVAL_BATCH_SIZE}"
    eval.report_to=none
    model.lora_rank="${LORA_RANK}"
    model.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
    model.max_prompt_length="${MAX_PROMPT_LENGTH}"
    model.max_completion_length="${MAX_COMPLETION_LENGTH}"
    ++eval.compute_policy_log_probs=false
    ++eval.compute_reward_model_scores=false
    ++eval.output_file="${EVAL_TRACE_FILE}"
)

echo "Evaluating OPSD GSM8K test pass@1"
echo "  checkpoint: ${CHECKPOINT_DIR}"
echo "  output:     ${EVAL_TRACE_FILE}"
echo "  log:        ${LOG_FILE}"
echo "  samples:    ${SAMPLING_N_SAMPLES}"
echo "  temp/top_p: ${SAMPLING_TEMPERATURE}/${SAMPLING_TOP_P}"
echo "  gpu:        ${GPU_NUM}"
echo "  command:    ${CMD[*]}"

if [[ "${RUN_EVAL}" != "1" ]]; then
    echo "RUN_EVAL=${RUN_EVAL}; command not launched."
    exit 0
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
rc=$?
if [[ ${rc} -ne 0 ]]; then
    echo "Evaluation failed with exit=${rc}. See ${LOG_FILE}"
    exit "${rc}"
fi

echo
echo "Final pass@k metrics from ${LOG_FILE}:"
grep -E "^(pass@|reward_weighted_pass@|success@|oracle@)" "${LOG_FILE}" || true
pass1="$(grep -E "^pass@1:" "${LOG_FILE}" | tail -n 1 | awk '{print $2}')"
if [[ -n "${pass1}" ]]; then
    echo "OPSD GSM8K pass@1: ${pass1}"
fi
