#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${MAX_EXAMPLES:=0}"  # 0 => all Table-5-valid prompts (currently 1234)
: "${START_INDEX:=0}"
: "${OUTPUT_FILE:=localisation/chatgpt_step_perturbations/gsm8k_qwen7b_sft_step_perturbations_full.jsonl}"
: "${SOURCE_PAIR_DETAILS:=localisation/qwen7b_full_localisation_from_qwen7b_sft/pair_details.jsonl}"
: "${MAX_RETRIES:=3}"
: "${NUM_WORKERS:=4}"
: "${SLEEP_SECONDS:=0}"
: "${RETRY_SLEEP_SECONDS:=1}"
: "${REASONING_EFFORT:=off}"
: "${FORCE:=0}"

cmd=(
    python src/eval/build_chatgpt_step_perturbation_dataset.py
    --source-pair-details "${SOURCE_PAIR_DETAILS}"
    --output-file "${OUTPUT_FILE}"
    --max-examples "${MAX_EXAMPLES}"
    --start-index "${START_INDEX}"
    --max-retries "${MAX_RETRIES}"
    --num-workers "${NUM_WORKERS}"
    --sleep-seconds "${SLEEP_SECONDS}"
    --retry-sleep-seconds "${RETRY_SLEEP_SECONDS}"
    --reasoning-effort "${REASONING_EFFORT}"
)

if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
fi

"${cmd[@]}"
