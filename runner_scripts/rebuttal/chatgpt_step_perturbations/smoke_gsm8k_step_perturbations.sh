#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${MAX_EXAMPLES:=3}"
: "${START_INDEX:=0}"
: "${OUTPUT_FILE:=localisation/chatgpt_step_perturbations/gsm8k_qwen7b_sft_step_perturbations_smoke.jsonl}"
: "${SOURCE_PAIR_DETAILS:=localisation/qwen7b_full_localisation_from_qwen7b_sft/pair_details.jsonl}"
: "${REASONING_EFFORT:=off}"
: "${FORCE:=0}"

cmd=(
    python src/eval/build_chatgpt_step_perturbation_dataset.py
    --source-pair-details "${SOURCE_PAIR_DETAILS}"
    --output-file "${OUTPUT_FILE}"
    --max-examples "${MAX_EXAMPLES}"
    --start-index "${START_INDEX}"
    --reasoning-effort "${REASONING_EFFORT}"
)

if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
fi

"${cmd[@]}"
