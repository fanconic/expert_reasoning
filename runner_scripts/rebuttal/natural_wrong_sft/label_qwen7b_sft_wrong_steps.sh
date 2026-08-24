#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${MAX_EXAMPLES:=0}"  # 0 => one wrong generation for each prompt with correct+wrong SFT samples
: "${START_INDEX:=0}"
: "${WRONG_PER_PROMPT:=1}"
: "${OUTPUT_FILE:=localisation/natural_wrong_sft/gsm8k_qwen7b_sft_wrong_step_labels_full.jsonl}"
: "${GENERATIONS_JSONL:=/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model/eval_results_math_qwen7b_sft_t0p5.jsonl}"
: "${REFERENCE_PAIR_DETAILS:=localisation/qwen7b_full_localisation_from_qwen7b_sft/pair_details.jsonl}"
: "${MAX_RETRIES:=3}"
: "${NUM_WORKERS:=4}"
: "${RETRY_SLEEP_SECONDS:=1}"
: "${REASONING_EFFORT:=off}"
: "${FORCE:=0}"

cmd=(
    python src/eval/label_natural_wrong_sft_steps.py
    --generations-jsonl "${GENERATIONS_JSONL}"
    --reference-pair-details "${REFERENCE_PAIR_DETAILS}"
    --output-file "${OUTPUT_FILE}"
    --max-examples "${MAX_EXAMPLES}"
    --start-index "${START_INDEX}"
    --wrong-per-prompt "${WRONG_PER_PROMPT}"
    --max-retries "${MAX_RETRIES}"
    --num-workers "${NUM_WORKERS}"
    --retry-sleep-seconds "${RETRY_SLEEP_SECONDS}"
    --reasoning-effort "${REASONING_EFFORT}"
)

if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
fi

echo "[wrong-sft-full] generations=${GENERATIONS_JSONL}"
echo "[wrong-sft-full] reference=${REFERENCE_PAIR_DETAILS}"
echo "[wrong-sft-full] output=${OUTPUT_FILE}"
echo "[wrong-sft-full] max_examples=${MAX_EXAMPLES} wrong_per_prompt=${WRONG_PER_PROMPT} workers=${NUM_WORKERS}"
"${cmd[@]}"
