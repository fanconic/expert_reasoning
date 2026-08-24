#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_NUM="${GPU_NUM:-2}"
export SEED="${SEED:-42}"
export DATASET="math"
export MODEL="llama8b"
export RUN_NAME="${RUN_NAME:-llama8b_gad_${DATASET}}"
export GAD_POLICY_GPU_MEMORY_UTILIZATION="${GAD_POLICY_GPU_MEMORY_UTILIZATION:-0.25}"
export GAD_EVAL_ONLY="${GAD_EVAL_ONLY:-1}"

if [[ "${GAD_EVAL_ONLY}" == "1" ]]; then
    export GAD_EVAL_MODEL_DIR="${GAD_EVAL_MODEL_DIR:-/mnt/pdata/caf83/neurips2026/math/outputs/llama8b_gad_math/checkpoint-300}"
    export GAD_EVAL_TRACE_FILE="${GAD_EVAL_TRACE_FILE:-${GAD_EVAL_MODEL_DIR}/eval_results_math_llama8b_gad_t0p5.jsonl}"
fi

bash "${SCRIPT_DIR}/run_qwen7b_gad.sh"
