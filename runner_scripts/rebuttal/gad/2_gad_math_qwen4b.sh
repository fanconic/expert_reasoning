#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_NUM="${GPU_NUM:-2}"
export SEED="${SEED:-42}"
export DATASET="math"
export MODEL="qwen4b"
export RUN_NAME="${RUN_NAME:-qwen4b_gad_${DATASET}}"
export GAD_POLICY_GPU_MEMORY_UTILIZATION="${GAD_POLICY_GPU_MEMORY_UTILIZATION:-0.3}"

bash "${SCRIPT_DIR}/run_qwen7b_gad.sh"
