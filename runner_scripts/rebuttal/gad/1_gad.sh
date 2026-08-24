#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_NUM="1"
export SEED="${SEED:-42}"
export DATASET="math"
export RUN_NAME="${RUN_NAME:-qwen7b_gad_${DATASET}}"
export GAD_POLICY_GPU_MEMORY_UTILIZATION="0.25"

bash "${SCRIPT_DIR}/run_qwen7b_gad.sh"
