#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_NUM="3"
export SEED="${SEED:-42}"
export DATASET="medicine"
export RUN_NAME="${RUN_NAME:-qwen7b_gad_${DATASET}}"
export WARMUP_REWARD_DIR="${WARMUP_REWARD_DIR:-/mnt/pdata/caf83/neurips2026/medicine/outputs/qwen7b_gad_medicine/reward_model_warmup}"
export GAD_POLICY_GPU_MEMORY_UTILIZATION="0.25"

bash "${SCRIPT_DIR}/run_qwen7b_gad.sh"
