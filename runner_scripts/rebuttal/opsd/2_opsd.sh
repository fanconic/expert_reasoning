#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_NUM="1"
export SEED="${SEED:-42}"
export DATASET="mmlu"
export OPSD_FAST="${OPSD_FAST:-1}"
export RUN_NAME="${RUN_NAME:-qwen7b_opsd_${DATASET}}"

bash "${SCRIPT_DIR}/run_qwen7b_opsd.sh"
