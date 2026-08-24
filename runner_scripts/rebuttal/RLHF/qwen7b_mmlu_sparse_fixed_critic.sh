#!/usr/bin/env bash
# Qwen2.5-7B MMLU-Pro sparse fixed-critic AIRL/RLHF rebuttal run.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_NUM="${GPU_NUM:-2}"
export DATASET="mmlu"
export MODEL="qwen7b"
export RUN_NAME="${RUN_NAME:-qwen7b_rlhf_sparse_fixed_critic_mmlu}"

bash "${SCRIPT_DIR}/run_sparse_fixed_critic_airl.sh"
