#!/usr/bin/env bash
# Qwen3-4B GSM8K sparse fixed-critic AIRL/RLHF rebuttal run.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_NUM="${GPU_NUM:-1}"
export DATASET="math"
export MODEL="qwen4b"
export RUN_NAME="${RUN_NAME:-qwen4b_rlhf_sparse_fixed_critic_math}"
export DEFAULT_WARMUP_REWARD_DIR="${DEFAULT_WARMUP_REWARD_DIR:-/mnt/pdata/caf83/icml_math/outputs/qwen4b_sparse/reward_model_warmup}"

bash "${SCRIPT_DIR}/run_sparse_fixed_critic_airl.sh"
