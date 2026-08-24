#!/usr/bin/env bash
# Qwen2.5-7B GSM8K sparse fixed-critic AIRL/RLHF rebuttal run.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_NUM="${GPU_NUM:-3}"
export DATASET="math"
export MODEL="qwen7b"
export RUN_NAME="${RUN_NAME:-qwen7b_rlhf_sparse_fixed_critic_math}"
export DEFAULT_WARMUP_REWARD_DIR="${DEFAULT_WARMUP_REWARD_DIR:-/mnt/pdata/caf83/icml_math/warmed_up_rewards/qwen7b/sparse}"

bash "${SCRIPT_DIR}/run_sparse_fixed_critic_airl.sh"
