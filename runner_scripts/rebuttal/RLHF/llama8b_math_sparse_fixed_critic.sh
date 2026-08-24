#!/usr/bin/env bash
# Llama-3.1-8B GSM8K sparse fixed-critic AIRL/RLHF rebuttal run.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_NUM="${GPU_NUM:-1}"
export DATASET="math"
export MODEL="llama8b"
export RUN_NAME="${RUN_NAME:-llama8b_rlhf_sparse_fixed_critic_math}"
export DEFAULT_WARMUP_REWARD_DIR="${DEFAULT_WARMUP_REWARD_DIR:-/mnt/pdata/caf83/icml_math/warmed_up_rewards/llama8b/sparse}"

bash "${SCRIPT_DIR}/run_sparse_fixed_critic_airl.sh"
