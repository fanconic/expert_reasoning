#!/usr/bin/env bash
# Qwen2.5-7B MedReason sparse fixed-critic AIRL/RLHF rebuttal run.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_NUM="${GPU_NUM:-3}"
export DATASET="medicine"
export MODEL="qwen7b"
export RUN_NAME="${RUN_NAME:-qwen7b_rlhf_sparse_fixed_critic_medicine}"
export DEFAULT_WARMUP_REWARD_DIR="${DEFAULT_WARMUP_REWARD_DIR:-/mnt/pdata/caf83/neurips2026/medicine/outputs/qwen7b_sparse/reward_model_warmup}"

bash "${SCRIPT_DIR}/run_sparse_fixed_critic_airl.sh"
