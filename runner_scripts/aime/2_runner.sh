#!/usr/bin/env bash
set -u
export GPU_NUM="2"
export MODEL="qwen3b"
export DATASET="aime" # Change this if running on Med or MMLU

# --- Constants ---
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
STEP_LIMIT="training.max_steps=400"
IRL_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.buffer_size=50"
REWARD_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0"

FAILED_RUNS=()
run_cmd() {
    local label="$1"; shift
    echo -e "\n▶ Starting: $label"
    "$@"
    [[ $? -ne 0 ]] && FAILED_RUNS+=("$label") && echo "  ✗ FAILED" || echo "  ✓ OK"
}

# Helper to run the 3-step evaluation (Standard, AIME24, AIME25)
run_triple_eval() {
    local variant="$1"
    local config_name="$2"
    local wname="${MODEL}_${variant}"
    local extra_flags="${3:-}"

    # 1. Standard Eval
    # run_cmd "${variant}_STD_EVAL" bash "$RUNNER" evaluate.py \
    #     --config-path="configs/${DATASET}/${MODEL}" --config-name="$config_name" \
    #     wandb.run_name="${wname}" $extra_flags

    # 2. AIME 2024
    run_cmd "${variant}_AIME24" bash "$RUNNER" evaluate.py \
        --config-path="configs/aime/${MODEL}" --config-name="$config_name" \
        wandb.run_name="${wname}" dataset.name="aime_2024" $extra_flags

    # 3. AIME 2025
    run_cmd "${variant}_AIME25" bash "$RUNNER" evaluate.py \
        --config-path="configs/aime/${MODEL}" --config-name="$config_name" \
        wandb.run_name="${wname}" dataset.name="aime_2025" $extra_flags
}

# --- Execution ---

# run_cmd "FIXED_TRAIN" bash "$RUNNER" train_irl.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run \
#     wandb.run_name="${MODEL}_partial_fixed" model.dense_rewards=partial_fixed $REWARD_FLAGS $IRL_PARAMS $STEP_LIMIT
run_triple_eval "grpo" "grpo_eval"

#run_cmd "SFT_TRAIN" bash "$RUNNER" train_sft.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train $STEP_LIMIT
# run_triple_eval "sft" "sft_eval"