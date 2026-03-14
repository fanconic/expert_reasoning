#!/bin/bash
export GPU_NUM="0"
export MODEL="qwen3b"
export DATASET="gsm8k_rebuttals"

# --- Helper Function ---
run_task() {
    local SUFFIX=$1
    local EXTRA_COMMON=$2
    local EXTRA_TRAIN=$3
    
    local BASE_NAME="${MODEL}_${SUFFIX}_new"
    local FULL_OVERRIDE="wandb.run_name=${BASE_NAME} ${EXTRA_COMMON}"
    
    # 1. Warmup Logic: Load for everything EXCEPT "partial"
    if [ "$SUFFIX" != "partial" ]; then
        local WARMUP_DIR="/mnt/pdata/caf83/icml_math/warmed_up_rewards/${MODEL}/${SUFFIX}/"
        FULL_OVERRIDE="${FULL_OVERRIDE} model.warmup_reward_dir=${WARMUP_DIR}"
    fi
    
    echo "-----------------------------------"
    echo "Starting task: ${MODEL}_${SUFFIX}"
    echo "-----------------------------------"
    
    # 2. Train: Includes EXTRA_TRAIN (updates, beta, buffer)
    echo "Running Training..."
    bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py \
        --config-path=configs/${DATASET}/${MODEL} \
        --config-name=good_run \
        $FULL_OVERRIDE $EXTRA_TRAIN
    
    # 3. Eval: Only uses FULL_OVERRIDE (common configs)
    echo "Running Evaluation..."
    bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
        --config-path=configs/${DATASET}/${MODEL} \
        --config-name=eval \
        $FULL_OVERRIDE
}

# --- Shared Training-Only Hyperparameters ---
TRAIN_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.buffer_size=50 training.max_steps=400"

# --- IRL Tasks ---
# Arguments: 1. Suffix | 2. Common/Model Configs | 3. Training-only Configs

run_task "partial"       "model.dense_rewards=partial model.reward_lb=-5.0 model.reward_ub=5.0"       "$TRAIN_PARAMS"
run_task "full"          "model.dense_rewards=full model.reward_lb=-5.0 model.reward_ub=5.0"          "$TRAIN_PARAMS"
run_task "partial_fixed" "model.dense_rewards=partial_fixed model.reward_lb=-5.0 model.reward_ub=5.0" "$TRAIN_PARAMS"
run_task "sparse"        "model.dense_rewards=false model.reward_lb=-5.0 model.reward_ub=5.0"         "$TRAIN_PARAMS"