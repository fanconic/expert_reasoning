#!/bin/bash
export GPU_NUM="1"
export MODEL="qwen7b"
export DATASET="gsm8k_rebuttals"

# --- Helper Function ---
run_task() {
    local SUFFIX=$1
    local EXTRA=$2
    local FULL_OVERRIDE="wandb.run_name=${MODEL}_${SUFFIX}_discounted_replay ${EXTRA}"
    
    # Check if the task is "full" and append the warmup directory
    # if [ "$SUFFIX" == "full" ]; then
    #     local WARMUP_DIR="/mnt/pdata/caf83/icml_math/warmed_up_rewards/${MODEL}/full/"
    #     FULL_OVERRIDE="${FULL_OVERRIDE} model.warmup_reward_dir=${WARMUP_DIR}"
    # fi
    
    echo "Starting task: ${MODEL}_${SUFFIX}"
    
    # Train
    bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py \
        --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $FULL_OVERRIDE
    
    # Eval
    bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
        --config-path=configs/${DATASET}/${MODEL} --config-name=eval $FULL_OVERRIDE
}

# # --- SFT & GRPO ---
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval

# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_train
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval

# --- IRL Tasks ---
run_task "stepwise"         "model.dense_rewards=partial model.advantage_calculation=discounted_dense model.dense_gamma=0.0 training.beta=0.1 model.reward_lb=-5.0 model.reward_ub=5.0 training.buffer_size=25 model.reward_updates_per_policy_step=3"
# run_task "partial"       "model.dense_rewards=partial"
# run_task "partial_fixed" "model.dense_rewards=partial_fixed"
# run_task "sparse"        "model.dense_rewards=false"