#!/bin/bash
export GPU_NUM="0"
export MODEL="qwen3b"
export DATASET="mmlu_rebuttals"

# --- Helper Function ---
# Usage: run_task <run_name_suffix> <extra_overrides>
run_task() {
    local SUFFIX=$1
    local EXTRA=$2
    local FULL_OVERRIDE="wandb.run_name=${MODEL}_${SUFFIX} ${EXTRA}"
    
    echo "Starting task: ${MODEL}_${SUFFIX}"
    
    # Train
    bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py \
        --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $FULL_OVERRIDE
    
    # Eval
    bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
        --config-path=configs/${DATASET}/${MODEL} --config-name=eval $FULL_OVERRIDE
}

# --- SFT & GRPO (Unique scripts/configs) ---
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval

# --- IRL Tasks (Consolidated) ---
run_task "full"          "model.dense_rewards=full"
run_task "partial"       "model.dense_rewards=partial"
run_task "partial_fixed" "model.dense_rewards=partial_fixed"
run_task "sparse"        "model.dense_rewards=false"