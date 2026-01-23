#!/bin/bash
export GPU_NUM="1"
export DATASET="gsm8k_rebuttals"

export OVERRIDE="wandb.run_name=qwen3b_ovr model.use_outcome_rewards=true"
export MODEL="qwen3b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE

export OVERRIDE="wandb.run_name=llama3b_ovr model.use_outcome_rewards=true"
export MODEL="llama3b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE

export OVERRIDE="wandb.run_name=qwen7b_ovr model.use_outcome_rewards=true"
export MODEL="qwen7b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE

export OVERRIDE="wandb.run_name=llama8b_ovr model.use_outcome_rewards=true"
export MODEL="llama8b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE