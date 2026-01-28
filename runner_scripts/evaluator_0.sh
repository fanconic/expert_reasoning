#!/bin/bash
export GPU_NUM="0"

export MODEL="llama8b/discounted_reward"
export DATASET="gsm8k_rebuttals"
#bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval

bash runner_scripts/0_run_gpu_node.sh evaluate.py --config-path=configs/gsm8k_rebuttals/llama8b --config-name=eval