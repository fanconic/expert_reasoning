#!/bin/bash
export GPU_NUM="2"

export MODEL="llama3b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=good_run $OVERRIDE

# export MODEL="qwen7b" 
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=good_run $OVERRIDE
