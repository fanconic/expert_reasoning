#!/bin/bash

export GPU_NUM="3" 

export MODEL="llama3b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${MODEL} --config-name=sft_3B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=sft_3B_config_eval

export MODEL="llama8b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${MODEL} --config-name=sft_8B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=sft_8B_config_eval


export MODEL="llama3b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${MODEL} --config-name=grpo_3B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=grpo_3B_config_eval

export MODEL="llama8b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${MODEL} --config-name=grpo_8B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=grpo_8B_config_eval