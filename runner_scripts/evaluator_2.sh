#!/bin/bash
export GPU_NUM="2" 

export MODEL="qwen3b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${MODEL} --config-name=sft_3B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=sft_3B_config_eval

export MODEL="qwen7b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${MODEL} --config-name=sft_7B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=sft_7B_config_eval


export MODEL="qwen3b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${MODEL} --config-name=grpo_3B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=grpo_3B_config_eval

export MODEL="qwen7b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${MODEL} --config-name=grpo_7B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=grpo_7B_config_eval