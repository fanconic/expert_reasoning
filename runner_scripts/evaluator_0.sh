#!/bin/bash
export GPU_NUM="0" 

export MODEL="llama3b"
#bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${MODEL} --config-name=grpo_3B_config_train
#bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=grpo_3B_config_eval

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/medreason/${MODEL} --config-name=grpo_3B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/medreason/${MODEL} --config-name=grpo_3B_config_eval