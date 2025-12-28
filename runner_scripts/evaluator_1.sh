#!/bin/bash

export GPU_NUM="1" 

export MODEL="llama8b"

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${MODEL} --config-name=grpo_8B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${MODEL} --config-name=grpo_8B_config_eval