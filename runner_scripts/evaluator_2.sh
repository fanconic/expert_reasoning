#!/bin/bash

export MODEL="ministral8b"
export GPU_NUM="2" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/medreason/${MODEL} --config-name=sft_8B_config_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/medreason/${MODEL} --config-name=sft_8B_config_eval