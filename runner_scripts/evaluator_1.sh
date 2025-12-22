#!/bin/bash


export MODEL="qwen3b"
export GPU_NUM="1" 

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs --config-name=config_irl_train