#!/bin/bash

export MODEL="minstral8b"
export GPU_NUM="0" 

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/medreason/${MODEL} --config-name=3B_1B_sft_long_config_irl_train_prime
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/medreason/${MODEL} --config-name=3B_1B_sft_long_config_eval_prime