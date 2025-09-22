#!/bin/bash
bash runner_scripts/0_run_gpu_node.sh sft_train.py --config-path=configs/llama3 --config-name=sft_3B_config_train
bash runner_scripts/0_run_gpu_node.sh evaluate.py --config-path=configs/llama3 --config-name=sft_3B_config_eval

bash runner_scripts/0_run_gpu_node.sh train.py --config-path=configs/llama3 --config-name=grpo_3B_config_train
bash runner_scripts/0_run_gpu_node.sh evaluate.py --config-path=configs/llama3 --config-name=grpo_3B_config_eval
