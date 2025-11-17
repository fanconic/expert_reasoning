#!/bin/bash
#bash runner_scripts/0_run_gpu_node.sh irl_train.py --config-path=configs/llama8b --config-name=8B_1B_config_irl_train_fixed
bash runner_scripts/0_run_gpu_node.sh evaluate.py --config-path=configs/llama8b --config-name=8B_1B_config_eval_fixed
