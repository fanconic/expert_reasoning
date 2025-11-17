#!/bin/bash
# export OVERRIDE="dataset.expert_error_rate=0.1"
# bash runner_scripts/1_run_gpu_node.sh irl_train.py --config-path=configs/llama8b --config-name=8B_1B_config_irl_train_error $OVERRIDE
# bash runner_scripts/1_run_gpu_node.sh evaluate.py --config-path=configs/llama8b --config-name=8B_1B_config_eval_error $OVERRIDE

export OVERRIDE="dataset.expert_error_rate=0.25"
bash runner_scripts/1_run_gpu_node.sh irl_train.py --config-path=configs/llama8b --config-name=8B_1B_config_irl_train_error $OVERRIDE
bash runner_scripts/1_run_gpu_node.sh evaluate.py --config-path=configs/llama8b --config-name=8B_1B_config_eval_error $OVERRIDE
