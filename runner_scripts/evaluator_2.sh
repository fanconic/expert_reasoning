#!/bin/bash
#export OVERRIDE="dataset.expert_error_rate=0.25"
#bash runner_scripts/2_run_gpu_node.sh evaluate.py --config-path=configs/qwen3b --config-name=3B_1B_config_eval_error $OVERRIDE

#bash runner_scripts/2_run_gpu_node.sh irl_train.py --config-path=configs/qwen7b --config-name=7B_1B_config_irl_train_fixed
bash runner_scripts/2_run_gpu_node.sh evaluate.py --config-path=configs/qwen7b --config-name=7B_1B_config_eval_fixed

export OVERRIDE="dataset.expert_error_rate=0.25"
bash runner_scripts/2_run_gpu_node.sh evaluate.py --config-path=configs/qwen3b --config-name=3B_1B_config_eval_error $OVERRIDE
