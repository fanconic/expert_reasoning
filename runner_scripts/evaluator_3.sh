#!/bin/bash
OVERRIDE="wandb.run_name=qwen7b_airl_09_bce_15math model.reward_name=Qwen/Qwen2.5-Math-1.5B-Instruct"
bash runner_scripts/3_run_gpu_node.sh evaluate.py $OVERRIDE

bash runner_scripts/3_run_gpu_node.sh sft_train.py --config-path=configs/phi --config-name=sft_7B_config_train
bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/phi --config-name=sft_7B_config_eval

bash runner_scripts/3_run_gpu_node.sh train.py --config-path=configs/phi --config-name=grpo_7B_config_train
bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/phi --config-name=grpo_7B_config_eval