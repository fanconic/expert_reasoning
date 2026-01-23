#!/bin/bash
#bash runner_scripts/0_run_gpu_node.sh irl_train.py

# OVERRIDE="wandb.run_name=airl_dense_1_09_help_smalldisc_prime"
# bash runner_scripts/0_run_gpu_node.sh evaluate_irl.py $OVERRIDE

OVERRIDE="wandb.run_name=airl_7b_sparse_per model.dense_rewards=false model.reward_name=Qwen/Qwen2.5-7B-Instruct"
bash runner_scripts/0_run_gpu_node.sh evaluate_irl.py $OVERRIDE

# OVERRIDE="wandb.run_name=airl_dense_1_09 model.reward_name=Qwen/Qwen2.5-7B-Instruct"
# bash runner_scripts/1_run_gpu_node.sh evaluate_irl.py $OVERRIDE