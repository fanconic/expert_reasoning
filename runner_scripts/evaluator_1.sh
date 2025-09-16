#!/bin/bash

OVERRIDE="wandb.run_name=airl_dense_per_09_help_smalldisc"
bash runner_scripts/1_run_gpu_node.sh evaluate_irl.py $OVERRIDE

OVERRIDE="wandb.run_name=airl_7b_sparse_per model.dense_rewards=false model.reward_name=Qwen/Qwen2.5-7B-Instruct"
bash runner_scripts/1_run_gpu_node.sh evaluate_irl.py $OVERRIDE