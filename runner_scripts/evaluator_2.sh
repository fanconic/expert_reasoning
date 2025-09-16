#!/bin/bash

OVERRIDE="wandb.run_name=airl_dense_per_099_help_smalldisc"
bash runner_scripts/2_run_gpu_node.sh evaluate_irl.py $OVERRIDE

OVERRIDE="wandb.run_name=airl_dense_per_09_help model.reward_name=Qwen/Qwen2.5-7B-Instruct"
bash runner_scripts/2_run_gpu_node.sh evaluate_irl.py $OVERRIDE