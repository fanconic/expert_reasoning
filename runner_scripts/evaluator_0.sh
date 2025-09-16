#!/bin/bash

OVERRIDE="wandb.run_name=airl_dense_1_09_help_smalldisc"
bash runner_scripts/0_run_gpu_node.sh evaluate_irl.py $OVERRIDE

OVERRIDE="wandb.run_name=airl_dense_1_09_help model.reward_name=Qwen/Qwen2.5-7B-Instruct"
bash runner_scripts/0_run_gpu_node.sh evaluate_irl.py $OVERRIDE