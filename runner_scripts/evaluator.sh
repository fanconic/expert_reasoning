#!/bin/bash

OVERRIDE="wandb.run_name=airl_dense_per_09_help_smalldisc"
bash runner_scripts/1_run_gpu_node.sh evaluate_irl.py $OVERRIDE

OVERRIDE="wandb.run_name=airl_dense_1_09_help_smalldisc"
bash runner_scripts/1_run_gpu_node.sh evaluate_irl.py $OVERRIDE

OVERRIDE="wandb.run_name=airl_dense_1_09_help"
bash runner_scripts/1_run_gpu_node.sh evaluate_irl.py $OVERRIDE