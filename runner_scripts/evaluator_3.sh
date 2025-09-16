#!/bin/bash

OVERRIDE="wandb.run_name=airl_dense_1_09_help_smalldisc_prime"
bash runner_scripts/3_run_gpu_node.sh evaluate_irl.py $OVERRIDE

OVERRIDE="wandb.run_name=airl_dense_per_099_help_smalldisc_prime"
bash runner_scripts/3_run_gpu_node.sh irl_train.py $OVERRIDE

OVERRIDE="wandb.run_name=airl_dense_per_099_help_smalldisc_prime"
bash runner_scripts/3_run_gpu_node.sh evaluate_irl.py $OVERRIDE