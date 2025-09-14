#!/bin/bash
bash runner_scripts/0_run_gpu_node.sh irl_train.py

OVERRIDE="run_name=airl_7b_sparse_per"
bash runner_scripts/0_run_gpu_node.sh evaluate_irl.py $OVERRIDE

OVERRIDE="run_name=airl_dense_1_09_help_smalldisc_prime"
bash runner_scripts/0_run_gpu_node.sh evaluate_irl.py $OVERRIDE