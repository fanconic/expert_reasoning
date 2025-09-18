#!/bin/bash

OVERRIDE="wandb.run_name=pre_trained_unsloth_dense_09_per model.add_expert_to_policy_optim=false"
bash runner_scripts/0_run_gpu_node.sh irl_train.py $OVERRIDE

OVERRIDE="wandb.run_name=unsloth_dense_075_per model.policy_name=Qwen/Qwen2.5-7B-Instruct model.dense_gamma=0.75"
bash runner_scripts/0_run_gpu_node.sh irl_train.py $OVERRIDE