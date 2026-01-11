#!/bin/bash
export GPU_NUM="1"
export MODEL="qwen3b" 

# export OVERRIDE="wandb.run_name=qwen3b_8ga_8gens_discounted model.advantage_calculation=discounted_dense"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=good_run $OVERRIDE
# export MODEL="qwen7b" 
# export OVERRIDE="wandb.run_name=qwen7b_1_warmup_partial model.dense_rewards=partial"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=1_warmup $OVERRIDE

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py