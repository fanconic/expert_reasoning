#!/bin/bash
export GPU_NUM="0"
export MODEL="qwen3b" 

export OVERRIDE="wandb.run_name=qwen3b_8ga_8gens_expert model.add_expert_to_policy_optim=true"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=good_run $OVERRIDE
# export MODEL="llama8b" 
# export OVERRIDE="wandb.run_name=llama8b_1_warmup_partial model.dense_rewards=partial"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=1_warmup $OVERRIDE