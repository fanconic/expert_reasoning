#!/bin/bash
export GPU_NUM="2"
export MODEL="qwen3b" 

export OVERRIDE="wandb.run_name=qwen3b_8ga_8gens_noswitch model.switch_label_if_correct=false"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=good_run $OVERRIDE
# export MODEL="llama3b" 
# export OVERRIDE="wandb.run_name=llama3b_1_warmup_partial model.dense_rewards=partial"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=1_warmup $OVERRIDE