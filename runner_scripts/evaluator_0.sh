#!/bin/bash
export GPU_NUM="0"
export MODEL="llama3b" 

export OVERRIDE="wandb.run_name=llama3b_1_warmup_temp10 model.disc_temperature=10"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=1_warmup $OVERRIDE


export MODEL="qwen3b" 
export OVERRIDE="wandb.run_name=qwen3b_1_warmup_temp10 model.disc_temperature=10"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=1_warmup $OVERRIDE