#!/bin/bash
export GPU_NUM="3"
export MODEL="qwen3b" 

export OVERRIDE="wandb.run_name=qwen3b_8ga_8gens_sft model.policy_name=/mnt/pdata/caf83/tabular_reasoning/outputs/qwen3b_sft/checkpoint-500"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=good_run $OVERRIDE