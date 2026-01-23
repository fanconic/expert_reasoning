#!/bin/bash
export GPU_NUM="2"
export DATASET="medreason_rebuttals"
export MODEL="qwen7b"

export OVERRIDE="wandb.run_name=qwen7b_8ga_8gens_clipped_sparse model.dense_rewards=false"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE

export OVERRIDE="wandb.run_name=qwen7b_8ga_8gens_clipped_full model.dense_rewards=full"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE