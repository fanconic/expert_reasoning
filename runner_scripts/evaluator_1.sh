#!/bin/bash
export GPU_NUM="1"
export DATASET="gsm8k_rebuttals"

export MODEL="qwen3b"
export OVERRIDE="wandb.run_name=qwen3b_partial_fixed model.dense_rewards=partial_fixed model.dense_partial_fixed_n=15"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

export DATASET="medreason_rebuttals"
export MODEL="qwen3b"
export OVERRIDE="wandb.run_name=qwen3b_partial_fixed model.dense_rewards=partial_fixed model.dense_partial_fixed_n=15"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE
