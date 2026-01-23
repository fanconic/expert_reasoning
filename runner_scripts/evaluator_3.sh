#!/bin/bash
bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/medreason_rebuttals/llama8b --config-name=eval wandb.run_name=llama8b_ovr

export GPU_NUM="3"
export DATASET="gsm8k_rebuttals"

export MODEL="llama3b"
export OVERRIDE="wandb.run_name=llama3b_partial_fixed model.dense_rewards=partial_fixed model.dense_partial_fixed_n=15"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE


export DATASET="medreason_rebuttals"
export MODEL="llama3b"
export OVERRIDE="wandb.run_name=llama3b_partial_fixed model.dense_rewards=partial_fixed model.dense_partial_fixed_n=15"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run $OVERRIDE
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE
