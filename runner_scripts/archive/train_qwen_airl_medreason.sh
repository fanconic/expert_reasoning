#!/bin/bash
export GPU_NUM="3"
export DATASET="medreason_rebuttals"

export MODEL="qwen3b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run

export MODEL="qwen7b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run