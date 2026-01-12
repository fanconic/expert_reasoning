#!/bin/bash
export GPU_NUM="0"
export DATASET="medreason_rebuttals"

export MODEL="llama3b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run

export MODEL="llama8b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run