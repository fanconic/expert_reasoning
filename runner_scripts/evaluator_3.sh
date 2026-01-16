#!/bin/bash
export GPU_NUM="3"
export DATASET="gsm8k_rebuttals"

export MODEL="llama8b" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run