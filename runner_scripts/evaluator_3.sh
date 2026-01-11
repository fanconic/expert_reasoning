#!/bin/bash
export GPU_NUM="3"
export DATASET="gsm8k_rebuttals"

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py