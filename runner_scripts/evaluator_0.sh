#!/bin/bash
export GPU_NUM="0"
export MODEL="llama3b" 

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/gsm8k_rebuttals/${MODEL} --config-name=1_warmup