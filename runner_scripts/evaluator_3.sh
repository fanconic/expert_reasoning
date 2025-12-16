#!/bin/bash

export MODEL="llama8b"
export GPU_NUM="3"  
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/medreason/${MODEL} --config-name=debug