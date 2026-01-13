#!/bin/bash
export GPU_NUM="1"
export MODEL="llama8b" 


bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py