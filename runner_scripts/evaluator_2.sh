#!/bin/bash
export GPU_NUM="2"

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py