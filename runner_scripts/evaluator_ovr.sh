#!/bin/bash
export GPU_NUM="3"

export DATASET="gsm8k_rebuttals"


# OVR
export MODEL="qwen3b"
export OVERRIDE="wandb.run_name=qwen3b_ovr"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

export MODEL="llama3b"
export OVERRIDE="wandb.run_name=llama3b_ovr"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

export MODEL="qwen7b"
export OVERRIDE="wandb.run_name=qwen7b_ovr"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

export MODEL="llama8b"
export OVERRIDE="wandb.run_name=llama8b_ovr"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE



export DATASET="medreason_rebuttals"

export MODEL="qwen3b"
export OVERRIDE="wandb.run_name=qwen3b_ovr"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

export MODEL="llama3b"
export OVERRIDE="wandb.run_name=llama3b_ovr"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

export MODEL="qwen7b"
export OVERRIDE="wandb.run_name=qwen7b_ovr"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

export MODEL="llama8b"
export OVERRIDE="wandb.run_name=llama8b_ovr"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE



