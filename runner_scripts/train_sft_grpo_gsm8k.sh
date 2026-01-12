#!/bin/bash
export GPU_NUM="3"
export DATASET="gsm8k_rebuttals"

# Qwen3B
export MODEL="qwen3b" 
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval


# Llama3B
export MODEL="llama3b" 
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval


# Qwen7B
export MODEL="qwen7b" 
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval


# Llama8B
export MODEL="llama8b" 
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval