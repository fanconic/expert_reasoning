#!/bin/bash
export GPU_NUM="1"

export DATASET="gsm8k_rebuttals"


export MODEL="qwen3b"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval


# export MODEL="llama3b"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval


export MODEL="qwen7b"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval


# export MODEL="llama8b"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval


# Sparse
# export MODEL="qwen3b"
# export OVERRIDE="wandb.run_name=qwen3b_8ga_8gens_clipped_sparse model.dense_rewards=false"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

# export MODEL="llama3b"
# export OVERRIDE="wandb.run_name=llama3b_8ga_8gens_clipped_sparse model.dense_rewards=false"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

# export MODEL="qwen7b"
# export OVERRIDE="wandb.run_name=qwen7b_8ga_8gens_clipped_sparse model.dense_rewards=false"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

export MODEL="llama8b"
export OVERRIDE="wandb.run_name=llama8b_8ga_8gens_clipped_sparse model.dense_rewards=false"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE


# Full
# export MODEL="qwen3b"
# export OVERRIDE="wandb.run_name=qwen3b_8ga_8gens_clipped_full model.dense_rewards=full"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

# export MODEL="llama3b"
# export OVERRIDE="wandb.run_name=llama3b_8ga_8gens_clipped_full model.dense_rewards=full"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

# export MODEL="qwen7b"
# export OVERRIDE="wandb.run_name=qwen7b_8ga_8gens_clipped_full model.dense_rewards=full"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE

# export MODEL="llama8b"
# export OVERRIDE="wandb.run_name=llama8b_8ga_8gens_clipped_full model.dense_rewards=full"
# bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval $OVERRIDE