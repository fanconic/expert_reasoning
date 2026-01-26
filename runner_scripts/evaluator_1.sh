#!/bin/bash
export GPU_NUM="1"

export MODEL="qwen7b/discounted_reward"
export DATASET="gsm8k_rebuttals"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval_${MODEL_VARIANT}
