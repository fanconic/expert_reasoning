#!/bin/bash
export GPU_NUM="3"

export MODEL="llama3b/discounted_reward"
export DATASET="gsm8k_rebuttals"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval_${MODEL_VARIANT}
