#!/bin/bash
export GPU_NUM="2"

export MODEL="switch_reward"
export DATASET="medreason_rebuttals"
export MODEL_VARIANT="qwen3b"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/${DATASET}/${MODEL} --config-name=good_run_${MODEL_VARIANT}
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/${DATASET}/${MODEL} --config-name=eval_${MODEL_VARIANT}
