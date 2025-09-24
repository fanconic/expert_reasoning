#!/bin/bash
# export OVERRIDE="wandb.run_name=qwen3_airl_wgan model.classifier_loss=wgan"
# bash runner_scripts/1_run_gpu_node.sh irl_train.py --config-path=configs/qwen_3B --config-name=3B_1B_config_irl_train $OVERRIDE
export OVERRIDE="wandb.run_name=qwen3_airl_wgan"
bash runner_scripts/1_run_gpu_node.sh evaluate.py --config-path=configs/qwen_3B --config-name=3B_1B_config_eval $OVERRIDE
