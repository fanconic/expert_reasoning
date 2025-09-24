#!/bin/bash
export OVERRIDE="wandb.run_name=llama8_airl_wgan"
#bash runner_scripts/0_run_gpu_node.sh irl_train.py --config-path=configs/llama --config-name=8B_1B_config_irl_train $OVERRIDE
bash runner_scripts/0_run_gpu_node.sh evaluate.py --config-path=configs/llama --config-name=8B_1B_config_eval $OVERRIDE
