#!/bin/bash
export OVERRIDE="wandb.run_name=llama3_airl_wgan"
#bash runner_scripts/3_run_gpu_node.sh irl_train.py --config-path=configs/llama3 --config-name=3B_1B_config_irl_train $OVERRIDE
bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/llama3 --config-name=3B_1B_config_eval $OVERRIDE
