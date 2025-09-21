#!/bin/bash

bash runner_scripts/2_run_gpu_node.sh irl_train.py --config-path=configs/qwen-discriminator --config-name=7B_3B_config_irl_train
bash runner_scripts/2_run_gpu_node.sh evaluate.py --config-path=configs/qwen-discriminator --config-name=7B_3B_config_eval