#!/bin/bash
bash runner_scripts/2_run_gpu_node.sh irl_train.py --config-path=configs/llama3 --config-name=3B_1B_config_irl_train
bash runner_scripts/2_run_gpu_node.sh evaluate.py --config-path=configs/llama3 --config-name=3B_1B_config_eval $OVERRIDE
