#!/bin/bash
# bash runner_scripts/3_run_gpu_node.sh irl_train.py --config-path=configs/gemma --config-name=9B_2B_config_irl_train
# bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/gemma --config-name=9B_2B_config_eval

bash runner_scripts/3_run_gpu_node.sh irl_train.py --config-path=configs/llama --config-name=8B_1B_config_irl_train_continued
bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/llama --config-name=8B_1B_config_eval_continued
