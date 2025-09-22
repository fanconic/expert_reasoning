#!/bin/bash
bash runner_scripts/1_run_gpu_node.sh irl_train.py --config-path=configs/llama3 --config-name=3B_1B_config_irl_train_continued
bash runner_scripts/1_run_gpu_node.sh evaluate.py --config-path=configs/llama3 --config-name=3B_1B_config_eval_continued
