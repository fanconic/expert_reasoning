#!/bin/bash
export OVERRIDE="dataset.expert_error_rate=0.25"
bash runner_scripts/1_run_gpu_node.sh evaluate.py --config-path=configs/qwen3b --config-name=3B_1B_config_eval_error $OVERRIDE
