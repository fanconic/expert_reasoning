#!/bin/bash
OVERRIDE="model.policy_learning_rate=1e-4 wandb.run_name=llama8_airl_2 eval.eval_steps=100"
bash runner_scripts/1_run_gpu_node.sh irl_train.py --config-path=configs/llama --config-name=8B_1B_config_irl_train $OVERRIDE
bash runner_scripts/1_run_gpu_node.sh evaluate.py --config-path=configs/llama --config-name=8B_1B_config_eval

# bash runner_scripts/1_run_gpu_node.sh irl_train.py --config-path=configs/phi --config-name=7B_2B_config_irl_train
# bash runner_scripts/1_run_gpu_node.sh irl_train.py --config-path=configs/phi --config-name=7B_2B_config_eval

bash runner_scripts/3_run_gpu_node.sh irl_train.py --config-path=configs/falcon --config-name=7B_2B_config_irl_train $OVERRIDE