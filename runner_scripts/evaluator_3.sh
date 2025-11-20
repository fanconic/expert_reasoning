# #!/bin/bash
# bash runner_scripts/3_run_gpu_node.sh sft_train.py --config-path=configs/medreason/llama8b --config-name=sft_8B_config_train
# bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/medreason/llama8b --config-name=sft_8B_config_eval

# # GRPO
# bash runner_scripts/3_run_gpu_node.sh train.py --config-path=configs/medreason/llama8b --config-name=grpo_8B_config_train
# bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/medreason/llama8b --config-name=grpo_8B_config_eval

#AIRL
bash runner_scripts/3_run_gpu_node.sh irl_train.py --config-path=configs/medreason/llama8b --config-name=8B_1B_config_irl_train
bash runner_scripts/3_run_gpu_node.sh evaluate.py --config-path=configs/medreason/llama8b --config-name=8B_1B_config_eval