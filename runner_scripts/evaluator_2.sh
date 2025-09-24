#!/bin/bash
export OVERRIDE="wandb.run_name=llama8_airl_noper_2 model.num_neg_perturbations_per_expert=0"
bash runner_scripts/2_run_gpu_node.sh irl_train.py --config-path=configs/llama --config-name=8B_1B_config_irl_train $OVERRIDE
export OVERRIDE="wandb.run_name=llama8_airl_noper_2"
bash runner_scripts/2_run_gpu_node.sh evaluate.py --config-path=configs/llama --config-name=8B_1B_config_eval $OVERRIDE