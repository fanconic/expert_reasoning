#!/bin/bash
export OVERRIDE="wandb.run_name=llama3_airl_noper model.num_neg_perturbations_per_expert=0"
bash runner_scripts/0_run_gpu_node.sh irl_train.py --config-path=configs/llama3 --config-name=3B_1B_config_irl_train $OVERRIDE
export OVERRIDE="wandb.run_name=llama3_airl_noper"
bash runner_scripts/0_run_gpu_node.sh evaluate.py --config-path=configs/llama3 --config-name=3B_1B_config_eval $OVERRIDE
