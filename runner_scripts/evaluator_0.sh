#!/bin/bash

export MODEL="llama3b"
export GPU_NUM="0" 

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/medreason/${MODEL} --config-name=3B_1B_perturb_config_irl_train
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/medreason/${MODEL} --config-name=3B_1B_perturb_config_eval

export MODEL="llama8b"
export OVERRIDE="wandb.run_name=${MODEL}_medical_airl_perturb2_sft model.policy_name=/mnt/pdata/caf83/medical_reasoning/outputs/${MODEL}_medical_sft/checkpoint-500" 
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py --config-path=configs/medreason/${MODEL} --config-name=8B_1B_perturb_config_irl_train $OVERRIDE
export OVERRIDE="wandb.run_name=${MODEL}_medical_airl_perturb2_sft"
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py --config-path=configs/medreason/${MODEL} --config-name=8B_1B_perturb_config_eval $OVERRIDE
