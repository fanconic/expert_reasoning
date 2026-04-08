#!/usr/bin/env bash
set -u
export GPU_NUM="3"
export MODEL="qwen4b"

STEP_LIMIT="training.max_steps=400"
IRL_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.buffer_size=50"
COMMON_REWARD_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0"

FAILED_RUNS=()
run_cmd() {
    local label="$1"; shift
    "$@"
    [[ $? -ne 0 ]] && FAILED_RUNS+=("$label") && echo "  ✗ FAILED" || echo "  ✓ OK"
}

for ds in "gsm8k_rebuttals" "medreason_rebuttals"; do
    for suffix in "partial" "partial_fixed"; do
        WNAME="${MODEL}_${suffix}"
        
        run_cmd "${ds}_${suffix}_TRAIN" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train_irl.py \
            --config-path=configs/${ds}/${MODEL} --config-name=good_run \
            wandb.run_name=$WNAME model.dense_rewards=$suffix $COMMON_REWARD_FLAGS $IRL_PARAMS $STEP_LIMIT
            
        run_cmd "${ds}_${suffix}_EVAL" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
            --config-path=configs/${ds}/${MODEL} --config-name=eval \
            wandb.run_name=$WNAME model.dense_rewards=$suffix $COMMON_REWARD_FLAGS
    done
done

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
[[ ${#FAILED_RUNS[@]} -eq 0 ]] && echo "All runs succeeded!" || printf "FAILURES:\n  %s\n" "${FAILED_RUNS[@]}"

bash runner_scripts/qwen4/3_runner_follower.sh