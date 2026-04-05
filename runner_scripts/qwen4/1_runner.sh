#!/usr/bin/env bash
set -u
export GPU_NUM="1"       # Set to 0, 1, or 2
export MODEL="qwen4b"
DATASET="medreason_rebuttals" # Set to gsm8k_rebuttals, medreason_rebuttals, or mmlu_rebuttals

# --- Shared Parameters ---
STEP_LIMIT="training.max_steps=400"
COMMON_REWARD_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0"
# IRL-specific params
IRL_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.buffer_size=50"

FAILED_RUNS=()
run_cmd() {
    local label="$1"; shift
    echo -e "\n▶ Starting: $label"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        FAILED_RUNS+=("$label (exit=$rc)")
        echo "  ✗ FAILED"
    else
        echo "  ✓ OK"
    fi
}

# 1. SFT (Now with max_steps)
run_cmd "${DATASET}_SFT" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh sft_train.py \
    --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train $STEP_LIMIT
run_cmd "${DATASET}_SFT_EVAL" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
    --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval

# 2. GRPO (Now with max_steps)
run_cmd "${DATASET}_GRPO" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train.py \
    --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_train $STEP_LIMIT
run_cmd "${DATASET}_GRPO_EVAL" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
    --config-path=configs/${DATASET}/${MODEL} --config-name=grpo_eval

# 3. IRL Tasks (Full & Sparse)
for suffix in "full" "sparse"; do
    DENSE="full" && [[ "$suffix" == "sparse" ]] && DENSE="false"
    WNAME="${MODEL}_${suffix}"
    
    run_cmd "${DATASET}_IRL_${suffix}" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py \
        --config-path=configs/${DATASET}/${MODEL} --config-name=good_run \
        wandb.run_name=$WNAME model.dense_rewards=$DENSE $COMMON_REWARD_FLAGS $IRL_PARAMS $STEP_LIMIT
        
    run_cmd "${DATASET}_IRL_${suffix}_EVAL" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
        --config-path=configs/${DATASET}/${MODEL} --config-name=eval \
        wandb.run_name=$WNAME model.dense_rewards=$DENSE $COMMON_REWARD_FLAGS
done

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
[[ ${#FAILED_RUNS[@]} -eq 0 ]] && echo "All runs succeeded!" || printf "FAILURES:\n  %s\n" "${FAILED_RUNS[@]}"



export GPU_NUM="1"
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

for ds in "mmlu_rebuttals"; do
    for suffix in "partial_fixed"; do
        WNAME="${MODEL}_${suffix}"
        
        run_cmd "${ds}_${suffix}_TRAIN" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh irl_train.py \
            --config-path=configs/${ds}/${MODEL} --config-name=good_run \
            wandb.run_name=$WNAME model.dense_rewards=$suffix $COMMON_REWARD_FLAGS $IRL_PARAMS $STEP_LIMIT
            
        run_cmd "${ds}_${suffix}_EVAL" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
            --config-path=configs/${ds}/${MODEL} --config-name=eval \
            wandb.run_name=$WNAME model.dense_rewards=$suffix $COMMON_REWARD_FLAGS
    done
done

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
[[ ${#FAILED_RUNS[@]} -eq 0 ]] && echo "All runs succeeded!" || printf "FAILURES:\n  %s\n" "${FAILED_RUNS[@]}"

bash runner_scripts/qwen4/1_runner_follower.sh