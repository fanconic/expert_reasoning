# #!/usr/bin/env bash
set -u
export GPU_NUM="3"       # Set to 0, 1, or 2
export MODEL="qwen4b"
DATASET="gsm8k_rebuttals" # Set to gsm8k_rebuttals, medreason_rebuttals, or mmlu_rebuttals

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


# 3. IRL Tasks (Full & Sparse)
for suffix in "partial_fixed"; do
    WNAME="${MODEL}_${suffix}"
    
    # run_cmd "${DATASET}_IRL_${suffix}" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train_irl.py \
    #     --config-path=configs/${DATASET}/${MODEL} --config-name=good_run \
    #     wandb.run_name=$WNAME model.dense_rewards=$DENSE $COMMON_REWARD_FLAGS $IRL_PARAMS $STEP_LIMIT
        
    run_cmd "${DATASET}_IRL_${suffix}_EVAL" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
        --config-path=configs/${DATASET}/${MODEL} --config-name=eval \
        wandb.run_name=$WNAME model.dense_rewards=$suffix $COMMON_REWARD_FLAGS
done

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
[[ ${#FAILED_RUNS[@]} -eq 0 ]] && echo "All runs succeeded!" || printf "FAILURES:\n  %s\n" "${FAILED_RUNS[@]}"