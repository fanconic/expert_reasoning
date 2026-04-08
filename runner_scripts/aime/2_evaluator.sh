#!/usr/bin/env bash
set -u
export GPU_NUM="2"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
REWARD_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0"

FAILED_RUNS=()
run_aime() {
    local MODEL="$1"
    local YEAR="$2"
    local VARIANT="$3"
    
    local WNAME="${MODEL}_${VARIANT}"
    local CONFIG="eval"
    local EXTRA=""

    # Logic to handle variant-specific configs
    case "$VARIANT" in
        sft)   CONFIG="sft_eval" ;;
        grpo)  CONFIG="grpo_eval" ;;
        sparse) EXTRA="model.dense_rewards=false ${REWARD_FLAGS}" ;;
        *)      EXTRA="model.dense_rewards=${VARIANT} ${REWARD_FLAGS}" ;;
    esac

    echo "▶ Eval: ${WNAME}"
    bash "$RUNNER" evaluate.py \
        --config-path="configs/aime/${MODEL}" \
        --config-name="$CONFIG" \
        wandb.run_name="$WNAME" \
        dataset.name="$YEAR" \
        $EXTRA
    
    [[ $? -ne 0 ]] && FAILED_RUNS+=("$WNAME")
}

# --- Execution Loop ---
for M in "qwen3b"; do
    for YEAR in "aime_2024"; do
        for V in "sft" "grpo" "full" "sparse" "partial" "partial_fixed"; do
            run_aime "$M" "$YEAR" "$V"
        done
    done
done


echo -e "\nSummary GPU 0: Failures: ${#FAILED_RUNS[@]}"
printf "  %s\n" "${FAILED_RUNS[@]}"