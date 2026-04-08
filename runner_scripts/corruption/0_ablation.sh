set -u 
export GPU_NUM="0" 
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

MODEL="qwen7b"
# Base training parameters that only go to train_irl.py
BASE_TRAIN_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.max_steps=400 training.buffer_size=50"
# Common flags that both train and eval can handle
COMMON_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0 model.dense_rewards=partial"

run_perturb_ablation() {
    local NAME_SUFFIX="$1"
    local PERTURB_LIST="$2"
    local SWITCH_LABEL="${3:-true}"
    
    # 1. REMOVE SPACES: Critical for Hydra to recognize the list
    local CLEAN_LIST=$(echo "$PERTURB_LIST" | tr -d ' ')
    
    local WNAME="${MODEL}_perturb_${NAME_SUFFIX}"
    
    # 2. DEFINING FLAGS: We don't quote these internal strings yet
    local TRAIN_ONLY_FLAGS="model.neg_perturb_fns=${CLEAN_LIST} model.switch_label_if_correct=${SWITCH_LABEL}"
    local SHARED_OVERRIDE="wandb.run_name=${WNAME} ${COMMON_FLAGS}"

    echo "------------------------------------------------"
    echo "▶ Starting: ${WNAME}"
    echo "------------------------------------------------"

    # 3. EXECUTION: Notice the REMOVAL of quotes around the variables here
    # This lets Bash split them into separate arguments for Hydra
    echo "Running Training..."
    bash "$RUNNER" train_irl.py \
        --config-path="configs/gsm8k_rebuttals/${MODEL}" \
        --config-name="good_run" \
        $SHARED_OVERRIDE $BASE_TRAIN_PARAMS $TRAIN_ONLY_FLAGS

    echo "Running Evaluation..."
    bash "$RUNNER" evaluate.py \
        --config-path="configs/gsm8k_rebuttals/${MODEL}" \
        --config-name="eval" \
        $SHARED_OVERRIDE
}

# Task: Empty List
run_perturb_ablation "none" "[]"