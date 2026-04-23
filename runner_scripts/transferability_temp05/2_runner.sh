#!/usr/bin/env bash
set -u

export GPU_NUM="2"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

# Transfer matrix in this repo uses partial by default.
: "${DENSITY:=partial_fixed}"

ASSIGNED_TASKS=(
    # "math math llama8b"
    # "math mmlu qwen3b"
    # "medicine math llama3b"
    # "medicine medicine llama8b"
    # "mmlu math qwen3b"
    # "mmlu medicine llama3b"
    # "mmlu mmlu llama8b"
    # "medicine medicine qwen4b"
    # "math mmlu qwen4b"
    "mmlu mmlu qwen7b"
    "mmlu medicine qwen7b"
    "mmlu math qwen7b"
    "medicine mmlu qwen7b"
    "medicine medicine qwen7b"
    "medicine math qwen7b"
    "math mmlu qwen7b"
    "math medicine qwen7b"
    "math math qwen7b"
)

REWARD_FLAGS=("model.reward_lb=-5.0" "model.reward_ub=5.0" "model.clip_reward_model=true")
FAILED_RUNS=()
LAUNCHED=
BASE_SFT_MODEL="llama8b_sft"

dataset_outputs_root() {
    local dataset="$1"
    case "$dataset" in
        math) echo "/mnt/pdata/caf83/icml_math/outputs" ;;
        medicine) echo "/mnt/pdata/caf83/icml_medicine/outputs" ;;
        mmlu) echo "/mnt/pdata/caf83/icml_mmlu/outputs" ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            return 1
            ;;
    esac
}

dataset_kd_name() {
    local dataset="$1"
    case "$dataset" in
        math) echo "gsm8k_kd" ;;
        medicine) echo "medical_kd" ;;
        mmlu) echo "mmlu_kd" ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            return 1
            ;;
    esac
}

run_cmd() {
    local label="$1"
    shift
    echo -e "\n▶ ${label}\n  $*"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        FAILED_RUNS+=("${label} (exit=${rc})")
        echo "  ✗ FAILED"
    else
        echo "  ✓ OK"
    fi
}

run_transfer_eval() {
    local POLICY_DATASET="$1"
    local REWARD_DATASET="$2"
    local REWARD_ARCH="$3"

    local DENSE_VAL="$DENSITY"
    [[ "$DENSITY" == "sparse" ]] && DENSE_VAL="false"

    local POLICY_ROOT
    POLICY_ROOT="$(dataset_outputs_root "${POLICY_DATASET}")" || return

    local POLICY_DATASET_NAME
    POLICY_DATASET_NAME="$(dataset_kd_name "${POLICY_DATASET}")" || return

    local POLICY_MODEL_DIR="${POLICY_ROOT}/${BASE_SFT_MODEL}/best_model"
    local TRACE_FILE="${POLICY_MODEL_DIR}/eval_results_${POLICY_DATASET}_${BASE_SFT_MODEL}_t0p5.jsonl"

    local WNAME="${REWARD_ARCH}_${DENSITY}"
    local TRANSFER_LABEL="transfer_${REWARD_ARCH}_${DENSITY}_P_${POLICY_DATASET}_R_${REWARD_DATASET}"
    local OUT_DIR="${POLICY_ROOT}/${TRANSFER_LABEL}/best_model"
    local OUTFILE="${OUT_DIR}/eval_results_new.jsonl"
    local MIRROR_FILE="${POLICY_MODEL_DIR}/${TRANSFER_LABEL}.jsonl"
    local LABEL="${TRANSFER_LABEL}_t0p5"

    mkdir -p "${OUT_DIR}"

    local CMD=(
        bash "$RUNNER" evaluate.py
        --config-path="configs/${REWARD_DATASET}/${REWARD_ARCH}"
        --config-name="irl_eval"
        "wandb.run_name=${WNAME}"
        "model.dense_rewards=${DENSE_VAL}"
        "${REWARD_FLAGS[@]}"
        "dataset.name=${POLICY_DATASET_NAME}"
        "model.policy_name=${POLICY_MODEL_DIR}"
        "sampling.temperature=0.5"
        "++eval.mode=pregenerated_policy_and_reward"
        "++eval.pregenerated_jsonl_path=${TRACE_FILE}"
        "++eval.compute_policy_log_probs=false"
        "++eval.compute_reward_model_scores=true"
        "++eval.output_file=${OUTFILE}"
    )

    run_cmd "${LABEL}" "${CMD[@]}"
    if [[ -f "${OUTFILE}" ]]; then
        run_cmd "${LABEL}_mirror" cp -f "${OUTFILE}" "${MIRROR_FILE}"
    fi
}

for TASK in "${ASSIGNED_TASKS[@]}"; do
    read -r POLICY_DATASET REWARD_DATASET REWARD_ARCH <<< "$TASK"
    run_transfer_eval "$POLICY_DATASET" "$REWARD_DATASET" "$REWARD_ARCH"
    ((LAUNCHED++))
done

echo -e "\n======================\nGPU ${GPU_NUM} (${DENSITY}) SUMMARY\n======================"
echo "Launched transfer evaluations: ${LAUNCHED}"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All assigned transfer evaluations succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
