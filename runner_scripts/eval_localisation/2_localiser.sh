#!/usr/bin/env bash
set -u

export GPU_NUM="2"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

# Shared defaults (override at launch if needed)
: "${TRACE_JSONL:=/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model/eval_results_math_qwen7b_sft_t0p5.jsonl}"
: "${OUTPUT_ROOT:=/mnt/pdata/caf83/workspace/caf83/expert_reasoning_clean/localisation}"
: "${MAX_EXAMPLES:=1300}"  # Empty => use full split.
: "${START_INDEX:=0}"
: "${MAX_SEVERITY:=5}"
: "${VARIANTS_PER_SEVERITY:=1}"
: "${LOCAL_WINDOW:=3}"
: "${MAX_MICRO_BATCH:=8}"
: "${AGGREGATE:=mean}"
: "${SEED:=42}"
: "${PARTIAL_FIXED_N:=0}"
: "${TRACE_SOURCES:=expert pregenerated}"
: "${PREGENERATED_PICK:=generation_idx}"
: "${PREGENERATED_GENERATION_IDX:=0}"
: "${CLEAN_CORRECT_POLICY:=require}"
: "${PERTURB_FNS:=flip_operator_in_one_step corrupt_numbers}"
: "${QWEN7B_FULL_REWARD_CHECKPOINT:=/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_full_rebuttal_restart/checkpoint-100}"

IFS=' ' read -r -a PERTURB_FN_ARR <<< "${PERTURB_FNS}"
IFS=' ' read -r -a TRACE_SOURCE_ARR <<< "${TRACE_SOURCES}"

ASSIGNED_TASKS=(
  "qwen7b full"
  "qwen7b partial_fixed"
)

FAILED_RUNS=()
LAUNCHED=0

model_config_path() {
    local model="$1"
    case "$model" in
        qwen4b) echo "configs/math/qwen4b/irl_eval.yaml" ;;
        qwen7b) echo "configs/math/qwen7b/irl_eval.yaml" ;;
        llama8b) echo "configs/math/llama8b/irl_eval.yaml" ;;
        *)
            echo "Unknown model: ${model}" >&2
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

run_localiser() {
    local model="$1"
    local density="$2"
    local trace_source="$3"

    local config_path
    config_path="$(model_config_path "${model}")" || return

    local checkpoint_dir="/mnt/pdata/caf83/icml_math/outputs/${model}_${density}/best_model"
    if [[ "${model}" == "qwen7b" && "${density}" == "full" ]]; then
        checkpoint_dir="${QWEN7B_FULL_REWARD_CHECKPOINT}"
    fi
    if [[ ! -f "${checkpoint_dir}/reward_model/adapter_config.json" ]]; then
        FAILED_RUNS+=("${model}_${density} (missing_checkpoint=${checkpoint_dir})")
        echo "Skipping ${model}_${density}: missing checkpoint ${checkpoint_dir}"
        return
    fi

    local source_suffix=""
    case "${trace_source}" in
        expert)
            source_suffix="expert"
            ;;
        pregenerated)
            source_suffix="from_qwen7b_sft"
            if [[ ! -f "${TRACE_JSONL}" ]]; then
                FAILED_RUNS+=("${model}_${density}_${trace_source} (missing_trace_jsonl=${TRACE_JSONL})")
                echo "Skipping ${model}_${density}_${trace_source}: missing trace jsonl ${TRACE_JSONL}"
                return
            fi
            ;;
        *)
            FAILED_RUNS+=("${model}_${density}_${trace_source} (unknown_trace_source)")
            echo "Skipping ${model}_${density}_${trace_source}: unknown trace source '${trace_source}'"
            return
            ;;
    esac

    local source_dir
    if [[ "${trace_source}" == "expert" ]]; then
        source_dir="expert"
    else
        source_dir="qwen7b_sft"
    fi

    local label="${model}_${density}_localisation_${source_suffix}"
    local out_dir="${OUTPUT_ROOT}/runs/${source_dir}/${model}/${density}"

    mkdir -p "${out_dir}"

    local cmd=(
        bash "${RUNNER}" src/eval/gsm8k_process_sensitivity.py
        --config "${config_path}"
        --checkpoint-dir "${checkpoint_dir}"
        --split "test"
        --trace-source "${trace_source}"
        --dense-reward-mode "${density}"
        --start-index "${START_INDEX}"
        --max-severity "${MAX_SEVERITY}"
        --variants-per-severity "${VARIANTS_PER_SEVERITY}"
        --aggregate "${AGGREGATE}"
        --local-window "${LOCAL_WINDOW}"
        --hit-ks 1 3 5
        --max-micro-batch "${MAX_MICRO_BATCH}"
        --seed "${SEED}"
        --output-dir "${out_dir}"
        --perturb-fns "${PERTURB_FN_ARR[@]}"
    )

    if [[ "${density}" == "partial_fixed" ]]; then
        cmd+=(--dense-partial-fixed-n "${PARTIAL_FIXED_N}")
    fi

    if [[ -n "${MAX_EXAMPLES}" ]]; then
        cmd+=(--max-examples "${MAX_EXAMPLES}")
    fi

    if [[ "${trace_source}" == "pregenerated" ]]; then
        cmd+=(
            --pregenerated-jsonl-path "${TRACE_JSONL}"
            --pregenerated-pick "${PREGENERATED_PICK}"
            --pregenerated-generation-idx "${PREGENERATED_GENERATION_IDX}"
            --clean-correct-policy "${CLEAN_CORRECT_POLICY}"
        )
    fi

    run_cmd "${label}" "${cmd[@]}"
}

for task in "${ASSIGNED_TASKS[@]}"; do
    read -r model density <<< "${task}"
    for trace_source in "${TRACE_SOURCE_ARR[@]}"; do
        run_localiser "${model}" "${density}" "${trace_source}"
        ((LAUNCHED++))
    done
done

echo -e "\n======================\nGPU ${GPU_NUM} LOCALISER SUMMARY\n======================"
echo "Launched localisation runs: ${LAUNCHED}"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All assigned localisation runs succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
