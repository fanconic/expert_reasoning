#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${EXPERT_REASONING_UV_BOOTSTRAPPED:-0}" != "1" \
    && -z "${VIRTUAL_ENV:-}" \
    && ( -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == "base" ) \
    && -d "${REPO_ROOT}/.venv" ]] \
    && command -v uv >/dev/null 2>&1; then
    echo "No project Conda/virtualenv detected; re-running under uv with the repo .venv."
    export EXPERT_REASONING_UV_BOOTSTRAPPED=1
    exec uv run bash "${SCRIPT_PATH}" "$@"
fi

: "${GPU_NUM:=0}"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

POLICY_MODEL_WAS_SET="${POLICY_MODEL+x}"
RUN_DIRS_WAS_SET="${RUN_DIRS+x}"
OUTPUT_SUBDIR_WAS_SET="${OUTPUT_SUBDIR+x}"

: "${LOCALISATION_ROOT:=/mnt/pdata/caf83/workspace/caf83/expert_reasoning_clean/localisation}"
: "${REFERENCE_RUN_DIRS:=runs/qwen7b_sft/qwen7b/full runs/qwen7b_sft/qwen7b/partial_fixed}"
: "${MICRO_BATCH:=4}"
: "${ENTROPY_TOKEN_CHUNK_SIZE:=32}"
: "${MAX_LENGTH:=1124}"
: "${SEVERITY:=1}"
: "${WINDOWS:=1 7}"
: "${SKIP_EXISTING:=1}"
: "${FORCE:=0}"

POLICY_NAMES=()
POLICY_MODELS=()
POLICY_RUN_DIRS=()

add_policy() {
    POLICY_NAMES+=("$1")
    POLICY_MODELS+=("$2")
    POLICY_RUN_DIRS+=("$3")
}

IFS=' ' read -r -a WINDOW_ARR <<< "${WINDOWS}"

if [[ -n "${POLICY_MODEL_WAS_SET}" || -n "${RUN_DIRS_WAS_SET}" ]]; then
    : "${POLICY_NAME:=custom_sft}"
    : "${POLICY_MODEL:=/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model}"
    : "${RUN_DIRS:=${REFERENCE_RUN_DIRS}}"
    add_policy "${POLICY_NAME}" "${POLICY_MODEL}" "${RUN_DIRS}"
else
    : "${POLICY_KEYS:=qwen7b qwen4b llama8b qwen7b_base qwen4b_base llama8b_base}"
    IFS=' ' read -r -a POLICY_KEY_ARR <<< "${POLICY_KEYS}"
    for policy_key in "${POLICY_KEY_ARR[@]}"; do
        case "${policy_key}" in
            qwen7b)
                add_policy \
                    "qwen7b_sft" \
                    "/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model" \
                    "${REFERENCE_RUN_DIRS}"
                ;;
            qwen4b)
                add_policy \
                    "qwen4b_sft" \
                    "/mnt/pdata/caf83/icml_math/outputs/qwen4b_sft/best_model" \
                    "${REFERENCE_RUN_DIRS}"
                ;;
            llama8b)
                add_policy \
                    "llama8b_sft" \
                    "/mnt/pdata/caf83/icml_math/outputs/llama8b_sft/best_model" \
                    "${REFERENCE_RUN_DIRS}"
                ;;
            qwen7b_base)
                add_policy \
                    "qwen7b_base" \
                    "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit" \
                    "${REFERENCE_RUN_DIRS}"
                ;;
            qwen4b_base)
                add_policy \
                    "qwen4b_base" \
                    "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit" \
                    "${REFERENCE_RUN_DIRS}"
                ;;
            llama8b_base)
                add_policy \
                    "llama8b_base" \
                    "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit" \
                    "${REFERENCE_RUN_DIRS}"
                ;;
            *)
                echo "Unknown POLICY_KEYS entry '${policy_key}'. Expected qwen7b, qwen4b, llama8b, qwen7b_base, qwen4b_base, or llama8b_base."
                exit 1
                ;;
        esac
    done
fi

FAILED_RUNS=()
LAUNCHED=0
SKIPPED=0

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

for i in "${!POLICY_NAMES[@]}"; do
    policy_name="${POLICY_NAMES[$i]}"
    policy_model="${POLICY_MODELS[$i]}"
    policy_run_dirs="${POLICY_RUN_DIRS[$i]}"

    if [[ "${policy_model}" == /* && ! -e "${policy_model}" ]]; then
        FAILED_RUNS+=("${policy_name} (missing_policy_model=${policy_model})")
        echo "Skipping ${policy_name}: missing policy model ${policy_model}"
        continue
    fi

    IFS=' ' read -r -a RUN_DIR_ARR <<< "${policy_run_dirs}"
    for run_name in "${RUN_DIR_ARR[@]}"; do
        run_dir="${LOCALISATION_ROOT}/${run_name}"
        if [[ -n "${OUTPUT_SUBDIR_WAS_SET}" ]]; then
            output_subdir="${OUTPUT_SUBDIR}"
        elif [[ "${policy_name}" == "qwen7b_sft" ]]; then
            output_subdir="policy_token_baselines"
        else
            output_subdir="policy_token_baselines_${policy_name}"
        fi
        output_dir="${run_dir}/${output_subdir}"
        summary_file="${output_dir}/policy_token_baselines_summary.json"

        if [[ ! -f "${run_dir}/pair_details.jsonl" ]]; then
            FAILED_RUNS+=("${policy_name}/${run_name} (missing_pair_details=${run_dir}/pair_details.jsonl)")
            echo "Skipping ${policy_name}/${run_name}: missing pair_details.jsonl"
            continue
        fi

        if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" && -f "${summary_file}" ]]; then
            if grep -q '"prob_largest_drop"' "${summary_file}"; then
                echo "Skipping ${policy_name}/${run_name}: existing ${summary_file}"
                ((SKIPPED++))
                continue
            fi
            echo "Recomputing ${policy_name}/${run_name}: existing summary lacks token-probability metrics."
        fi

        cmd=(
            bash "${RUNNER}" src/eval/localisation_policy_token_baselines.py
            --run-dir "${run_dir}"
            --output-dir "${output_dir}"
            --policy-model "${policy_model}"
            --max-length "${MAX_LENGTH}"
            --micro-batch "${MICRO_BATCH}"
            --entropy-token-chunk-size "${ENTROPY_TOKEN_CHUNK_SIZE}"
            --severity "${SEVERITY}"
            --windows "${WINDOW_ARR[@]}"
        )

        run_cmd "${policy_name}_${run_name}_policy_token_baselines" "${cmd[@]}"
        ((LAUNCHED++))
    done
done

echo -e "\n======================\nPOLICY TOKEN LOCALISATION BASELINES\n======================"
echo "Launched runs: ${LAUNCHED}"
echo "Skipped existing runs: ${SKIPPED}"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All policy-token baseline runs succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
