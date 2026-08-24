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
: "${WINDOWS:=1 7}"
: "${MAX_EXAMPLES:=0}"
: "${START_INDEX:=0}"
: "${MAX_LENGTH:=1124}"
: "${POLICY_MICRO_BATCH:=4}"
: "${REWARD_MICRO_BATCH:=8}"
: "${ENTROPY_TOKEN_CHUNK_SIZE:=32}"
: "${PARTIAL_FIXED_N:=15}"
: "${REWARD_GPU_MEMORY_UTILIZATION:=0.4}"
: "${TARGET_POSITION_SOURCE:=target_char_span}"
: "${REQUIRE_NATURAL_WRONG_FINAL_ANSWER_MISMATCH:=1}"
: "${SKIP_EXISTING:=1}"
: "${FORCE:=0}"
: "${INCLUDE_TEXT:=0}"
: "${LIVE_LOG:=1}"
: "${LOG_TAIL_LINES:=8}"
: "${QWEN7B_FULL_REWARD_CHECKPOINT:=/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_full_rebuttal_restart/best_model}"
: "${QWEN7B_PARTIAL_FIXED_REWARD_CHECKPOINT:=/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_partial_fixed_rebuttal_restart/best_model}"

RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
IFS=' ' read -r -a WINDOW_ARR <<< "${WINDOWS}"

DATASETS=(
    "expert_step|localisation/expert_step_perturbations/gsm8k_expert_step_perturbations_full.jsonl|localisation/expert_step_perturbations/scores"
    "natural_wrong_sft|localisation/natural_wrong_sft/gsm8k_qwen7b_sft_wrong_step_labels_full.jsonl|localisation/natural_wrong_sft/scores"
)

FAILED_RUNS=()
LAUNCHED=0
SKIPPED=0

policy_model_path() {
    local model="$1"
    case "${model}" in
        qwen7b) echo "/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model" ;;
        qwen4b) echo "/mnt/pdata/caf83/icml_math/outputs/qwen4b_sft/best_model" ;;
        llama8b) echo "/mnt/pdata/caf83/icml_math/outputs/llama8b_sft/best_model" ;;
        *)
            echo "Unknown policy model: ${model}" >&2
            return 1
            ;;
    esac
}

reward_config_path() {
    local model="$1"
    case "${model}" in
        qwen7b) echo "configs/math/qwen7b/irl_eval.yaml" ;;
        qwen4b) echo "configs/math/qwen4b/irl_eval.yaml" ;;
        llama8b) echo "configs/math/llama8b/irl_eval.yaml" ;;
        *)
            echo "Unknown reward model: ${model}" >&2
            return 1
            ;;
    esac
}

reward_checkpoint_dir() {
    local model="$1"
    local density="$2"
    if [[ "${model}" == "qwen7b" && "${density}" == "full" ]]; then
        echo "${QWEN7B_FULL_REWARD_CHECKPOINT}"
        return
    fi
    if [[ "${model}" == "qwen7b" && "${density}" == "partial_fixed" ]]; then
        echo "${QWEN7B_PARTIAL_FIXED_REWARD_CHECKPOINT}"
        return
    fi
    echo "/mnt/pdata/caf83/icml_math/outputs/${model}_${density}/best_model"
}

summary_source_matches() {
    local summary_file="$1"
    local source_key="$2"
    local expected="$3"
    python - "${summary_file}" "${source_key}" "${expected}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
source_key = sys.argv[2]
expected = sys.argv[3]
try:
    with summary_path.open("r") as f:
        summary = json.load(f)
except Exception:
    sys.exit(1)
sys.exit(0 if str(summary.get(source_key)) == expected else 1)
PY
}

run_cmd() {
    local log_dir="$1"
    local label="$2"
    shift 2
    local log_file="${log_dir}/${label}.log"
    echo -e "\n▶ ${label}"
    echo "  GPU: ${GPU_NUM}"
    echo "  log: ${log_file}"
    echo "  cmd: $*"
    local rc
    if [[ "${LIVE_LOG}" == "1" ]]; then
        stdbuf -oL -eL "$@" 2>&1 | tee "${log_file}"
        rc=${PIPESTATUS[0]}
    else
        "$@" > "${log_file}" 2>&1
        rc=$?
    fi
    if [[ ${rc} -ne 0 ]]; then
        FAILED_RUNS+=("${label} (exit=${rc})")
        echo "  ✗ FAILED; last log lines:"
        tail -n 40 "${log_file}" || true
    else
        echo "  ✓ OK; last log lines:"
        tail -n "${LOG_TAIL_LINES}" "${log_file}" || true
    fi
    return "${rc}"
}

run_policy() {
    local dataset_name="$1"
    local pairs_jsonl="$2"
    local output_root="$3"
    local log_dir="$4"
    local model="$5"

    local policy_name="${model}_sft"
    local model_path
    model_path="$(policy_model_path "${model}")" || return
    local output_dir="${output_root}/${policy_name}_policy_token_baselines"
    local summary_file="${output_dir}/policy_token_baselines_summary.json"

    echo -e "\nPreparing policy-token scoring: dataset=${dataset_name}, model=${model}"
    echo "  pairs=${pairs_jsonl}"
    echo "  model_path=${model_path}"
    echo "  output_dir=${output_dir}"

    if [[ "${model_path}" == /* && ! -e "${model_path}" ]]; then
        FAILED_RUNS+=("${dataset_name}_${policy_name} (missing_model=${model_path})")
        echo "Skipping ${dataset_name}/${policy_name}: missing policy model ${model_path}"
        return
    fi
    if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" && -f "${summary_file}" ]]; then
        if grep -q '"prob_largest_drop"' "${summary_file}"; then
            if summary_source_matches "${summary_file}" "source_pair_details" "${pairs_jsonl}"; then
                echo "Skipping ${dataset_name}/${policy_name}: existing ${summary_file}"
                ((SKIPPED++))
                return
            fi
            echo "Recomputing ${dataset_name}/${policy_name}: existing summary used a different input file."
        else
            echo "Recomputing ${dataset_name}/${policy_name}: existing summary lacks token-probability metrics."
        fi
    fi

    local cmd=(
        bash "${RUNNER}" src/eval/localisation_policy_token_baselines.py
        --pair-details "${pairs_jsonl}"
        --output-dir "${output_dir}"
        --policy-model "${model_path}"
        --max-length "${MAX_LENGTH}"
        --max-examples "${MAX_EXAMPLES}"
        --start-index "${START_INDEX}"
        --micro-batch "${POLICY_MICRO_BATCH}"
        --entropy-token-chunk-size "${ENTROPY_TOKEN_CHUNK_SIZE}"
        --all-severities
        --no-require-table-valid
        --target-position-source "${TARGET_POSITION_SOURCE}"
        --windows "${WINDOW_ARR[@]}"
    )
    if [[ "${INCLUDE_TEXT}" == "1" ]]; then
        cmd+=(--include-text)
    fi

    run_cmd "${log_dir}" "${dataset_name}_${policy_name}_policy_token_baselines_gpu${GPU_NUM}" "${cmd[@]}"
    ((LAUNCHED++))
}

run_reward() {
    local dataset_name="$1"
    local pairs_jsonl="$2"
    local output_root="$3"
    local log_dir="$4"
    local model="$5"
    local density="$6"

    local config_path
    config_path="$(reward_config_path "${model}")" || return
    local checkpoint_dir
    checkpoint_dir="$(reward_checkpoint_dir "${model}" "${density}")"
    local output_dir="${output_root}/${model}_${density}_reward_localisation"
    local summary_file="${output_dir}/summary.json"

    echo -e "\nPreparing reward scoring: dataset=${dataset_name}, model=${model}, density=${density}"
    echo "  pairs=${pairs_jsonl}"
    echo "  config=${config_path}"
    echo "  checkpoint=${checkpoint_dir}"
    echo "  output_dir=${output_dir}"

    if [[ ! -f "${checkpoint_dir}/reward_model/adapter_config.json" ]]; then
        FAILED_RUNS+=("${dataset_name}_${model}_${density} (missing_checkpoint=${checkpoint_dir})")
        echo "Skipping ${dataset_name}/${model}_${density}: missing reward checkpoint ${checkpoint_dir}"
        return
    fi
    if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" && -f "${summary_file}" ]]; then
        if ! summary_source_matches "${summary_file}" "source_pairs_jsonl" "${pairs_jsonl}"; then
            echo "Recomputing ${dataset_name}/${model}_${density}: existing summary used a different input file."
        elif [[ "${model}" == "qwen7b" && "${density}" == "full" ]] \
            && ! grep -q "\"checkpoint_dir\": \"${checkpoint_dir}\"" "${summary_file}"; then
            echo "Recomputing ${dataset_name}/${model}_${density}: existing summary used a different checkpoint."
        else
            echo "Skipping ${dataset_name}/${model}_${density}: existing ${summary_file}"
            ((SKIPPED++))
            return
        fi
    fi

    local cmd=(
        bash "${RUNNER}" src/eval/localisation_reward_on_pairs.py
        --pairs-jsonl "${pairs_jsonl}"
        --output-dir "${output_dir}"
        --config "${config_path}"
        --checkpoint-dir "${checkpoint_dir}"
        --dense-reward-mode "${density}"
        --dense-partial-fixed-n "${PARTIAL_FIXED_N}"
        --reward-gpu-memory-utilization "${REWARD_GPU_MEMORY_UTILIZATION}"
        --max-examples "${MAX_EXAMPLES}"
        --start-index "${START_INDEX}"
        --max-micro-batch "${REWARD_MICRO_BATCH}"
        --target-position-source "${TARGET_POSITION_SOURCE}"
        --windows "${WINDOW_ARR[@]}"
    )
    if [[ "${INCLUDE_TEXT}" == "1" ]]; then
        cmd+=(--include-text)
    fi

    run_cmd "${log_dir}" "${dataset_name}_${model}_${density}_reward_localisation_gpu${GPU_NUM}" "${cmd[@]}"
    ((LAUNCHED++))
}

prepare_valid_pairs() {
    local dataset_name="$1"
    local raw_pairs_jsonl="$2"
    local output_root="$3"
    local input_dir="${output_root}/_inputs"
    local filter_suffix="valid_target_char_span"
    if [[ "${dataset_name}" == "natural_wrong_sft" && "${REQUIRE_NATURAL_WRONG_FINAL_ANSWER_MISMATCH}" == "1" ]]; then
        filter_suffix="${filter_suffix}_actual_wrong_answer"
    fi
    local filtered_pairs="${input_dir}/${dataset_name}_${filter_suffix}.jsonl"
    mkdir -p "${input_dir}"

    if [[ "${FORCE}" != "1" && -f "${filtered_pairs}" ]]; then
        echo "${filtered_pairs}"
        return
    fi

    python - "${raw_pairs_jsonl}" "${filtered_pairs}" "${dataset_name}" "${REQUIRE_NATURAL_WRONG_FINAL_ANSWER_MISMATCH}" <<'PY'
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dataset_name = sys.argv[3]
require_natural_answer_mismatch = sys.argv[4] == "1"

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}|boxed\{([^{}]+)\}", re.IGNORECASE)
SIMPLE_NUM_RE = re.compile(r"^\s*[$]?[-+]?\d[\d,]*(?:\.\d+)?\s*%?\s*$")


def _normalized_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _decimal(value):
    raw = str(value or "").strip().replace("$", "").replace(",", "")
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1].strip()
    try:
        parsed = Decimal(raw)
    except InvalidOperation:
        return None
    if percent:
        parsed /= Decimal("100")
    return parsed.normalize()


def _last_boxed(value):
    matches = list(BOX_RE.finditer(value or ""))
    if not matches:
        return None
    match = matches[-1]
    return (match.group(1) or match.group(2) or "").strip()


def _extract_answer(text):
    answer_matches = list(ANSWER_RE.finditer(text or ""))
    if answer_matches:
        answer = answer_matches[-1].group(1).strip()
        boxed = _last_boxed(answer)
        return boxed if boxed is not None else answer
    boxed = _last_boxed(text or "")
    return boxed if boxed is not None else ""


def _comparable_answer(value):
    value = str(value or "").strip()
    boxed = _last_boxed(value)
    if boxed is not None:
        value = boxed
    if SIMPLE_NUM_RE.match(value):
        parsed = _decimal(value)
        if parsed is not None:
            return ("num", parsed)
    return ("text", _normalized_text(value))


def _same_final_answer(predicted, gold):
    pred_kind, pred_value = _comparable_answer(predicted)
    gold_kind, gold_value = _comparable_answer(gold)
    if pred_kind == "num" and gold_kind == "num":
        return pred_value == gold_value
    return _normalized_text(predicted) == _normalized_text(gold)


def _is_actual_wrong_final_answer(row, pert):
    gold = str(row.get("gold_answer") or row.get("answer") or "").strip()
    predicted = _extract_answer(pert)
    if not gold or not predicted:
        return False
    return not _same_final_answer(predicted, gold)


rows = 0
kept = 0
answer_filtered = 0
with src.open("r") as f_in, dst.open("w") as f_out:
    for line in f_in:
        raw = line.strip()
        if not raw:
            continue
        rows += 1
        row = json.loads(raw)
        span = row.get("target_char_span")
        clean = row.get("clean_text")
        pert = row.get("pert_text") or row.get("wrong_text")
        valid_span = (
            isinstance(span, list)
            and len(span) == 2
            and isinstance(span[0], int)
            and isinstance(span[1], int)
            and span[1] > span[0]
        )
        if row.get("error") is not None or not isinstance(clean, str) or not isinstance(pert, str) or not valid_span:
            continue
        if (
            dataset_name == "natural_wrong_sft"
            and require_natural_answer_mismatch
            and not _is_actual_wrong_final_answer(row, pert)
        ):
            answer_filtered += 1
            continue
        row["pert_text"] = pert
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1
extra = f", final-answer-filtered {answer_filtered}" if answer_filtered else ""
print(f"Prepared {dst}: kept {kept}/{rows} rows{extra}", file=sys.stderr)
PY
    echo "${filtered_pairs}"
}

score_dataset() {
    local dataset_spec="$1"
    local dataset_name raw_pairs_jsonl output_root
    IFS='|' read -r dataset_name raw_pairs_jsonl output_root <<< "${dataset_spec}"
    local log_dir="${output_root}/logs"
    mkdir -p "${output_root}" "${log_dir}"

    if [[ ! -f "${raw_pairs_jsonl}" ]]; then
        FAILED_RUNS+=("${dataset_name} (missing_pairs=${raw_pairs_jsonl})")
        echo "Missing pairs JSONL for ${dataset_name}: ${raw_pairs_jsonl}"
        return
    fi
    local pairs_jsonl
    pairs_jsonl="$(prepare_valid_pairs "${dataset_name}" "${raw_pairs_jsonl}" "${output_root}")"

    echo -e "\n======================"
    echo "SCORING DATASET: ${dataset_name}"
    echo "======================"
    echo "GPU: ${GPU_NUM}"
    echo "Raw pairs: ${raw_pairs_jsonl}"
    echo "Valid-span pairs: ${pairs_jsonl}"
    echo "Outputs: ${output_root}"
    echo "Logs: ${log_dir}"
    echo "Windows: ${WINDOWS}"
    echo "Max examples: ${MAX_EXAMPLES}"
    echo "Target position source: ${TARGET_POSITION_SOURCE}"

    for model in qwen7b qwen4b llama8b; do
        run_policy "${dataset_name}" "${pairs_jsonl}" "${output_root}" "${log_dir}" "${model}"
    done

    for model in qwen7b qwen4b llama8b; do
        for density in full partial_fixed; do
            run_reward "${dataset_name}" "${pairs_jsonl}" "${output_root}" "${log_dir}" "${model}" "${density}"
        done
    done
}

if [[ ! -f "${RUNNER}" ]]; then
    echo "Missing GPU runner: ${RUNNER}"
    exit 1
fi

echo "======================"
echo "ONE-GPU NATURAL-ERROR SCORING"
echo "======================"
echo "GPU_NUM=${GPU_NUM}"
echo "Runner=${RUNNER}"
echo "Qwen7B dense reward checkpoint=${QWEN7B_FULL_REWARD_CHECKPOINT}"
echo "Qwen7B interval reward checkpoint=${QWEN7B_PARTIAL_FIXED_REWARD_CHECKPOINT}"
echo "Natural wrong SFT requires actual final-answer mismatch=${REQUIRE_NATURAL_WRONG_FINAL_ANSWER_MISMATCH}"
echo "Policy scoring: SFT log-probability and entropy for qwen7b/qwen4b/llama8b."
echo "Reward scoring: dense full and fixed interval for qwen7b/qwen4b/llama8b."

for dataset_spec in "${DATASETS[@]}"; do
    score_dataset "${dataset_spec}"
done

echo -e "\n======================"
echo "ONE-GPU NATURAL-ERROR SCORING SUMMARY"
echo "======================"
echo "Launched runs: ${LAUNCHED}"
echo "Skipped existing runs: ${SKIPPED}"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All scoring runs succeeded."
else
    echo "Failures: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
