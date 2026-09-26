#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="localisation/synthetic_perturbations/results"
RUNS_DIR="localisation/synthetic_perturbations/runs"
mkdir -p "${RESULTS_DIR}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  echo "Python environment is missing numpy. Activate the project environment or set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

"${PYTHON_BIN}" src/plot_generators/table_localisation.py \
  --root-dir "${RUNS_DIR}" \
  --window 1 \
  --source pregenerated \
  --output-file "${RESULTS_DIR}/controlled_gsm8k_reward_hit1_token.tex"

"${PYTHON_BIN}" src/plot_generators/table_localisation.py \
  --root-dir "${RUNS_DIR}" \
  --window 7 \
  --source pregenerated \
  --output-file "${RESULTS_DIR}/controlled_gsm8k_reward_hit7_token.tex"

"${PYTHON_BIN}" src/plot_generators/table_original_synthetic_rebuttal_hit1_hit7.py \
  --old-root-dir "localisation/synthetic_perturbations" \
  --output "${RESULTS_DIR}/controlled_gsm8k_hit1_hit7_source_table.tex"

echo "Wrote synthetic localisation tables to ${RESULTS_DIR}"
