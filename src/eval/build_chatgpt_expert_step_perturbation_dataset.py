"""Build ChatGPT-edited GSM8K perturbations from expert traces.

This is a thin expert-trace wrapper around
``build_chatgpt_step_perturbation_dataset.py``. The underlying builder performs
the Azure OpenAI call, validation, resume logic, and JSONL writing.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import build_chatgpt_step_perturbation_dataset as base  # noqa: E402

base.DEFAULT_SOURCE = (
    PROJECT_ROOT / "localisation/qwen7b_full_localisation_expert/pair_details.jsonl"
)
base.DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "localisation/expert_step_perturbations/"
    / "gsm8k_expert_step_perturbations_smoke.jsonl"
)


if __name__ == "__main__":
    base.main()
