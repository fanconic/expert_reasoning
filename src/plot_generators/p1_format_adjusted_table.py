"""Build the main pass@1 table with format-adjusted correctness checks.

This is a rebuttal helper for recomputing the table in
``figures/answer_only/results_p1_temp05.txt`` directly from eval JSONLs when the
raw traces are available. Missing raw traces fall back to the existing figure
tables and are recorded in the metadata JSON.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import extract_hash_answer
from src.rewards.reward_functions import mc_answer_equal, mc_answer_equal_2


OUT_DIR = Path("figures/answer_only")
OUT_TEX = OUT_DIR / "results_p1_temp05_format_adjusted.txt"
OUT_META = OUT_DIR / "results_p1_temp05_format_adjusted_metadata.json"

MODELS = [
    ("qwen7b", r"\texttt{Qwen2.5-7B}"),
    ("llama8b", r"\texttt{Llama3.1-8B}"),
    ("qwen4b", r"\texttt{Qwen3-4B}"),
]
DATASETS = [
    ("math", r"\textbf{\textsc{GSM8K}}"),
    ("mmlu", r"\textbf{\textsc{MMLU-Pro}}"),
    ("medicine", r"\textbf{\textsc{MedReason}}"),
]
ROWS = [
    ("SFT", "SFT", ""),
    ("GAD", "GAD", ""),
    ("sparse", "R-AIRL", r"\textit{Sparse}"),
    ("partial_fixed", "R-AIRL", r"\textit{Interval}"),
    ("full", "R-AIRL", r"\textit{Dense}"),
]

VALUE_PATTERN = re.compile(r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])")
NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


@dataclass
class CellResult:
    mean: float | None
    low: float | None
    high: float | None
    source: str
    source_path: str | None = None
    n_prompts: int | None = None
    n_rows: int | None = None

    @property
    def value_str(self) -> str | None:
        if self.mean is None or self.low is None or self.high is None:
            return None
        return f"{self.mean:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _norm_question(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _question_from_prompt(prompt: Any) -> str | None:
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                return content if isinstance(content, str) else None
    if isinstance(prompt, str):
        return prompt
    return None


def _last_xml_answer(text: str) -> str | None:
    if "<answer>" not in text and "</answer>" not in text:
        return None
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def _numbers(text: str | None) -> list[Decimal]:
    values: list[Decimal] = []
    for match in NUMBER_PATTERN.finditer(text or ""):
        token = match.group(0).replace("$", "").replace(",", "")
        try:
            values.append(Decimal(token))
        except InvalidOperation:
            pass
    return values


def _final_number_equals(text: str | None, gold: str) -> bool:
    pred_nums = _numbers(text)
    gold_nums = _numbers(gold)
    return bool(pred_nums and gold_nums and pred_nums[-1] == gold_nums[-1])


def _load_gold_maps() -> dict[str, dict[str, str]]:
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    med = load_from_disk(
        "/mnt/pdata/caf83/data/expert_reasoning/"
        "medreason_corrupted_full_token_filtered_no_violations"
    )["test"]
    mmlu = load_from_disk(
        "/mnt/pdata/caf83/data/expert_reasoning/mmlu_pro_filtered"
    )["test"]

    return {
        "math": {
            _norm_question(row["question"]): extract_hash_answer(row["answer"])
            for row in gsm8k
        },
        "medicine": {
            _norm_question(row["question"]): row["answer"]
            for row in med
        },
        "mmlu": {
            _norm_question(row["question"]): row["answer"]
            for row in mmlu
        },
    }


def _adjusted_correct(domain: str, generation: str, gold: str) -> bool:
    answer = _last_xml_answer(generation)
    if domain == "math":
        return _final_number_equals(answer, gold)

    # MC domains are already fairly robust, but this allows extra text/formatting
    # inside the final answer span. If no answer tag exists, try the completion.
    predicted = answer if answer is not None else generation
    if domain == "medicine":
        return bool(mc_answer_equal(predicted, gold))
    return bool(mc_answer_equal_2(predicted, gold))


def _bootstrap_pass1(group_scores: list[float], n_boot: int = 1000) -> tuple[float, float, float]:
    arr = np.asarray(group_scores, dtype=float)
    mean = float(np.mean(arr)) if arr.size else float("nan")
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots.append(float(np.mean(arr[idx])))
    low, high = np.percentile(boots, [2.5, 97.5])
    return mean, float(low), float(high)


def _eval_path(domain: str, model: str, algo: str) -> Path | None:
    if domain == "math" and model == "qwen7b" and algo == "full":
        return Path(
            "/mnt/pdata/caf83/neurips2026/math/outputs/"
            "qwen7b_full_rebuttal_restart/best_model/"
            "eval_results_math_qwen7b_full_t0p5.jsonl"
        )
    if domain == "math" and model == "qwen7b" and algo == "partial_fixed":
        return Path(
            "/mnt/pdata/caf83/neurips2026/math/outputs/"
            "qwen7b_partial_fixed_rebuttal_restart/best_model/"
            "eval_results_math_qwen7b_partial_fixed_t0p5.jsonl"
        )
    if domain == "math" and model == "qwen7b" and algo == "GAD":
        return Path(
            "/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_gad_math/"
            "best_model/eval_results_math_qwen7b_gad_t0p5.jsonl"
        )

    if domain == "math":
        root = Path("/mnt/pdata/caf83/icml_math/outputs")
        run = f"{model}_sft" if algo == "SFT" else f"{model}_{algo}"
        variant = "sft" if algo == "SFT" else algo
        return root / run / "best_model" / f"eval_results_math_{model}_{variant}_t0p5.jsonl"

    if domain == "medicine":
        root = Path("/mnt/pdata/caf83/neurips2026/medicine/outputs")
        if algo == "SFT":
            return (
                root
                / f"{model}_sft"
                / "best_model"
                / f"eval_results_medicine_{model}_sft_t0p5.jsonl"
            )
        if algo == "GAD":
            return (
                root
                / f"{model}_gad_medicine"
                / "best_model"
                / f"eval_results_medicine_{model}_gad_t0p5.jsonl"
            )
        return root / f"{model}_{algo}" / "best_model" / "eval_results_medical_kd.jsonl"

    if domain == "mmlu":
        root = Path("/mnt/pdata/caf83/neurips2026/mmlu/outputs")
        if algo == "SFT":
            candidates = [
                root / f"{model}_sft" / "best_model" / "eval_results_mmlu_kd.jsonl",
                root / f"{model}_mmlu_sft" / "best_model" / f"eval_results_mmlu_{model}_sft_t0p5.jsonl",
            ]
            return next((p for p in candidates if p.exists()), candidates[0])
        if algo == "GAD":
            return (
                root
                / f"{model}_gad_mmlu"
                / "best_model"
                / f"eval_results_mmlu_{model}_gad_t0p5.jsonl"
            )
        candidates = [
            root / f"{model}_{algo}" / "best_model" / "eval_results_mmlu_kd.jsonl",
            root / f"{model}_mmlu_{algo}" / "best_model" / "eval_results_mmlu_kd.jsonl",
        ]
        return next((p for p in candidates if p.exists()), candidates[0])

    return None


def _fallback_table_path(domain: str, model: str, algo: str) -> tuple[Path | None, str]:
    if algo == "GAD":
        return None, ""

    base = OUT_DIR / domain / "standard"
    if algo == "SFT":
        for variant in ("sparse", "partial_fixed", "full"):
            path = base / f"{model}_{variant}" / "pass_at_k_table.txt"
            if path.exists():
                return path, "SFT"
        return None, "SFT"

    path = base / f"{model}_{algo}" / "pass_at_k_table.txt"
    return path, "AIRL"


def _extract_fallback_cell(domain: str, model: str, algo: str) -> CellResult:
    path, label = _fallback_table_path(domain, model, algo)
    if path is None or not path.exists():
        return CellResult(None, None, None, "missing")

    rows = path.read_text().replace("\n", " ").split(r"\\")
    for row in rows:
        if label not in row:
            continue
        matches = VALUE_PATTERN.findall(row)
        if not matches:
            continue
        mean_s, interval = matches[0].split(" ", 1)
        low_s, high_s = interval.strip("[]").split(",")
        return CellResult(
            float(mean_s),
            float(low_s.strip()),
            float(high_s.strip()),
            "fallback_existing_table",
            str(path),
        )
    return CellResult(None, None, None, "missing", str(path))


def _compute_from_jsonl(path: Path, domain: str, gold: dict[str, str]) -> CellResult:
    groups: OrderedDict[str, list[bool]] = OrderedDict()
    n_rows = 0
    missing_gold = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            question = _norm_question(_question_from_prompt(row.get("prompt")))
            answer = gold.get(question)
            if answer is None:
                missing_gold += 1
                continue
            generation = row.get("generation")
            if isinstance(generation, dict):
                generation_text = generation.get("content", "")
            else:
                generation_text = str(generation or "")
            groups.setdefault(question, []).append(
                _adjusted_correct(domain, generation_text, answer)
            )

    if not groups:
        return CellResult(None, None, None, "raw_jsonl_no_matched_prompts", str(path), 0, n_rows)

    group_scores = [sum(flags) / len(flags) for flags in groups.values() if flags]
    mean, low, high = _bootstrap_pass1(group_scores)
    source = "raw_jsonl_format_adjusted"
    if missing_gold:
        source += f"_missing_gold_rows_{missing_gold}"
    return CellResult(mean, low, high, source, str(path), len(group_scores), n_rows)


def _compute_all() -> dict[str, dict[str, dict[str, CellResult]]]:
    gold_maps = _load_gold_maps()
    data: dict[str, dict[str, dict[str, CellResult]]] = {
        model: {
            row_key: {domain: CellResult(None, None, None, "missing") for domain, _ in DATASETS}
            for row_key, _method, _granularity in ROWS
        }
        for model, _label in MODELS
    }

    for model, _model_label in MODELS:
        for row_key, _method, _granularity in ROWS:
            for domain, _domain_label in DATASETS:
                path = _eval_path(domain, model, row_key)
                if path is not None and path.exists():
                    data[model][row_key][domain] = _compute_from_jsonl(
                        path, domain, gold_maps[domain]
                    )
                else:
                    data[model][row_key][domain] = _extract_fallback_cell(
                        domain, model, row_key
                    )
    return data


def _mean(cell: CellResult) -> float:
    return cell.mean if cell.mean is not None else -1.0


def _format_cell(cell: CellResult, best: bool, second: bool) -> str:
    if cell.mean is None or cell.low is None or cell.high is None:
        return "-"

    mean_pp = cell.mean * 100
    half_width = (cell.high - cell.low) * 50
    mean_text = f"{mean_pp:.1f}"
    ci_text = rf"{{\scriptsize\color{{gray}}$\pm$ {half_width:.1f}}}"

    if 0 <= cell.mean < 0.20:
        return rf"\textcolor{{gray}}{{{mean_text}$^*$ {{\scriptsize $\pm$ {half_width:.1f}}}}}"
    if best:
        mean_text = rf"\textbf{{{mean_text}}}"
    elif second:
        mean_text = rf"\underline{{{mean_text}}}"
    return f"{mean_text} {ci_text}"


def _format_data(data: dict[str, dict[str, dict[str, CellResult]]]) -> dict[str, dict[str, dict[str, str]]]:
    formatted: dict[str, dict[str, dict[str, str]]] = {
        model: {row_key: {} for row_key, _method, _granularity in ROWS}
        for model, _label in MODELS
    }

    for model, _model_label in MODELS:
        for domain, _domain_label in DATASETS:
            values = sorted(
                {
                    _mean(data[model][row_key][domain])
                    for row_key, _method, _granularity in ROWS
                    if _mean(data[model][row_key][domain]) >= 0
                },
                reverse=True,
            )
            best = values[0] if values else -1.0
            second = values[1] if len(values) > 1 else -1.0
            for row_key, _method, _granularity in ROWS:
                value = _mean(data[model][row_key][domain])
                formatted[model][row_key][domain] = _format_cell(
                    data[model][row_key][domain],
                    best=value == best and value >= 0,
                    second=value == second and value >= 0,
                )
    return formatted


def _build_latex(formatted: dict[str, dict[str, dict[str, str]]]) -> str:
    total_columns = 2 + len(DATASETS) * len(MODELS)
    lines: list[str] = [
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{graphicx}, \usepackage{multirow}, \usepackage{arydshln}",
        r"\begin{table*}[h!]",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l c c c c c c c c c}",
        r"\toprule",
    ]
    dataset_header = " & ".join(
        rf"\multicolumn{{{len(MODELS)}}}{{c}}{{{label}}}"
        for _domain, label in DATASETS
    )
    lines.append(
        r"\textbf{Method} & \textbf{Granularity} & " + dataset_header + r" \\"
    )
    cmidrules = []
    for idx, _ in enumerate(DATASETS):
        start = 3 + idx * len(MODELS)
        end = start + len(MODELS) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append("".join(cmidrules))
    model_header = " & ".join(
        rf"\textbf{{\tiny{model_label}}}"
        for _domain, _domain_label in DATASETS
        for _model, model_label in MODELS
    )
    lines.extend([r"& & " + model_header + r" \\", r"\midrule"])

    def row_cells(row_key: str) -> list[str]:
        return [
            formatted[model][row_key][domain]
            for domain, _domain_label in DATASETS
            for model, _model_label in MODELS
        ]

    lines.append(r"%\rowcolor{black!6}")
    lines.append("SFT &  & " + " & ".join(row_cells("SFT")) + r" \\")
    lines.append("GAD &  & " + " & ".join(row_cells("GAD")) + r" \\")
    lines.append(rf"\cdashline{{1-{total_columns}}}[0.5pt/1.8pt]")

    rairl = [
        ("sparse", r"\textit{Sparse}"),
        ("partial_fixed", r"\textit{Interval}"),
        ("full", r"\textit{Dense}"),
    ]
    for idx, (row_key, granularity) in enumerate(rairl):
        prefix = (
            rf"\multirow{{{len(rairl)}}}{{*}}{{R-AIRL}} & {granularity} & "
            if idx == 0
            else rf"& {granularity} & "
        )
        lines.append(prefix + " & ".join(row_cells(row_key)) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\vspace{0.6em}",
            (
                r"\caption{\textbf{Held-out pass@1 with format-adjusted checking.} "
                r"\textbf{Bold} indicates the best performance, and "
                r"\underline{underlined} the second best, compared between SFT, GAD, "
                r"and R-AIRL variants in the demonstration-only setting. Values are "
                r"reported as mean $\pm$ half-width of the 95\% confidence interval "
                r"bootstrapped over the test set.}"
            ),
            r"\label{tab:p1_results_format_adjusted}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    data = _compute_all()
    formatted = _format_data(data)
    latex = _build_latex(formatted)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(latex + "\n", encoding="utf-8")
    OUT_META.write_text(
        json.dumps(
            {
                "checker": {
                    "math": "final numeric value in last <answer> span equals GSM8K gold number",
                    "mmlu": "last <answer> span passed through mc_answer_equal_2",
                    "medicine": "last <answer> span passed through A-D mc_answer_equal",
                },
                "results": {
                    model: {
                        row_key: {
                            domain: asdict(cell)
                            for domain, cell in domain_map.items()
                        }
                        for row_key, domain_map in row_map.items()
                    }
                    for model, row_map in data.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(latex)
    print(f"\nWrote {OUT_TEX}")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
