"""Patch the main Best-of-16 reranking table with adjusted answer checks.

GSM8K and MedReason columns are recomputed from raw eval JSONLs. The GAD
MMLU-Pro row is also recomputed from raw eval JSONLs; other MMLU-Pro columns
are copied from ``figures/answer_only/reranking_results_main.txt``.
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
from typing import Any, Callable

import numpy as np
from datasets import load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import extract_hash_answer
from src.plot_generators.plot_helpers import (
    aggregate_dense_rewards,
    bootstrap_ci,
    compute_pass_at_k,
    compute_success_at_k_from_scores,
)
from src.rewards.reward_functions import mc_answer_equal, mc_answer_equal_2


OLD_TABLE = Path("figures/answer_only/reranking_results_main.txt")
OUT_TABLE = Path("figures/answer_only/reranking_results_main_gsm8k_format_adjusted.txt")
OUT_META = Path("figures/answer_only/reranking_results_main_gsm8k_format_adjusted_metadata.json")

MODELS = [
    ("qwen7b", r"\texttt{Qwen2.5}"),
    ("llama8b", r"\texttt{Llama3.1}"),
    ("qwen4b", r"\texttt{Qwen3}"),
]
ALGOS = [
    ("sparse", r"\textit{Sparse}"),
    ("partial_fixed", r"\textit{Interval}"),
    ("full", r"\textit{Dense}"),
]

NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
DELTA_PATTERN = re.compile(r"([+-]\d+(?:\.\d+)?)")


@dataclass
class Metric:
    mean: float
    low: float
    high: float

    @property
    def half_width_pp(self) -> float:
        return (self.high - self.low) * 50.0


@dataclass
class RunMetrics:
    random: Metric
    reward: Metric
    oracle: Metric
    delta: float
    oracle_delta: float
    source_path: str
    n_prompts: int
    n_rows: int


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
    out: list[Decimal] = []
    for match in NUMBER_PATTERN.finditer(text or ""):
        token = match.group(0).replace("$", "").replace(",", "")
        try:
            out.append(Decimal(token))
        except InvalidOperation:
            pass
    return out


def _final_number_equals(text: str | None, gold: str) -> bool:
    pred_nums = _numbers(text)
    gold_nums = _numbers(gold)
    return bool(pred_nums and gold_nums and pred_nums[-1] == gold_nums[-1])


def _gsm8k_gold() -> dict[str, str]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return {
        _norm_question(row["question"]): extract_hash_answer(row["answer"])
        for row in ds
    }


def _mmlu_gold() -> dict[str, str]:
    ds = load_from_disk("/mnt/pdata/caf83/data/expert_reasoning/mmlu_pro_filtered")["test"]
    return {_norm_question(row["question"]): row["answer"] for row in ds}


def _medicine_gold() -> dict[str, str]:
    ds = load_from_disk(
        "/mnt/pdata/caf83/data/expert_reasoning/"
        "medreason_corrupted_full_token_filtered_no_violations"
    )["test"]
    return {_norm_question(row["question"]): row["answer"] for row in ds}


def _gsm8k_correct(generation_text: str, gold: str) -> bool:
    return _final_number_equals(_last_xml_answer(generation_text), gold)


def _mmlu_correct(generation_text: str, gold: str) -> bool:
    answer = _last_xml_answer(generation_text)
    predicted = answer if answer is not None else generation_text
    return bool(mc_answer_equal_2(predicted, gold))


def _medicine_correct(generation_text: str, gold: str) -> bool:
    answer = _last_xml_answer(generation_text)
    predicted = answer if answer is not None else generation_text
    return bool(mc_answer_equal(predicted, gold))


def _eval_path(model: str, algo: str) -> Path:
    if model == "qwen7b" and algo == "full":
        return Path(
            "/mnt/pdata/caf83/neurips2026/math/outputs/"
            "qwen7b_full_rebuttal_restart/best_model/"
            "eval_results_math_qwen7b_full_t0p5.jsonl"
        )
    if model == "qwen7b" and algo == "partial_fixed":
        return Path(
            "/mnt/pdata/caf83/neurips2026/math/outputs/"
            "qwen7b_partial_fixed_rebuttal_restart/best_model/"
            "eval_results_math_qwen7b_partial_fixed_t0p5.jsonl"
        )
    root = Path("/mnt/pdata/caf83/icml_math/outputs")
    return root / f"{model}_{algo}" / "best_model" / f"eval_results_math_{model}_{algo}_t0p5.jsonl"


def _medicine_eval_path(model: str, algo: str) -> Path:
    root = Path("/mnt/pdata/caf83/neurips2026/medicine/outputs")
    return root / f"{model}_{algo}" / "best_model" / "eval_results_medical_kd.jsonl"


def _gad_path() -> Path:
    return Path(
        "/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_gad_math/"
        "best_model/eval_results_math_qwen7b_gad_t0p5.jsonl"
    )


def _gad_mmlu_path() -> Path:
    return Path(
        "/mnt/pdata/caf83/neurips2026/mmlu/outputs/qwen7b_gad_mmlu/"
        "best_model/eval_results_mmlu_qwen7b_gad_t0p5.jsonl"
    )


def _gad_medicine_path() -> Path:
    return Path(
        "/mnt/pdata/caf83/neurips2026/medicine/outputs/qwen7b_gad_medicine/"
        "best_model/eval_results_medicine_qwen7b_gad_t0p5.jsonl"
    )


def _score_from_reward_trace(values: Any, *, model: str) -> float:
    if not isinstance(values, list) or not values:
        return float("nan")
    rewards = values
    if "qwen" in model:
        # Mirror read_and_enhance() for Qwen traces.
        rewards = [rewards[0]] + rewards
    return aggregate_dense_rewards(rewards, mode="discounted_mean", gamma=0.95)


def _metric_ci(metric_fn, flags, ks, scores=None) -> dict[int, tuple[float, float]]:
    if scores is None:
        return bootstrap_ci(metric_fn, flags, ks)
    return bootstrap_ci(metric_fn, flags, ks, all_scores=scores)


def _bootstrap_mean(values: list[float], n_boot: int = 1000) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr)) if arr.size else float("nan")
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots.append(float(np.mean(arr[idx])))
    low, high = np.percentile(boots, [2.5, 97.5])
    return mean, float(low), float(high)


def _pass_at_k_value(flags: list[bool], k: int) -> float:
    n = len(flags)
    m = sum(flags)
    if m == 0 or k > n:
        return 0.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def _reward_selected_success(flags: list[bool], scores: list[float]) -> float:
    if not flags or not scores:
        return 0.0
    order = np.argsort(np.asarray(scores, dtype=float))[::-1]
    return float(bool(flags[int(order[0])]))


def _compute_run(
    path: Path,
    *,
    model: str,
    gold: dict[str, str],
    correct_fn: Callable[[str, str], bool],
) -> RunMetrics:
    groups: OrderedDict[str, dict[str, list[Any]]] = OrderedDict()
    n_rows = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            question = _norm_question(_question_from_prompt(row.get("prompt")))
            answer = gold.get(question)
            if answer is None:
                continue
            generation = row.get("generation")
            if isinstance(generation, dict):
                generation_text = generation.get("content", "")
            else:
                generation_text = str(generation or "")
            correct = correct_fn(generation_text, answer)
            score = _score_from_reward_trace(row.get("reward_model_score"), model=model)
            entry = groups.setdefault(question, {"flags": [], "scores": []})
            entry["flags"].append(bool(correct))
            entry["scores"].append(score)

    flags = [entry["flags"] for entry in groups.values()]
    scores = [entry["scores"] for entry in groups.values()]

    random_vals = [sum(row_flags) / len(row_flags) for row_flags in flags if row_flags]
    reward_vals = [
        _reward_selected_success(row_flags, row_scores)
        for row_flags, row_scores in zip(flags, scores)
    ]
    oracle_vals = [_pass_at_k_value(row_flags, 16) for row_flags in flags]

    random = Metric(*_bootstrap_mean(random_vals))
    reward = Metric(*_bootstrap_mean(reward_vals))
    oracle = Metric(*_bootstrap_mean(oracle_vals))
    return RunMetrics(
        random=random,
        reward=reward,
        oracle=oracle,
        delta=reward.mean - random.mean,
        oracle_delta=oracle.mean - random.mean,
        source_path=str(path),
        n_prompts=len(flags),
        n_rows=n_rows,
    )


def _format_metric(metric: Metric) -> str:
    return (
        f"{metric.mean * 100:.1f} "
        rf"{{\scriptsize\color{{gray}}$\pm$ {metric.half_width_pp:.1f}}}"
    )


def _format_delta(delta: float) -> str:
    delta_pp = delta * 100
    if abs(delta_pp) < 0.05:
        delta_pp = 0.0
        return rf"{{\color{{gray}}{delta_pp:+.1f}}}"
    if delta_pp > 0:
        return rf"{{\color{{insightteal}}\textbf{{$\uparrow${delta_pp:+.1f}}}}}"
    if delta_pp < 0:
        return rf"{{\color{{purple}}$\downarrow${delta_pp:+.1f}}}"
    return rf"{{\color{{gray}}{delta_pp:+.1f}}}"


def _compact_legacy_delta(cell: str) -> str:
    match = DELTA_PATTERN.search(cell)
    if match is None:
        return cell
    return _format_delta(float(match.group(1)) / 100.0)


def _delta_from_cell(cell: str) -> float:
    match = DELTA_PATTERN.search(cell)
    return float(match.group(1)) / 100.0 if match else float("nan")


def _average_delta(*deltas: float) -> float:
    finite = [delta for delta in deltas if math.isfinite(delta)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _parse_old_non_gsm() -> dict[tuple[str, str], dict[str, list[str]]]:
    current_model: str | None = None
    parsed: dict[tuple[str, str], dict[str, list[str]]] = {}
    label_to_model = {
        r"\texttt{Qwen2.5-7B}": "qwen7b",
        r"\texttt{Llama3.1-8B}": "llama8b",
        r"\texttt{Qwen3-4B}": "qwen4b",
    }
    algo_labels = {
        r"\textit{Sparse}": "sparse",
        r"\textit{Interval}": "partial_fixed",
        r"\textit{Dense}": "full",
    }

    for raw in OLD_TABLE.read_text(encoding="utf-8").splitlines():
        if "&" not in raw or r"\textit{" not in raw:
            continue
        model_match = re.search(r"\\multirow\{3\}\{\*\}\{([^}]*(?:}[^}]*)*)\}", raw)
        if model_match:
            label = model_match.group(1)
            for model_label, model_key in label_to_model.items():
                if model_label in label:
                    current_model = model_key
                    break
        if current_model is None:
            continue
        algo_key = None
        for algo_label, candidate in algo_labels.items():
            if algo_label in raw:
                algo_key = candidate
                break
        if algo_key is None:
            continue
        cells = [cell.strip() for cell in raw.rstrip("\\").split("&")]
        # Layout: model, granularity, GSM random/reward/delta, MMLU 3, Med 3.
        if len(cells) < 11:
            continue
        mmlu = cells[5:8]
        medicine = cells[8:11]
        mmlu[2] = _compact_legacy_delta(mmlu[2])
        medicine[2] = _compact_legacy_delta(medicine[2])
        parsed[(current_model, algo_key)] = {"mmlu": mmlu, "medicine": medicine}
    return parsed


def _build_table(
    gsm: dict[tuple[str, str], RunMetrics],
    medicine: dict[tuple[str, str], RunMetrics],
    gad: dict[str, RunMetrics],
    old_non_gsm: dict[tuple[str, str], dict[str, list[str]]],
) -> str:
    lines = [
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{graphicx}, \usepackage{multirow}, \usepackage{arydshln}",
        r"\begin{table*}[h!]",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}ll*{10}{c}@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Granularity} & \multicolumn{3}{c}{\textbf{\textsc{GSM8K}}} & \multicolumn{3}{c}{\textbf{\textsc{MMLU-Pro}}} & \multicolumn{3}{c}{\textbf{\textsc{MedReason}}} & \textbf{Avg.} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-12}",
        r"& & \textbf{Random} & \textbf{Reward} & $\mathbf{\Delta}$ & \textbf{Random} & \textbf{Reward} & $\mathbf{\Delta}$ & \textbf{Random} & \textbf{Reward} & $\mathbf{\Delta}$ & $\mathbf{\Delta}$ \\",
        r"\midrule",
    ]

    gad_avg_delta = _average_delta(
        gad["math"].delta,
        gad["mmlu"].delta,
        gad["medicine"].delta,
    )
    gad_cells = [
        _format_metric(gad["math"].random),
        _format_metric(gad["math"].reward),
        _format_delta(gad["math"].delta),
        _format_metric(gad["mmlu"].random),
        _format_metric(gad["mmlu"].reward),
        _format_delta(gad["mmlu"].delta),
        _format_metric(gad["medicine"].random),
        _format_metric(gad["medicine"].reward),
        _format_delta(gad["medicine"].delta),
        _format_delta(gad_avg_delta),
    ]
    lines.append(r"\texttt{Qwen2.5} & \textit{GAD} & " + " & ".join(gad_cells) + r" \\")
    lines.append(r"\cdashline{1-12}[0.5pt/1.8pt]")

    for model_idx, (model, model_label) in enumerate(MODELS):
        for algo_idx, (algo, algo_label) in enumerate(ALGOS):
            run = gsm[(model, algo)]
            gsm_cells = [
                _format_metric(run.random),
                _format_metric(run.reward),
                _format_delta(run.delta),
            ]
            medicine_run = medicine[(model, algo)]
            medicine_cells = [
                _format_metric(medicine_run.random),
                _format_metric(medicine_run.reward),
                _format_delta(medicine_run.delta),
            ]
            non_gsm = old_non_gsm[(model, algo)]
            avg_delta = _average_delta(
                run.delta,
                _delta_from_cell(non_gsm["mmlu"][2]),
                medicine_run.delta,
            )
            row_cells = gsm_cells + non_gsm["mmlu"] + medicine_cells + [
                _format_delta(avg_delta)
            ]
            prefix = (
                rf"\multirow{{3}}{{*}}{{{model_label}}} & {algo_label} & "
                if algo_idx == 0
                else rf" & {algo_label} & "
            )
            lines.append(prefix + " & ".join(row_cells) + r" \\")
        if model_idx != len(MODELS) - 1:
            lines.append(r"\cdashline{1-12}[0.5pt/1.8pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\vspace{0.6em}",
            (
                r"\caption{\textbf{Best-of-16 reranking performance (\%).} "
                r"GSM8K columns use the format-adjusted numeric checker; MedReason "
                r"columns and the GAD MMLU-Pro cells use the adjusted multiple-choice "
                r"checker. Other MMLU-Pro columns are kept from the previous table. "
                r"$\Delta$ reports percentage-point change (reward minus random); "
                r"Avg. $\Delta$ averages the three dataset-level deltas within each row.}"
            ),
            r"\label{tab:reranking_main_gsm8k_format_adjusted}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    gsm_gold = _gsm8k_gold()
    medicine_gold = _medicine_gold()
    gsm: dict[tuple[str, str], RunMetrics] = {}
    medicine: dict[tuple[str, str], RunMetrics] = {}
    for model, _model_label in MODELS:
        for algo, _algo_label in ALGOS:
            gsm[(model, algo)] = _compute_run(
                _eval_path(model, algo),
                model=model,
                gold=gsm_gold,
                correct_fn=_gsm8k_correct,
            )
            medicine[(model, algo)] = _compute_run(
                _medicine_eval_path(model, algo),
                model=model,
                gold=medicine_gold,
                correct_fn=_medicine_correct,
            )

    gad = {
        "math": _compute_run(
            _gad_path(),
            model="qwen7b",
            gold=gsm_gold,
            correct_fn=_gsm8k_correct,
        ),
        "mmlu": _compute_run(
            _gad_mmlu_path(),
            model="qwen7b",
            gold=_mmlu_gold(),
            correct_fn=_mmlu_correct,
        ),
        "medicine": _compute_run(
            _gad_medicine_path(),
            model="qwen7b",
            gold=medicine_gold,
            correct_fn=_medicine_correct,
        ),
    }
    old_non_gsm = _parse_old_non_gsm()
    table = _build_table(gsm, medicine, gad, old_non_gsm)

    OUT_TABLE.write_text(table + "\n", encoding="utf-8")
    OUT_META.write_text(
        json.dumps(
            {
                "checker": {
                    "gsm8k": "final numeric value in last <answer> equals gold",
                    "mmlu": "last <answer> span passed through mc_answer_equal_2",
                    "medicine": "last <answer> span passed through A-D mc_answer_equal",
                },
                "gsm8k": {
                    f"{model}_{algo}": asdict(run)
                    for (model, algo), run in gsm.items()
                },
                "medicine": {
                    f"{model}_{algo}": asdict(run)
                    for (model, algo), run in medicine.items()
                },
                "gad_qwen7b": {domain: asdict(run) for domain, run in gad.items()},
                "non_gsm_columns": (
                    "AIRL MMLU-Pro metric cells copied from "
                    "figures/answer_only/reranking_results_main.txt; delta cells "
                    "compacted without changing values. AIRL MedReason recomputed "
                    "from raw JSONL with adjusted checker."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(table)
    for domain, run in gad.items():
        print(f"\nGAD Qwen2.5-7B {domain}")
        print(f"Random: {_format_metric(run.random)}")
        print(f"Reward: {_format_metric(run.reward)}")
        print(f"Delta: {run.delta * 100:+.1f} pp")
        print(f"Oracle best-of-16: {_format_metric(run.oracle)}")
        print(f"Potential oracle gain over random: {run.oracle_delta * 100:+.1f} pp")
    print(f"\nWrote {OUT_TABLE}")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
