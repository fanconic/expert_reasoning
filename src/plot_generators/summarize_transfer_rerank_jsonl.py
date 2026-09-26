"""Summarize corrected transfer reranking JSONLs.

This is a lightweight companion to the plot generator for the rebuttal-time
transfer reruns. It reads pregenerated SFT traces that have been rescored by
``src/eval/score_pregenerated_trace_signals.py`` and reports Best-of-N reward
reranking deltas.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable


DEFAULT_INPUT_ROOT = Path("/mnt/pdata/caf83/neurips2026/transfer_rerank_corrected_qwen7b_sft")
DEFAULT_OUTPUT_DIR = Path("figures/transferability_ablation_corrected")
DEFAULT_DOMAINS = ("math", "mmlu", "medicine")
DEFAULT_ARCHES = ("llama8b", "qwen4b", "qwen7b")
DOMAIN_LABELS = {
    "math": "GSM8K",
    "mmlu": "MMLU-Pro",
    "medicine": "MedReason",
}
ARCH_LABELS = {
    "llama8b": "Llama3.1-8B",
    "qwen4b": "Qwen3-4B",
    "qwen7b": "Qwen2.5-7B",
}
NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


@dataclass
class Cell:
    policy_dataset: str
    reward_dataset: str
    reward_arch: str
    variant: str
    path: str
    n_prompts: int
    n_rows: int
    baseline: float
    baseline_low: float
    baseline_high: float
    reward: float
    reward_low: float
    reward_high: float
    delta: float
    correctness_source: str
    reward_checkpoint_dir: str | None
    reward_adapter_dir: str | None
    trace_file: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-datasets", type=str, default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--reward-datasets", type=str, default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--reward-arches", type=str, default=",".join(DEFAULT_ARCHES))
    parser.add_argument("--variant", type=str, default="partial_fixed")
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--recompute-correctness",
        choices=["auto", "stored"],
        default="auto",
        help="auto recomputes GSM8K/MC correctness when gold labels are available.",
    )
    parser.add_argument(
        "--qwen-compat-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror read_and_enhance() by prepending the first reward for Qwen reward traces.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def _prompt_key(row: dict[str, Any]) -> str:
    question = _question_from_prompt(row.get("prompt"))
    if question:
        return _norm_question(question)
    return json.dumps(row.get("prompt"), sort_keys=True, ensure_ascii=False)


def _generation_text(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if isinstance(generation, dict):
        return str(generation.get("content", ""))
    return str(generation or "")


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
            continue
    return out


def _final_number_equals(text: str | None, gold: str) -> bool:
    pred_nums = _numbers(text)
    gold_nums = _numbers(gold)
    return bool(pred_nums and gold_nums and pred_nums[-1] == gold_nums[-1])


def _extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return text.strip()
    return text.split("####", 1)[1].strip()


def _simple_answer_equal(predicted: str | None, solution: str | None) -> bool:
    if predicted is None or solution is None:
        return False
    pred = re.sub(r"\s+", " ", str(predicted)).strip().lower()
    sol = re.sub(r"\s+", " ", str(solution)).strip().lower()
    return pred == sol or pred[:1] == sol[:1]


def _mc_match(domain: str, predicted: str | None, solution: str | None) -> bool:
    try:
        from src.rewards.reward_functions import mc_answer_equal, mc_answer_equal_2

        if domain == "medicine":
            return bool(mc_answer_equal(predicted, solution))
        return bool(mc_answer_equal_2(predicted, solution))
    except Exception:
        return _simple_answer_equal(predicted, solution)


def _gold_for_domain(domain: str) -> dict[str, str]:
    if domain == "math":
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return {
            _norm_question(row["question"]): _extract_hash_answer(row["answer"])
            for row in ds
        }
    if domain == "mmlu":
        from datasets import load_from_disk

        ds = load_from_disk("/mnt/pdata/caf83/data/expert_reasoning/mmlu_pro_filtered")["test"]
        return {_norm_question(row["question"]): str(row["answer"]) for row in ds}
    if domain == "medicine":
        from datasets import load_from_disk

        ds = load_from_disk(
            "/mnt/pdata/caf83/data/expert_reasoning/"
            "medreason_corrupted_full_token_filtered_no_violations"
        )["test"]
        return {_norm_question(row["question"]): str(row["answer"]) for row in ds}
    return {}


def _stored_correct(row: dict[str, Any]) -> bool | None:
    for key in ("correctness_reward_func", "correct", "is_correct"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        try:
            return float(value) > 0.0
        except Exception:
            continue
    return None


def _correctness_fn(
    domain: str,
    gold: dict[str, str] | None,
) -> Callable[[dict[str, Any]], tuple[bool | None, str]]:
    def correct(row: dict[str, Any]) -> tuple[bool | None, str]:
        if gold:
            question = _prompt_key(row)
            answer = gold.get(question)
            if answer is not None:
                generation_text = _generation_text(row)
                extracted = _last_xml_answer(generation_text)
                if domain == "math":
                    return _final_number_equals(extracted, answer), "gold_recomputed"
                predicted = extracted if extracted is not None else generation_text
                return _mc_match(domain, predicted, answer), "gold_recomputed"
        return _stored_correct(row), "stored"

    return correct


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _discounted_mean(values: list[float], gamma: float) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    n = len(values)
    weights: list[float] = []
    kept: list[float] = []
    for i, value in enumerate(values):
        try:
            value_f = float(value)
        except Exception:
            continue
        if not math.isfinite(value_f):
            continue
        weights.append(float(gamma) ** (n - 1 - i))
        kept.append(value_f)
    denom = sum(weights)
    if denom <= 0.0:
        return float("nan")
    return sum(v * w for v, w in zip(kept, weights)) / denom


def _reward_score(row: dict[str, Any], *, reward_arch: str, gamma: float, qwen_shift: bool) -> float:
    values = row.get("reward_model_score")
    if not isinstance(values, list) or not values:
        scalar = row.get("reward_score_scalar")
        try:
            return float(scalar)
        except Exception:
            return float("nan")
    rewards = []
    for value in values:
        try:
            rewards.append(float(value))
        except Exception:
            rewards.append(float("nan"))
    if qwen_shift and "qwen" in reward_arch and rewards:
        rewards = [rewards[0]] + rewards
    return _discounted_mean(rewards, gamma)


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _bootstrap(values: list[float], n_samples: int, seed: int) -> tuple[float, float, float]:
    mean = _mean(values)
    if not values or not math.isfinite(mean) or n_samples <= 0:
        return mean, float("nan"), float("nan")
    rng = random.Random(seed)
    boots: list[float] = []
    n = len(values)
    for _ in range(n_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(_mean(sample))
    boots.sort()
    lo_idx = int(0.025 * (len(boots) - 1))
    hi_idx = int(0.975 * (len(boots) - 1))
    return mean, boots[lo_idx], boots[hi_idx]


def summarize_cell(
    path: Path,
    *,
    policy_dataset: str,
    reward_dataset: str,
    reward_arch: str,
    variant: str,
    num_generations: int,
    gamma: float,
    correctness_fn: Callable[[dict[str, Any]], tuple[bool | None, str]],
    qwen_shift: bool,
    bootstrap_samples: int,
    seed: int,
) -> Cell:
    rows = _load_rows(path)
    groups: OrderedDict[str, list[tuple[int, dict[str, Any]]]] = OrderedDict()
    for row_idx, row in enumerate(rows):
        try:
            generation_idx = int(row.get("generation_idx", row_idx))
        except Exception:
            generation_idx = row_idx
        if generation_idx >= num_generations:
            continue
        groups.setdefault(_prompt_key(row), []).append((row_idx, row))

    baseline_values: list[float] = []
    reward_values: list[float] = []
    source_counts: dict[str, int] = {}

    for entries in groups.values():
        entries.sort(
            key=lambda item: (
                int(item[1].get("generation_idx", item[0]))
                if str(item[1].get("generation_idx", item[0])).isdigit()
                else item[0]
            )
        )
        candidates = [row for _, row in entries[:num_generations]]
        flags: list[bool] = []
        scores: list[float] = []
        for row in candidates:
            correct, source = correctness_fn(row)
            source_counts[source] = source_counts.get(source, 0) + 1
            if correct is None:
                continue
            flags.append(bool(correct))
            scores.append(
                _reward_score(
                    row,
                    reward_arch=reward_arch,
                    gamma=gamma,
                    qwen_shift=qwen_shift,
                )
            )
        if not flags:
            continue
        baseline_values.append(sum(float(x) for x in flags) / len(flags))
        finite_scores = [
            (idx, score) for idx, score in enumerate(scores) if math.isfinite(score)
        ]
        if finite_scores:
            best_idx = max(finite_scores, key=lambda item: item[1])[0]
        else:
            best_idx = 0
        reward_values.append(float(flags[best_idx]))

    baseline, baseline_low, baseline_high = _bootstrap(
        baseline_values,
        bootstrap_samples,
        seed,
    )
    reward, reward_low, reward_high = _bootstrap(
        reward_values,
        bootstrap_samples,
        seed + 17,
    )

    first = rows[0] if rows else {}
    if len(source_counts) == 1:
        correctness_source = next(iter(source_counts))
    elif source_counts:
        correctness_source = ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
    else:
        correctness_source = "none"

    return Cell(
        policy_dataset=policy_dataset,
        reward_dataset=reward_dataset,
        reward_arch=reward_arch,
        variant=variant,
        path=str(path),
        n_prompts=len(baseline_values),
        n_rows=len(rows),
        baseline=baseline,
        baseline_low=baseline_low,
        baseline_high=baseline_high,
        reward=reward,
        reward_low=reward_low,
        reward_high=reward_high,
        delta=reward - baseline,
        correctness_source=correctness_source,
        reward_checkpoint_dir=first.get("reward_checkpoint_dir"),
        reward_adapter_dir=first.get("reward_adapter_dir"),
        trace_file=first.get("source_trace_file"),
    )


def _fmt_pct(value: float) -> str:
    return "--" if not math.isfinite(value) else f"{value * 100:.1f}"


def _fmt_delta(value: float) -> str:
    return "--" if not math.isfinite(value) else f"{value * 100:+.1f}"


def _fmt_ci_half(low: float, high: float) -> str:
    if not math.isfinite(low) or not math.isfinite(high):
        return "--"
    return f"{(high - low) * 50:.1f}"


def write_outputs(
    cells: dict[tuple[str, str, str], Cell],
    missing: list[dict[str, str]],
    *,
    policy_datasets: list[str],
    reward_datasets: list[str],
    reward_arches: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cells": [asdict(cell) for cell in cells.values()],
        "missing": missing,
    }
    (output_dir / "transfer_values.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tsv_lines = [
        "\t".join(
            [
                "policy_dataset",
                "reward_dataset",
                "reward_arch",
                "variant",
                "n_prompts",
                "n_rows",
                "baseline_pct",
                "baseline_ci_half_pct",
                "reward_pct",
                "reward_ci_half_pct",
                "delta_pct",
                "correctness_source",
                "reward_checkpoint_dir",
                "trace_file",
                "path",
            ]
        )
    ]
    for key in sorted(cells):
        cell = cells[key]
        tsv_lines.append(
            "\t".join(
                [
                    cell.policy_dataset,
                    cell.reward_dataset,
                    cell.reward_arch,
                    cell.variant,
                    str(cell.n_prompts),
                    str(cell.n_rows),
                    _fmt_pct(cell.baseline),
                    _fmt_ci_half(cell.baseline_low, cell.baseline_high),
                    _fmt_pct(cell.reward),
                    _fmt_ci_half(cell.reward_low, cell.reward_high),
                    _fmt_delta(cell.delta),
                    cell.correctness_source,
                    cell.reward_checkpoint_dir or "",
                    cell.trace_file or "",
                    cell.path,
                ]
            )
        )
    (output_dir / "transfer_values.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    header = ["Policy Dataset", "Baseline"]
    for arch in reward_arches:
        for reward_dataset in reward_datasets:
            header.append(f"{ARCH_LABELS.get(arch, arch)} / {DOMAIN_LABELS.get(reward_dataset, reward_dataset)}")
    lines = ["# Corrected Transfer Reranking Values", "", "| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for policy_dataset in policy_datasets:
        row = [DOMAIN_LABELS.get(policy_dataset, policy_dataset)]
        baseline_cell = next(
            (
                cells[key]
                for key in cells
                if key[0] == policy_dataset
            ),
            None,
        )
        if baseline_cell is None:
            row.append("--")
        else:
            row.append(
                f"{_fmt_pct(baseline_cell.baseline)} +/- "
                f"{_fmt_ci_half(baseline_cell.baseline_low, baseline_cell.baseline_high)}"
            )
        for arch in reward_arches:
            for reward_dataset in reward_datasets:
                cell = cells.get((policy_dataset, reward_dataset, arch))
                if cell is None:
                    row.append("--")
                else:
                    row.append(f"{_fmt_pct(cell.reward)} ({_fmt_delta(cell.delta)})")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Provenance", ""])
    for key in sorted(cells):
        cell = cells[key]
        lines.append(
            f"- {cell.policy_dataset}/{cell.reward_dataset}/{cell.reward_arch}: "
            f"reward={cell.reward_checkpoint_dir}; trace={cell.trace_file}; "
            f"correctness={cell.correctness_source}; n_prompts={cell.n_prompts}"
        )
    if missing:
        lines.extend(["", "## Missing Or Skipped", ""])
        for item in missing:
            lines.append(
                f"- {item['policy_dataset']}/{item['reward_dataset']}/{item['reward_arch']}: "
                f"{item['reason']}"
            )
    (output_dir / "transfer_values.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    colspec = "l" + "r" + ("c" * (len(reward_arches) * len(reward_datasets)))
    tex = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    tex.append("Policy Dataset & Baseline & " + " & ".join(header[2:]) + r" \\")
    tex.append(r"\midrule")
    for policy_dataset in policy_datasets:
        row = [DOMAIN_LABELS.get(policy_dataset, policy_dataset)]
        baseline_cell = next(
            (
                cells[key]
                for key in cells
                if key[0] == policy_dataset
            ),
            None,
        )
        row.append("--" if baseline_cell is None else _fmt_pct(baseline_cell.baseline))
        for arch in reward_arches:
            for reward_dataset in reward_datasets:
                cell = cells.get((policy_dataset, reward_dataset, arch))
                row.append("--" if cell is None else _fmt_delta(cell.delta))
        tex.append(" & ".join(row) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    (output_dir / "transfer_matrix.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    policy_datasets = _csv(args.policy_datasets)
    reward_datasets = _csv(args.reward_datasets)
    reward_arches = _csv(args.reward_arches)

    gold_by_domain: dict[str, dict[str, str] | None] = {}
    if args.recompute_correctness == "auto":
        for domain in policy_datasets:
            try:
                gold_by_domain[domain] = _gold_for_domain(domain)
                print(f"[INFO] Loaded gold labels for {domain}: {len(gold_by_domain[domain] or {})}")
            except Exception as exc:
                gold_by_domain[domain] = None
                print(
                    f"[WARNING] Could not load gold labels for {domain}; "
                    f"falling back to stored correctness. ({type(exc).__name__}: {exc})"
                )
    else:
        gold_by_domain = {domain: None for domain in policy_datasets}

    cells: dict[tuple[str, str, str], Cell] = {}
    missing: list[dict[str, str]] = []
    for policy_dataset in policy_datasets:
        correct_fn = _correctness_fn(policy_dataset, gold_by_domain.get(policy_dataset))
        for reward_dataset in reward_datasets:
            for reward_arch in reward_arches:
                path = (
                    args.input_root
                    / f"P_{policy_dataset}"
                    / f"R_{reward_dataset}"
                    / f"{reward_arch}_{args.variant}"
                    / "eval_results_new.jsonl"
                )
                if not path.exists():
                    missing.append(
                        {
                            "policy_dataset": policy_dataset,
                            "reward_dataset": reward_dataset,
                            "reward_arch": reward_arch,
                            "reason": f"missing {path}",
                        }
                    )
                    continue
                cells[(policy_dataset, reward_dataset, reward_arch)] = summarize_cell(
                    path,
                    policy_dataset=policy_dataset,
                    reward_dataset=reward_dataset,
                    reward_arch=reward_arch,
                    variant=args.variant,
                    num_generations=int(args.num_generations),
                    gamma=float(args.gamma),
                    correctness_fn=correct_fn,
                    qwen_shift=bool(args.qwen_compat_shift),
                    bootstrap_samples=int(args.bootstrap_samples),
                    seed=int(args.seed),
                )

    write_outputs(
        cells,
        missing,
        policy_datasets=policy_datasets,
        reward_datasets=reward_datasets,
        reward_arches=reward_arches,
        output_dir=args.output_dir,
    )
    print(f"Wrote {args.output_dir / 'transfer_values.md'}")
    print(f"Wrote {args.output_dir / 'transfer_values.tsv'}")
    print(f"Wrote {args.output_dir / 'transfer_values.json'}")


if __name__ == "__main__":
    main()
