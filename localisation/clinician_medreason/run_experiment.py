"""Run the MedReason clinician localisation experiment.

This is the single entrypoint for the clinician-labelled MedReason
localisation analysis. It reads anonymised expert annotations, aligns them with
the clinician-visible Qwen2.5-7B SFT traces, loads token-level reward or policy
scores, and reports Hit@1 / Hit@+/-1 for the first-error localisation task.

Default scoring policy:

* each annotation row marks the first wrong reasoning unit;
* rows marked as no-clear or final-answer-only are mapped to the final visible
  reasoning unit, matching the paper analysis;
* reward and log-probability signals use the largest token-level drop;
* entropy uses the largest token-level increase;
* chance is the expected score of a uniform random reasoning-unit prediction.

The default external score paths are cluster paths used for the paper. Override
them with --interval-root, --dense-root, or --policy-jsonl if needed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_DIR = ROOT / "annotations"
DEFAULT_METADATA = ROOT / "metadata" / "clinician_cases.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "results"

DEFAULT_INTERVAL_ROOT = Path("/mnt/pdata/caf83/neurips2026/transfer_rerank_corrected_qwen7b_sft")
DEFAULT_DENSE_ROOT = Path("/mnt/pdata/caf83/neurips2026/medreason_full_dense_reward_scores_qwen7b_sft")
DEFAULT_POLICY_JSONL = (
    ROOT.parent
    / "trace_scoring/medicine_qwen7b_sft_t0p5_qwen7b_domain_signals/"
    / "qwen7b_medicine_sft_policy_logprob_entropy_on_medicine_sft_trace.jsonl"
)

FINAL_ANSWER_ONLY_PATTERNS = [
    re.compile(r"\bfinal answer\b.*\b(wrong|incorrect|error|problem)", re.I),
    re.compile(r"\b(answer|option|letter)\b.*\bwrong\b.*\b(reasoning|rationale)\b.*\b(correct|right)", re.I),
    re.compile(r"\b(reasoning|rationale)\b.*\b(correct|right)\b.*\b(answer|option|letter)\b.*\b(wrong|incorrect)", re.I),
    re.compile(r"\bonly\b.*\b(final answer|answer|option|letter)\b.*\b(wrong|incorrect|error)", re.I),
    re.compile(r"\bcontains?\b.*\b(correct|right) answer\b.*\bselected answer is wrong\b", re.I),
    re.compile(r"\bknows? the answer\b.*\bselects?\b.*\bwrong option\b", re.I),
]


@dataclass(frozen=True)
class ScoreSpec:
    key: str
    label: str
    path: Path
    field: str
    direction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--interval-root", type=Path, default=DEFAULT_INTERVAL_ROOT)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--policy-jsonl", type=Path, default=DEFAULT_POLICY_JSONL)
    parser.add_argument(
        "--signals",
        choices=("all", "interval", "dense", "policy"),
        default="all",
        help="Which default signal family to score.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--keep-details",
        action="store_true",
        help="Also write one per-case prediction row per scored signal.",
    )
    return parser.parse_args()


def score_specs(args: argparse.Namespace) -> list[ScoreSpec]:
    specs: list[ScoreSpec] = []
    if args.signals in {"all", "interval"}:
        for reward_dataset in ("math", "medicine", "mmlu"):
            for arch in ("llama8b", "qwen4b", "qwen7b"):
                specs.append(
                    ScoreSpec(
                        key=f"interval_{arch}_R_{reward_dataset}",
                        label=f"interval {arch} R={reward_dataset}",
                        path=(
                            args.interval_root
                            / "P_medicine"
                            / f"R_{reward_dataset}"
                            / f"{arch}_partial_fixed"
                            / "eval_results_new.jsonl"
                        ),
                        field="reward_model_score",
                        direction="drop",
                    )
                )
    if args.signals in {"all", "dense"}:
        for arch in ("llama8b", "qwen4b", "qwen7b"):
            for granularity in ("dense", "full"):
                specs.append(
                    ScoreSpec(
                        key=f"dense_{arch}_{granularity}",
                        label=f"dense {arch} {granularity}",
                        path=args.dense_root / f"{arch}_{granularity}" / "eval_results_new.jsonl",
                        field="reward_model_score",
                        direction="drop",
                    )
                )
    if args.signals in {"all", "policy"}:
        specs.extend(
            [
                ScoreSpec(
                    key="policy_qwen7b_sft_logprob_drop",
                    label="qwen7b SFT log-prob drop",
                    path=args.policy_jsonl,
                    field="policy_log_probs",
                    direction="drop",
                ),
                ScoreSpec(
                    key="policy_qwen7b_sft_entropy_increase",
                    label="qwen7b SFT entropy increase",
                    path=args.policy_jsonl,
                    field="policy_entropies",
                    direction="increase",
                ),
            ]
        )
    return specs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generation_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value or "")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "x"}


def as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def note_implies_final_answer_only(note: str) -> bool:
    return bool(note) and any(pattern.search(note) for pattern in FINAL_ANSWER_ONLY_PATTERNS)


def load_annotations(annotation_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(annotation_dir.glob("expert_*.jsonl")):
        for row in load_jsonl(path):
            row = dict(row)
            row.setdefault("expert_id", path.stem)
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No expert_*.jsonl annotations found in {annotation_dir}")
    return rows


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        if case_id in by_case:
            raise ValueError(f"Duplicate case_id in metadata: {case_id}")
        by_case[case_id] = row
    return by_case


def approx_score_span_for_char_span(
    *,
    text: str,
    scores: list[Any],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    if not scores:
        return 0, 0
    text_len = max(1, len(text or ""))
    score_len = len(scores)
    start = int(math.floor(max(0, char_start) / text_len * score_len))
    end = int(math.ceil(min(text_len, max(char_start, char_end)) / text_len * score_len))
    start = max(0, min(score_len - 1, start))
    end = max(start + 1, min(score_len, end))
    return start, end


def numeric_scores(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        out.append(number if math.isfinite(number) else float("nan"))
    return out


def step_spans(row: dict[str, Any], scores: list[float]) -> list[tuple[int, int, int]]:
    text = generation_text(row.get("generation"))
    spans: list[tuple[int, int, int]] = []
    for step in row.get("steps") or []:
        step_id = as_int(step.get("step_id"))
        span = step.get("char_span") or [0, 0]
        if step_id is None:
            continue
        start, end = approx_score_span_for_char_span(
            text=text,
            scores=scores,
            char_start=int(span[0]),
            char_end=int(span[1]),
        )
        spans.append((step_id, start, end))
    return spans


def change(prev: float, cur: float, direction: str) -> float:
    if direction == "increase":
        return cur - prev
    return prev - cur


def predict_token_step(row: dict[str, Any], scores: list[float], direction: str) -> dict[str, Any]:
    spans = step_spans(row, scores)
    if len(scores) < 2 or not spans:
        return {"pred_step_id": None, "token_idx": None, "score_change": None}

    lo = max(1, min(start for _step_id, start, _end in spans) + 1)
    hi = min(len(scores), max(end for _step_id, _start, end in spans))
    best_idx: int | None = None
    best_change: float | None = None
    for idx in range(lo, hi):
        prev = scores[idx - 1]
        cur = scores[idx]
        if not math.isfinite(prev) or not math.isfinite(cur):
            continue
        delta = change(prev, cur, direction)
        if best_change is None or delta > best_change:
            best_change = delta
            best_idx = idx

    if best_idx is None:
        return {"pred_step_id": None, "token_idx": None, "score_change": None}

    nearest_step: int | None = None
    nearest_distance: int | None = None
    for step_id, start, end in spans:
        if start <= best_idx < end:
            return {"pred_step_id": step_id, "token_idx": best_idx, "score_change": best_change}
        distance = min(abs(best_idx - start), abs(best_idx - max(start, end - 1)))
        if nearest_distance is None or distance < nearest_distance:
            nearest_step = step_id
            nearest_distance = distance
    return {"pred_step_id": nearest_step, "token_idx": best_idx, "score_change": best_change}


def metric_label(annotation: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    selected = as_int(annotation.get("selected_step_id"))
    note = str(annotation.get("note") or "")
    no_clear = as_bool(annotation.get("no_clear_localisable_error"))
    explicit_final = as_bool(annotation.get("final_answer_only_error"))
    inferred_final = note_implies_final_answer_only(note)
    step_ids = [as_int(step.get("step_id")) for step in metadata.get("steps") or []]
    step_ids = [step_id for step_id in step_ids if step_id is not None]
    last_reasoning = max(step_ids) if step_ids else None

    mapped_to_last = False
    if (no_clear or explicit_final or inferred_final) and last_reasoning is not None:
        selected = last_reasoning
        mapped_to_last = True

    return {
        "selected_step_id": selected,
        "raw_selected_step_id": as_int(annotation.get("selected_step_id")),
        "no_clear_localisable_error": no_clear,
        "final_answer_only_error": bool(explicit_final or inferred_final),
        "final_answer_only_explicit": explicit_final,
        "final_answer_only_note_inferred": inferred_final,
        "mapped_to_last_reasoning": mapped_to_last,
        "last_reasoning_step_id": last_reasoning,
        "n_steps": len(step_ids),
    }


def random_expectation(label: int | None, n_steps: int) -> tuple[float | None, float | None]:
    if label is None or n_steps <= 0:
        return None, None
    hit1 = 1.0 / n_steps
    hit_pm1 = sum(1 for step_id in range(1, n_steps + 1) if abs(step_id - label) <= 1) / n_steps
    return hit1, hit_pm1


def needed_source_lines(metadata_by_case: dict[str, dict[str, Any]], annotations: list[dict[str, Any]]) -> set[int]:
    lines: set[int] = set()
    for row in annotations:
        metadata = metadata_by_case.get(str(row.get("case_id")))
        if metadata is None:
            continue
        line_no = as_int(metadata.get("source_line_no"))
        if line_no is not None:
            lines.add(line_no)
    return lines


def load_score_rows(path: Path, needed_lines: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no not in needed_lines:
                continue
            if line.strip():
                rows[line_no] = json.loads(line)
            if len(rows) == len(needed_lines):
                break
    return rows


def evaluate_signal(
    spec: ScoreSpec,
    annotations: list[dict[str, Any]],
    metadata_by_case: dict[str, dict[str, Any]],
    source_lines: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score_rows = load_score_rows(spec.path, source_lines)
    details: list[dict[str, Any]] = []
    missing = 0
    mismatched = 0
    replaced = 0
    score_lengths: list[int] = []

    for annotation in annotations:
        case_id = str(annotation.get("case_id"))
        metadata = metadata_by_case.get(case_id)
        if metadata is None:
            continue

        line_no = as_int(metadata.get("source_line_no"))
        score_row = score_rows.get(line_no or -1)
        if score_row is None:
            missing += 1
            continue

        if generation_text(score_row.get("generation")) != generation_text(metadata.get("generation")):
            mismatched += 1
            continue

        scores = numeric_scores(score_row.get(spec.field))
        if not scores:
            missing += 1
            continue
        replaced += 1
        score_lengths.append(len(scores))

        label = metric_label(annotation, metadata)
        predicted = predict_token_step(metadata, scores, spec.direction)
        selected = label["selected_step_id"]
        pred_step = predicted["pred_step_id"]
        hit1 = selected is not None and pred_step == selected
        hit_pm1 = selected is not None and pred_step is not None and abs(int(pred_step) - int(selected)) <= 1
        chance_1, chance_pm1 = random_expectation(selected, int(label["n_steps"] or 0))

        details.append(
            {
                "signal_key": spec.key,
                "signal_label": spec.label,
                "expert_id": annotation.get("expert_id"),
                "case_id": case_id,
                "case_rank": annotation.get("case_rank"),
                "selected_step_id": selected,
                "raw_selected_step_id": label["raw_selected_step_id"],
                "pred_step_id": pred_step,
                "pred_token_idx": predicted["token_idx"],
                "pred_score_change": predicted["score_change"],
                "hit_at_1": bool(hit1),
                "hit_at_pm1": bool(hit_pm1),
                "chance_hit_at_1": chance_1,
                "chance_hit_at_pm1": chance_pm1,
                "n_steps": label["n_steps"],
                "mapped_to_last_reasoning": label["mapped_to_last_reasoning"],
                "no_clear_localisable_error": label["no_clear_localisable_error"],
                "final_answer_only_error": label["final_answer_only_error"],
                "note": annotation.get("note", ""),
            }
        )

    alignment = {
        "score_jsonl": str(spec.path),
        "score_field": spec.field,
        "score_direction": spec.direction,
        "score_rows_loaded_for_needed_lines": len(score_rows),
        "score_rows_replaced": replaced,
        "score_rows_missing": missing,
        "score_rows_generation_mismatch": mismatched,
        "score_length_min": min(score_lengths) if score_lengths else None,
        "score_length_median": sorted(score_lengths)[len(score_lengths) // 2] if score_lengths else None,
        "score_length_max": max(score_lengths) if score_lengths else None,
    }
    return details, alignment


def proportion(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row[key]) / len(rows)


def mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def bootstrap_ci(
    rows: list[dict[str, Any]],
    key: str,
    *,
    samples: int,
    seed: int,
    stratify_by: str | None = "expert_id",
) -> list[float] | None:
    if not rows:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    if stratify_by is None:
        groups = {"all": rows}
    else:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row.get(stratify_by))].append(row)

    group_values = list(groups.values())
    for _ in range(samples):
        sample_rows: list[dict[str, Any]] = []
        for group in group_values:
            sample_rows.extend(rng.choice(group) for _ in group)
        values.append(float(proportion(sample_rows, key) or 0.0))
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return [lo, hi]


def bootstrap_mean_ci(
    rows: list[dict[str, Any]],
    key: str,
    *,
    samples: int,
    seed: int,
    stratify_by: str | None = "expert_id",
) -> list[float] | None:
    if not rows:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    if stratify_by is None:
        groups = {"all": rows}
    else:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row.get(stratify_by))].append(row)

    group_values = list(groups.values())
    for _ in range(samples):
        sample_rows: list[dict[str, Any]] = []
        for group in group_values:
            sample_rows.extend(rng.choice(group) for _ in group)
        value = mean_optional(sample_rows, key)
        values.append(float(value or 0.0))
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return [lo, hi]


def summarise_signal(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    by_expert: dict[str, Any] = {}
    for expert_id in sorted({str(row.get("expert_id")) for row in rows}):
        expert_rows = [row for row in rows if str(row.get("expert_id")) == expert_id]
        by_expert[expert_id] = summarise_rows(
            expert_rows,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            stratify_by=None,
        )
    pooled = summarise_rows(
        rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        stratify_by="expert_id",
    )
    return {"pooled": pooled, "by_expert": by_expert}


def summarise_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    stratify_by: str | None,
) -> dict[str, Any]:
    n = len(rows)
    hit1_count = sum(1 for row in rows if row["hit_at_1"])
    hit_pm1_count = sum(1 for row in rows if row["hit_at_pm1"])
    return {
        "n": n,
        "hit_at_1": {
            "count": hit1_count,
            "rate": hit1_count / n if n else None,
            "ci95": bootstrap_ci(
                rows,
                "hit_at_1",
                samples=bootstrap_samples,
                seed=seed + 1,
                stratify_by=stratify_by,
            ),
        },
        "hit_at_pm1": {
            "count": hit_pm1_count,
            "rate": hit_pm1_count / n if n else None,
            "ci95": bootstrap_ci(
                rows,
                "hit_at_pm1",
                samples=bootstrap_samples,
                seed=seed + 2,
                stratify_by=stratify_by,
            ),
        },
    }


def chance_summary(reference_rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    return {
        "n": len(reference_rows),
        "hit_at_1": {
            "rate": mean_optional(reference_rows, "chance_hit_at_1"),
            "ci95": bootstrap_mean_ci(
                reference_rows,
                "chance_hit_at_1",
                samples=bootstrap_samples,
                seed=seed + 3,
                stratify_by="expert_id",
            ),
        },
        "hit_at_pm1": {
            "rate": mean_optional(reference_rows, "chance_hit_at_pm1"),
            "ci95": bootstrap_mean_ci(
                reference_rows,
                "chance_hit_at_pm1",
                samples=bootstrap_samples,
                seed=seed + 4,
                stratify_by="expert_id",
            ),
        },
    }


def percent(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def metric_cell(metric: dict[str, Any], include_count: bool = True) -> str:
    rate = percent(metric.get("rate"))
    ci = metric.get("ci95")
    ci_text = ""
    if ci is not None:
        ci_text = f" [{percent(ci[0])}, {percent(ci[1])}]"
    if include_count and metric.get("count") is not None:
        return f"{metric['count']}/{metric.get('n', '')} ({rate}){ci_text}"
    return f"{rate}{ci_text}"


def result_cell(summary: dict[str, Any]) -> str:
    n = summary["n"]
    hit1 = dict(summary["hit_at_1"])
    hit_pm1 = dict(summary["hit_at_pm1"])
    hit1["n"] = n
    hit_pm1["n"] = n
    return f"{metric_cell(hit1)} / {metric_cell(hit_pm1)}"


def chance_cell(summary: dict[str, Any]) -> str:
    return f"{metric_cell(summary['hit_at_1'], include_count=False)} / {metric_cell(summary['hit_at_pm1'], include_count=False)}"


def write_markdown(
    path: Path,
    *,
    summaries: dict[str, Any],
    chance: dict[str, Any] | None,
    alignments: dict[str, Any],
    annotation_count: int,
    metadata_count: int,
) -> None:
    lines: list[str] = [
        "# MedReason Clinician Localisation Metrics",
        "",
        (
            "Anonymised expert annotations are scored against token-level signals on "
            "the clinician-visible Qwen2.5-7B SFT wrong traces. No-clear and "
            "final-answer-only labels are mapped to the final visible reasoning unit."
        ),
        "",
        f"- annotation rows: {annotation_count}",
        f"- metadata rows: {metadata_count}",
        "- metric cell: `Hit@1 / Hit@+/-1`, with 95% percentile bootstrap CIs",
        "",
        "## Pooled",
        "",
        "| signal | pooled |",
        "|---|---:|",
    ]
    if chance is not None:
        lines.append(f"| chance | {chance_cell(chance)} |")
    for key, summary in summaries.items():
        lines.append(f"| {summary['label']} | {result_cell(summary['metrics']['pooled'])} |")

    lines.extend(["", "## By Expert", ""])
    expert_ids = sorted(
        {
            expert_id
            for summary in summaries.values()
            for expert_id in summary["metrics"]["by_expert"]
        }
    )
    header = "| signal | " + " | ".join(expert_ids) + " |"
    lines.append(header)
    lines.append("|---" + "|---:" * len(expert_ids) + "|")
    for key, summary in summaries.items():
        cells = []
        for expert_id in expert_ids:
            expert_summary = summary["metrics"]["by_expert"].get(expert_id)
            cells.append(result_cell(expert_summary) if expert_summary else "NA")
        lines.append(f"| {summary['label']} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Alignment", "", "| signal | matched | missing | mismatched | score length |", "|---|---:|---:|---:|---:|"])
    for key, summary in summaries.items():
        alignment = alignments[key]
        length = (
            f"{alignment['score_length_min']}/{alignment['score_length_median']}/{alignment['score_length_max']}"
            if alignment["score_length_min"] is not None
            else "NA"
        )
        lines.append(
            f"| {summary['label']} | {alignment['score_rows_replaced']} | "
            f"{alignment['score_rows_missing']} | {alignment['score_rows_generation_mismatch']} | {length} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations_dir)
    metadata_by_case = load_metadata(args.metadata)
    source_lines = needed_source_lines(metadata_by_case, annotations)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    alignments: dict[str, Any] = {}
    all_details: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] | None = None

    for spec in score_specs(args):
        if not spec.path.exists():
            print(f"Skipping missing score file: {spec.path}")
            continue
        details, alignment = evaluate_signal(spec, annotations, metadata_by_case, source_lines)
        if not details:
            print(f"Skipping {spec.key}: no matched details")
            continue
        metrics = summarise_signal(details, args.bootstrap_samples, args.seed)
        summaries[spec.key] = {"label": spec.label, "metrics": metrics}
        alignments[spec.key] = alignment
        all_details.extend(details)
        if reference_rows is None:
            reference_rows = details

    if not summaries:
        raise RuntimeError("No score files could be evaluated.")

    chance = (
        chance_summary(reference_rows, args.bootstrap_samples, args.seed)
        if reference_rows is not None
        else None
    )
    result = {
        "annotation_policy": {
            "no_clear_target": "last_visible_reasoning_unit",
            "final_answer_only_target": "last_visible_reasoning_unit",
            "prediction": "largest token-level drop for reward/logprob, largest token-level increase for entropy",
        },
        "annotation_rows": len(annotations),
        "metadata_rows": len(metadata_by_case),
        "score_alignments": alignments,
        "chance": chance,
        "signals": summaries,
    }

    json_path = args.results_dir / "clinician_localisation_metrics.json"
    md_path = args.results_dir / "clinician_localisation_metrics.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_markdown(
        md_path,
        summaries=summaries,
        chance=chance,
        alignments=alignments,
        annotation_count=len(annotations),
        metadata_count=len(metadata_by_case),
    )

    if args.keep_details:
        details_path = args.results_dir / "clinician_localisation_details.jsonl"
        with details_path.open("w", encoding="utf-8") as handle:
            for row in all_details:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(f"Wrote details: {details_path}")

    print(f"Wrote metrics: {json_path}")
    print(f"Wrote table: {md_path}")


if __name__ == "__main__":
    main()
