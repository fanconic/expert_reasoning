"""Summarise cross-domain natural-error localisation label sets.

This script is intentionally CPU-only. It summarises the new MedReason and
MMLU-Pro natural-error label files, including usable span counts and random
chance levels for the same token/interval localisation grids used by the model
scorers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_generators.table_chatgpt_step_extra_metrics import (  # noqa: E402
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    _expected_random_average_precision,
    _mean_ci,
)


DEFAULT_OUTPUT_JSON = Path(
    "localisation/cross_domain_natural_wrong_sft/localisation_dataset_summary.json"
)
DEFAULT_OUTPUT_MD = Path(
    "localisation/cross_domain_natural_wrong_sft/localisation_dataset_summary.md"
)
DEFAULT_WINDOWS = [1, 3, 5, 7]
DEFAULT_STRIDE = 15

DEFAULT_DATASETS = {
    "medreason": {
        "label": "MedReason",
        "full_jsonl": Path(
            "localisation/medreason_natural_wrong_sft/"
            "medreason_qwen7b_sft_wrong_step_labels_full.jsonl"
        ),
        "invalid_jsonl": Path(
            "localisation/medreason_natural_wrong_sft/"
            "medreason_qwen7b_sft_wrong_step_labels_invalid_step_only.jsonl"
        ),
        "tokenizer": "/mnt/pdata/caf83/neurips2026/medicine/outputs/"
        "qwen7b_medicine_sft/best_model",
        "score_root": Path("localisation/medreason_natural_wrong_sft/scores"),
    },
    "mmlu_pro": {
        "label": "MMLU-Pro",
        "full_jsonl": Path(
            "localisation/mmlu_pro_natural_wrong_sft/"
            "mmlu_pro_llama8b_sft_wrong_step_labels_full.jsonl"
        ),
        "invalid_jsonl": Path(
            "localisation/mmlu_pro_natural_wrong_sft/"
            "mmlu_pro_llama8b_sft_wrong_step_labels_invalid_step_only.jsonl"
        ),
        "tokenizer": "/mnt/pdata/caf83/neurips2026/mmlu/outputs/"
        "llama8b_mmlu_sft/best_model",
        "score_root": Path("localisation/mmlu_pro_natural_wrong_sft/scores"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-length", type=int, default=1124)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _tokenizer_source(path: str | Path) -> str:
    p = Path(str(path))
    adapter_cfg = p / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            obj = json.loads(adapter_cfg.read_text())
            base_model = obj.get("base_model_name_or_path")
            if base_model:
                return str(base_model)
        except Exception:
            pass
    return str(path)


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _finite(values: Iterable[float]) -> list[float]:
    out = []
    for value in values:
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            out.append(parsed)
    return out


def _summary_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    vals = _finite(values)
    if not vals:
        return {"mean": None, "median": None, "p90": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.shape[0]),
    }


def _token_positions_from_char_span(
    tokenizer,
    text: str,
    span: Sequence[int] | None,
    max_length: int,
) -> tuple[list[int], int]:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = enc.get("offset_mapping", [])
    seq_len = int(len(enc.get("input_ids", [])))
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return [], seq_len
    start = _to_int(span[0])
    end = _to_int(span[1])
    if start is None or end is None or end <= start:
        return [], seq_len

    positions: list[int] = []
    for idx, offset in enumerate(offsets):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            continue
        tok_start = _to_int(offset[0])
        tok_end = _to_int(offset[1])
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            continue
        if tok_end > start and tok_start < end:
            positions.append(int(idx))
    return positions, seq_len


def _text_span(text: str, region: str) -> tuple[int, int]:
    if region == "all":
        return 0, len(text)

    low = text.lower()
    answer_start = low.find("<answer>")
    if region == "pre_answer":
        return 0, answer_start if answer_start >= 0 else len(text)

    if region != "think":
        raise ValueError(f"Unknown region: {region}")

    think_start = low.find("<think>")
    if think_start >= 0:
        start = think_start + len("<think>")
        end = low.find("</think>", start)
        if end < 0:
            end = answer_start
        if end < 0:
            end = len(text)
        return start, max(start, end)
    return 0, answer_start if answer_start >= 0 else len(text)


def _candidate_token_positions(
    tokenizer,
    text: str,
    region: str,
    seq_len: int,
    max_length: int,
) -> list[int]:
    start, end = _text_span(text, region)
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    positions = []
    for idx, offset in enumerate(enc.get("offset_mapping", [])):
        if idx <= 0 or idx >= seq_len:
            continue
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            continue
        tok_start = _to_int(offset[0])
        tok_end = _to_int(offset[1])
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            continue
        if tok_end > start and tok_start < end:
            positions.append(int(idx))
    return positions


def _hit_chance_at_k(
    candidate_indices: Sequence[int],
    targets: Sequence[int],
    window: int,
    k: int,
) -> float:
    n = len(candidate_indices)
    if n <= 0 or not targets:
        return float("nan")
    target_set = {int(t) for t in targets}
    good = [
        int(idx)
        for idx in candidate_indices
        if any(abs(int(idx) - target) <= int(window) for target in target_set)
    ]
    m = len(set(good))
    k = min(max(1, int(k)), n)
    if m <= 0:
        return 0.0
    if n - m < k:
        return 1.0
    return float(1.0 - math.comb(n - m, k) / math.comb(n, k))


def _chance_metrics_for_rows(
    rows: Sequence[dict[str, Any]],
    tokenizer,
    *,
    windows: Sequence[int],
    stride: int,
    max_length: int,
    region: str,
    grid: str,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "chance_hit1": [],
        "chance_hit3": [],
        "chance_hit5": [],
        "chance_map": [],
    }
    for window in windows:
        values[f"chance_hit1_at_{window}"] = []
        values[f"chance_hit3_at_{window}"] = []
        values[f"chance_hit5_at_{window}"] = []
    candidate_counts = []
    target_counts = []
    seq_lens = []
    skipped = 0

    for row in rows:
        text = str(row.get("wrong_text") or row.get("pert_text") or "")
        targets, seq_len = _token_positions_from_char_span(
            tokenizer,
            text,
            row.get("target_char_span"),
            max_length=max_length,
        )
        if seq_len <= 1 or not targets:
            skipped += 1
            continue
        candidates = _candidate_token_positions(
            tokenizer,
            text,
            region=region,
            seq_len=seq_len,
            max_length=max_length,
        )
        if grid == "interval15":
            n_units = int(math.ceil(seq_len / float(stride)))
            candidates = sorted({int(c) // stride for c in candidates if int(c) // stride > 0})
            targets = sorted({int(t) // stride for t in targets if 0 <= int(t) < seq_len})
            unit_windows = {int(w): int(math.ceil(max(0, int(w)) / float(stride))) for w in windows}
        elif grid == "token":
            targets = sorted({int(t) for t in targets if 0 <= int(t) < seq_len})
            unit_windows = {int(w): int(w) for w in windows}
            n_units = seq_len
        else:
            raise ValueError(f"Unknown grid: {grid}")

        if not candidates or not targets:
            skipped += 1
            continue

        candidate_counts.append(len(candidates))
        target_counts.append(len(targets))
        seq_lens.append(n_units)
        relevant_exact = len(set(candidates).intersection(targets))
        values["chance_map"].append(
            _expected_random_average_precision(len(candidates), relevant_exact)
        )
        for window in windows:
            unit_w = unit_windows[int(window)]
            for k in (1, 3, 5):
                values[f"chance_hit{k}_at_{int(window)}"].append(
                    _hit_chance_at_k(candidates, targets, unit_w, k)
                )

    metrics = {
        key: _mean_ci(vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed)
        for key, vals in values.items()
        if vals
    }
    return {
        "region": region,
        "grid": grid,
        "n_input": len(rows),
        "n_scored": int(metrics.get("chance_map", {}).get("n") or 0),
        "n_skipped": int(skipped),
        "candidate_count": _summary_stats(candidate_counts),
        "target_count": _summary_stats(target_counts),
        "sequence_units": _summary_stats(seq_lens),
        "metrics": metrics,
    }


def _score_file_status(score_root: Path) -> dict[str, Any]:
    expected = {
        "dense_reward": [
            score_root / "qwen7b_full_reward_localisation" / "summary.json",
            score_root / "llama8b_full_reward_localisation" / "summary.json",
        ],
        "interval_reward": [
            score_root / "qwen7b_partial_fixed_reward_localisation" / "summary.json",
            score_root / "llama8b_partial_fixed_reward_localisation" / "summary.json",
        ],
        "policy_baselines": [
            score_root / "qwen7b_sft_policy_token_baselines" / "policy_token_baselines_summary.json",
            score_root / "qwen7b_medicine_sft_policy_token_baselines" / "policy_token_baselines_summary.json",
            score_root / "llama8b_sft_policy_token_baselines" / "policy_token_baselines_summary.json",
            score_root / "llama8b_mmlu_sft_policy_token_baselines" / "policy_token_baselines_summary.json",
        ],
    }
    found = []
    for paths in expected.values():
        for path in paths:
            if path.exists():
                found.append(str(path))
    return {
        "score_root": str(score_root),
        "has_model_scores": bool(found),
        "found_summaries": found,
    }


def _dataset_summary(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    full_rows = _load_jsonl(Path(cfg["full_jsonl"]))
    invalid_rows = _load_jsonl(Path(cfg["invalid_jsonl"]))
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(str(cfg["tokenizer"])),
        trust_remote_code=True,
    )

    span_chars = []
    step_indices = []
    token_span_lens = []
    token_seq_lens = []
    for row in invalid_rows:
        span = row.get("target_char_span")
        if isinstance(span, list) and len(span) == 2:
            span_chars.append(float(int(span[1]) - int(span[0])))
        if row.get("first_wrong_step_index") is not None:
            step_indices.append(float(row["first_wrong_step_index"]))
        targets, seq_len = _token_positions_from_char_span(
            tokenizer,
            str(row.get("wrong_text") or row.get("pert_text") or ""),
            row.get("target_char_span"),
            max_length=int(args.max_length),
        )
        if seq_len > 0:
            token_seq_lens.append(float(seq_len))
        if targets:
            token_span_lens.append(float(len(targets)))

    chance = []
    for region in ("all", "pre_answer", "think"):
        for grid in ("token", "interval15"):
            chance.append(
                _chance_metrics_for_rows(
                    invalid_rows,
                    tokenizer,
                    windows=args.windows,
                    stride=int(args.stride),
                    max_length=int(args.max_length),
                    region=region,
                    grid=grid,
                    bootstrap_samples=int(args.bootstrap_samples),
                    bootstrap_alpha=float(args.bootstrap_alpha),
                    bootstrap_seed=int(args.bootstrap_seed),
                )
            )

    labels = {}
    for row in full_rows:
        label = str(row.get("label") or "missing")
        labels[label] = labels.get(label, 0) + 1

    return {
        "name": name,
        "label": cfg["label"],
        "full_jsonl": str(cfg["full_jsonl"]),
        "invalid_step_only_jsonl": str(cfg["invalid_jsonl"]),
        "tokenizer": str(cfg["tokenizer"]),
        "n_full": len(full_rows),
        "n_invalid_step": labels.get("invalid_step", 0),
        "n_answer_only": labels.get("answer_only", 0),
        "n_ambiguous": labels.get("ambiguous", 0),
        "label_counts": labels,
        "manual_annotations": sum(1 for row in full_rows if row.get("manual_annotation")),
        "span_chars": _summary_stats(span_chars),
        "span_tokens": _summary_stats(token_span_lens),
        "completion_tokens": _summary_stats(token_seq_lens),
        "first_wrong_step_index": _summary_stats(step_indices),
        "chance": chance,
        "model_score_status": _score_file_status(Path(cfg["score_root"])),
    }


def _fmt_count_metric(metric: dict[str, Any]) -> str:
    mean = metric.get("mean")
    med = metric.get("median")
    if mean is None or med is None:
        return "-"
    return f"{float(mean):.1f} / {float(med):.1f}"


def _fmt_pct(metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("mean") is None:
        return "-"
    mean = 100.0 * float(metric["mean"])
    ci = metric.get("ci_halfwidth")
    return f"{mean:.2f}" + (f" ± {100.0 * float(ci):.2f}" if ci is not None else "")


def _write_markdown(path: Path, payload: dict[str, Any], windows: Sequence[int]) -> None:
    lines = [
        "# Cross-Domain Natural-Error Localisation Summary",
        "",
        "## Label Sets",
        "",
        "| Dataset | Full rows | Invalid-step spans | Answer-only | Manual | Span tokens mean/median | Completion tokens mean/median | First bad step mean/median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in payload["datasets"]:
        span_tokens = _fmt_count_metric(ds["span_tokens"])
        completion_tokens = _fmt_count_metric(ds["completion_tokens"])
        first_wrong_step = _fmt_count_metric(ds["first_wrong_step_index"])
        lines.append(
            f"| {ds['label']} | {ds['n_full']} | {ds['n_invalid_step']} | "
            f"{ds['n_answer_only']} | {ds['manual_annotations']} | "
            f"{span_tokens} | {completion_tokens} | {first_wrong_step} |"
        )

    lines.extend(
        [
            "",
            "## Random-Chance Localisation Baselines",
            "",
            "Chance is computed over transition candidates in the requested region. "
            "MAP chance is exact random-ranking AP for exact target positions.",
            "",
            "| Dataset | Region | Grid | n | Candidates mean | Targets mean | Chance MAP | "
            + " | ".join(f"Hit@1@{int(w)}" for w in windows)
            + " |",
            "|---|---|---|---:|---:|---:|---:|"
            + "|".join(["---:"] * len(windows))
            + "|",
        ]
    )
    for ds in payload["datasets"]:
        for row in ds["chance"]:
            metrics = row["metrics"]
            hit_cells = [
                _fmt_pct(metrics.get(f"chance_hit1_at_{int(window)}"))
                for window in windows
            ]
            lines.append(
                f"| {ds['label']} | {row['region']} | {row['grid']} | {row['n_scored']} | "
                f"{float(row['candidate_count']['mean'] or 0.0):.1f} | "
                f"{float(row['target_count']['mean'] or 0.0):.1f} | "
                f"{_fmt_pct(metrics.get('chance_map'))} | "
                + " | ".join(hit_cells)
                + " |"
            )

    lines.extend(["", "## Model Score Status", ""])
    for ds in payload["datasets"]:
        status = ds["model_score_status"]
        if status["has_model_scores"]:
            lines.append(f"- {ds['label']}: found {len(status['found_summaries'])} score summary file(s).")
        else:
            lines.append(f"- {ds['label']}: no model score summaries yet under `{status['score_root']}`.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = [
        _dataset_summary(name, cfg, args)
        for name, cfg in DEFAULT_DATASETS.items()
    ]
    payload = {
        "windows": [int(w) for w in args.windows],
        "stride": int(args.stride),
        "max_length": int(args.max_length),
        "datasets": datasets,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(args.output_md, payload, args.windows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    for ds in datasets:
        print(
            f"{ds['label']}: {ds['n_invalid_step']}/{ds['n_full']} usable spans; "
            f"{ds['n_answer_only']} answer-only"
        )


if __name__ == "__main__":
    main()
