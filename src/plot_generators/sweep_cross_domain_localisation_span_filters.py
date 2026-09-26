"""Sweep cross-domain natural-error localisation under span-length filters."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_generators import summarize_cross_domain_scored_localisation as base  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("localisation/cross_domain_natural_wrong_sft")
DEFAULT_THRESHOLDS = [15, 30, 45, 60, 96]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=["native", "token", "interval15"],
        default=["native", "token", "interval15"],
        help=(
            "native scores each method on its native grid; token/interval15 force "
            "all methods onto a common grid."
        ),
    )
    parser.add_argument("--region", default="think", choices=list(base.REGIONS))
    parser.add_argument("--stride", type=int, default=base.INTERVAL_STRIDE)
    parser.add_argument("--max-length", type=int, default=base.MAX_LENGTH)
    parser.add_argument("--bootstrap-samples", type=int, default=base.BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=base.BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=base.BOOTSTRAP_SEED)
    return parser.parse_args()


def _read_first_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                return json.loads(raw)
    return None


def _policy_tokenizer_for_dataset(dataset: dict[str, Any]):
    policy_path = (
        Path(dataset["score_root"])
        / f"{dataset['score_prefix']}_policy_token_baselines"
        / "policy_token_baselines.jsonl"
    )
    first = _read_first_jsonl(policy_path)
    if first is None:
        raise FileNotFoundError(f"Missing policy score file for tokenizer: {policy_path}")
    return base._tokenizer(first.get("policy_model"))


def _span_token_count(row: dict[str, Any], tokenizer) -> int | None:
    text = row.get("pert_text") or row.get("wrong_text")
    span = row.get("target_char_span")
    if not isinstance(text, str) or not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    try:
        start = int(span[0])
        end = int(span[1])
    except Exception:
        return None
    if end <= start:
        return None
    ids = tokenizer(text[start:end], add_special_tokens=False)["input_ids"]
    return int(len(ids))


def _span_inside_region(row: dict[str, Any], region: str) -> bool:
    text = row.get("pert_text") or row.get("wrong_text")
    span = row.get("target_char_span")
    if not isinstance(text, str) or not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    try:
        start = int(span[0])
        end = int(span[1])
    except Exception:
        return False
    region_start, region_end = base._text_span(text, region)
    return start >= region_start and end <= region_end


def _metric_mean(row: dict[str, Any], key: str) -> float | None:
    value = row["metrics"][key]["mean"]
    return None if value is None else float(value)


def _pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{100.0 * value:.2f}"


def _delta(metric: dict[str, Any], chance: dict[str, Any]) -> str:
    if metric.get("mean") is None or chance.get("mean") is None:
        return "-"
    return f"{100.0 * (float(metric['mean']) - float(chance['mean'])):+.2f}"


def _score_for_threshold(
    dataset: dict[str, Any],
    label_rows: list[dict[str, Any]],
    threshold: int,
    *,
    region: str,
    views: list[str],
    stride: int,
    max_length: int,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in label_rows
        if row.get("_span_tokens") is not None
        and int(row["_span_tokens"]) <= int(threshold)
        and _span_inside_region(row, region)
    ]
    selected_keys = {base._row_key(row) for row in selected}
    text_by_key = {
        base._row_key(row): row.get("pert_text") or row.get("wrong_text")
        for row in selected
        if isinstance(row.get("pert_text") or row.get("wrong_text"), str)
    }

    out: list[dict[str, Any]] = []
    for spec in base._method_specs(dataset):
        path = Path(spec["path"])
        if not path.exists():
            continue
        rows = [row for row in base._load_jsonl(path) if base._row_key(row) in selected_keys]
        for view in views:
            grids = [spec["native_grid"]] if view == "native" else [view]
            for grid in grids:
                result = base._score_method(
                    rows=rows,
                    spec=spec,
                    text_by_key=text_by_key,
                    dataset=dataset,
                    region=region,
                    grid=grid,
                    stride=stride,
                    max_length=max_length,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_alpha=bootstrap_alpha,
                    bootstrap_seed=bootstrap_seed,
                )
                result["threshold_span_tokens"] = int(threshold)
                result["view"] = view
                result["n_label_rows_selected"] = len(selected)
                out.append(result)
    return out


def _markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No rows._\n"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "") for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _compact_table(results: list[dict[str, Any]], *, view: str) -> list[dict[str, str]]:
    table = []
    for row in results:
        if row["view"] != view:
            continue
        m = row["metrics"]
        hit = m["hit1_w7"]
        chance_hit = m["chance_hit1_w7"]
        map_metric = m["map_tie_aware"]
        chance_map = m["chance_map"]
        table.append(
            {
                "Dataset": row["dataset"],
                "Max Span Tok": str(row["threshold_span_tokens"]),
                "Signal": row["signal"],
                "Grid": row["grid"],
                "N": str(row["n_scored"]),
                "Hit@7": _pct(_metric_mean(row, "hit1_w7")),
                "Hit Ch.": _pct(chance_hit.get("mean")),
                "Hit Delta": _delta(hit, chance_hit),
                "Norm Hit": _pct(_metric_mean(row, "norm_hit1_w7")),
                "MAP": _pct(map_metric.get("mean")),
                "MAP Ch.": _pct(chance_map.get("mean")),
                "Norm MAP": _pct(m["norm_map_tie_aware"].get("mean")),
                "MRR": _pct(m["mrr"].get("mean")),
            }
        )
    return table


def _best_reward_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    table = []
    reward_keys = {"reward_dense", "reward_interval"}
    for row in results:
        if row["view"] != "native" or row["method_key"] not in reward_keys:
            continue
        m = row["metrics"]
        table.append(
            {
                "Dataset": row["dataset"],
                "Max Span Tok": str(row["threshold_span_tokens"]),
                "Signal": row["signal"],
                "Grid": row["grid"],
                "N": str(row["n_scored"]),
                "Hit@7": _pct(m["hit1_w7"].get("mean")),
                "Hit Ch.": _pct(m["chance_hit1_w7"].get("mean")),
                "Hit Delta": _delta(m["hit1_w7"], m["chance_hit1_w7"]),
                "Norm Hit": _pct(m["norm_hit1_w7"].get("mean")),
                "MAP": _pct(m["map_tie_aware"].get("mean")),
                "MAP Ch.": _pct(m["chance_map"].get("mean")),
                "Norm MAP": _pct(m["norm_map_tie_aware"].get("mean")),
            }
        )
    return table


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Span-Filtered Cross-Domain Localisation Sweep",
        "",
        (
            "Rows use only labelled spans fully inside `<think>`, and candidate "
            "drops/spikes are restricted to `<think>`. Span length is measured "
            "with the dataset policy tokenizer. MAP is tie-aware AP; within an "
            "equal-score block, expected AP is used instead of stable-sort tie "
            "breaking. Native view scores dense/log-prob/entropy on token grid "
            "and interval reward on the 15-token grid."
        ),
        "",
        "## Reward-Only Native View",
        "",
        _markdown_table(payload["tables"]["reward_native"]),
    ]
    for view in payload["views"]:
        lines.extend(
            [
                "",
                f"## View `{view}`",
                "",
                _markdown_table(payload["tables"][view]),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    dataset_summaries = []
    for dataset in base.DATASETS:
        tokenizer = _policy_tokenizer_for_dataset(dataset)
        labels = base._load_jsonl(Path(dataset["labels"]))
        span_lengths = []
        enriched = []
        for row in labels:
            row = dict(row)
            count = _span_token_count(row, tokenizer)
            row["_span_tokens"] = count
            if count is not None:
                span_lengths.append(int(count))
            enriched.append(row)
        dataset_summaries.append(
            {
                "key": dataset["key"],
                "label": dataset["label"],
                "n_labels": len(labels),
                "span_tokens": {
                    "mean": float(sum(span_lengths) / len(span_lengths)) if span_lengths else None,
                    "median": float(sorted(span_lengths)[len(span_lengths) // 2]) if span_lengths else None,
                    "max": max(span_lengths) if span_lengths else None,
                    "threshold_counts": {
                        str(threshold): int(sum(1 for value in span_lengths if value <= threshold))
                        for threshold in args.thresholds
                    },
                },
            }
        )
        for threshold in args.thresholds:
            results.extend(
                _score_for_threshold(
                    dataset,
                    enriched,
                    int(threshold),
                    region=str(args.region),
                    views=list(args.views),
                    stride=int(args.stride),
                    max_length=int(args.max_length),
                    bootstrap_samples=int(args.bootstrap_samples),
                    bootstrap_alpha=float(args.bootstrap_alpha),
                    bootstrap_seed=int(args.bootstrap_seed),
                )
            )
    tables = {
        "reward_native": _best_reward_rows(results),
        **{view: _compact_table(results, view=view) for view in args.views},
    }
    return {
        "region": args.region,
        "views": list(args.views),
        "thresholds": [int(x) for x in args.thresholds],
        "dataset_summaries": dataset_summaries,
        "results": results,
        "tables": tables,
    }


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.output_dir / "span_filtered_localisation_sweep.json"
    output_md = args.output_md or args.output_dir / "span_filtered_localisation_sweep.md"
    payload = build_payload(args)
    base._write_json(output_json, payload)
    _write_markdown(output_md, payload)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print("\nReward-only native view:")
    print(_markdown_table(payload["tables"]["reward_native"]))


if __name__ == "__main__":
    main()
