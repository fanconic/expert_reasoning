"""Span-filtered reward localisation sweep for GSM8K natural SFT errors."""

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

from src.plot_generators import sweep_natural_error_locator_variants as sweep  # noqa: E402


DEFAULT_ROOT = Path("localisation/natural_wrong_sft/scores")
DEFAULT_STRICT_INPUT = (
    DEFAULT_ROOT / "_inputs" / "natural_wrong_sft_valid_target_char_span_actual_wrong_answer.jsonl"
)
DEFAULT_OUTPUT_JSON = Path(
    "localisation/natural_wrong_sft/localisation_natural_wrong_sft_reward_span_filters.json"
)
DEFAULT_OUTPUT_MD = Path(
    "localisation/natural_wrong_sft/localisation_natural_wrong_sft_reward_span_filters.md"
)
DEFAULT_THRESHOLDS = [15, 30, 45, 60, 96]
DEFAULT_MODELS = ["qwen4b", "llama8b"]
DEFAULT_VIEWS = ["native", "token", "interval15"]
SPAN_TOKENIZER = "/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--strict-input", type=Path, default=DEFAULT_STRICT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--views", nargs="+", default=DEFAULT_VIEWS, choices=DEFAULT_VIEWS)
    parser.add_argument("--region", default="think", choices=list(sweep.REGIONS))
    parser.add_argument("--span-tokenizer", default=SPAN_TOKENIZER)
    parser.add_argument("--stride", type=int, default=sweep.STRIDE)
    parser.add_argument("--window", type=int, default=sweep.WINDOW)
    parser.add_argument("--bootstrap-samples", type=int, default=sweep.BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=sweep.BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=sweep.BOOTSTRAP_SEED)
    return parser.parse_args()


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
    return int(len(tokenizer(text[start:end], add_special_tokens=False)["input_ids"]))


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
    region_start, region_end = sweep._text_span(text, region)
    return start >= region_start and end <= region_end


def _fmt_pct(metric: dict[str, Any] | float | None, *, signed: bool = False) -> str:
    if isinstance(metric, dict):
        value = metric.get("mean")
    else:
        value = metric
    if value is None:
        return "-"
    value = float(value)
    if not math.isfinite(value):
        return "-"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{100.0 * value:.2f}"


def _delta(metric: dict[str, Any], chance: dict[str, Any]) -> str:
    if metric.get("mean") is None or chance.get("mean") is None:
        return "-"
    return f"{100.0 * (float(metric['mean']) - float(chance['mean'])):+.2f}"


def _score_filtered(
    *,
    root_dir: Path,
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]],
    selected_keys: set[tuple[Any, ...]],
    models: set[str],
    views: list[str],
    region: str,
    stride: int,
    window: int,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    text_by_key = {
        key: row.get("pert_text") or row.get("wrong_text")
        for key, row in rows_by_key.items()
        if key in selected_keys and isinstance(row.get("pert_text") or row.get("wrong_text"), str)
    }
    results = []
    for spec in sweep._method_specs(root_dir):
        if spec["model_key"] not in models or spec["method_key"] not in {"reward_dense", "reward_interval"}:
            continue
        path = Path(spec["path"])
        if not path.exists():
            continue
        score_rows = [row for row in sweep._load_jsonl(path) if sweep._row_key(row) in selected_keys]
        for view in views:
            grids = [spec.get("native_grid", "token")] if view == "native" else [view]
            # The upstream method specs do not carry native_grid.
            if view == "native":
                grids = ["interval15"] if spec["method_key"] == "reward_interval" else ["token"]
            for grid in grids:
                row = sweep._score_method(
                    rows=score_rows,
                    spec=spec,
                    text_by_key=text_by_key,
                    region=region,
                    grid=grid,
                    window=window,
                    stride=stride,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_alpha=bootstrap_alpha,
                    bootstrap_seed=bootstrap_seed,
                )
                row["view"] = view
                row["threshold_span_tokens"] = None
                row["n_label_rows_selected"] = len(selected_keys)
                results.append(row)
    return results


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


def _table_rows(results: list[dict[str, Any]], *, view: str) -> list[dict[str, str]]:
    rows = []
    for row in results:
        if row["view"] != view:
            continue
        metrics = row["metrics"]
        rows.append(
            {
                "Model": row["model"],
                "Max Span Tok": str(row["threshold_span_tokens"]),
                "Signal": row["signal"],
                "Grid": row["grid"],
                "N": str(row["n_scored"]),
                "Hit@7": _fmt_pct(metrics["hit1"]),
                "Hit Ch.": _fmt_pct(metrics["chance_hit1"]),
                "Hit Delta": _delta(metrics["hit1"], metrics["chance_hit1"]),
                "Norm Hit": _fmt_pct(metrics["norm_hit1"], signed=True),
                "MAP": _fmt_pct(metrics["map_tie_aware"]),
                "MAP Ch.": _fmt_pct(metrics["chance_map"]),
                "Norm MAP": _fmt_pct(metrics["norm_map_tie_aware"], signed=True),
                "MRR": _fmt_pct(metrics["mrr"]),
            }
        )
    return rows


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    span_tokenizer = sweep._tokenizer(args.span_tokenizer)
    strict_rows = sweep._load_jsonl(args.strict_input)
    rows_by_key = {}
    span_lengths = []
    for row in strict_rows:
        row = dict(row)
        span_tokens = _span_token_count(row, span_tokenizer)
        row["_span_tokens"] = span_tokens
        rows_by_key[sweep._row_key(row)] = row
        if span_tokens is not None:
            span_lengths.append(int(span_tokens))

    all_results = []
    for threshold in args.thresholds:
        selected_keys = {
            key
            for key, row in rows_by_key.items()
            if row.get("_span_tokens") is not None
            and int(row["_span_tokens"]) <= int(threshold)
            and _span_inside_region(row, args.region)
        }
        threshold_results = _score_filtered(
            root_dir=args.root_dir,
            rows_by_key=rows_by_key,
            selected_keys=selected_keys,
            models=set(args.models),
            views=list(args.views),
            region=str(args.region),
            stride=int(args.stride),
            window=int(args.window),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_alpha=float(args.bootstrap_alpha),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        for row in threshold_results:
            row["threshold_span_tokens"] = int(threshold)
        all_results.extend(threshold_results)

    tables = {view: _table_rows(all_results, view=view) for view in args.views}
    return {
        "root_dir": str(args.root_dir),
        "strict_input": str(args.strict_input),
        "region": args.region,
        "models": list(args.models),
        "views": list(args.views),
        "thresholds": [int(x) for x in args.thresholds],
        "span_tokenizer": sweep._tokenizer_source(args.span_tokenizer),
        "span_token_summary": {
            "n": len(span_lengths),
            "mean": float(sum(span_lengths) / len(span_lengths)) if span_lengths else None,
            "median": float(sorted(span_lengths)[len(span_lengths) // 2]) if span_lengths else None,
            "max": max(span_lengths) if span_lengths else None,
            "threshold_counts": {
                str(threshold): int(sum(1 for value in span_lengths if value <= threshold))
                for threshold in args.thresholds
            },
        },
        "results": all_results,
        "tables": tables,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GSM8K Natural Wrong SFT Reward Span Filters",
        "",
        (
            "Rows are filtered to first-invalid spans fully inside `<think>`. "
            "Candidate reward drops are also restricted to `<think>`. Span length "
            "is measured with the original Qwen2.5-7B SFT tokenizer so all models "
            "use the same example subset. MAP is tie-aware AP."
        ),
        "",
        "## Native View",
        "",
        _markdown_table(payload["tables"].get("native", [])),
    ]
    for view in payload["views"]:
        if view == "native":
            continue
        lines.extend(["", f"## View `{view}`", "", _markdown_table(payload["tables"].get(view, []))])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_md, payload)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print("\nNative view:")
    print(_markdown_table(payload["tables"].get("native", [])))


if __name__ == "__main__":
    main()
