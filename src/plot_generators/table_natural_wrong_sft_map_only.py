"""Build a MAP-only localisation table for naturally wrong Qwen7B-SFT traces."""

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

from src.plot_generators.table_chatgpt_step_extra_metrics import (  # noqa: E402
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    _load_jsonl,
)
from src.plot_generators.table_chatgpt_step_localisation import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
)
from src.plot_generators.table_natural_error_fair_grid_metrics import (  # noqa: E402
    DEFAULT_STRIDE,
    DEFAULT_WINDOW,
    GRID_LABELS,
    _apply_marker,
    _fmt_pct,
    _method_specs,
    _metric_cell,
    _metric_mean,
    _random_row,
    _rank_markers,
    _score_rows,
)


DEFAULT_ROOT = Path("localisation/natural_wrong_sft/scores")
DEFAULT_STRICT_INPUT = (
    DEFAULT_ROOT
    / "_inputs"
    / "natural_wrong_sft_valid_target_char_span_actual_wrong_answer.jsonl"
)
DEFAULT_OUTPUT_JSON = Path(
    "localisation/natural_wrong_sft/localisation_natural_wrong_sft_map_only.json"
)
DEFAULT_OUTPUT_TEX = Path(
    "localisation/natural_wrong_sft/localisation_natural_wrong_sft_map_only.tex"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--strict-input", type=Path, default=DEFAULT_STRICT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _row_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        row.get("prompt_idx"),
        row.get("variant_idx"),
        row.get("clean_generation_idx"),
    )


def _strict_keys(path: Path) -> set[tuple[Any, Any, Any]]:
    rows = _load_jsonl(path)
    keys = {_row_key(row) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"Strict input has duplicate row keys: {path}")
    return keys


def _filter_rows(rows: list[dict[str, Any]], keys: set[tuple[Any, Any, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _row_key(row) in keys]


def build_results(args: argparse.Namespace) -> dict[str, Any]:
    strict_keys = _strict_keys(args.strict_input)
    methods: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "strict_input_rows": len(strict_keys),
        "row_counts_before_filter": set(),
        "row_counts_after_filter": set(),
        "target_sources": set(),
    }

    for grid in ["token", "interval15"]:
        random_by_prompt: dict[Any, dict[str, list[float]]] = {}
        for method_spec in _method_specs(args.root_dir):
            if not method_spec["path"].exists():
                raise FileNotFoundError(f"Missing detail file: {method_spec['path']}")
            all_rows = _load_jsonl(method_spec["path"])
            rows = _filter_rows(all_rows, strict_keys)
            if len(rows) != len(strict_keys):
                raise ValueError(
                    f"{method_spec['path']} has {len(rows)} strict rows, expected {len(strict_keys)}."
                )
            scored = _score_rows(
                rows,
                grid=grid,
                stride=int(args.stride),
                window=int(args.window),
                seq_key=method_spec["seq_key"],
                detector=method_spec["detector"],
                is_policy=bool(method_spec["is_policy"]),
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            )
            metadata["row_counts_before_filter"].add(len(all_rows))
            metadata["row_counts_after_filter"].add(int(scored["n"]))
            metadata["target_sources"].update(scored["target_sources"])
            methods.append(
                {
                    "grid": grid,
                    "model_key": method_spec["model_key"],
                    "model": method_spec["model"],
                    "method_key": method_spec["method_key"],
                    "signal": method_spec[f"signal_{grid}"],
                    "detail_path": str(method_spec["path"]),
                    "n_rows": scored["n"],
                    "n": scored["map"]["n"],
                    "metrics": {
                        "map": scored["map"],
                        "norm_map": scored["norm_map"],
                    },
                    "is_random": False,
                }
            )
            for prompt_key, _hit_chance, ap_chance in scored["random_items"]:
                bucket = random_by_prompt.setdefault(prompt_key, {"hit7": [], "map": []})
                if math.isfinite(float(ap_chance)):
                    bucket["map"].append(float(ap_chance))
                    bucket["hit7"].append(0.0)

        random = _random_row(
            grid,
            random_by_prompt,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_alpha=float(args.bootstrap_alpha),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        random["metrics"] = {
            "map": random["metrics"]["map"],
            "norm_map": random["metrics"]["norm_map"],
        }
        methods.append(random)

    return {
        "dataset": "natural_wrong_sft",
        "dataset_title": "Naturally wrong Qwen7B-SFT GSM8K generations",
        "root_dir": str(args.root_dir),
        "strict_input": str(args.strict_input),
        "window": int(args.window),
        "stride": int(args.stride),
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "alpha": float(args.bootstrap_alpha),
            "seed": int(args.bootstrap_seed),
        },
        "metadata": {
            "strict_input_rows": int(metadata["strict_input_rows"]),
            "row_counts_before_filter": sorted(metadata["row_counts_before_filter"]),
            "row_counts_after_filter": sorted(metadata["row_counts_after_filter"]),
            "target_sources": sorted(metadata["target_sources"]),
        },
        "methods": methods,
    }


def build_latex(results: dict[str, Any]) -> str:
    stride = int(results["stride"])
    n_rows = int(results["metadata"]["strict_input_rows"])
    target_sources = results["metadata"]["target_sources"]
    target = target_sources[0] if len(target_sources) == 1 else ", ".join(target_sources)
    target_latex = target.replace("_", r"\_")
    metric_order = ["map", "norm_map"]

    latex: list[str] = []
    latex.append(r"% Requires \usepackage{booktabs,multirow,arydshln}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(
        r"\caption{\textbf{MAP localisation on naturally wrong Qwen7B-SFT GSM8K traces.} "
        rf"We label the first mathematically invalid reasoning step in {n_rows} Qwen7B-SFT generations "
        r"whose final answer is actually incorrect, then rank candidate transition positions by each "
        r"method's error signal. The token-grid panel evaluates all methods over token transitions, "
        rf"expanding interval rewards to tokens. The interval-grid panel evaluates common {stride}-token "
        r"intervals, mean-pooling dense reward and SFT token baselines while using the interval reward "
        rf"natively. MAP uses the labelled step span ({target_latex}); Norm. MAP subtracts the expected "
        r"random-ranking MAP, $(m-r)/(1-r)$. Values are percentages with 95\% bootstrap CI half-widths; "
        r"$n$ is the number of examples with finite MAP on that tokenizer/grid. "
        r"\textbf{Bold} marks the best and \underline{underlining} the second-best value within each model and grid.}"
    )
    latex.append(r"\label{tab:natural_wrong_sft_map_only}")
    latex.append(r"\begin{tabular}{lllccc}")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Grid} & \textbf{Model} & \textbf{Signal} & "
        r"\textbf{MAP} & \textbf{Norm. MAP} & \textbf{n} \\"
    )
    latex.append(r"\midrule")

    first_grid = True
    for grid in ["token", "interval15"]:
        grid_methods = [m for m in results["methods"] if m["grid"] == grid]
        if not grid_methods:
            continue
        if not first_grid:
            latex.append(r"\midrule")
        first_grid = False
        grid_label = GRID_LABELS[grid]
        nonrandom_count = sum(1 for m in grid_methods if not m.get("is_random"))
        grid_cell_used = False
        first_model = True

        for model_key in MODEL_ORDER:
            group = [m for m in grid_methods if m.get("model_key") == model_key]
            if not group:
                continue
            if not first_model:
                latex.append(r"\cdashline{2-6}")
            first_model = False
            markers_by_metric = {
                metric: _rank_markers([_metric_mean(m, metric) for m in group])
                for metric in metric_order
            }
            for idx, method in enumerate(group):
                grid_cell = (
                    rf"\multirow{{{nonrandom_count}}}{{*}}{{{grid_label}}}"
                    if not grid_cell_used
                    else ""
                )
                grid_cell_used = True
                model_cell = (
                    rf"\multirow{{{len(group)}}}{{*}}{{{MODEL_LABELS[model_key]}}}"
                    if idx == 0
                    else ""
                )
                line = [grid_cell, model_cell, method["signal"]]
                for metric in metric_order:
                    line.append(
                        _apply_marker(
                            _metric_cell(method, metric),
                            markers_by_metric[metric][idx],
                        )
                    )
                line.append(str(method["n"]) if method["n"] is not None else "-")
                latex.append(" & ".join(line) + r" \\")

        random_rows = [m for m in grid_methods if m.get("is_random")]
        for random in random_rows:
            line = [
                "",
                random["model"],
                random["signal"],
                _fmt_pct(random["metrics"]["map"]),
                _fmt_pct(random["metrics"]["norm_map"]),
                str(random["n"]) if random["n"] is not None else "-",
            ]
            latex.append(" & ".join(line) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    return "\n".join(latex)


def main() -> None:
    args = parse_args()
    results = build_results(args)
    _write_json(args.output_json, results)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(build_latex(results))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_tex}")
    for method in results["methods"]:
        metrics = method["metrics"]
        print(
            f"{method['grid']:>10} | {method['model']} | {method['signal']:<32} | "
            f"MAP={_fmt_pct(metrics['map'])} | NormMAP={_fmt_pct(metrics['norm_map'])} | "
            f"n={method['n']}"
        )


if __name__ == "__main__":
    main()
