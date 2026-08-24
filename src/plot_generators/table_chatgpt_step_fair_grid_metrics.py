"""Compare natural-error localisation on token and 15-token interval grids."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_generators.table_chatgpt_step_extra_metrics import (  # noqa: E402
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    _average_precision,
    _expected_random_average_precision,
    _hit_at_window_from_scores,
    _load_jsonl,
    _mean_ci,
    _row_targets,
    _transition_exact_relevant_count,
    _transition_scores,
    _unit_sequence_and_targets,
)
from src.plot_generators.table_chatgpt_step_localisation import (  # noqa: E402
    DEFAULT_ROOT,
    MODEL_LABELS,
    MODEL_ORDER,
)


DEFAULT_OUTPUT_TEX = (
    Path("localisation/chatgpt_step_perturbations")
    / "localisation_chatgpt_step_fair_grid_metrics.tex"
)
DEFAULT_OUTPUT_JSON = (
    Path("localisation/chatgpt_step_perturbations")
    / "localisation_chatgpt_step_fair_grid_metrics.json"
)
DEFAULT_WINDOW = 7
DEFAULT_STRIDE = 15

GRID_LABELS = {
    "token": "Token grid",
    "interval15": "15-token interval grid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _bold_mean(cell: str) -> str:
    return re.sub(r"(-?\d+(?:\.\d+)?)", r"\\textbf{\1}", cell, count=1)


def _underline_mean(cell: str) -> str:
    return re.sub(r"(-?\d+(?:\.\d+)?)", r"\\underline{\1}", cell, count=1)


def _fmt_pct(metric: dict[str, Any]) -> str:
    mean = metric.get("mean")
    ci = metric.get("ci_halfwidth")
    if mean is None:
        return "-"
    if ci is None:
        return f"{100.0 * float(mean):.2f}"
    return f"{100.0 * float(mean):.2f} $\\pm$ {100.0 * float(ci):.2f}"


def _rank_markers(values: Iterable[float | None]) -> list[str]:
    values_list = list(values)
    unique = sorted({round(float(v), 12) for v in values_list if v is not None}, reverse=True)
    best = unique[0] if unique else None
    second = unique[1] if len(unique) > 1 else None
    markers = []
    for value in values_list:
        if value is None:
            markers.append("")
        elif best is not None and abs(float(value) - best) <= 1e-12:
            markers.append("best")
        elif second is not None and abs(float(value) - second) <= 1e-12:
            markers.append("second")
        else:
            markers.append("")
    return markers


def _apply_marker(cell: str, marker: str) -> str:
    if marker == "best":
        return _bold_mean(cell)
    if marker == "second":
        return _underline_mean(cell)
    return cell


def _transition_chance(n_units: int, targets: list[int], window: int) -> float:
    if n_units <= 1 or not targets:
        return float("nan")
    indices = np.arange(1, n_units, dtype=np.int64)
    mask = np.zeros(indices.shape[0], dtype=bool)
    for target in targets:
        mask |= np.abs(indices - int(target)) <= int(window)
    return float(mask.mean())


def _normalize(value: float, chance: float) -> float:
    if not math.isfinite(value) or not math.isfinite(chance):
        return float("nan")
    denom = 1.0 - chance
    if denom <= 1e-12:
        return float("nan")
    return float((value - chance) / denom)


def _row_for_grid(row: dict[str, Any], grid: str, stride: int) -> dict[str, Any]:
    out = copy.copy(row)
    if grid == "token":
        out["localization_mode"] = "token"
        out["partial_fixed_stride"] = None
    elif grid == "interval15":
        out["localization_mode"] = "bucket"
        out["partial_fixed_stride"] = int(stride)
    else:
        raise ValueError(f"Unknown grid: {grid}")
    return out


def _method_specs(root_dir: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        specs.extend(
            [
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "method_key": "reward_dense",
                    "signal_token": "Reward dense",
                    "signal_interval15": "Reward dense, mean/15",
                    "path": root_dir / f"{model_key}_full_reward_localisation" / "pair_details.jsonl",
                    "seq_key": "pert_score_seq",
                    "detector": "largest_drop",
                    "is_policy": False,
                },
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "method_key": "reward_interval",
                    "signal_token": "Reward interval, expanded",
                    "signal_interval15": "Reward interval",
                    "path": root_dir / f"{model_key}_partial_fixed_reward_localisation" / "pair_details.jsonl",
                    "seq_key": "pert_score_seq",
                    "detector": "largest_drop",
                    "is_policy": False,
                },
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "method_key": "sft_logprob",
                    "signal_token": "SFT log-probability",
                    "signal_interval15": "SFT log-probability, mean/15",
                    "path": root_dir / f"{model_key}_sft_policy_token_baselines" / "policy_token_baselines.jsonl",
                    "seq_key": "pert_policy_log_probs",
                    "detector": "largest_drop",
                    "is_policy": True,
                },
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "method_key": "sft_entropy",
                    "signal_token": "SFT entropy",
                    "signal_interval15": "SFT entropy, mean/15",
                    "path": root_dir / f"{model_key}_sft_policy_token_baselines" / "policy_token_baselines.jsonl",
                    "seq_key": "pert_policy_entropies",
                    "detector": "largest_spike",
                    "is_policy": True,
                },
            ]
        )
    return specs


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    grid: str,
    stride: int,
    window: int,
    seq_key: str,
    detector: str,
    is_policy: bool,
    method_key: str,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    hit_values = []
    norm_hit_values = []
    ap_values = []
    norm_ap_values = []
    random_items: list[tuple[Any, float, float]] = []
    target_sources = set()

    for idx, row in enumerate(rows):
        seq = row.get(seq_key, [])
        if not isinstance(seq, list):
            continue
        work_row = _row_for_grid(row, grid, stride)
        target_sources.add(str(row.get("target_position_source")))

        first_targets = _row_targets(row, "first_diff", is_policy=is_policy)
        unit_first, first_units, first_w = _unit_sequence_and_targets(
            work_row, seq, first_targets, int(window)
        )
        first_scores, first_indices = _transition_scores(unit_first, detector)
        use_native_interval_hit = grid == "interval15" and method_key == "reward_interval"
        if use_native_interval_hit:
            stored_hit = ((row.get("reward_hit1_at_window") or {}).get(str(window)))
            stored_chance = ((row.get("random_hit1_at_window") or {}).get(str(window)))
            hit = float(stored_hit) if stored_hit is not None else float("nan")
            hit_chance = float(stored_chance) if stored_chance is not None else float("nan")
        else:
            hit = _hit_at_window_from_scores(first_scores, first_indices, first_units, first_w)
            hit_chance = _transition_chance(int(unit_first.shape[0]), first_units, first_w)

        edit_targets = _row_targets(row, "edit_span", is_policy=is_policy)
        unit_edit, edit_units, _edit_w = _unit_sequence_and_targets(work_row, seq, edit_targets, 0)
        edit_scores, edit_indices = _transition_scores(unit_edit, detector)
        ap = _average_precision(edit_scores, edit_indices, edit_units)
        n_candidates = max(0, int(unit_edit.shape[0]) - 1)
        n_relevant = _transition_exact_relevant_count(int(unit_edit.shape[0]), edit_units)
        ap_chance = _expected_random_average_precision(n_candidates, n_relevant)

        hit_values.append(hit)
        norm_hit_values.append(_normalize(hit, hit_chance))
        ap_values.append(ap)
        norm_ap_values.append(_normalize(ap, ap_chance))
        random_items.append((row.get("prompt_idx", idx), hit_chance, ap_chance))

    return {
        "n": len(rows),
        "target_sources": sorted(target_sources),
        "hit7": _mean_ci(hit_values, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
        "norm_hit7": _mean_ci(norm_hit_values, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
        "map": _mean_ci(ap_values, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
        "norm_map": _mean_ci(norm_ap_values, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
        "random_items": random_items,
    }


def _random_row(
    grid: str,
    random_by_prompt: dict[Any, dict[str, list[float]]],
    *,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    hit_values = [float(np.mean(v["hit7"])) for v in random_by_prompt.values() if v["hit7"]]
    map_values = [float(np.mean(v["map"])) for v in random_by_prompt.values() if v["map"]]
    zero_values = [0.0 for _ in hit_values]
    hit_metric = _mean_ci(hit_values, bootstrap_samples, bootstrap_alpha, bootstrap_seed)
    map_metric = _mean_ci(map_values, bootstrap_samples, bootstrap_alpha, bootstrap_seed)
    zero_metric = _mean_ci(zero_values, bootstrap_samples, bootstrap_alpha, bootstrap_seed)
    return {
        "grid": grid,
        "model_key": "__random__",
        "model": r"\textsc{Random}",
        "method_key": "random",
        "signal": "Expected chance",
        "n": hit_metric.get("n"),
        "metrics": {
            "hit7": hit_metric,
            "norm_hit7": zero_metric,
            "map": map_metric,
            "norm_map": zero_metric,
        },
        "is_random": True,
    }


def build_results(args: argparse.Namespace) -> dict[str, Any]:
    methods: list[dict[str, Any]] = []
    metadata = {"row_counts": set(), "target_sources": set()}

    specs = _method_specs(args.root_dir)
    for grid in ["token", "interval15"]:
        random_by_prompt: dict[Any, dict[str, list[float]]] = {}
        for spec in specs:
            if not spec["path"].exists():
                raise FileNotFoundError(f"Missing detail file: {spec['path']}")
            rows = _load_jsonl(spec["path"])
            scored = _score_rows(
                rows,
                grid=grid,
                stride=int(args.stride),
                window=int(args.window),
                seq_key=spec["seq_key"],
                detector=spec["detector"],
                is_policy=bool(spec["is_policy"]),
                method_key=str(spec["method_key"]),
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            )
            metadata["row_counts"].add(int(scored["n"]))
            metadata["target_sources"].update(scored["target_sources"])
            methods.append(
                {
                    "grid": grid,
                    "model_key": spec["model_key"],
                    "model": spec["model"],
                    "method_key": spec["method_key"],
                    "signal": spec[f"signal_{grid}"],
                    "n": scored["n"],
                    "metrics": {
                        "hit7": scored["hit7"],
                        "norm_hit7": scored["norm_hit7"],
                        "map": scored["map"],
                        "norm_map": scored["norm_map"],
                    },
                    "is_random": False,
                }
            )
            for prompt_key, hit_chance, ap_chance in scored["random_items"]:
                bucket = random_by_prompt.setdefault(prompt_key, {"hit7": [], "map": []})
                if math.isfinite(float(hit_chance)):
                    bucket["hit7"].append(float(hit_chance))
                if math.isfinite(float(ap_chance)):
                    bucket["map"].append(float(ap_chance))
        methods.append(
            _random_row(
                grid,
                random_by_prompt,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            )
        )

    return {
        "root_dir": str(args.root_dir),
        "window": int(args.window),
        "stride": int(args.stride),
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "alpha": float(args.bootstrap_alpha),
            "seed": int(args.bootstrap_seed),
        },
        "metadata": {
            "row_counts": sorted(metadata["row_counts"]),
            "target_sources": sorted(metadata["target_sources"]),
        },
        "methods": methods,
    }


def _metric_cell(method: dict[str, Any], metric: str) -> str:
    return _fmt_pct(method["metrics"][metric])


def _metric_mean(method: dict[str, Any], metric: str) -> float | None:
    mean = method["metrics"][metric].get("mean")
    return None if mean is None else float(mean)


def build_latex(results: dict[str, Any]) -> str:
    window = int(results["window"])
    stride = int(results["stride"])
    row_counts = results["metadata"]["row_counts"]
    n_rows = row_counts[0] if len(row_counts) == 1 else "mixed"
    target_sources = results["metadata"]["target_sources"]
    target = target_sources[0] if len(target_sources) == 1 else ", ".join(target_sources)
    target_latex = target.replace("_", r"\_")
    metric_order = ["hit7", "norm_hit7", "map", "norm_map"]

    latex: list[str] = []
    latex.append(r"% Requires \usepackage{booktabs,multirow}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\setlength{\tabcolsep}{3pt}")
    latex.append(
        r"\caption{\textbf{Fair-grid natural-error localisation metrics.} "
        rf"All rows use the same {n_rows} ChatGPT-edited GSM8K clean/perturbed pairs. "
        rf"The token-grid panel evaluates all methods over token transition candidates, expanding interval scores to tokens. "
        rf"The interval-grid panel evaluates all methods over common {stride}-token intervals: dense reward and SFT token "
        rf"baselines are mean-pooled, while the interval reward uses its native intervals. "
        rf"Hit@{window} uses the first changed token target ({target_latex}); MAP uses the broader edited token span. "
        rf"Normalized metrics subtract the expected random value for the corresponding grid, "
        rf"$(m-r)/(1-r)$. Values are percentages with 95\% bootstrap CI half-widths. "
        rf"\textbf{{Bold}} marks the best and \underline{{underlining}} the second-best value within each model, grid, and metric.}}"
    )
    latex.append(r"\label{tab:localisation_chatgpt_step_fair_grid_metrics}")
    latex.append(r"\begin{tabular}{lllccccc}")
    latex.append(r"\toprule")
    latex.append(
        rf"\textbf{{Grid}} & \textbf{{Model}} & \textbf{{Signal}} & "
        rf"\textbf{{Hit@{window}}} & \textbf{{Norm. Hit@{window}}} & "
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
        grid_nonrandom_count = sum(1 for m in grid_methods if not m.get("is_random"))
        grid_cell_used = False
        first_model = True

        for model_key in MODEL_ORDER:
            group = [m for m in grid_methods if m.get("model_key") == model_key]
            if not group:
                continue
            if not first_model:
                latex.append(r"\addlinespace[1pt]")
            first_model = False
            markers_by_metric = {
                metric: _rank_markers([_metric_mean(m, metric) for m in group])
                for metric in metric_order
            }
            for idx, method in enumerate(group):
                grid_cell = (
                    rf"\multirow{{{grid_nonrandom_count}}}{{*}}{{{grid_label}}}"
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
                    marker = markers_by_metric[metric][idx]
                    line.append(_apply_marker(_metric_cell(method, metric), marker))
                line.append(str(method["n"]) if method["n"] is not None else "-")
                latex.append(" & ".join(line) + r" \\")

        random_rows = [m for m in grid_methods if m.get("is_random")]
        if random_rows:
            for random in random_rows:
                line = [
                    "",
                    random["model"],
                    random["signal"],
                    *[_metric_cell(random, metric) for metric in metric_order],
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
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_json, results)
    args.output_tex.write_text(build_latex(results))

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_tex}")
    for method in results["methods"]:
        metrics = method["metrics"]
        print(
            f"{method['grid']:>10} | {method['model']} | {method['signal']} | "
            f"Hit@{results['window']}={_fmt_pct(metrics['hit7'])} | "
            f"NormHit={_fmt_pct(metrics['norm_hit7'])} | "
            f"MAP={_fmt_pct(metrics['map'])} | "
            f"NormMAP={_fmt_pct(metrics['norm_map'])} | n={method['n']}"
        )


if __name__ == "__main__":
    main()
