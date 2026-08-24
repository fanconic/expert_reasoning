"""Build fair-grid localisation metric tables for the new natural-error datasets."""

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
    _transition_exact_relevant_count,
    _transition_scores,
    _unit_sequence_and_targets,
)
from src.plot_generators.table_chatgpt_step_localisation import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
)


DATASET_SPECS = {
    "expert_step": {
        "title": "Expert-trace edited GSM8K errors",
        "root": Path("localisation/expert_step_perturbations/scores"),
        "output_stem": "localisation_expert_step_fair_grid_metrics",
        "caption_name": "expert-trace edited GSM8K pairs",
    },
    "natural_wrong_sft": {
        "title": "Naturally wrong Qwen7B-SFT GSM8K generations",
        "root": Path("localisation/natural_wrong_sft/scores"),
        "output_stem": "localisation_natural_wrong_sft_fair_grid_metrics",
        "caption_name": "naturally wrong Qwen7B-SFT GSM8K pairs",
        "strict_input": Path(
            "localisation/natural_wrong_sft/scores/_inputs/"
            "natural_wrong_sft_valid_target_char_span_actual_wrong_answer.jsonl"
        ),
    },
}

DEFAULT_WINDOW = 7
DEFAULT_STRIDE = 15

GRID_LABELS = {
    "token": "Token grid",
    "interval15": "15-token interval grid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS),
        choices=sorted(DATASET_SPECS),
    )
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


def _label_targets(row: dict[str, Any], *, is_policy: bool) -> list[int]:
    key = "policy_changed_token_positions" if is_policy else "changed_token_positions"
    values = row.get(key, [])
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            pass
    return out


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("prompt_idx"),
        row.get("variant_idx"),
        row.get("clean_generation_idx"),
    )


def _strict_keys(path: Path | None) -> set[tuple[Any, ...]] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Missing strict input file: {path}")
    rows = _load_jsonl(path)
    keys = {_row_key(row) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"Strict input has duplicate row keys: {path}")
    return keys


def _filter_to_keys(rows: list[dict[str, Any]], keys: set[tuple[Any, ...]] | None) -> list[dict[str, Any]]:
    if keys is None:
        return rows
    return [row for row in rows if _row_key(row) in keys]


def _example_key(row: dict[str, Any], idx: int) -> tuple[Any, ...]:
    prompt_idx = row.get("prompt_idx")
    variant_idx = row.get("variant_idx")
    clean_generation_idx = row.get("clean_generation_idx")
    if prompt_idx is None:
        return ("row", idx)
    if variant_idx is None:
        variant_idx = 0
    if clean_generation_idx is None:
        clean_generation_idx = 0
    return ("prompt", prompt_idx, "variant", variant_idx, "clean", clean_generation_idx)


def _method_specs(root_dir: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        interval_path = root_dir / f"{model_key}_partial_fixed_reward_localisation" / "pair_details.jsonl"
        if model_key == "qwen7b":
            rebuttal_interval_path = (
                root_dir
                / "qwen7b_partial_fixed_rebuttal_restart_reward_localisation"
                / "pair_details.jsonl"
            )
            if rebuttal_interval_path.exists():
                interval_path = rebuttal_interval_path
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
                    "path": interval_path,
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
        targets = _label_targets(row, is_policy=is_policy)

        unit_hit, hit_units, hit_w = _unit_sequence_and_targets(work_row, seq, targets, int(window))
        hit_scores, hit_indices = _transition_scores(unit_hit, detector)
        hit = _hit_at_window_from_scores(hit_scores, hit_indices, hit_units, hit_w)
        hit_chance = _transition_chance(int(unit_hit.shape[0]), hit_units, hit_w)

        unit_map, map_units, _ = _unit_sequence_and_targets(work_row, seq, targets, 0)
        map_scores, map_indices = _transition_scores(unit_map, detector)
        ap = _average_precision(map_scores, map_indices, map_units)
        n_candidates = max(0, int(unit_map.shape[0]) - 1)
        n_relevant = _transition_exact_relevant_count(int(unit_map.shape[0]), map_units)
        ap_chance = _expected_random_average_precision(n_candidates, n_relevant)

        hit_values.append(hit)
        norm_hit_values.append(_normalize(hit, hit_chance))
        ap_values.append(ap)
        norm_ap_values.append(_normalize(ap, ap_chance))
        random_items.append((_example_key(row, idx), hit_chance, ap_chance))

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


def build_results(
    dataset_key: str,
    *,
    window: int,
    stride: int,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset_key]
    root_dir = Path(spec["root"])
    strict_key_set = _strict_keys(spec.get("strict_input"))
    methods: list[dict[str, Any]] = []
    metadata = {"row_counts": set(), "target_sources": set()}

    for grid in ["token", "interval15"]:
        random_by_prompt: dict[Any, dict[str, list[float]]] = {}
        for method_spec in _method_specs(root_dir):
            if not method_spec["path"].exists():
                raise FileNotFoundError(f"Missing detail file: {method_spec['path']}")
            rows = _filter_to_keys(_load_jsonl(method_spec["path"]), strict_key_set)
            if strict_key_set is not None and len(rows) != len(strict_key_set):
                raise ValueError(
                    f"{method_spec['path']} has {len(rows)} strict rows, "
                    f"expected {len(strict_key_set)}."
                )
            scored = _score_rows(
                rows,
                grid=grid,
                stride=int(stride),
                window=int(window),
                seq_key=method_spec["seq_key"],
                detector=method_spec["detector"],
                is_policy=bool(method_spec["is_policy"]),
                bootstrap_samples=int(bootstrap_samples),
                bootstrap_alpha=float(bootstrap_alpha),
                bootstrap_seed=int(bootstrap_seed),
            )
            metadata["row_counts"].add(int(scored["n"]))
            metadata["target_sources"].update(scored["target_sources"])
            methods.append(
                {
                    "grid": grid,
                    "model_key": method_spec["model_key"],
                    "model": method_spec["model"],
                    "method_key": method_spec["method_key"],
                    "signal": method_spec[f"signal_{grid}"],
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
                bootstrap_samples=int(bootstrap_samples),
                bootstrap_alpha=float(bootstrap_alpha),
                bootstrap_seed=int(bootstrap_seed),
            )
        )

    return {
        "dataset": dataset_key,
        "dataset_title": spec["title"],
        "root_dir": str(root_dir),
        "window": int(window),
        "stride": int(stride),
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "alpha": float(bootstrap_alpha),
            "seed": int(bootstrap_seed),
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
    dataset_spec = DATASET_SPECS[str(results["dataset"])]
    window = int(results["window"])
    stride = int(results["stride"])
    row_counts = results["metadata"]["row_counts"]
    n_rows = row_counts[0] if len(row_counts) == 1 else "mixed"
    target_sources = results["metadata"]["target_sources"]
    target = target_sources[0] if len(target_sources) == 1 else ", ".join(target_sources)
    target_latex = target.replace("_", r"\_")
    metric_order = ["hit7", "norm_hit7", "map", "norm_map"]

    latex: list[str] = []
    latex.append(r"% Requires \usepackage{booktabs,multirow,arydshln}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\setlength{\tabcolsep}{3pt}")
    latex.append(
        rf"\caption{{\textbf{{Fair-grid localisation on {dataset_spec['title']}.}} "
        rf"All rows use the same {n_rows} labelled {dataset_spec['caption_name']}. "
        rf"The token-grid panel evaluates all methods over token transition candidates, expanding interval scores to tokens. "
        rf"The interval-grid panel evaluates all methods over common {stride}-token intervals: dense reward and SFT token "
        rf"baselines are mean-pooled, while the interval reward uses its native intervals. "
        rf"Hit@{window} and MAP both use the labelled target span ({target_latex}); MAP ranks all transition positions. "
        rf"Normalized metrics subtract the expected random value for the corresponding grid, $(m-r)/(1-r)$. "
        rf"Values are percentages with 95\% bootstrap CI half-widths. "
        rf"\textbf{{Bold}} marks the best and \underline{{underlining}} the second-best value within each model, grid, and metric.}}"
    )
    latex.append(rf"\label{{tab:{dataset_spec['output_stem']}}}")
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
                latex.append(r"\cdashline{2-8}")
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
    for dataset_key in args.datasets:
        results = build_results(
            dataset_key,
            window=int(args.window),
            stride=int(args.stride),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_alpha=float(args.bootstrap_alpha),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        spec = DATASET_SPECS[dataset_key]
        out_dir = Path(spec["root"]).parent
        json_path = out_dir / f"{spec['output_stem']}.json"
        tex_path = out_dir / f"{spec['output_stem']}.tex"
        _write_json(json_path, results)
        tex_path.write_text(build_latex(results))
        print(f"Wrote {json_path}")
        print(f"Wrote {tex_path}")
        for method in results["methods"]:
            metrics = method["metrics"]
            print(
                f"{dataset_key:>17} | {method['grid']:>10} | {method['model']} | {method['signal']} | "
                f"Hit@{results['window']}={_fmt_pct(metrics['hit7'])} | "
                f"NormHit={_fmt_pct(metrics['norm_hit7'])} | "
                f"MAP={_fmt_pct(metrics['map'])} | "
                f"NormMAP={_fmt_pct(metrics['norm_map'])} | n={method['n']}"
            )


if __name__ == "__main__":
    main()
