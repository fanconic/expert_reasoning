"""Build natural-error Hit@7/MAP table after mean-bucketing scores into 15-token intervals."""

from __future__ import annotations

import argparse
import copy
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


DEFAULT_OUTPUT = (
    Path("localisation/chatgpt_step_perturbations")
    / "localisation_chatgpt_step_sft_interval15_hit7_map.tex"
)
DEFAULT_WINDOW = 7
DEFAULT_STRIDE = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--include-random",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one empirical random-location row averaged over the same 15-token grids.",
    )
    return parser.parse_args()


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


def _chance_for_units(n_units: int, targets: list[int], window: int) -> float:
    if n_units <= 0 or not targets:
        return float("nan")
    mask = np.zeros(n_units, dtype=bool)
    for target in targets:
        if 0 <= int(target) < n_units:
            lo = max(0, int(target) - int(window))
            hi = min(n_units - 1, int(target) + int(window))
            mask[lo : hi + 1] = True
    return float(mask.mean())


def _force_interval_row(row: dict[str, Any], stride: int) -> dict[str, Any]:
    forced = copy.copy(row)
    forced["localization_mode"] = "bucket"
    forced["partial_fixed_stride"] = int(stride)
    return forced


def _method_specs(root_dir: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        specs.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "signal": "Reward dense, mean/15",
                "path": root_dir / f"{model_key}_full_reward_localisation" / "pair_details.jsonl",
                "seq_key": "pert_score_seq",
                "detector": "largest_drop",
                "is_policy": False,
                "force_interval": True,
                "hit_key": None,
            }
        )
        specs.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "signal": "Reward interval",
                "path": root_dir / f"{model_key}_partial_fixed_reward_localisation" / "pair_details.jsonl",
                "seq_key": "pert_score_seq",
                "detector": "largest_drop",
                "is_policy": False,
                "force_interval": False,
                "hit_key": "reward_hit1_at_window",
            }
        )
        specs.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "signal": "SFT log-probability, mean/15",
                "path": root_dir / f"{model_key}_sft_policy_token_baselines" / "policy_token_baselines.jsonl",
                "seq_key": "pert_policy_log_probs",
                "detector": "largest_drop",
                "is_policy": True,
                "force_interval": True,
                "hit_key": None,
            }
        )
        specs.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "signal": "SFT entropy, mean/15",
                "path": root_dir / f"{model_key}_sft_policy_token_baselines" / "policy_token_baselines.jsonl",
                "seq_key": "pert_policy_entropies",
                "detector": "largest_spike",
                "is_policy": True,
                "force_interval": True,
                "hit_key": None,
            }
        )
    return specs


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    seq_key: str,
    detector: str,
    is_policy: bool,
    force_interval: bool,
    hit_key: str | None,
    stride: int,
    window: int,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    hit_vals = []
    ap_vals = []
    random_items: list[tuple[Any, float, float]] = []
    target_sources = set()

    for idx, row in enumerate(rows):
        seq = row.get(seq_key, [])
        if not isinstance(seq, list):
            continue
        work_row = _force_interval_row(row, stride) if force_interval else row
        target_sources.add(str(row.get("target_position_source")))

        first_targets = _row_targets(row, "first_diff", is_policy=is_policy)
        unit_first, first_units, first_w = _unit_sequence_and_targets(
            work_row, seq, first_targets, int(window)
        )
        first_scores, first_indices = _transition_scores(unit_first, detector)
        stored_hit = None
        if hit_key:
            stored_hit = ((row.get(hit_key) or {}).get(str(window)))
        if stored_hit is None:
            hit_vals.append(_hit_at_window_from_scores(first_scores, first_indices, first_units, first_w))
            random_hit = _chance_for_units(int(unit_first.shape[0]), first_units, first_w)
        else:
            hit_vals.append(float(stored_hit))
            random_hit = ((row.get("random_hit1_at_window") or {}).get(str(window)))
            random_hit = float(random_hit) if random_hit is not None else float("nan")

        edit_targets = _row_targets(row, "edit_span", is_policy=is_policy)
        unit_edit, edit_units, _edit_w = _unit_sequence_and_targets(work_row, seq, edit_targets, 0)
        edit_scores, edit_indices = _transition_scores(unit_edit, detector)
        ap_vals.append(_average_precision(edit_scores, edit_indices, edit_units))

        n_candidates = max(0, int(unit_edit.shape[0]) - 1)
        n_relevant = _transition_exact_relevant_count(int(unit_edit.shape[0]), edit_units)
        random_ap = _expected_random_average_precision(n_candidates, n_relevant)
        random_items.append((row.get("prompt_idx", idx), random_hit, random_ap))

    return {
        "n": len(rows),
        "target_sources": sorted(target_sources),
        "hit": _mean_ci(hit_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
        "map": _mean_ci(ap_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
        "random_items": random_items,
    }


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table_rows: list[dict[str, Any]] = []
    random_by_prompt: dict[Any, dict[str, list[float]]] = {}
    target_sources = set()
    row_counts = set()

    for spec in _method_specs(args.root_dir):
        if not spec["path"].exists():
            raise FileNotFoundError(f"Missing detail file: {spec['path']}")
        detail_rows = _load_jsonl(spec["path"])
        scored = _score_rows(
            detail_rows,
            seq_key=spec["seq_key"],
            detector=spec["detector"],
            is_policy=bool(spec["is_policy"]),
            force_interval=bool(spec["force_interval"]),
            hit_key=spec.get("hit_key"),
            stride=int(args.stride),
            window=int(args.window),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_alpha=float(args.bootstrap_alpha),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        target_sources.update(scored["target_sources"])
        row_counts.add(int(scored["n"]))

        table_rows.append(
            {
                "model_key": spec["model_key"],
                "model": spec["model"],
                "signal": spec["signal"],
                "hit_cell": _fmt_pct(scored["hit"]),
                "hit_mean": scored["hit"].get("mean"),
                "map_cell": _fmt_pct(scored["map"]),
                "map_mean": scored["map"].get("mean"),
                "n": scored["n"],
                "is_random": False,
            }
        )

        if bool(args.include_random):
            for prompt_key, random_hit, random_map in scored["random_items"]:
                bucket = random_by_prompt.setdefault(prompt_key, {"hit": [], "map": []})
                if math.isfinite(float(random_hit)):
                    bucket["hit"].append(float(random_hit))
                if math.isfinite(float(random_map)):
                    bucket["map"].append(float(random_map))

    if bool(args.include_random) and random_by_prompt:
        random_hit = [float(np.mean(v["hit"])) for v in random_by_prompt.values() if v["hit"]]
        random_map = [float(np.mean(v["map"])) for v in random_by_prompt.values() if v["map"]]
        hit_metric = _mean_ci(
            random_hit,
            int(args.bootstrap_samples),
            float(args.bootstrap_alpha),
            int(args.bootstrap_seed),
        )
        map_metric = _mean_ci(
            random_map,
            int(args.bootstrap_samples),
            float(args.bootstrap_alpha),
            int(args.bootstrap_seed),
        )
        table_rows.append(
            {
                "model_key": "__random__",
                "model": r"\textsc{Random}",
                "signal": "Shared 15-token interval chance",
                "hit_cell": _fmt_pct(hit_metric),
                "hit_mean": hit_metric.get("mean"),
                "map_cell": _fmt_pct(map_metric),
                "map_mean": map_metric.get("mean"),
                "n": hit_metric.get("n"),
                "is_random": True,
            }
        )

    metadata = {
        "row_counts": sorted(row_counts),
        "target_sources": sorted(target_sources),
    }
    return table_rows, metadata


def build_latex(rows: list[dict[str, Any]], metadata: dict[str, Any], stride: int, window: int) -> str:
    n_rows = metadata["row_counts"][0] if len(metadata["row_counts"]) == 1 else "mixed"
    targets = metadata["target_sources"]
    target = targets[0] if len(targets) == 1 else ", ".join(targets)
    target_latex = target.replace("_", r"\_")

    latex: list[str] = []
    latex.append(r"% Requires \usepackage{multirow}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{5pt}")
    latex.append(
        r"\caption{\textbf{Interval-resolution natural-error localisation.} "
        rf"Dense reward and SFT token baselines are averaged into non-overlapping {stride}-token intervals before detection; "
        rf"the fixed-interval reward uses its native interval scores. All methods are scored on the same {n_rows} "
        rf"ChatGPT-edited GSM8K clean/perturbed pairs. Hit@{window} uses the first changed token target "
        rf"({target_latex}); MAP uses the broader edited token span. "
        rf"Values are percentages with 95\% bootstrap CI half-widths. "
        rf"\textbf{{Bold}} marks the best and \underline{{underlining}} the second-best value within each model and metric.}}"
    )
    latex.append(r"\label{tab:localisation_chatgpt_step_sft_interval15_hit7_map}")
    latex.append(r"\begin{tabular}{llccc}")
    latex.append(r"\toprule")
    latex.append(
        rf"\textbf{{Model}} & \textbf{{Signal}} & \textbf{{Hit@{window} (\%)}} & "
        r"\textbf{MAP (\%)} & \textbf{n} \\"
    )
    latex.append(r"\midrule")

    first_group = True
    for model_key in MODEL_ORDER:
        group_rows = [row for row in rows if row.get("model_key") == model_key]
        if not group_rows:
            continue
        hit_markers = _rank_markers([row.get("hit_mean") for row in group_rows])
        map_markers = _rank_markers([row.get("map_mean") for row in group_rows])
        if not first_group:
            latex.append(r"\midrule")
        first_group = False
        for idx, row in enumerate(group_rows):
            model_cell = (
                rf"\multirow{{{len(group_rows)}}}{{*}}{{{MODEL_LABELS[model_key]}}}"
                if idx == 0
                else ""
            )
            line = [
                model_cell,
                row["signal"],
                _apply_marker(row["hit_cell"], hit_markers[idx]),
                _apply_marker(row["map_cell"], map_markers[idx]),
                str(row["n"]) if row["n"] is not None else "-",
            ]
            latex.append(" & ".join(line) + r" \\")

    random_rows = [row for row in rows if row.get("model_key") == "__random__"]
    if random_rows:
        latex.append(r"\midrule")
        for row in random_rows:
            line = [
                row["model"],
                row["signal"],
                row["hit_cell"],
                row["map_cell"],
                str(row["n"]) if row["n"] is not None else "-",
            ]
            latex.append(" & ".join(line) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    return "\n".join(latex)


def main() -> None:
    args = parse_args()
    rows, metadata = collect_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_latex(
            rows=rows,
            metadata=metadata,
            stride=int(args.stride),
            window=int(args.window),
        )
    )
    print(f"Wrote {args.output}")
    print(f"Row counts: {metadata['row_counts']}")
    print(f"Target sources: {metadata['target_sources']}")
    for row in rows:
        print(
            f"{row['model']} | {row['signal']} | "
            f"Hit@{int(args.window)}={row['hit_cell']} | MAP={row['map_cell']} | n={row['n']}"
        )


if __name__ == "__main__":
    main()
