"""Build a LaTeX table for ChatGPT-step localisation scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("localisation/chatgpt_step_perturbations/scores")
DEFAULT_OUTPUT = (
    Path("localisation/chatgpt_step_perturbations")
    / "localisation_chatgpt_step_combined_hit1_hit7.tex"
)
DEFAULT_WINDOWS = [1, 7]
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42

MODEL_ORDER = ["qwen7b", "llama8b", "qwen4b"]
MODEL_LABELS = {
    "qwen7b": r"\textsc{Qwen2.5-7B}",
    "llama8b": r"\textsc{Llama3.1-8B}",
    "qwen4b": r"\textsc{Qwen3-4B}",
}
SFT_MODEL_LABELS = {
    "qwen7b": r"\textsc{Qwen2.5-7B-SFT}",
    "llama8b": r"\textsc{Llama3.1-8B-SFT}",
    "qwen4b": r"\textsc{Qwen3-4B-SFT}",
}
POLICY_VARIANTS = [
    ("base", "Base"),
    ("sft", "SFT"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--include-policy-random",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one shared generator-token random-location baseline at the end.",
    )
    parser.add_argument(
        "--include-reward-random",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also include reward-token-space chance rows for each reward run.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _mean_ci(values: list[float], samples: int = BOOTSTRAP_SAMPLES) -> dict[str, float | int | None]:
    finite_values = []
    for value in values:
        try:
            parsed = float(value)
        except Exception:
            continue
        if np.isfinite(parsed):
            finite_values.append(parsed)
    vals = np.asarray(finite_values, dtype=np.float64)
    n = int(vals.shape[0])
    if n == 0:
        return {"mean": None, "ci_halfwidth": None, "n": 0}
    mean = float(vals.mean())
    if n == 1:
        return {"mean": mean, "ci_halfwidth": 0.0, "n": 1}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(max(100, int(samples)), n))
    boot_means = vals[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, BOOTSTRAP_ALPHA / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - BOOTSTRAP_ALPHA / 2.0))
    return {"mean": mean, "ci_halfwidth": float((hi - lo) / 2.0), "n": n}


def _metric_cell(summary: dict[str, Any], metric: str, window: int) -> tuple[str, int | None, float | None]:
    vals = ((summary.get("metrics") or {}).get(metric) or {}).get(str(window), {})
    mean = vals.get("mean")
    ci = vals.get("ci_halfwidth")
    if mean is None:
        return "-", vals.get("n"), None
    if ci is None:
        return f"{100.0 * float(mean):.2f}", vals.get("n"), float(mean)
    return f"{100.0 * float(mean):.2f} $\\pm$ {100.0 * float(ci):.2f}", vals.get("n"), float(mean)


def _row_cells(summary: dict[str, Any], metric: str, windows: list[int]) -> tuple[list[str], int | None, list[float | None]]:
    cells = []
    means = []
    n_vals = []
    for window in windows:
        cell, n, mean = _metric_cell(summary, metric, window)
        cells.append(cell)
        means.append(mean)
        if n is not None:
            n_vals.append(int(n))
    n = n_vals[0] if n_vals and all(x == n_vals[0] for x in n_vals) else None
    return cells, n, means


def _cells_from_window_metrics(metrics: dict[str, dict[str, float | int | None]], windows: list[int]) -> tuple[list[str], int | None, list[float | None]]:
    cells = []
    means = []
    n_vals = []
    for window in windows:
        vals = metrics.get(str(window), {})
        mean = vals.get("mean")
        ci = vals.get("ci_halfwidth")
        if mean is None:
            cells.append("-")
            means.append(None)
        elif ci is None:
            cells.append(f"{100.0 * float(mean):.2f}")
            means.append(float(mean))
        else:
            cells.append(f"{100.0 * float(mean):.2f} $\\pm$ {100.0 * float(ci):.2f}")
            means.append(float(mean))
        if vals.get("n") is not None:
            n_vals.append(int(vals["n"]))
    n = n_vals[0] if n_vals and all(x == n_vals[0] for x in n_vals) else None
    return cells, n, means


def _bold_mean(cell: str) -> str:
    return re.sub(r"(-?\d+(?:\.\d+)?)", r"\\textbf{\1}", cell, count=1)


def _latex_texttt(text: str) -> str:
    escaped = (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )
    return rf"\texttt{{{escaped}}}"


def _check_common_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_counts = sorted({int(r["summary"].get("n_rows_scored", -1)) for r in rows})
    targets = sorted({str(r["summary"].get("target_position_source")) for r in rows})
    windows = sorted({tuple(r["summary"].get("windows", [])) for r in rows})
    return {
        "row_counts": row_counts,
        "targets": targets,
        "windows": windows,
    }


def _shared_policy_random_row(root_dir: Path, windows: list[int]) -> dict[str, Any]:
    values_by_key: dict[Any, dict[int, list[float]]] = {}
    summaries = []
    for model in MODEL_ORDER:
        for variant, _variant_label in POLICY_VARIANTS:
            detail_path = root_dir / f"{model}_{variant}_policy_token_baselines" / "policy_token_baselines.jsonl"
            summary_path = root_dir / f"{model}_{variant}_policy_token_baselines" / "policy_token_baselines_summary.json"
            if summary_path.exists():
                summaries.append(_load_json(summary_path))
            if not detail_path.exists():
                continue
            for idx, row in enumerate(_load_jsonl(detail_path)):
                key = row.get("prompt_idx", idx)
                random_hit = row.get("random_hit1_at_window") or {}
                bucket = values_by_key.setdefault(key, {window: [] for window in windows})
                for window in windows:
                    value = random_hit.get(str(window))
                    if value is not None:
                        bucket[window].append(float(value))

    metrics: dict[str, dict[str, float | int | None]] = {}
    if values_by_key:
        for window in windows:
            averaged_values = [
                float(np.mean(vals[window]))
                for vals in values_by_key.values()
                if vals.get(window)
            ]
            metrics[str(window)] = _mean_ci(averaged_values)
    else:
        for window in windows:
            vals = []
            for summary in summaries:
                metric = ((summary.get("metrics") or {}).get("random_location") or {}).get(str(window), {})
                if metric.get("mean") is not None:
                    vals.append(float(metric["mean"]))
            metrics[str(window)] = {
                "mean": float(np.mean(vals)) if vals else None,
                "ci_halfwidth": None,
                "n": int(summaries[0].get("n_rows_scored", 0)) if summaries else None,
            }

    cells, n, means = _cells_from_window_metrics(metrics, windows)
    summary = summaries[0] if summaries else {"n_rows_scored": n, "target_position_source": None, "windows": windows}
    return {
        "section": "random",
        "model_key": "__random__",
        "model": r"\textsc{Random}",
        "signal": "Shared token-grid chance",
        "cells": cells,
        "means": means,
        "n": n,
        "summary": summary,
        "is_random": True,
    }


def collect_rows(root_dir: Path, windows: list[int], include_policy_random: bool, include_reward_random: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        for density, signal_label in [
            ("full", "Reward dense"),
            ("partial_fixed", "Reward interval"),
        ]:
            summary_path = root_dir / f"{model}_{density}_reward_localisation" / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing reward summary: {summary_path}")
            summary = _load_json(summary_path)
            cells, n, means = _row_cells(summary, "reward_largest_drop", windows)
            rows.append(
                {
                    "section": "reward",
                    "model_key": model,
                    "model": MODEL_LABELS[model],
                    "signal": signal_label,
                    "cells": cells,
                    "means": means,
                    "n": n,
                    "summary": summary,
                    "is_random": False,
                }
            )
            if include_reward_random:
                cells, n, means = _row_cells(summary, "random_location", windows)
                rows.append(
                    {
                        "section": "reward",
                        "model_key": model,
                        "model": MODEL_LABELS[model],
                        "signal": f"Random ({signal_label.lower()} space)",
                        "cells": cells,
                        "means": means,
                        "n": n,
                        "summary": summary,
                        "is_random": True,
                    }
                )

    for model in MODEL_ORDER:
        for variant, variant_label in POLICY_VARIANTS:
            summary_path = root_dir / f"{model}_{variant}_policy_token_baselines" / "policy_token_baselines_summary.json"
            if not summary_path.exists():
                if variant == "base":
                    continue
                raise FileNotFoundError(f"Missing policy summary: {summary_path}")
            summary = _load_json(summary_path)
            for metric, signal_label in [
                ("logprob_largest_drop", f"{variant_label} token log-probability"),
                ("entropy_largest_spike", f"{variant_label} token entropy"),
            ]:
                cells, n, means = _row_cells(summary, metric, windows)
                rows.append(
                    {
                        "section": "policy",
                        "model_key": model,
                        "model": SFT_MODEL_LABELS[model] if variant == "sft" else MODEL_LABELS[model],
                        "signal": signal_label,
                        "cells": cells,
                        "means": means,
                        "n": n,
                        "summary": summary,
                        "is_random": False,
                    }
                )

    if include_policy_random:
        rows.append(_shared_policy_random_row(root_dir, windows))

    return rows


def build_latex(rows: list[dict[str, Any]], windows: list[int], metadata: dict[str, Any]) -> str:
    n_cols = 3 + len(windows)
    target = metadata["targets"][0] if len(metadata["targets"]) == 1 else ", ".join(metadata["targets"])
    target_latex = _latex_texttt(target)
    n_rows = metadata["row_counts"][0] if len(metadata["row_counts"]) == 1 else "mixed"

    latex: list[str] = []
    latex.append(r"% Requires \usepackage{multirow}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{5pt}")
    latex.append(
        r"\caption{\textbf{Localisation on ChatGPT-edited GSM8K reasoning traces.} "
        rf"All methods are scored on the same {n_rows} successful clean/perturbed pairs. "
        rf"The target location is the first perturbed-step token that differs from the clean step "
        rf"({target_latex}). Values are Hit@1@W in percent with 95\% bootstrap CI half-widths. "
        rf"The random row is the empirical chance baseline averaged across available policy token grids.}}"
    )
    latex.append(r"\label{tab:localisation_chatgpt_step_combined_hit1_hit7}")
    col_spec = "ll" + ("c" * len(windows)) + "c"
    latex.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")
    header = [r"\textbf{Model}", r"\textbf{Signal}"]
    header.extend([rf"\textbf{{Hit@{w} (\%)}}" for w in windows])
    header.append(r"\textbf{n}")
    latex.append(" & ".join(header) + r" \\")
    latex.append(r"\midrule")

    first_group = True
    for model_key in MODEL_ORDER:
        group_rows = [row for row in rows if row.get("model_key") == model_key]
        if not group_rows:
            continue
        best_by_col: list[float | None] = []
        method_rows = [row for row in group_rows if not row.get("is_random")]
        for col_idx in range(len(windows)):
            vals = [
                float(row["means"][col_idx])
                for row in method_rows
                if row.get("means") and row["means"][col_idx] is not None
            ]
            best_by_col.append(max(vals) if vals else None)
        if not first_group:
            latex.append(r"\midrule")
        first_group = False
        for idx, row in enumerate(group_rows):
            model_cell = (
                rf"\multirow{{{len(group_rows)}}}{{*}}{{{MODEL_LABELS[model_key]}}}"
                if idx == 0
                else ""
            )
            line = [model_cell, row["signal"]]
            formatted_cells = []
            for col_idx, cell in enumerate(row["cells"]):
                mean = row["means"][col_idx]
                best = best_by_col[col_idx]
                if mean is not None and best is not None and abs(float(mean) - float(best)) <= 1e-12:
                    formatted_cells.append(_bold_mean(cell))
                else:
                    formatted_cells.append(cell)
            line.extend(formatted_cells)
            line.append(str(row["n"]) if row["n"] is not None else "-")
            latex.append(" & ".join(line) + r" \\")

    random_rows = [row for row in rows if row.get("model_key") == "__random__"]
    if random_rows:
        latex.append(r"\midrule")
        for row in random_rows:
            line = [
                row["model"],
                row["signal"],
                *row["cells"],
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
    windows = [int(w) for w in args.windows]
    rows = collect_rows(
        root_dir=args.root_dir,
        windows=windows,
        include_policy_random=bool(args.include_policy_random),
        include_reward_random=bool(args.include_reward_random),
    )
    metadata = _check_common_metadata(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latex = build_latex(rows, windows=windows, metadata=metadata)
    args.output.write_text(latex)

    print(f"Wrote {args.output}")
    print(f"Row counts: {metadata['row_counts']}")
    print(f"Target sources: {metadata['targets']}")
    print(f"Windows: {metadata['windows']}")
    for row in rows:
        print(
            f"{row['section']:>6} | {row['model']} | {row['signal']} | "
            + " | ".join(row["cells"])
            + f" | n={row['n']}"
        )


if __name__ == "__main__":
    main()
