"""Build a natural-error table with Hit@7 and MAP plus SFT token baselines."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_generators.table_chatgpt_step_extra_metrics import (  # noqa: E402
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    build_results as build_extra_results,
)
from src.plot_generators.table_chatgpt_step_localisation import (  # noqa: E402
    DEFAULT_ROOT,
    MODEL_LABELS,
    MODEL_ORDER,
    _check_common_metadata,
    _latex_texttt,
    collect_rows,
)


DEFAULT_OUTPUT = (
    Path("localisation/chatgpt_step_perturbations")
    / "localisation_chatgpt_step_sft_hit7_map.tex"
)
DEFAULT_WINDOW = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--include-random",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one shared generator-token random-location baseline at the end.",
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


def _keep_rebuttal_row(row: dict[str, Any]) -> bool:
    if row.get("model_key") == "__random__":
        return True
    return row.get("signal") in {
        "Reward dense",
        "Reward interval",
        "SFT token log-probability",
        "SFT token entropy",
    }


def _rank_markers(values: list[float | None]) -> list[str]:
    unique = sorted({round(float(v), 12) for v in values if v is not None}, reverse=True)
    best = unique[0] if unique else None
    second = unique[1] if len(unique) > 1 else None
    markers = []
    for value in values:
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


def _extra_result_map(args: argparse.Namespace, window: int) -> dict[tuple[str, str], dict[str, Any]]:
    extra_args = SimpleNamespace(
        root_dir=args.root_dir,
        windows=[int(window)],
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_alpha=float(args.bootstrap_alpha),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    results = build_extra_results(extra_args)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for method in results["methods"]:
        if not _keep_rebuttal_row(method):
            continue
        out[(str(method.get("model_key")), str(method.get("signal")))] = method
    return out


def _combine_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window = int(args.window)
    hit_rows = collect_rows(
        root_dir=args.root_dir,
        windows=[window],
        include_policy_random=bool(args.include_random),
        include_reward_random=False,
    )
    hit_rows = [row for row in hit_rows if _keep_rebuttal_row(row)]
    metadata = _check_common_metadata(hit_rows)
    extra_by_key = _extra_result_map(args, window)

    combined: list[dict[str, Any]] = []
    for row in hit_rows:
        key = (str(row.get("model_key")), str(row.get("signal")))
        extra = extra_by_key.get(key)
        if extra is None:
            raise KeyError(f"Missing MAP metrics for {key}")
        map_metric = extra["metrics"]["ranking"]["edit_span_map"]
        combined.append(
            {
                **row,
                "hit7_cell": row["cells"][0],
                "hit7_mean": row["means"][0],
                "map_cell": _fmt_pct(map_metric),
                "map_mean": map_metric.get("mean"),
                "map_n": map_metric.get("n"),
            }
        )
    return combined, metadata


def build_latex(rows: list[dict[str, Any]], metadata: dict[str, Any], window: int) -> str:
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
        r"\caption{\textbf{Natural-error localisation with SFT token baselines.} "
        rf"All methods are scored on the same {n_rows} ChatGPT-edited GSM8K clean/perturbed pairs. "
        rf"Hit@{window} is measured at the first perturbed-step token that differs from the clean step "
        rf"({target_latex}). MAP is average precision over the broader edited token span, ranking positions by "
        rf"reward/log-probability drops or entropy spikes. Values are percentages with 95\% bootstrap CI half-widths. "
        rf"\textbf{{Bold}} marks the best and \underline{{underlining}} the second-best value within each model and metric.}}"
    )
    latex.append(r"\label{tab:localisation_chatgpt_step_sft_hit7_map}")
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
        hit_markers = _rank_markers([row.get("hit7_mean") for row in group_rows])
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
            n = row.get("n")
            line = [
                model_cell,
                row["signal"],
                _apply_marker(row["hit7_cell"], hit_markers[idx]),
                _apply_marker(row["map_cell"], map_markers[idx]),
                str(n) if n is not None else "-",
            ]
            latex.append(" & ".join(line) + r" \\")

    random_rows = [row for row in rows if row.get("model_key") == "__random__"]
    if random_rows:
        latex.append(r"\midrule")
        for row in random_rows:
            n = row.get("n")
            line = [
                row["model"],
                row["signal"],
                row["hit7_cell"],
                row["map_cell"],
                str(n) if n is not None else "-",
            ]
            latex.append(" & ".join(line) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    return "\n".join(latex)


def main() -> None:
    args = parse_args()
    rows, metadata = _combine_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_latex(rows=rows, metadata=metadata, window=int(args.window)))
    print(f"Wrote {args.output}")
    print(f"Row counts: {metadata['row_counts']}")
    print(f"Target sources: {metadata['targets']}")
    for row in rows:
        print(
            f"{row['model']} | {row['signal']} | "
            f"Hit@{int(args.window)}={row['hit7_cell']} | MAP={row['map_cell']} | n={row['n']}"
        )


if __name__ == "__main__":
    main()
