"""Build a compact natural-error Hit@1/Hit@7 table with SFT token baselines."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_generators.table_chatgpt_step_localisation import (
    DEFAULT_ROOT,
    DEFAULT_WINDOWS,
    MODEL_LABELS,
    MODEL_ORDER,
    _check_common_metadata,
    _latex_texttt,
    collect_rows,
)


DEFAULT_OUTPUT = (
    Path("localisation/chatgpt_step_perturbations")
    / "localisation_chatgpt_step_sft_hit1_hit7.tex"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
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


def _is_second(mean: float, best: float | None, second: float | None) -> bool:
    return (
        best is not None
        and second is not None
        and abs(float(mean) - float(best)) > 1e-12
        and abs(float(mean) - float(second)) <= 1e-12
    )


def _best_second_by_col(rows: list[dict[str, Any]], n_cols: int) -> list[tuple[float | None, float | None]]:
    result: list[tuple[float | None, float | None]] = []
    for col_idx in range(n_cols):
        vals = sorted(
            {
                round(float(row["means"][col_idx]), 12)
                for row in rows
                if row.get("means") and row["means"][col_idx] is not None
            },
            reverse=True,
        )
        best = vals[0] if vals else None
        second = vals[1] if len(vals) > 1 else None
        result.append((best, second))
    return result


def _keep_rebuttal_row(row: dict[str, Any]) -> bool:
    if row.get("model_key") == "__random__":
        return True
    signal = str(row.get("signal", ""))
    if signal.startswith("Base token"):
        return False
    return signal in {
        "Reward dense",
        "Reward interval",
        "SFT token log-probability",
        "SFT token entropy",
    }


def build_latex(rows: list[dict[str, Any]], windows: list[int], metadata: dict[str, Any]) -> str:
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
        rf"The target is the first perturbed-step token that differs from the clean step ({target_latex}). "
        rf"Reward and SFT log-probability use the largest-drop detector; SFT entropy uses the largest-spike detector. "
        rf"Values are Hit@1@W in percent with 95\% bootstrap CI half-widths. "
        rf"\textbf{{Bold}} marks the best and \underline{{underlining}} the second-best value within each model and metric.}}"
    )
    latex.append(r"\label{tab:localisation_chatgpt_step_sft_hit1_hit7}")
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
        best_second = _best_second_by_col(group_rows, len(windows))
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
                best, second = best_second[col_idx]
                if mean is None:
                    formatted_cells.append(cell)
                elif best is not None and abs(float(mean) - float(best)) <= 1e-12:
                    formatted_cells.append(_bold_mean(cell))
                elif _is_second(float(mean), best, second):
                    formatted_cells.append(_underline_mean(cell))
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
    windows = [int(window) for window in args.windows]
    rows = collect_rows(
        root_dir=args.root_dir,
        windows=windows,
        include_policy_random=bool(args.include_random),
        include_reward_random=False,
    )
    rows = [row for row in rows if _keep_rebuttal_row(row)]
    metadata = _check_common_metadata(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_latex(rows=rows, windows=windows, metadata=metadata))

    print(f"Wrote {args.output}")
    print(f"Row counts: {metadata['row_counts']}")
    print(f"Target sources: {metadata['targets']}")
    for row in rows:
        print(
            f"{row['model']} | {row['signal']} | "
            + " | ".join(row["cells"])
            + f" | n={row['n']}"
        )


if __name__ == "__main__":
    main()
