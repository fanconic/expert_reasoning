import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path("localisation")
DEFAULT_RUN_DIRS = [
    "runs/qwen7b_sft/qwen7b/full",
    "runs/qwen7b_sft/qwen7b/partial_fixed",
]
DEFAULT_OUTPUT = ROOT_DIR / "localisation_policy_token_baselines_hit1_hit7.tex"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42

METRICS = [
    (
        "prob_largest_drop",
        "SFT token probability",
        "prob_hit1_at_window",
    ),
    (
        "logprob_largest_drop",
        "SFT log-probability",
        "logprob_hit1_at_window",
    ),
    (
        "entropy_largest_spike",
        "SFT entropy",
        "entropy_hit1_at_window",
    ),
    (
        "random_location",
        "Random location",
        "random_hit1_at_window",
    ),
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _row_id(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("prompt_idx"),
        row.get("severity"),
        row.get("variant_idx"),
        row.get("clean_generation_idx"),
        tuple(row.get("perturb_fns", []) or []),
    )


def _selected_rows(rows: list[dict[str, Any]], windows: list[int]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        keep = True
        for _metric_name, _label, key in METRICS:
            values = row.get(key)
            if not isinstance(values, dict):
                keep = False
                break
            for window in windows:
                if values.get(str(window)) is None:
                    keep = False
                    break
            if not keep:
                break
        if keep:
            selected.append(row)
    return selected


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _ci_halfwidth(values: list[float], samples: int, alpha: float, seed: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n_boot = max(100, int(samples))
    alpha = min(max(float(alpha), 1e-6), 0.5)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float((hi - lo) / 2.0)


def _fmt_pct(mean: float | None, ci: float | None) -> str:
    if mean is None:
        return "-"
    mean_pct = 100.0 * mean
    if ci is None:
        return f"{mean_pct:.2f}"
    return f"{mean_pct:.2f} $\\pm$ {100.0 * ci:.2f}"


def _compute_table(
    rows: list[dict[str, Any]],
    windows: list[int],
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for metric_name, label, key in METRICS:
        by_window: dict[str, Any] = {"label": label}
        for window in windows:
            vals = [float(row[key][str(window)]) for row in rows]
            by_window[str(window)] = {
                "mean": _mean(vals),
                "ci": _ci_halfwidth(
                    vals,
                    samples=bootstrap_samples,
                    alpha=bootstrap_alpha,
                    seed=bootstrap_seed,
                ),
                "n": len(vals),
            }
        table[metric_name] = by_window
    return table


def _render_latex(table: dict[str, dict[str, Any]], windows: list[int], n_rows: int) -> str:
    conf_level = (1.0 - float(BOOTSTRAP_ALPHA)) * 100.0
    if windows != [1, 7]:
        window_header = " & ".join([rf"\textbf{{Hit@1@{w} (\%)}}" for w in windows])
    else:
        window_header = r"\textbf{Hit@1 (\%)} & \textbf{Hit@7 (\%)}"

    latex: list[str] = []
    latex.append(r"% Requires \usepackage{booktabs}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\renewcommand{\arraystretch}{1.08}")
    latex.append(r"\begin{tabular}{l" + " r" * len(windows) + r"}")
    latex.append(r"\toprule")
    latex.append(rf"\textbf{{Baseline}} & {window_header} \\")
    latex.append(r"\midrule")
    for metric_name, _label, _key in METRICS:
        row = table[metric_name]
        cells = [
            _fmt_pct(row[str(window)]["mean"], row[str(window)]["ci"])
            for window in windows
        ]
        latex.append(f"{row['label']} & " + " & ".join(cells) + r" \\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(
        rf"\caption{{\textbf{{Generator-side localisation baselines on GSM8K "
        rf"pregenerated traces.}} Log-probability uses the largest-drop detector; "
        rf"entropy uses the largest-spike detector; random is the analytic "
        rf"uniform-location baseline. Hit@1 and Hit@7 are computed from the exact "
        rf"same {n_rows} severity-1 Table-5-valid corrupted traces, reported as "
        rf"percentage mean $\pm$ bootstrapped {conf_level:.0f}\% CI half-width.}}"
    )
    latex.append(r"\label{tab:localisation_policy_token_baselines_hit1_hit7}")
    latex.append(r"\end{table}")
    return "\n".join(latex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LaTeX table for generator log-probability, entropy, and random localisation baselines."
    )
    parser.add_argument("--root-dir", type=Path, default=ROOT_DIR)
    parser.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 7])
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = [int(w) for w in args.windows]

    loaded: list[tuple[str, list[dict[str, Any]]]] = []
    for run_name in args.run_dirs:
        path = args.root_dir / run_name / "policy_token_baselines" / "policy_token_baselines.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing policy baseline details: {path}")
        rows = _selected_rows(_load_jsonl(path), windows)
        if not rows:
            raise ValueError(f"No rows with all requested metric windows in {path}")
        loaded.append((run_name, rows))

    ref_name, ref_rows = loaded[0]
    ref_ids = [_row_id(row) for row in ref_rows]
    for run_name, rows in loaded[1:]:
        ids = [_row_id(row) for row in rows]
        if ids != ref_ids:
            raise ValueError(
                f"Row filter mismatch between {ref_name} and {run_name}; "
                "refusing to build a mixed-filter table."
            )

    global BOOTSTRAP_SAMPLES, BOOTSTRAP_ALPHA, BOOTSTRAP_SEED
    BOOTSTRAP_SAMPLES = int(args.bootstrap_samples)
    BOOTSTRAP_ALPHA = float(args.bootstrap_alpha)
    BOOTSTRAP_SEED = int(args.bootstrap_seed)

    table = _compute_table(
        ref_rows,
        windows=windows,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_alpha=BOOTSTRAP_ALPHA,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    latex = _render_latex(table, windows=windows, n_rows=len(ref_rows))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as f:
        f.write(latex)
        f.write("\n")

    print(f"Wrote LaTeX table to {args.output_file}")
    print(f"Runs checked: {', '.join(name for name, _rows in loaded)}")
    print(f"Shared rows: {len(ref_rows)}")
    for metric_name, row in table.items():
        parts = []
        for window in windows:
            vals = row[str(window)]
            parts.append(f"Hit@{window}={vals['mean']:.4f} +/- {vals['ci']:.4f}")
        print(f"{metric_name}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
