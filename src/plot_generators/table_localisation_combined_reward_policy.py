import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path("localisation")
DEFAULT_OUTPUT = ROOT_DIR / "localisation_combined_reward_policy_hit1_hit7.tex"
WINDOWS = [1, 7]
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42

MODEL_ORDER = [
    ("qwen7b", r"\textsc{Qwen2.5-7B}"),
    ("llama8b", r"\textsc{Llama3.1-8B}"),
    ("qwen4b", r"\textsc{Qwen3-4B}"),
]
MODE_ORDER = [
    ("full", "Reward (dense)"),
    ("partial_fixed", "Reward (interval)"),
]
RUN_NAME = "runs/qwen7b_sft/{model}/{mode}"

BASELINE_RUN_DIRS = [
    "runs/qwen7b_sft/qwen7b/full",
    "runs/qwen7b_sft/qwen7b/partial_fixed",
]
POLICY_BASELINES = [
    (
        r"\textsc{Qwen2.5-7B-SFT}",
        "policy_token_baselines",
    ),
    (
        r"\textsc{Llama3.1-8B-SFT}",
        "policy_token_baselines_llama8b_sft",
    ),
    (
        r"\textsc{Qwen3-4B-SFT}",
        "policy_token_baselines_qwen4b_sft",
    ),
]
BASELINE_SIGNALS = [
    (
        "Token log-probability",
        "logprob_hit1_at_window",
    ),
    (
        "Token entropy",
        "entropy_hit1_at_window",
    ),
    (
        "Random location",
        "random_hit1_at_window",
    ),
]


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


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _row_id(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("prompt_idx"),
        row.get("severity"),
        row.get("variant_idx"),
        row.get("clean_generation_idx"),
        tuple(row.get("perturb_fns", []) or []),
    )


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
    if ci is None:
        return f"{100.0 * mean:.2f}"
    return f"{100.0 * mean:.2f} $\\pm$ {100.0 * ci:.2f}"


def _run_stride(run_cfg: dict[str, Any]) -> int:
    args_cfg = run_cfg.get("args", {}) if isinstance(run_cfg, dict) else {}
    for value in [
        run_cfg.get("partial_fixed_stride"),
        args_cfg.get("partial_fixed_stride") if isinstance(args_cfg, dict) else None,
        args_cfg.get("dense_partial_fixed_n") if isinstance(args_cfg, dict) else None,
    ]:
        parsed = _to_int(value)
        if parsed is not None and parsed > 0:
            return int(parsed)
    return 15


def _reward_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if _to_int(row.get("severity")) != 1:
            continue
        pert_seq = row.get("pert_score_seq", [])
        changed_positions = row.get("changed_token_positions", [])
        if not isinstance(pert_seq, list) or len(pert_seq) < 2:
            continue
        if not isinstance(changed_positions, list):
            continue
        changed = {
            int(c)
            for c in changed_positions
            if _to_int(c) is not None and 0 <= int(c) < len(pert_seq)
        }
        if not changed:
            continue
        selected.append(row)
    return selected


def _reward_hit(row: dict[str, Any], window: int, mode: str, stride: int) -> float:
    pert_seq = row["pert_score_seq"]
    changed = sorted({int(c) for c in row["changed_token_positions"]})
    z = np.asarray([float(v) for v in pert_seq], dtype=np.float64)
    drops = np.maximum(0.0, z[:-1] - z[1:])
    pred = int(np.argmax(drops)) + 1

    if mode == "partial_fixed":
        t_len = len(pert_seq)
        n_buckets = int(math.ceil(t_len / float(stride)))
        pred_bucket = int(pred // stride)
        changed_buckets = sorted({int(c // stride) for c in changed if 0 <= int(c) < t_len})
        w_bucket = int(math.ceil(max(0, int(window)) / float(stride)))
        if not changed_buckets or n_buckets <= 0:
            return float("nan")
        return float(min(abs(pred_bucket - b) for b in changed_buckets) <= w_bucket)

    w = int(max(0, window))
    changed = [c for c in changed if 0 <= c < len(pert_seq)]
    if not changed:
        return float("nan")
    return float(min(abs(pred - c) for c in changed) <= w)


def _reward_metrics(
    rows: list[dict[str, Any]],
    mode: str,
    stride: int,
    windows: list[int],
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for window in windows:
        vals = [
            _reward_hit(row, window=window, mode=mode, stride=stride)
            for row in rows
        ]
        vals = [float(v) for v in vals if math.isfinite(float(v))]
        out[str(window)] = {
            "mean": _mean(vals),
            "ci": _ci_halfwidth(
                vals,
                samples=bootstrap_samples,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            ),
            "n": len(vals),
        }
    return out


def _baseline_rows(rows: list[dict[str, Any]], windows: list[int]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        keep = True
        for _signal, key in BASELINE_SIGNALS:
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


def _baseline_metrics(
    rows: list[dict[str, Any]],
    key: str,
    windows: list[int],
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for window in windows:
        vals = [float(row[key][str(window)]) for row in rows]
        out[str(window)] = {
            "mean": _mean(vals),
            "ci": _ci_halfwidth(
                vals,
                samples=bootstrap_samples,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            ),
            "n": len(vals),
        }
    return out


def _render_latex(table_rows: list[dict[str, Any]], n_rows: int, windows: list[int]) -> str:
    conf_level = (1.0 - float(BOOTSTRAP_ALPHA)) * 100.0
    latex: list[str] = []
    latex.append(r"% Requires \usepackage{booktabs}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\renewcommand{\arraystretch}{1.08}")
    latex.append(r"\begin{tabular}{l l r r}")
    latex.append(r"\toprule")
    latex.append(
        rf"\textbf{{Scorer}} & \textbf{{Signal}} & "
        rf"\textbf{{Hit@{windows[0]} (\%)}} & \textbf{{Hit@{windows[1]} (\%)}} \\"
    )
    latex.append(r"\midrule")

    prev_group = None
    for row in table_rows:
        group = row["group"]
        if prev_group is not None and group != prev_group:
            latex.append(r"\midrule")
        prev_group = group
        cells = [
            _fmt_pct(row["metrics"][str(window)]["mean"], row["metrics"][str(window)]["ci"])
            for window in windows
        ]
        latex.append(f"{row['scorer']} & {row['signal']} & " + " & ".join(cells) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(
        rf"\caption{{\textbf{{Localisation on GSM8K pregenerated traces with learned "
        rf"reward signals and simple generator-side baselines.}} All rows are computed "
        rf"on the exact same {n_rows} severity-1 Table-5-valid corrupted traces. "
        rf"Reward and log-probability rows use the largest-drop detector; entropy uses "
        rf"the largest-spike detector. SFT rows score the same traces with the indicated "
        rf"policy/tokenizer, and each random row is the corresponding analytic "
        rf"uniform-location baseline in that tokenizer space. "
        rf"Values are percentage mean $\pm$ bootstrapped {conf_level:.0f}\% CI half-width.}}"
    )
    latex.append(r"\label{tab:localisation_combined_reward_policy_hit1_hit7}")
    latex.append(r"\end{table}")
    return "\n".join(latex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined reward-model and generator-baseline localisation table."
    )
    parser.add_argument("--root-dir", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", nargs=2, type=int, default=WINDOWS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = [int(w) for w in args.windows]

    global BOOTSTRAP_SAMPLES, BOOTSTRAP_ALPHA, BOOTSTRAP_SEED
    BOOTSTRAP_SAMPLES = int(args.bootstrap_samples)
    BOOTSTRAP_ALPHA = float(args.bootstrap_alpha)
    BOOTSTRAP_SEED = int(args.bootstrap_seed)

    table_rows: list[dict[str, Any]] = []
    ref_ids: list[tuple[Any, ...]] | None = None
    ref_name = None

    for model_key, model_label in MODEL_ORDER:
        for mode, signal in MODE_ORDER:
            run_name = RUN_NAME.format(model=model_key, mode=mode)
            run_dir = args.root_dir / run_name
            pair_path = run_dir / "pair_details.jsonl"
            cfg_path = run_dir / "run_config.json"
            if not pair_path.exists():
                raise FileNotFoundError(f"Missing pair details: {pair_path}")
            rows = _reward_table_rows(_load_jsonl(pair_path))
            ids = [_row_id(row) for row in rows]
            if ref_ids is None:
                ref_ids = ids
                ref_name = run_name
            elif ids != ref_ids:
                raise ValueError(
                    f"Row filter mismatch between {ref_name} and {run_name}; "
                    "refusing to build a mixed-filter table."
                )
            run_cfg = _load_json(cfg_path) if cfg_path.exists() else {}
            stride = _run_stride(run_cfg)
            table_rows.append(
                {
                    "group": "reward",
                    "scorer": model_label,
                    "signal": signal,
                    "metrics": _reward_metrics(
                        rows,
                        mode=mode,
                        stride=stride,
                        windows=windows,
                        bootstrap_samples=BOOTSTRAP_SAMPLES,
                        bootstrap_alpha=BOOTSTRAP_ALPHA,
                        bootstrap_seed=BOOTSTRAP_SEED,
                    ),
                }
            )

    assert ref_ids is not None
    n_rows = len(ref_ids)

    for scorer, subdir in POLICY_BASELINES:
        baseline_ref_rows: list[dict[str, Any]] | None = None
        for baseline_run in BASELINE_RUN_DIRS:
            path = args.root_dir / baseline_run / subdir / "policy_token_baselines.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing policy baseline details: {path}")
            rows = _baseline_rows(_load_jsonl(path), windows=windows)
            ids = [_row_id(row) for row in rows]
            if ids != ref_ids:
                raise ValueError(
                    f"Row filter mismatch between {ref_name} and {baseline_run}/{subdir}; "
                    "refusing to build a mixed-filter table."
                )
            if baseline_ref_rows is None:
                baseline_ref_rows = rows
            else:
                prev_by_id = {_row_id(row): row for row in baseline_ref_rows}
                for row in rows:
                    prev = prev_by_id[_row_id(row)]
                    for _signal, key in BASELINE_SIGNALS:
                        if row.get(key) != prev.get(key):
                            raise ValueError(
                                "Policy baseline values differ across full/interval output dirs; "
                                f"first mismatch in {subdir}/{key} for row {_row_id(row)}."
                            )

        assert baseline_ref_rows is not None
        for signal, key in BASELINE_SIGNALS:
            table_rows.append(
                {
                    "group": "baseline",
                    "scorer": scorer,
                    "signal": signal,
                    "metrics": _baseline_metrics(
                        baseline_ref_rows,
                        key=key,
                        windows=windows,
                        bootstrap_samples=BOOTSTRAP_SAMPLES,
                        bootstrap_alpha=BOOTSTRAP_ALPHA,
                        bootstrap_seed=BOOTSTRAP_SEED,
                    ),
                }
            )

    latex = _render_latex(table_rows, n_rows=n_rows, windows=windows)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as f:
        f.write(latex)
        f.write("\n")

    print(f"Wrote LaTeX table to {args.output_file}")
    print(f"Shared rows: {n_rows}")
    for row in table_rows:
        parts = []
        for window in windows:
            vals = row["metrics"][str(window)]
            parts.append(f"Hit@{window}={vals['mean']:.4f} +/- {vals['ci']:.4f}")
        print(f"{row['scorer']} | {row['signal']}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
