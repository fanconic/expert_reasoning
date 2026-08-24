"""Build Hit@1/Hit@7 table for original synthetic rebuttal localisation runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SOURCE = Path("outputs/gsm8k_process_sensitivity_pregen/pair_details.jsonl")
DEFAULT_ROOT = Path("outputs/gsm8k_process_sensitivity_pregen/rebuttal_scores")
DEFAULT_OLD_ROOT = Path("localisation")
DEFAULT_OUTPUT = Path("localisation/localisation_original_synthetic_rebuttal_hit1_hit7.tex")
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
POLICY_DIRS = {
    "qwen7b": "qwen7b_base_policy_token_baselines",
    "llama8b": "llama8b_base_policy_token_baselines",
    "qwen4b": "qwen4b_base_policy_token_baselines",
}
REWARD_DIRS = {
    "qwen7b": "qwen7b_full_rebuttal_reward_localisation",
}
OLD_REWARD_RUN = "{model}_{mode}_localisation_from_qwen7b_sft"
OLD_REWARD_MODES = [
    ("full", "Reward dense (old)"),
    ("partial_fixed", "Reward interval (old)"),
]
OLD_POLICY_DIRS = {
    "qwen7b": "policy_token_baselines",
    "llama8b": "policy_token_baselines_llama8b_sft",
    "qwen4b": "policy_token_baselines_qwen4b_sft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pairs", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--old-root-dir", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("prompt_idx"),
        row.get("severity"),
        row.get("variant_idx"),
        row.get("clean_generation_idx"),
    )


def _old_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("prompt_idx"),
        row.get("severity"),
        row.get("variant_idx"),
        row.get("clean_generation_idx"),
        tuple(row.get("perturb_fns", []) or []),
    )


def _check_same_rows(name: str, rows: list[dict[str, Any]], ref_keys: list[tuple[Any, ...]]) -> None:
    keys = [_row_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate row keys in {name}; refusing to build table.")
    if keys != ref_keys:
        missing = len(set(ref_keys) - set(keys))
        extra = len(set(keys) - set(ref_keys))
        raise ValueError(
            f"Row mismatch for {name}: expected {len(ref_keys)} rows, got {len(keys)} "
            f"(missing={missing}, extra={extra})."
        )


def _mean_ci(
    values: list[float],
    samples: int,
    alpha: float,
    seed: int,
) -> dict[str, float | int | None]:
    vals = []
    for value in values:
        try:
            parsed = float(value)
        except Exception:
            continue
        if np.isfinite(parsed):
            vals.append(parsed)
    arr = np.asarray(vals, dtype=np.float64)
    n = int(arr.shape[0])
    if n == 0:
        return {"mean": None, "ci_halfwidth": None, "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci_halfwidth": 0.0, "n": 1}
    rng = np.random.default_rng(int(seed))
    n_boot = max(100, int(samples))
    alpha = min(max(float(alpha), 1e-6), 0.5)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {"mean": mean, "ci_halfwidth": float((hi - lo) / 2.0), "n": n}


def _metric_from_rows(
    rows: list[dict[str, Any]],
    key: str,
    windows: list[int],
    samples: int,
    alpha: float,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for window in windows:
        vals = []
        for row in rows:
            hit = row.get(key)
            if isinstance(hit, dict) and hit.get(str(window)) is not None:
                vals.append(float(hit[str(window)]))
        out[str(window)] = _mean_ci(vals, samples=samples, alpha=alpha, seed=seed)
    return out


def _cells(
    metrics: dict[str, dict[str, float | int | None]],
    windows: list[int],
) -> tuple[list[str], list[float | None], int | None]:
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
    return cells, means, n


def _bold_mean(cell: str) -> str:
    return re.sub(r"(-?\d+(?:\.\d+)?)", r"\\textbf{\1}", cell, count=1)


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


def _old_reward_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        if changed:
            selected.append(row)
    return selected


def _old_reward_hit(row: dict[str, Any], window: int, mode: str, stride: int) -> float:
    pert_seq = row["pert_score_seq"]
    changed = sorted({int(c) for c in row["changed_token_positions"]})
    z = np.asarray([float(v) for v in pert_seq], dtype=np.float64)
    drops = np.maximum(0.0, z[:-1] - z[1:])
    pred = int(np.argmax(drops)) + 1

    if mode == "partial_fixed":
        t_len = len(pert_seq)
        pred_bucket = int(pred // stride)
        changed_buckets = sorted({int(c // stride) for c in changed if 0 <= int(c) < t_len})
        w_bucket = int(math.ceil(max(0, int(window)) / float(stride)))
        if not changed_buckets:
            return float("nan")
        return float(min(abs(pred_bucket - b) for b in changed_buckets) <= w_bucket)

    w = int(max(0, window))
    changed = [c for c in changed if 0 <= c < len(pert_seq)]
    if not changed:
        return float("nan")
    return float(min(abs(pred - c) for c in changed) <= w)


def _old_reward_metrics(
    rows: list[dict[str, Any]],
    mode: str,
    stride: int,
    windows: list[int],
    args: argparse.Namespace,
) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for window in windows:
        vals = [_old_reward_hit(row, window=window, mode=mode, stride=stride) for row in rows]
        out[str(window)] = _mean_ci(
            vals,
            samples=int(args.bootstrap_samples),
            alpha=float(args.bootstrap_alpha),
            seed=int(args.bootstrap_seed),
        )
    return out


def _old_policy_rows(rows: list[dict[str, Any]], windows: list[int]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        keep = True
        for key in ["logprob_hit1_at_window", "entropy_hit1_at_window", "random_hit1_at_window"]:
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


def _append_row(
    rows: list[dict[str, Any]],
    model_key: str,
    signal: str,
    metrics: dict[str, dict[str, float | int | None]],
    windows: list[int],
) -> None:
    cells, means, n = _cells(metrics, windows)
    rows.append(
        {
            "model_key": model_key,
            "signal": signal,
            "cells": cells,
            "means": means,
            "n": n,
        }
    )


def _collect_old_rows(args: argparse.Namespace, windows: list[int]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    ref_keys: list[tuple[Any, ...]] | None = None
    ref_name = None

    for model in MODEL_ORDER:
        for mode, signal in OLD_REWARD_MODES:
            run_name = OLD_REWARD_RUN.format(model=model, mode=mode)
            run_dir = args.old_root_dir / run_name
            pair_path = run_dir / "pair_details.jsonl"
            cfg_path = run_dir / "run_config.json"
            if not pair_path.exists():
                raise FileNotFoundError(f"Missing old reward details: {pair_path}")
            reward_rows = _old_reward_table_rows(_load_jsonl(pair_path))
            keys = [_old_row_key(row) for row in reward_rows]
            if ref_keys is None:
                ref_keys = keys
                ref_name = run_name
            elif keys != ref_keys:
                raise ValueError(f"Old row mismatch between {ref_name} and {run_name}.")
            run_cfg = _load_json(cfg_path) if cfg_path.exists() else {}
            metrics = _old_reward_metrics(
                reward_rows,
                mode=mode,
                stride=_run_stride(run_cfg),
                windows=windows,
                args=args,
            )
            _append_row(rows, model, signal, metrics, windows)

    if ref_keys is None:
        raise ValueError("No old reward rows found.")

    random_by_key: dict[tuple[Any, ...], dict[int, list[float]]] = {
        key: {window: [] for window in windows} for key in ref_keys
    }
    old_policy_base = args.old_root_dir / "qwen7b_full_localisation_from_qwen7b_sft"
    for model in MODEL_ORDER:
        policy_dir = OLD_POLICY_DIRS[model]
        policy_path = old_policy_base / policy_dir / "policy_token_baselines.jsonl"
        if not policy_path.exists():
            raise FileNotFoundError(f"Missing old policy details: {policy_path}")
        policy_rows = _old_policy_rows(_load_jsonl(policy_path), windows)
        keys = [_old_row_key(row) for row in policy_rows]
        if keys != ref_keys:
            raise ValueError(f"Old policy row mismatch for {policy_path}.")
        for row in policy_rows:
            key = _old_row_key(row)
            random_hit = row.get("random_hit1_at_window") or {}
            for window in windows:
                if random_hit.get(str(window)) is not None:
                    random_by_key[key][window].append(float(random_hit[str(window)]))
        for hit_key, signal in [
            ("logprob_hit1_at_window", "SFT token log-probability"),
            ("entropy_hit1_at_window", "SFT token entropy"),
        ]:
            metrics = _metric_from_rows(
                policy_rows,
                key=hit_key,
                windows=windows,
                samples=int(args.bootstrap_samples),
                alpha=float(args.bootstrap_alpha),
                seed=int(args.bootstrap_seed),
            )
            _append_row(rows, model, signal, metrics, windows)

    random_metrics = {}
    for window in windows:
        vals = [
            float(np.mean(per_window[window]))
            for per_window in random_by_key.values()
            if per_window[window]
        ]
        random_metrics[str(window)] = _mean_ci(
            vals,
            samples=int(args.bootstrap_samples),
            alpha=float(args.bootstrap_alpha),
            seed=int(args.bootstrap_seed),
        )
    _append_row(rows, "__random__", "Chance (old SFT grids)", random_metrics, windows)
    return rows, len(ref_keys)


def _collect_new_rows(args: argparse.Namespace, windows: list[int]) -> tuple[list[dict[str, Any]], int]:
    windows = [int(w) for w in args.windows]
    source_rows = _load_jsonl(args.source_pairs)
    ref_keys = [_row_key(row) for row in source_rows]
    if len(set(ref_keys)) != len(ref_keys):
        raise ValueError(f"Duplicate row keys in source pairs: {args.source_pairs}")

    table_rows: list[dict[str, Any]] = []
    random_by_key: dict[tuple[Any, ...], dict[int, list[float]]] = {
        key: {window: [] for window in windows} for key in ref_keys
    }

    for model in MODEL_ORDER:
        reward_dir = REWARD_DIRS.get(model)
        if reward_dir is not None:
            reward_path = args.root_dir / reward_dir / "pair_details.jsonl"
            if not reward_path.exists():
                raise FileNotFoundError(f"Missing reward details: {reward_path}")
            reward_rows = _load_jsonl(reward_path)
            _check_same_rows(str(reward_path), reward_rows, ref_keys)
            metrics = _metric_from_rows(
                reward_rows,
                key="reward_hit1_at_window",
                windows=windows,
                samples=int(args.bootstrap_samples),
                alpha=float(args.bootstrap_alpha),
                seed=int(args.bootstrap_seed),
            )
            _append_row(table_rows, model, "Reward dense (new)", metrics, windows)

        policy_dir = POLICY_DIRS[model]
        policy_path = args.root_dir / policy_dir / "policy_token_baselines.jsonl"
        if not policy_path.exists():
            raise FileNotFoundError(f"Missing policy details: {policy_path}")
        policy_rows = _load_jsonl(policy_path)
        _check_same_rows(str(policy_path), policy_rows, ref_keys)

        for row in policy_rows:
            key = _row_key(row)
            random_hit = row.get("random_hit1_at_window") or {}
            for window in windows:
                if random_hit.get(str(window)) is not None:
                    random_by_key[key][window].append(float(random_hit[str(window)]))

        for hit_key, signal in [
            ("logprob_hit1_at_window", "Base token log-probability"),
            ("entropy_hit1_at_window", "Base token entropy"),
        ]:
            metrics = _metric_from_rows(
                policy_rows,
                key=hit_key,
                windows=windows,
                samples=int(args.bootstrap_samples),
                alpha=float(args.bootstrap_alpha),
                seed=int(args.bootstrap_seed),
            )
            _append_row(table_rows, model, signal, metrics, windows)

    random_metrics = {}
    for window in windows:
        vals = [
            float(np.mean(per_window[window]))
            for per_window in random_by_key.values()
            if per_window[window]
        ]
        random_metrics[str(window)] = _mean_ci(
            vals,
            samples=int(args.bootstrap_samples),
            alpha=float(args.bootstrap_alpha),
            seed=int(args.bootstrap_seed),
        )
    _append_row(table_rows, "__random__", "Chance (new base grids)", random_metrics, windows)
    return table_rows, len(ref_keys)


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int, int]:
    windows = [int(w) for w in args.windows]
    old_rows, old_n = _collect_old_rows(args, windows)
    new_rows, new_n = _collect_new_rows(args, windows)
    by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_ORDER}
    random_rows = []
    for row in old_rows + new_rows:
        if row["model_key"] == "__random__":
            random_rows.append(row)
        else:
            by_model[row["model_key"]].append(row)

    rows = []
    for model in MODEL_ORDER:
        rows.extend(by_model[model])
    rows.extend(random_rows)
    return rows, old_n, new_n


def build_latex(rows: list[dict[str, Any]], old_n: int, new_n: int, windows: list[int]) -> str:
    conf_level = (1.0 - BOOTSTRAP_ALPHA) * 100.0
    latex: list[str] = []
    latex.append(r"% Requires \usepackage{booktabs,multirow}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{5pt}")
    latex.append(
        r"\caption{\textbf{Localisation on synthetic GSM8K perturbations.} "
        rf"Old rows use the previous {old_n} severity-1 Table-5-valid pregenerated traces; "
        rf"new rebuttal rows use the {new_n} original synthetic perturbations from "
        r"\texttt{gsm8k\_process\_sensitivity\_pregen}. "
        r"Reward and log-probability use the largest-drop detector; entropy uses the largest-spike detector. "
        rf"Values are Hit@1@W in percent with bootstrapped {conf_level:.0f}\% CI half-widths.}}"
    )
    latex.append(r"\label{tab:localisation_original_synthetic_rebuttal_hit1_hit7}")
    col_spec = "ll" + ("c" * len(windows)) + "c"
    latex.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")
    header = [r"\textbf{Model}", r"\textbf{Signal}"]
    header.extend([rf"\textbf{{Hit@{w} (\%)}}" for w in windows])
    header.append(r"\textbf{n}")
    latex.append(" & ".join(header) + r" \\")
    latex.append(r"\midrule")

    first = True
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model_key"] == model]
        if not model_rows:
            continue
        if not first:
            latex.append(r"\midrule")
        first = False
        best_by_col = []
        for col_idx in range(len(windows)):
            vals = [
                float(row["means"][col_idx])
                for row in model_rows
                if row["means"][col_idx] is not None
            ]
            best_by_col.append(max(vals) if vals else None)
        for idx, row in enumerate(model_rows):
            model_cell = (
                rf"\multirow{{{len(model_rows)}}}{{*}}{{{MODEL_LABELS[model]}}}"
                if idx == 0
                else ""
            )
            formatted_cells = []
            for col_idx, cell in enumerate(row["cells"]):
                mean = row["means"][col_idx]
                best = best_by_col[col_idx]
                if mean is not None and best is not None and abs(float(mean) - float(best)) <= 1e-12:
                    formatted_cells.append(_bold_mean(cell))
                else:
                    formatted_cells.append(cell)
            line = [model_cell, row["signal"], *formatted_cells, str(row["n"] or "-")]
            latex.append(" & ".join(line) + r" \\")

    random_rows = [row for row in rows if row["model_key"] == "__random__"]
    if random_rows:
        latex.append(r"\midrule")
        for row in random_rows:
            line = [r"\textsc{Random}", row["signal"], *row["cells"], str(row["n"] or "-")]
            latex.append(" & ".join(line) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    return "\n".join(latex)


def main() -> None:
    args = parse_args()
    windows = [int(w) for w in args.windows]
    rows, old_n, new_n = collect_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_latex(rows, old_n=old_n, new_n=new_n, windows=windows))
    print(f"Wrote {args.output}")
    print(f"Old rows: {old_n}")
    print(f"New rows: {new_n}")
    for row in rows:
        print(
            f"{row['model_key']:>10} | {row['signal']} | "
            + " | ".join(row["cells"])
            + f" | n={row['n']}"
        )


if __name__ == "__main__":
    main()
