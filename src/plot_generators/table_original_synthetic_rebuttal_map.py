"""Build MAP table for original synthetic GSM8K localisation runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_SOURCE = Path("outputs/gsm8k_process_sensitivity_pregen/pair_details.jsonl")
DEFAULT_ROOT = Path("outputs/gsm8k_process_sensitivity_pregen/rebuttal_scores")
DEFAULT_OLD_ROOT = Path("localisation")
DEFAULT_OUTPUT = Path("localisation/localisation_original_synthetic_rebuttal_map.tex")
DEFAULT_JSON = Path("localisation/localisation_original_synthetic_rebuttal_map.json")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42

MODEL_ORDER = ["qwen7b", "llama8b", "qwen4b"]
MODEL_LABELS = {
    "qwen7b": r"\textsc{Qwen2.5-7B}",
    "llama8b": r"\textsc{Llama3.1-8B}",
    "qwen4b": r"\textsc{Qwen3-4B}",
}

OLD_REWARD_RUN = "runs/qwen7b_sft/{model}/{mode}"
OLD_REWARD_MODES = [
    ("full", "Reward dense (old)"),
    ("partial_fixed", "Reward interval (old)"),
]
OLD_POLICY_DIRS = {
    "qwen7b": "policy_token_baselines",
    "llama8b": "policy_token_baselines_llama8b_sft",
    "qwen4b": "policy_token_baselines_qwen4b_sft",
}
NEW_POLICY_DIRS = {
    "qwen7b": "qwen7b_base_policy_token_baselines",
    "llama8b": "llama8b_base_policy_token_baselines",
    "qwen4b": "qwen4b_base_policy_token_baselines",
}
NEW_REWARD_DIRS = {
    "qwen7b": "qwen7b_full_rebuttal_reward_localisation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pairs", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--old-root-dir", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
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


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


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


def _mean_ci(
    values: Iterable[float],
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


def _fmt_pct(metric: dict[str, float | int | None]) -> str:
    mean = metric.get("mean")
    ci = metric.get("ci_halfwidth")
    if mean is None:
        return "-"
    if ci is None:
        return f"{100.0 * float(mean):.2f}"
    return f"{100.0 * float(mean):.2f} $\\pm$ {100.0 * float(ci):.2f}"


def _fmt_pct_mean(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def _bold_mean(cell: str) -> str:
    return re.sub(r"(-?\d+(?:\.\d+)?)", r"\\textbf{\1}", cell, count=1)


def _underline_mean(cell: str) -> str:
    return re.sub(r"(-?\d+(?:\.\d+)?)", r"\\underline{\1}", cell, count=1)


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


def _finite_array(seq: Any) -> np.ndarray:
    if not isinstance(seq, list):
        return np.asarray([], dtype=np.float64)
    vals = []
    for value in seq:
        try:
            parsed = float(value)
        except Exception:
            continue
        if np.isfinite(parsed):
            vals.append(parsed)
    return np.asarray(vals, dtype=np.float64)


def _unit_sequence_and_targets(
    seq: Any,
    targets: list[int],
    *,
    mode: str,
    stride: int,
) -> tuple[np.ndarray, list[int]]:
    arr = _finite_array(seq)
    if arr.shape[0] == 0:
        return arr, []
    if mode == "bucket":
        stride = max(1, int(stride))
        n_buckets = int(math.ceil(arr.shape[0] / float(stride)))
        unit = np.zeros(n_buckets, dtype=np.float64)
        for b in range(n_buckets):
            s = b * stride
            e = min(arr.shape[0], (b + 1) * stride)
            unit[b] = float(np.nanmean(arr[s:e])) if e > s else float("nan")
        unit = unit[np.isfinite(unit)]
        target_units = sorted({int(t) // stride for t in targets if 0 <= int(t) < arr.shape[0]})
        return unit, target_units
    target_units = sorted({int(t) for t in targets if 0 <= int(t) < arr.shape[0]})
    return arr, target_units


def _transition_scores(unit_seq: np.ndarray, detector: str) -> tuple[np.ndarray, np.ndarray]:
    if unit_seq.shape[0] < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)
    if detector == "largest_drop":
        scores = np.maximum(0.0, unit_seq[:-1] - unit_seq[1:])
    elif detector == "largest_spike":
        scores = np.maximum(0.0, unit_seq[1:] - unit_seq[:-1])
    else:
        raise ValueError(f"Unknown detector: {detector}")
    indices = np.arange(1, unit_seq.shape[0], dtype=np.int64)
    return scores.astype(np.float64), indices


def _average_precision(scores: np.ndarray, indices: np.ndarray, targets: list[int]) -> float:
    target_set = {int(t) for t in targets}
    labels = np.asarray([1 if int(idx) in target_set else 0 for idx in indices.tolist()], dtype=np.int64)
    n_pos = int(labels.sum())
    if scores.size == 0 or n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = 0
    precisions = []
    for rank, pos in enumerate(order.tolist(), start=1):
        if labels[pos]:
            hits += 1
            precisions.append(float(hits / rank))
    return float(sum(precisions) / n_pos)


def _expected_random_average_precision(n_items: int, n_relevant: int) -> float:
    if n_items <= 0 or n_relevant <= 0:
        return float("nan")
    if n_items == 1:
        return 1.0 if n_relevant > 0 else float("nan")
    harmonic = float(sum(1.0 / k for k in range(1, n_items + 1)))
    return float((harmonic + ((n_relevant - 1.0) / (n_items - 1.0)) * (n_items - harmonic)) / n_items)


def _transition_relevant_count(n_units: int, targets: list[int]) -> int:
    if n_units <= 1 or not targets:
        return 0
    target_set = {int(t) for t in targets}
    return int(sum(1 for idx in range(1, n_units) if idx in target_set))


def _targets(row: dict[str, Any], *, is_policy: bool) -> list[int]:
    key = "policy_changed_token_positions" if is_policy else "changed_token_positions"
    vals = row.get(key, [])
    if not isinstance(vals, list):
        return []
    out = []
    for value in vals:
        parsed = _to_int(value)
        if parsed is not None:
            out.append(int(parsed))
    return out


def _row_map_values(
    row: dict[str, Any],
    *,
    seq_key: str,
    detector: str,
    is_policy: bool,
    mode: str,
    stride: int,
) -> tuple[float, float, float]:
    targets = _targets(row, is_policy=is_policy)
    unit, target_units = _unit_sequence_and_targets(
        row.get(seq_key, []),
        targets,
        mode=mode,
        stride=stride,
    )
    scores, indices = _transition_scores(unit, detector)
    ap = _average_precision(scores, indices, target_units)
    n_candidates = max(0, int(unit.shape[0]) - 1)
    n_relevant = _transition_relevant_count(int(unit.shape[0]), target_units)
    random_ap = _expected_random_average_precision(n_candidates, n_relevant)
    if math.isfinite(ap) and math.isfinite(random_ap) and (1.0 - random_ap) > 1e-12:
        norm_ap = float((ap - random_ap) / (1.0 - random_ap))
    else:
        norm_ap = float("nan")
    return ap, random_ap, norm_ap


def _summarize_map(
    rows: list[dict[str, Any]],
    *,
    seq_key: str,
    detector: str,
    is_policy: bool,
    mode: str,
    stride: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ap_vals = []
    random_vals = []
    norm_vals = []
    for row in rows:
        ap, random_ap, norm_ap = _row_map_values(
            row,
            seq_key=seq_key,
            detector=detector,
            is_policy=is_policy,
            mode=mode,
            stride=stride,
        )
        ap_vals.append(ap)
        random_vals.append(random_ap)
        norm_vals.append(norm_ap)
    return {
        "map": _mean_ci(ap_vals, int(args.bootstrap_samples), float(args.bootstrap_alpha), int(args.bootstrap_seed)),
        "random_map": _mean_ci(random_vals, int(args.bootstrap_samples), float(args.bootstrap_alpha), int(args.bootstrap_seed)),
        "norm_map": _mean_ci(norm_vals, int(args.bootstrap_samples), float(args.bootstrap_alpha), int(args.bootstrap_seed)),
    }


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


def _old_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if isinstance(row.get("pert_policy_log_probs"), list) and isinstance(row.get("policy_changed_token_positions"), list):
            selected.append(row)
    return selected


def _check_keys(name: str, rows: list[dict[str, Any]], ref_keys: list[tuple[Any, ...]], *, old: bool) -> None:
    key_fn = _old_row_key if old else _row_key
    keys = [key_fn(row) for row in rows]
    if keys != ref_keys:
        raise ValueError(
            f"Row mismatch for {name}: got {len(keys)} rows, expected {len(ref_keys)}."
        )


def _append_method(
    methods: list[dict[str, Any]],
    *,
    source: str,
    model_key: str,
    signal: str,
    metrics: dict[str, Any],
) -> None:
    methods.append(
        {
            "source": source,
            "model_key": model_key,
            "model": MODEL_LABELS.get(model_key, model_key),
            "signal": signal,
            "map": metrics["map"],
            "random_map": metrics["random_map"],
            "norm_map": metrics["norm_map"],
            "n": metrics["map"].get("n"),
        }
    )


def _collect_old(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    methods: list[dict[str, Any]] = []
    ref_keys: list[tuple[Any, ...]] | None = None
    for model in MODEL_ORDER:
        for mode, signal in OLD_REWARD_MODES:
            run_name = OLD_REWARD_RUN.format(model=model, mode=mode)
            run_dir = args.old_root_dir / run_name
            path = run_dir / "pair_details.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing old reward details: {path}")
            rows = _old_reward_table_rows(_load_jsonl(path))
            keys = [_old_row_key(row) for row in rows]
            if ref_keys is None:
                ref_keys = keys
            elif keys != ref_keys:
                raise ValueError(f"Old reward row mismatch for {path}")
            run_cfg = _load_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
            native_mode = "bucket" if mode == "partial_fixed" else "token"
            metrics = _summarize_map(
                rows,
                seq_key="pert_score_seq",
                detector="largest_drop",
                is_policy=False,
                mode=native_mode,
                stride=_run_stride(run_cfg),
                args=args,
            )
            _append_method(methods, source="old", model_key=model, signal=signal, metrics=metrics)

    if ref_keys is None:
        raise ValueError("No old reward rows found.")

    old_policy_base = args.old_root_dir / "runs/qwen7b_sft/qwen7b/full"
    for model in MODEL_ORDER:
        path = old_policy_base / OLD_POLICY_DIRS[model] / "policy_token_baselines.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing old policy details: {path}")
        rows = _old_policy_rows(_load_jsonl(path))
        _check_keys(str(path), rows, ref_keys, old=True)
        for seq_key, detector, signal in [
            ("pert_policy_log_probs", "largest_drop", "SFT token log-probability"),
            ("pert_policy_entropies", "largest_spike", "SFT token entropy"),
        ]:
            metrics = _summarize_map(
                rows,
                seq_key=seq_key,
                detector=detector,
                is_policy=True,
                mode="token",
                stride=15,
                args=args,
            )
            _append_method(methods, source="old", model_key=model, signal=signal, metrics=metrics)
    return methods, len(ref_keys)


def _collect_new(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    methods: list[dict[str, Any]] = []
    source_rows = _load_jsonl(args.source_pairs)
    ref_keys = [_row_key(row) for row in source_rows]
    if len(set(ref_keys)) != len(ref_keys):
        raise ValueError(f"Duplicate source row keys: {args.source_pairs}")

    for model in MODEL_ORDER:
        reward_dir = NEW_REWARD_DIRS.get(model)
        if reward_dir is not None:
            path = args.root_dir / reward_dir / "pair_details.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing new reward details: {path}")
            rows = _load_jsonl(path)
            _check_keys(str(path), rows, ref_keys, old=False)
            metrics = _summarize_map(
                rows,
                seq_key="pert_score_seq",
                detector="largest_drop",
                is_policy=False,
                mode="token",
                stride=15,
                args=args,
            )
            _append_method(methods, source="new", model_key=model, signal="Reward dense (new)", metrics=metrics)

        path = args.root_dir / NEW_POLICY_DIRS[model] / "policy_token_baselines.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing new policy details: {path}")
        rows = _load_jsonl(path)
        _check_keys(str(path), rows, ref_keys, old=False)
        for seq_key, detector, signal in [
            ("pert_policy_log_probs", "largest_drop", "Base token log-probability"),
            ("pert_policy_entropies", "largest_spike", "Base token entropy"),
        ]:
            metrics = _summarize_map(
                rows,
                seq_key=seq_key,
                detector=detector,
                is_policy=True,
                mode="token",
                stride=15,
                args=args,
            )
            _append_method(methods, source="new", model_key=model, signal=signal, metrics=metrics)
    return methods, len(ref_keys)


def build_results(args: argparse.Namespace) -> dict[str, Any]:
    old_methods, old_n = _collect_old(args)
    new_methods, new_n = _collect_new(args)
    methods = []
    for model in MODEL_ORDER:
        methods.extend([m for m in old_methods + new_methods if m["model_key"] == model])
    return {
        "old_n": old_n,
        "new_n": new_n,
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "alpha": float(args.bootstrap_alpha),
            "seed": int(args.bootstrap_seed),
        },
        "methods": methods,
    }


def build_latex(results: dict[str, Any]) -> str:
    conf_level = (1.0 - BOOTSTRAP_ALPHA) * 100.0
    latex: list[str] = []
    latex.append(r"% Requires \usepackage{booktabs,multirow,arydshln}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{5pt}")
    latex.append(
        r"\caption{\textbf{MAP on synthetic GSM8K perturbations.} "
        rf"Old rows use the previous {results['old_n']} severity-1 Table-5-valid pregenerated traces; "
        rf"new rebuttal rows use the {results['new_n']} original synthetic perturbations from "
        r"\texttt{gsm8k\_process\_sensitivity\_pregen}. "
        r"MAP is average precision over exact changed-token positions. "
        r"Reward and log-probability rank largest drops; entropy ranks largest spikes. "
        r"Rand. MAP is the expected random-ranking AP in the corresponding token or interval grid; "
        r"Norm. MAP is $(\mathrm{MAP}-\mathrm{Rand})/(1-\mathrm{Rand})$. "
        rf"Values are percentages with bootstrapped {conf_level:.0f}\% CI half-widths. "
        r"\textbf{Bold} marks the best and \underline{underlining} the second-best Norm. MAP within each model.}"
    )
    latex.append(r"\label{tab:localisation_original_synthetic_rebuttal_map}")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Model} & \textbf{Signal} & \textbf{MAP} & "
        r"\textbf{Rand. MAP} & \textbf{Norm. MAP} & \textbf{n} \\"
    )
    latex.append(r"\midrule")

    first = True
    for model in MODEL_ORDER:
        rows = [m for m in results["methods"] if m["model_key"] == model]
        if not rows:
            continue
        if not first:
            latex.append(r"\midrule")
        first = False
        markers = _rank_markers([
            None if row["norm_map"].get("mean") is None else float(row["norm_map"]["mean"])
            for row in rows
        ])
        for idx, row in enumerate(rows):
            model_cell = (
                rf"\multirow{{{len(rows)}}}{{*}}{{{MODEL_LABELS[model]}}}"
                if idx == 0
                else ""
            )
            norm_cell = _apply_marker(_fmt_pct(row["norm_map"]), markers[idx])
            line = [
                model_cell,
                row["signal"],
                _fmt_pct(row["map"]),
                _fmt_pct_mean(row["random_map"].get("mean")),
                norm_cell,
                str(row["n"]) if row["n"] is not None else "-",
            ]
            latex.append(" & ".join(line) + r" \\")
            if idx + 1 < len(rows):
                latex.append(r"\cdashline{2-6}")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    return "\n".join(latex)


def main() -> None:
    args = parse_args()
    results = build_results(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_json, results)
    args.output.write_text(build_latex(results))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output}")
    print(f"Old rows: {results['old_n']}")
    print(f"New rows: {results['new_n']}")
    for row in results["methods"]:
        print(
            f"{row['model_key']:>10} | {row['signal']} | "
            f"MAP={_fmt_pct(row['map'])} | "
            f"Rand={_fmt_pct_mean(row['random_map'].get('mean'))} | "
            f"Norm={_fmt_pct(row['norm_map'])} | n={row['n']}"
        )


if __name__ == "__main__":
    main()
