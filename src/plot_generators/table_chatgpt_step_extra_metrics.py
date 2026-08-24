"""Compute additional localisation metrics for ChatGPT-step score files."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_ROOT = Path("localisation/chatgpt_step_perturbations/scores")
DEFAULT_JSON = Path("localisation/chatgpt_step_perturbations/localisation_chatgpt_step_extra_metrics.json")
DEFAULT_TEX = Path("localisation/chatgpt_step_perturbations/localisation_chatgpt_step_extra_metrics.tex")
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
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TEX)
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


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    out = []
    for value in values:
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def _mean_ci(values: Sequence[float], samples: int, alpha: float, seed: int) -> dict[str, float | int | None]:
    vals = _finite_values(values)
    n = int(vals.shape[0])
    if n == 0:
        return {"mean": None, "ci_halfwidth": None, "n": 0}
    mean = float(vals.mean())
    if n == 1:
        return {"mean": mean, "ci_halfwidth": 0.0, "n": 1}
    rng = np.random.default_rng(int(seed))
    n_boot = max(100, int(samples))
    alpha = min(max(float(alpha), 1e-6), 0.5)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = vals[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {"mean": mean, "ci_halfwidth": float((hi - lo) / 2.0), "n": n}


def _bold_mean(cell: str) -> str:
    return re.sub(r"(-?\d+(?:\.\d+)?)", r"\\textbf{\1}", cell, count=1)


def _fmt_pct(metric: dict[str, float | int | None], bold: bool = False) -> str:
    mean = metric.get("mean")
    ci = metric.get("ci_halfwidth")
    if mean is None:
        return "-"
    if ci is None:
        cell = f"{100.0 * float(mean):.2f}"
    else:
        cell = f"{100.0 * float(mean):.2f} $\\pm$ {100.0 * float(ci):.2f}"
    return _bold_mean(cell) if bold else cell


def _unit_sequence_and_targets(
    row: dict[str, Any],
    seq: Sequence[float],
    target_positions: Sequence[int],
    window: int,
) -> tuple[np.ndarray, list[int], int]:
    arr = np.asarray(seq, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    localization_mode = row.get("localization_mode")
    stride = _to_int(row.get("partial_fixed_stride")) or 15
    if localization_mode == "bucket" and stride > 1:
        n_buckets = int(math.ceil(arr.shape[0] / float(stride)))
        unit = np.zeros(n_buckets, dtype=np.float64)
        for b in range(n_buckets):
            s = b * stride
            e = min(arr.shape[0], (b + 1) * stride)
            unit[b] = float(np.nanmean(arr[s:e])) if e > s else float("nan")
        targets = sorted({int(t) // stride for t in target_positions if 0 <= int(t) < arr.shape[0]})
        unit_window = int(math.ceil(max(0, int(window)) / float(stride)))
        return unit[np.isfinite(unit)], targets, unit_window

    targets = sorted({int(t) for t in target_positions if 0 <= int(t) < arr.shape[0]})
    return arr, targets, int(max(0, window))


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


def _hit_at_window_from_scores(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int], window: int) -> float:
    if scores.size == 0 or not targets:
        return float("nan")
    pred = int(indices[int(np.argmax(scores))])
    return float(any(abs(pred - int(t)) <= int(window) for t in targets))


def _chance_for_targets(n_units: int, targets: Sequence[int], window: int) -> float:
    if n_units <= 0 or not targets:
        return float("nan")
    mask = np.zeros(n_units, dtype=bool)
    for target in targets:
        if 0 <= int(target) < n_units:
            lo = max(0, int(target) - int(window))
            hi = min(n_units - 1, int(target) + int(window))
            mask[lo : hi + 1] = True
    return float(mask.mean())


def _mass_at_window(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int], window: int) -> float:
    if scores.size == 0 or not targets:
        return float("nan")
    total = float(scores.sum())
    if total <= 0.0:
        return float("nan")
    mask = np.zeros(scores.shape[0], dtype=bool)
    for target in targets:
        mask |= np.abs(indices - int(target)) <= int(window)
    return float(scores[mask].sum() / total)


def _mrr(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int]) -> float:
    target_set = {int(t) for t in targets}
    valid = [i for i, idx in enumerate(indices.tolist()) if int(idx) in target_set]
    if scores.size == 0 or not valid:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    for rank, pos in enumerate(order.tolist(), start=1):
        if int(indices[pos]) in target_set:
            return float(1.0 / rank)
    return float("nan")


def _average_precision(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int]) -> float:
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


def _row_targets(row: dict[str, Any], region: str, is_policy: bool) -> list[int]:
    if region == "first_diff":
        key = "policy_changed_token_positions" if is_policy else "changed_token_positions"
    elif region == "edit_span":
        key = "policy_diff_changed_token_positions" if is_policy else "diff_changed_token_positions"
    else:
        raise ValueError(f"Unknown target region: {region}")
    values = row.get(key, [])
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            out.append(int(parsed))
    return out


def _transition_chance_for_targets(n_units: int, targets: Sequence[int], window: int) -> float:
    if n_units <= 1 or not targets:
        return float("nan")
    indices = np.arange(1, n_units, dtype=np.int64)
    mask = np.zeros(indices.shape[0], dtype=bool)
    for target in targets:
        mask |= np.abs(indices - int(target)) <= int(window)
    return float(mask.mean())


def _transition_exact_relevant_count(n_units: int, targets: Sequence[int]) -> int:
    if n_units <= 1 or not targets:
        return 0
    target_set = {int(t) for t in targets}
    return int(sum(1 for idx in range(1, n_units) if idx in target_set))


def _normalized_existing_hit(row: dict[str, Any], hit_key: str, window: int) -> float:
    hit = ((row.get(hit_key) or {}).get(str(window)))
    chance = ((row.get("random_hit1_at_window") or {}).get(str(window)))
    if hit is None or chance is None:
        return float("nan")
    chance = float(chance)
    denom = 1.0 - chance
    if denom <= 1e-12:
        return float("nan")
    return float((float(hit) - chance) / denom)


def _compute_for_rows(
    rows: list[dict[str, Any]],
    seq_key: str,
    hit_key: str,
    detector: str,
    windows: list[int],
    is_policy: bool,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    per_window: dict[str, Any] = {}
    for window in windows:
        norm_hit_vals = []
        first_mass_vals = []
        edit_mass_vals = []
        edit_hit_vals = []
        edit_norm_hit_vals = []
        for row in rows:
            seq = row.get(seq_key, [])
            if not isinstance(seq, list):
                continue
            first_targets = _row_targets(row, "first_diff", is_policy=is_policy)
            edit_targets = _row_targets(row, "edit_span", is_policy=is_policy)

            unit_first, first_units, first_w = _unit_sequence_and_targets(row, seq, first_targets, window)
            first_scores, first_indices = _transition_scores(unit_first, detector)
            unit_edit, edit_units, edit_w = _unit_sequence_and_targets(row, seq, edit_targets, window)
            edit_scores, edit_indices = _transition_scores(unit_edit, detector)

            norm_hit_vals.append(_normalized_existing_hit(row, hit_key, window))
            first_mass_vals.append(_mass_at_window(first_scores, first_indices, first_units, first_w))
            edit_mass_vals.append(_mass_at_window(edit_scores, edit_indices, edit_units, edit_w))
            edit_hit = _hit_at_window_from_scores(edit_scores, edit_indices, edit_units, edit_w)
            edit_chance = _chance_for_targets(int(unit_edit.shape[0]), edit_units, edit_w)
            edit_hit_vals.append(edit_hit)
            if math.isfinite(edit_hit) and math.isfinite(edit_chance) and (1.0 - edit_chance) > 1e-12:
                edit_norm_hit_vals.append(float((edit_hit - edit_chance) / (1.0 - edit_chance)))
            else:
                edit_norm_hit_vals.append(float("nan"))

        per_window[str(window)] = {
            "first_diff_normalized_hit": _mean_ci(
                norm_hit_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed
            ),
            "first_diff_mass": _mean_ci(
                first_mass_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed
            ),
            "edit_span_hit": _mean_ci(
                edit_hit_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed
            ),
            "edit_span_normalized_hit": _mean_ci(
                edit_norm_hit_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed
            ),
            "edit_span_mass": _mean_ci(
                edit_mass_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed
            ),
        }

    first_mrr_vals = []
    first_ap_vals = []
    edit_mrr_vals = []
    edit_ap_vals = []
    for row in rows:
        seq = row.get(seq_key, [])
        if not isinstance(seq, list):
            continue
        first_targets = _row_targets(row, "first_diff", is_policy=is_policy)
        edit_targets = _row_targets(row, "edit_span", is_policy=is_policy)
        unit_first, first_units, _first_w = _unit_sequence_and_targets(row, seq, first_targets, 0)
        first_scores, first_indices = _transition_scores(unit_first, detector)
        unit_edit, edit_units, _edit_w = _unit_sequence_and_targets(row, seq, edit_targets, 0)
        edit_scores, edit_indices = _transition_scores(unit_edit, detector)
        first_mrr_vals.append(_mrr(first_scores, first_indices, first_units))
        first_ap_vals.append(_average_precision(first_scores, first_indices, first_units))
        edit_mrr_vals.append(_mrr(edit_scores, edit_indices, edit_units))
        edit_ap_vals.append(_average_precision(edit_scores, edit_indices, edit_units))

    return {
        "n_rows": len(rows),
        "windows": per_window,
        "ranking": {
            "first_diff_mrr": _mean_ci(first_mrr_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
            "first_diff_map": _mean_ci(first_ap_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
            "edit_span_mrr": _mean_ci(edit_mrr_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
            "edit_span_map": _mean_ci(edit_ap_vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed),
        },
    }


def _shared_policy_random_metrics(
    root_dir: Path,
    windows: list[int],
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values_by_key: dict[Any, dict[str, list[float]]] = {}

    def bucket_for(row: dict[str, Any], idx: int) -> dict[str, list[float]]:
        key = row.get("prompt_idx", idx)
        return values_by_key.setdefault(key, {})

    for model in MODEL_ORDER:
        for variant, _variant_label in POLICY_VARIANTS:
            path = root_dir / f"{model}_{variant}_policy_token_baselines" / "policy_token_baselines.jsonl"
            if not path.exists():
                continue
            for idx, row in enumerate(_load_jsonl(path)):
                seq = row.get("pert_policy_log_probs", [])
                if not isinstance(seq, list):
                    continue
                edit_targets = _row_targets(row, "edit_span", is_policy=True)
                bucket = bucket_for(row, idx)
                ranking_values = bucket.setdefault("edit_span_map", [])
                unit_edit, edit_units_zero, _ = _unit_sequence_and_targets(row, seq, edit_targets, 0)
                n_candidates = max(0, int(unit_edit.shape[0]) - 1)
                n_relevant = _transition_exact_relevant_count(int(unit_edit.shape[0]), edit_units_zero)
                ranking_values.append(_expected_random_average_precision(n_candidates, n_relevant))

                for window in windows:
                    unit_edit_w, edit_units_w, edit_w = _unit_sequence_and_targets(row, seq, edit_targets, window)
                    edit_chance = _transition_chance_for_targets(int(unit_edit_w.shape[0]), edit_units_w, edit_w)
                    bucket.setdefault(f"first_diff_normalized_hit_{window}", []).append(0.0)
                    bucket.setdefault(f"edit_span_hit_{window}", []).append(edit_chance)
                    bucket.setdefault(f"edit_span_mass_{window}", []).append(edit_chance)

    def averaged(metric_key: str) -> list[float]:
        values = []
        for metric_values in values_by_key.values():
            vals = metric_values.get(metric_key, [])
            vals = [float(v) for v in vals if math.isfinite(float(v))]
            if vals:
                values.append(float(np.mean(vals)))
        return values

    per_window: dict[str, Any] = {}
    for window in windows:
        per_window[str(window)] = {
            "first_diff_normalized_hit": _mean_ci(
                averaged(f"first_diff_normalized_hit_{window}"),
                bootstrap_samples,
                bootstrap_alpha,
                bootstrap_seed,
            ),
            "first_diff_mass": {"mean": None, "ci_halfwidth": None, "n": 0},
            "edit_span_hit": _mean_ci(
                averaged(f"edit_span_hit_{window}"),
                bootstrap_samples,
                bootstrap_alpha,
                bootstrap_seed,
            ),
            "edit_span_normalized_hit": {"mean": None, "ci_halfwidth": None, "n": 0},
            "edit_span_mass": _mean_ci(
                averaged(f"edit_span_mass_{window}"),
                bootstrap_samples,
                bootstrap_alpha,
                bootstrap_seed,
            ),
        }

    return {
        "n_rows": len(values_by_key),
        "windows": per_window,
        "ranking": {
            "first_diff_mrr": {"mean": None, "ci_halfwidth": None, "n": 0},
            "first_diff_map": {"mean": None, "ci_halfwidth": None, "n": 0},
            "edit_span_mrr": {"mean": None, "ci_halfwidth": None, "n": 0},
            "edit_span_map": _mean_ci(
                averaged("edit_span_map"),
                bootstrap_samples,
                bootstrap_alpha,
                bootstrap_seed,
            ),
        },
    }


def _method_specs(root_dir: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for density, signal_label in [("full", "Reward dense"), ("partial_fixed", "Reward interval")]:
            specs.append(
                {
                    "key": f"{model}_{density}_reward",
                    "section": "reward",
                    "model_key": model,
                    "model": MODEL_LABELS[model],
                    "signal": signal_label,
                    "path": root_dir / f"{model}_{density}_reward_localisation" / "pair_details.jsonl",
                    "seq_key": "pert_score_seq",
                    "hit_key": "reward_hit1_at_window",
                    "detector": "largest_drop",
                    "is_policy": False,
                }
            )

    for model in MODEL_ORDER:
        for variant, variant_label in POLICY_VARIANTS:
            path = root_dir / f"{model}_{variant}_policy_token_baselines" / "policy_token_baselines.jsonl"
            if not path.exists() and variant == "base":
                continue
            base = {
                "section": "policy",
                "model_key": model,
                "model": SFT_MODEL_LABELS[model] if variant == "sft" else MODEL_LABELS[model],
                "path": path,
                "is_policy": True,
            }
            specs.append(
                {
                    **base,
                    "key": f"{model}_{variant}_logprob",
                    "signal": f"{variant_label} token log-probability",
                    "seq_key": "pert_policy_log_probs",
                    "hit_key": "logprob_hit1_at_window",
                    "detector": "largest_drop",
                }
            )
            specs.append(
                {
                    **base,
                    "key": f"{model}_{variant}_entropy",
                    "signal": f"{variant_label} token entropy",
                    "seq_key": "pert_policy_entropies",
                    "hit_key": "entropy_hit1_at_window",
                    "detector": "largest_spike",
                }
            )
    return specs


def build_results(args: argparse.Namespace) -> dict[str, Any]:
    methods = []
    for spec in _method_specs(args.root_dir):
        if not spec["path"].exists():
            raise FileNotFoundError(f"Missing detail file: {spec['path']}")
        rows = _load_jsonl(spec["path"])
        metrics = _compute_for_rows(
            rows=rows,
            seq_key=spec["seq_key"],
            hit_key=spec["hit_key"],
            detector=spec["detector"],
            windows=[int(w) for w in args.windows],
            is_policy=bool(spec["is_policy"]),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_alpha=float(args.bootstrap_alpha),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        methods.append(
            {
                "key": spec["key"],
                "section": spec["section"],
                "model_key": spec["model_key"],
                "model": spec["model"],
                "signal": spec["signal"],
                "detail_path": str(spec["path"]),
                "detector": spec["detector"],
                "metrics": metrics,
            }
        )
    random_metrics = _shared_policy_random_metrics(
        root_dir=args.root_dir,
        windows=[int(w) for w in args.windows],
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_alpha=float(args.bootstrap_alpha),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    if random_metrics["n_rows"] > 0:
        methods.append(
            {
                "key": "shared_token_grid_random",
                "section": "random",
                "model_key": "__random__",
                "model": r"\textsc{Random}",
                "signal": "Shared token-grid chance",
                "detail_path": None,
                "detector": "uniform",
                "metrics": random_metrics,
            }
        )
    return {
        "root_dir": str(args.root_dir),
        "windows": [int(w) for w in args.windows],
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "alpha": float(args.bootstrap_alpha),
            "seed": int(args.bootstrap_seed),
        },
        "methods": methods,
    }


def build_latex(results: dict[str, Any]) -> str:
    window = 7
    latex: list[str] = []
    latex.append(r"% Requires \usepackage{multirow}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(
        r"\caption{\textbf{Additional localisation metrics on ChatGPT-edited GSM8K traces.} "
        r"Norm. Hit@7 is chance-normalized Hit@1@7 at the first differing token. "
        r"Edit Hit@7 and Edit Mass@7 use the broader token-diff edit span. "
        r"MRR and MAP rank positions by reward drop or entropy spike/log-probability drop. "
        r"The random row is averaged across available policy token grids.}"
    )
    latex.append(r"\label{tab:localisation_chatgpt_step_extra_metrics}")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Model} & \textbf{Signal} & \textbf{Norm. Hit@7} & "
        r"\textbf{Edit Hit@7} & \textbf{Edit Mass@7} & \textbf{Edit MAP} \\"
    )
    latex.append(r"\midrule")
    metric_getters = [
        lambda method: method["metrics"]["windows"][str(window)]["first_diff_normalized_hit"],
        lambda method: method["metrics"]["windows"][str(window)]["edit_span_hit"],
        lambda method: method["metrics"]["windows"][str(window)]["edit_span_mass"],
        lambda method: method["metrics"]["ranking"]["edit_span_map"],
    ]
    first_group = True
    for model_key in MODEL_ORDER:
        group_methods = [method for method in results["methods"] if method.get("model_key") == model_key]
        if not group_methods:
            continue
        best_by_col = []
        for getter in metric_getters:
            vals = []
            for method in group_methods:
                mean = getter(method).get("mean")
                if mean is not None and math.isfinite(float(mean)):
                    vals.append(float(mean))
            best_by_col.append(max(vals) if vals else None)
        if not first_group:
            latex.append(r"\midrule")
        first_group = False
        for idx, method in enumerate(group_methods):
            w_metrics = method["metrics"]["windows"][str(window)]
            ranking = method["metrics"]["ranking"]
            model_cell = (
                rf"\multirow{{{len(group_methods)}}}{{*}}{{{MODEL_LABELS[model_key]}}}"
                if idx == 0
                else ""
            )
            line = [
                model_cell,
                method["signal"],
                _fmt_pct(
                    w_metrics["first_diff_normalized_hit"],
                    bold=(
                        best_by_col[0] is not None
                        and w_metrics["first_diff_normalized_hit"].get("mean") is not None
                        and abs(float(w_metrics["first_diff_normalized_hit"]["mean"]) - best_by_col[0]) <= 1e-12
                    ),
                ),
                _fmt_pct(
                    w_metrics["edit_span_hit"],
                    bold=(
                        best_by_col[1] is not None
                        and w_metrics["edit_span_hit"].get("mean") is not None
                        and abs(float(w_metrics["edit_span_hit"]["mean"]) - best_by_col[1]) <= 1e-12
                    ),
                ),
                _fmt_pct(
                    w_metrics["edit_span_mass"],
                    bold=(
                        best_by_col[2] is not None
                        and w_metrics["edit_span_mass"].get("mean") is not None
                        and abs(float(w_metrics["edit_span_mass"]["mean"]) - best_by_col[2]) <= 1e-12
                    ),
                ),
                _fmt_pct(
                    ranking["edit_span_map"],
                    bold=(
                        best_by_col[3] is not None
                        and ranking["edit_span_map"].get("mean") is not None
                        and abs(float(ranking["edit_span_map"]["mean"]) - best_by_col[3]) <= 1e-12
                    ),
                ),
            ]
            latex.append(" & ".join(line) + r" \\")
    random_methods = [method for method in results["methods"] if method.get("model_key") == "__random__"]
    if random_methods:
        latex.append(r"\midrule")
        for method in random_methods:
            w_metrics = method["metrics"]["windows"][str(window)]
            ranking = method["metrics"]["ranking"]
            line = [
                method["model"],
                method["signal"],
                _fmt_pct(w_metrics["first_diff_normalized_hit"]),
                _fmt_pct(w_metrics["edit_span_hit"]),
                _fmt_pct(w_metrics["edit_span_mass"]),
                _fmt_pct(ranking["edit_span_map"]),
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
        w7 = method["metrics"]["windows"]["7"]
        ranking = method["metrics"]["ranking"]
        print(
            f"{method['section']:>6} | {method['model']} | {method['signal']} | "
            f"norm_hit7={_fmt_pct(w7['first_diff_normalized_hit'])} | "
            f"edit_hit7={_fmt_pct(w7['edit_span_hit'])} | "
            f"edit_mass7={_fmt_pct(w7['edit_span_mass'])} | "
            f"edit_map={_fmt_pct(ranking['edit_span_map'])}"
        )


if __name__ == "__main__":
    main()
