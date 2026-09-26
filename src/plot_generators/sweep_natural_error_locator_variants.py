"""Sweep natural-error localisation filters and metrics.

This is a diagnostic script for the rebuttal/paper localisation section.  It
compares reward and token-baseline signals under several candidate regions and
grids, including exact chance estimates for top-k Hit@window.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_generators.table_chatgpt_step_extra_metrics import (  # noqa: E402
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    _average_precision,
    _expected_random_average_precision,
    _mean_ci,
    _transition_scores,
)
from src.plot_generators.table_chatgpt_step_localisation import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
)


DEFAULT_ROOT = Path("localisation/natural_wrong_sft/scores")
DEFAULT_OUTPUT = Path("localisation/natural_wrong_sft/localisation_natural_wrong_sft_locator_sweep.json")
DEFAULT_STRICT_INPUT = (
    DEFAULT_ROOT / "_inputs" / "natural_wrong_sft_valid_target_char_span_actual_wrong_answer.jsonl"
)
DEFAULT_ALL_INPUT = DEFAULT_ROOT / "_inputs" / "natural_wrong_sft_valid_target_char_span.jsonl"

REGIONS = ("all", "pre_answer", "think")
GRIDS = ("token", "interval15")
WINDOW = 7
STRIDE = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--strict-input", type=Path, default=DEFAULT_STRICT_INPUT)
    parser.add_argument("--all-input", type=Path, default=DEFAULT_ALL_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--print-top", type=int, default=0)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("prompt_idx"), row.get("variant_idx"), row.get("clean_generation_idx"))


def _tokenizer_source(path: str | Path | None) -> str:
    if path is None:
        raise ValueError("Missing tokenizer/model path.")
    p = Path(str(path))
    if (p / "tokenizer_config.json").exists() or (p / "config.json").exists():
        return str(p)
    adapter_cfg = p / "adapter_config.json"
    if adapter_cfg.exists():
        obj = json.loads(adapter_cfg.read_text())
        return str(obj.get("base_model_name_or_path") or p)
    return str(path)


_TOKENIZER_CACHE: dict[str, Any] = {}


def _tokenizer(path: str | Path | None):
    src = _tokenizer_source(path)
    if src not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[src] = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    return _TOKENIZER_CACHE[src]


def _text_span(text: str, region: str) -> tuple[int, int]:
    if region == "all":
        return 0, len(text)

    low = text.lower()
    answer_start = low.find("<answer>")
    if region == "pre_answer":
        return 0, answer_start if answer_start >= 0 else len(text)

    if region != "think":
        raise ValueError(f"Unknown region: {region}")

    think_start = low.find("<think>")
    if think_start >= 0:
        start = think_start + len("<think>")
        end = low.find("</think>", start)
        if end < 0:
            end = answer_start
        if end < 0:
            end = len(text)
        return start, max(start, end)
    return 0, answer_start if answer_start >= 0 else len(text)


def _allowed_token_positions(
    text: str,
    tokenizer,
    *,
    region: str,
    seq_len: int,
    max_length: int,
    append_eos: bool,
) -> set[int]:
    start, end = _text_span(text, region)
    completion = text + ((tokenizer.eos_token or "") if append_eos else "")
    enc = tokenizer(
        completion,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    out: set[int] = set()
    for idx, offset in enumerate(enc.get("offset_mapping", [])):
        if idx >= seq_len:
            break
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            continue
        tok_start, tok_end = int(offset[0]), int(offset[1])
        if tok_end <= tok_start:
            continue
        if tok_end > start and tok_start < end:
            out.add(idx)
    return out


def _method_specs(root_dir: Path) -> list[dict[str, Any]]:
    specs = []
    for model_key in MODEL_ORDER:
        interval_path = root_dir / f"{model_key}_partial_fixed_reward_localisation" / "pair_details.jsonl"
        rebuttal_interval_path = (
            root_dir / "qwen7b_partial_fixed_rebuttal_restart_reward_localisation" / "pair_details.jsonl"
        )
        if model_key == "qwen7b" and rebuttal_interval_path.exists():
            interval_path = rebuttal_interval_path
        specs.extend(
            [
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "method_key": "reward_dense",
                    "signal": "Reward dense",
                    "path": root_dir / f"{model_key}_full_reward_localisation" / "pair_details.jsonl",
                    "seq_key": "pert_score_seq",
                    "detector": "largest_drop",
                    "is_policy": False,
                    "append_eos": True,
                },
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "method_key": "reward_interval",
                    "signal": "Reward interval",
                    "path": interval_path,
                    "seq_key": "pert_score_seq",
                    "detector": "largest_drop",
                    "is_policy": False,
                    "append_eos": True,
                },
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "method_key": "sft_logprob",
                    "signal": "SFT log-probability",
                    "path": root_dir / f"{model_key}_sft_policy_token_baselines" / "policy_token_baselines.jsonl",
                    "seq_key": "pert_policy_log_probs",
                    "detector": "largest_drop",
                    "is_policy": True,
                    "append_eos": False,
                },
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "method_key": "sft_entropy",
                    "signal": "SFT entropy",
                    "path": root_dir / f"{model_key}_sft_policy_token_baselines" / "policy_token_baselines.jsonl",
                    "seq_key": "pert_policy_entropies",
                    "detector": "largest_spike",
                    "is_policy": True,
                    "append_eos": False,
                },
            ]
        )
    return specs


def _targets(row: dict[str, Any], is_policy: bool, seq_len: int) -> list[int]:
    key = "policy_changed_token_positions" if is_policy else "changed_token_positions"
    values = row.get(key, [])
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        try:
            idx = int(value)
        except Exception:
            continue
        if 0 <= idx < seq_len:
            out.append(idx)
    return sorted(set(out))


def _unit_sequence_and_targets(
    seq: Sequence[float],
    targets: Sequence[int],
    allowed_tokens: set[int],
    *,
    grid: str,
    stride: int,
) -> tuple[np.ndarray, list[int], set[int]]:
    arr = np.asarray(seq, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if grid == "token":
        return arr, sorted({int(t) for t in targets if 0 <= int(t) < arr.shape[0]}), set(allowed_tokens)

    if grid != "interval15":
        raise ValueError(f"Unknown grid: {grid}")

    n_buckets = int(math.ceil(arr.shape[0] / float(stride)))
    unit = np.zeros(n_buckets, dtype=np.float64)
    for bucket in range(n_buckets):
        start = bucket * stride
        end = min(arr.shape[0], (bucket + 1) * stride)
        unit[bucket] = float(np.nanmean(arr[start:end])) if end > start else float("nan")
    unit_targets = sorted({int(t) // stride for t in targets if 0 <= int(t) < arr.shape[0]})
    unit_allowed = sorted({int(t) // stride for t in allowed_tokens if 0 <= int(t) < arr.shape[0]})
    return unit[np.isfinite(unit)], unit_targets, set(unit_allowed)


def _hit_at_k(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int], window: int, k: int) -> float:
    if scores.size == 0 or not targets:
        return float("nan")
    top = np.argsort(-scores, kind="stable")[: max(1, int(k))]
    preds = [int(indices[pos]) for pos in top.tolist()]
    return float(any(abs(pred - int(target)) <= int(window) for pred in preds for target in targets))


def _hit_chance_at_k(indices: np.ndarray, targets: Sequence[int], window: int, k: int) -> float:
    n = int(indices.shape[0])
    if n <= 0 or not targets:
        return float("nan")
    mask = np.zeros(n, dtype=bool)
    for target in targets:
        mask |= np.abs(indices - int(target)) <= int(window)
    m = int(mask.sum())
    k = min(max(1, int(k)), n)
    if m <= 0:
        return 0.0
    if n - m < k:
        return 1.0
    return float(1.0 - math.comb(n - m, k) / math.comb(n, k))


def _tie_aware_average_precision(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int]) -> float:
    """Expected AP when positions with identical scores are randomly ordered.

    Expanded interval scores create large tied blocks on the token grid. Stable
    sorting those ties can make the AP depend on tokenizer/order artifacts. This
    tie-aware variant gives a constant signal exactly random AP, as it should.
    """
    if scores.size == 0 or not targets:
        return float("nan")
    target_set = {int(target) for target in targets}
    labels = np.asarray([1 if int(idx) in target_set else 0 for idx in indices.tolist()], dtype=np.int64)
    total_pos = int(labels.sum())
    if total_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ap_sum = 0.0
    prev_hits = 0.0
    rank_before = 0
    pos = 0
    while pos < sorted_scores.shape[0]:
        score = sorted_scores[pos]
        end = pos + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == score:
            end += 1

        group = sorted_labels[pos:end]
        group_size = int(end - pos)
        group_pos = int(group.sum())
        if group_pos > 0:
            if group_size == 1:
                ap_sum += (prev_hits + 1.0) / float(rank_before + 1)
            else:
                for offset in range(1, group_size + 1):
                    prob_current_is_positive = group_pos / float(group_size)
                    expected_positive_before = (offset - 1) * (group_pos - 1) / float(group_size - 1)
                    ap_sum += prob_current_is_positive * (
                        prev_hits + expected_positive_before + 1.0
                    ) / float(rank_before + offset)

        prev_hits += group_pos
        rank_before += group_size
        pos = end

    return float(ap_sum / total_pos)


def _normalize(value: float, chance: float) -> float:
    if not math.isfinite(value) or not math.isfinite(chance):
        return float("nan")
    denom = 1.0 - chance
    if denom <= 1e-12:
        return float("nan")
    return float((value - chance) / denom)


def _score_method(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    text_by_key: dict[tuple[Any, ...], str],
    *,
    region: str,
    grid: str,
    window: int,
    stride: int,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if spec["is_policy"]:
        tokenizer_path = rows[0].get("policy_model") if rows else None
    else:
        tokenizer_path = rows[0].get("reward_checkpoint_dir") if rows else None
    tok = _tokenizer(tokenizer_path)
    unit_window = int(math.ceil(max(0, int(window)) / float(stride))) if grid == "interval15" else int(window)

    values: dict[str, list[float]] = {
        "hit1": [],
        "hit3": [],
        "hit5": [],
        "chance_hit1": [],
        "chance_hit3": [],
        "chance_hit5": [],
        "norm_hit1": [],
        "norm_hit3": [],
        "norm_hit5": [],
        "map": [],
        "map_tie_aware": [],
        "chance_map": [],
        "norm_map": [],
        "norm_map_tie_aware": [],
        "mrr": [],
    }
    pred_fracs = []
    candidate_counts = []
    total_transition_counts = []
    skipped = 0

    for row in rows:
        seq = row.get(spec["seq_key"], [])
        if not isinstance(seq, list) or len(seq) < 2:
            skipped += 1
            continue
        text = text_by_key.get(_row_key(row))
        if not isinstance(text, str):
            skipped += 1
            continue
        seq_len = len(seq)
        targets = _targets(row, bool(spec["is_policy"]), seq_len)
        if not targets:
            skipped += 1
            continue
        allowed_tokens = _allowed_token_positions(
            text,
            tok,
            region=region,
            seq_len=seq_len,
            max_length=1124,
            append_eos=bool(spec["append_eos"]),
        )
        unit_seq, unit_targets, unit_allowed = _unit_sequence_and_targets(
            seq,
            targets,
            allowed_tokens,
            grid=grid,
            stride=stride,
        )
        scores, indices = _transition_scores(unit_seq, str(spec["detector"]))
        if scores.size == 0:
            skipped += 1
            continue
        if region != "all":
            mask = np.asarray([int(idx) in unit_allowed for idx in indices.tolist()], dtype=bool)
            scores = scores[mask]
            indices = indices[mask]
        if scores.size == 0:
            skipped += 1
            continue

        total_transition_counts.append(max(0, int(unit_seq.shape[0]) - 1))
        candidate_counts.append(int(indices.shape[0]))
        pred = int(indices[int(np.argmax(scores))])
        pred_fracs.append(float(pred / max(1, int(unit_seq.shape[0]))))

        for k in (1, 3, 5):
            hit = _hit_at_k(scores, indices, unit_targets, unit_window, k)
            chance = _hit_chance_at_k(indices, unit_targets, unit_window, k)
            values[f"hit{k}"].append(hit)
            values[f"chance_hit{k}"].append(chance)
            values[f"norm_hit{k}"].append(_normalize(hit, chance))

        ap = _average_precision(scores, indices, unit_targets)
        ap_tie_aware = _tie_aware_average_precision(scores, indices, unit_targets)
        exact_relevant = int(sum(1 for idx in indices.tolist() if int(idx) in set(unit_targets)))
        ap_chance = _expected_random_average_precision(int(indices.shape[0]), exact_relevant)
        values["map"].append(ap)
        values["map_tie_aware"].append(ap_tie_aware)
        values["chance_map"].append(ap_chance)
        values["norm_map"].append(_normalize(ap, ap_chance))
        values["norm_map_tie_aware"].append(_normalize(ap_tie_aware, ap_chance))

        order = np.argsort(-scores, kind="stable")
        target_set = set(unit_targets)
        reciprocal = float("nan")
        for rank, pos in enumerate(order.tolist(), start=1):
            if int(indices[pos]) in target_set:
                reciprocal = float(1.0 / rank)
                break
        values["mrr"].append(reciprocal)

    metrics = {
        name: _mean_ci(vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed)
        for name, vals in values.items()
    }
    metrics["pred_fraction"] = {
        "median": float(np.median(pred_fracs)) if pred_fracs else None,
        "q25": float(np.percentile(pred_fracs, 25)) if pred_fracs else None,
        "q75": float(np.percentile(pred_fracs, 75)) if pred_fracs else None,
    }
    metrics["candidate_counts"] = {
        "mean": float(np.mean(candidate_counts)) if candidate_counts else None,
        "total_transition_mean": float(np.mean(total_transition_counts)) if total_transition_counts else None,
    }
    return {
        "region": region,
        "grid": grid,
        "model_key": spec["model_key"],
        "model": spec["model"],
        "method_key": spec["method_key"],
        "signal": spec["signal"],
        "n_input": len(rows),
        "n_scored": int(metrics["map"]["n"] or 0),
        "n_skipped": skipped,
        "tokenizer_source": _tokenizer_source(tokenizer_path),
        "metrics": metrics,
    }


def _fmt_pct(metric: dict[str, Any]) -> str:
    mean = metric.get("mean")
    ci = metric.get("ci_halfwidth")
    if mean is None:
        return "-"
    return f"{100.0 * float(mean):.2f}" + (f" ± {100.0 * float(ci):.2f}" if ci is not None else "")


def _print_slice(results: list[dict[str, Any]], *, region: str, grid: str, metric: str) -> None:
    print(f"\n=== region={region} grid={grid} metric={metric} ===")
    for model_key in MODEL_ORDER:
        rows = [r for r in results if r["region"] == region and r["grid"] == grid and r["model_key"] == model_key]
        if not rows:
            continue
        print(MODEL_LABELS[model_key])
        for row in rows:
            metric_obj = row["metrics"][metric]
            chance_key = f"chance_{metric}"
            chance = row["metrics"].get(chance_key)
            if chance is not None:
                delta = float(metric_obj["mean"]) - float(chance["mean"])
                print(
                    f"  {row['signal']}: {_fmt_pct(metric_obj)} "
                    f"(chance {_fmt_pct(chance)}, delta {100.0 * delta:+.2f})"
                )
            else:
                print(f"  {row['signal']}: {_fmt_pct(metric_obj)}")


def main() -> None:
    args = parse_args()
    strict_rows = _load_jsonl(args.strict_input)
    strict_keys = {_row_key(row) for row in strict_rows}
    text_by_key = {
        _row_key(row): row["pert_text"]
        for row in _load_jsonl(args.all_input) + strict_rows
        if isinstance(row.get("pert_text"), str)
    }

    results = []
    for spec in _method_specs(args.root_dir):
        if not Path(spec["path"]).exists():
            continue
        rows = [row for row in _load_jsonl(Path(spec["path"])) if _row_key(row) in strict_keys]
        if not rows:
            continue
        for region in REGIONS:
            for grid in GRIDS:
                results.append(
                    _score_method(
                        rows,
                        spec,
                        text_by_key,
                        region=region,
                        grid=grid,
                        window=int(args.window),
                        stride=int(args.stride),
                        bootstrap_samples=int(args.bootstrap_samples),
                        bootstrap_alpha=float(args.bootstrap_alpha),
                        bootstrap_seed=int(args.bootstrap_seed),
                    )
                )

    payload = {
        "root_dir": str(args.root_dir),
        "strict_input": str(args.strict_input),
        "n_strict": len(strict_keys),
        "window": int(args.window),
        "stride": int(args.stride),
        "regions": list(REGIONS),
        "grids": list(GRIDS),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output_json}")

    _print_slice(results, region="think", grid="token", metric="map")
    _print_slice(results, region="think", grid="token", metric="hit1")
    _print_slice(results, region="think", grid="token", metric="hit3")
    _print_slice(results, region="think", grid="interval15", metric="map")
    _print_slice(results, region="think", grid="interval15", metric="hit1")
    _print_slice(results, region="all", grid="interval15", metric="map")


if __name__ == "__main__":
    main()
