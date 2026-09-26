"""Summarize natural-error localisation scores for MedReason and MMLU-Pro.

The GPU scorers write per-example token sequences for reward, log-probability,
and entropy.  This script evaluates those sequences under the same filtered
ranking protocol used for the GSM8K localisation sweeps: region filters
(`all`, `pre_answer`, `think`), token and 15-token interval grids, Hit@K within
a local window, MAP, tie-aware MAP, MRR, and exact random baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_DIR = Path("localisation/cross_domain_natural_wrong_sft")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42
REGIONS = ("all", "pre_answer", "think")
GRIDS = ("token", "interval15")
HIT_KS = (1, 3, 5)
WINDOWS = (1, 3, 5, 7)
INTERVAL_STRIDE = 15
MAX_LENGTH = 1124


DATASETS = [
    {
        "key": "medreason",
        "label": "MedReason",
        "model": "Qwen2.5-7B-SFT",
        "labels": Path(
            "localisation/medreason_natural_wrong_sft/"
            "medreason_qwen7b_sft_wrong_step_labels_invalid_step_only.jsonl"
        ),
        "score_root": Path("localisation/medreason_natural_wrong_sft/scores"),
        "score_prefix": "qwen7b_medicine_sft",
    },
    {
        "key": "mmlu_pro",
        "label": "MMLU-Pro",
        "model": "Llama3.1-8B-SFT",
        "labels": Path(
            "localisation/mmlu_pro_natural_wrong_sft/"
            "mmlu_pro_llama8b_sft_wrong_step_labels_invalid_step_only.jsonl"
        ),
        "score_root": Path("localisation/mmlu_pro_natural_wrong_sft/scores"),
        "score_prefix": "llama8b_mmlu_sft",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    parser.add_argument("--grids", nargs="+", default=list(GRIDS), choices=list(GRIDS))
    parser.add_argument("--stride", type=int, default=INTERVAL_STRIDE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("prompt_idx"), row.get("variant_idx"), row.get("clean_generation_idx"))


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


def _tokenizer_source(path: str | Path | None) -> str:
    if path is None:
        raise ValueError("Missing tokenizer/model path.")
    p = Path(str(path))
    for candidate in (
        p,
        p / "reward_model",
        p / "policy_model",
    ):
        if (candidate / "tokenizer_config.json").exists() or (candidate / "config.json").exists():
            return str(candidate)
        adapter_cfg = candidate / "adapter_config.json"
        if adapter_cfg.exists():
            obj = json.loads(adapter_cfg.read_text())
            base = obj.get("base_model_name_or_path")
            if base:
                return str(base)
    return str(path)


_TOKENIZER_CACHE: dict[str, Any] = {}


def _tokenizer(path: str | Path | None):
    source = _tokenizer_source(path)
    if source not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[source] = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    return _TOKENIZER_CACHE[source]


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
            out.add(int(idx))
    return out


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
    if not np.isfinite(arr).all():
        # The score files are expected to be finite; if not, keep alignment by
        # replacing invalid values with the sequence mean instead of shortening.
        finite = arr[np.isfinite(arr)]
        fill = float(finite.mean()) if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)

    if grid == "token":
        unit_targets = sorted({int(t) for t in targets if 0 <= int(t) < arr.shape[0]})
        return arr, unit_targets, set(allowed_tokens)

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
    return unit, unit_targets, set(unit_allowed)


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


def _average_precision(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int]) -> float:
    target_set = {int(target) for target in targets}
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


def _tie_aware_average_precision(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int]) -> float:
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
                    expected_positive_before = (
                        (offset - 1) * (group_pos - 1) / float(group_size - 1)
                    )
                    ap_sum += prob_current_is_positive * (
                        prev_hits + expected_positive_before + 1.0
                    ) / float(rank_before + offset)

        prev_hits += group_pos
        rank_before += group_size
        pos = end

    return float(ap_sum / total_pos)


def _expected_random_average_precision(n_items: int, n_relevant: int) -> float:
    if n_items <= 0 or n_relevant <= 0:
        return float("nan")
    if n_items == 1:
        return 1.0
    harmonic = float(sum(1.0 / k for k in range(1, n_items + 1)))
    return float(
        (harmonic + ((n_relevant - 1.0) / (n_items - 1.0)) * (n_items - harmonic))
        / n_items
    )


def _expected_random_mrr(n_items: int, n_relevant: int) -> float:
    if n_items <= 0 or n_relevant <= 0:
        return float("nan")
    if n_relevant >= n_items:
        return 1.0
    no_previous = 1.0
    expected = 0.0
    max_rank = n_items - n_relevant + 1
    for rank in range(1, max_rank + 1):
        remaining = n_items - rank + 1
        expected += no_previous * (n_relevant / float(remaining)) / float(rank)
        no_previous *= (n_items - n_relevant - (rank - 1)) / float(n_items - (rank - 1))
    return float(expected)


def _mrr(scores: np.ndarray, indices: np.ndarray, targets: Sequence[int]) -> float:
    target_set = {int(t) for t in targets}
    if scores.size == 0 or not target_set:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    for rank, pos in enumerate(order.tolist(), start=1):
        if int(indices[pos]) in target_set:
            return float(1.0 / rank)
    return float("nan")


def _normalize(value: float, chance: float) -> float:
    if not math.isfinite(value) or not math.isfinite(chance):
        return float("nan")
    denom = 1.0 - chance
    if denom <= 1e-12:
        return float("nan")
    return float((value - chance) / denom)


def _method_specs(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(dataset["score_root"])
    prefix = str(dataset["score_prefix"])
    return [
        {
            "key": "reward_dense",
            "signal": "Dense reward drop",
            "path": root / f"{prefix}_full_reward_localisation" / "pair_details.jsonl",
            "seq_key": "pert_score_seq",
            "detector": "largest_drop",
            "is_policy": False,
            "append_eos": True,
            "native_grid": "token",
        },
        {
            "key": "reward_interval",
            "signal": "Interval reward drop",
            "path": root / f"{prefix}_partial_fixed_reward_localisation" / "pair_details.jsonl",
            "seq_key": "pert_score_seq",
            "detector": "largest_drop",
            "is_policy": False,
            "append_eos": True,
            "native_grid": "interval15",
        },
        {
            "key": "sft_logprob",
            "signal": "SFT log-prob drop",
            "path": root / f"{prefix}_policy_token_baselines" / "policy_token_baselines.jsonl",
            "seq_key": "pert_policy_log_probs",
            "detector": "largest_drop",
            "is_policy": True,
            "append_eos": False,
            "native_grid": "token",
        },
        {
            "key": "sft_entropy",
            "signal": "SFT entropy spike",
            "path": root / f"{prefix}_policy_token_baselines" / "policy_token_baselines.jsonl",
            "seq_key": "pert_policy_entropies",
            "detector": "largest_spike",
            "is_policy": True,
            "append_eos": False,
            "native_grid": "token",
        },
    ]


def _tokenizer_path(rows: list[dict[str, Any]], is_policy: bool) -> str | Path | None:
    if not rows:
        return None
    key = "policy_model" if is_policy else "reward_checkpoint_dir"
    return rows[0].get(key)


def _score_method(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    text_by_key: dict[tuple[Any, ...], str],
    *,
    dataset: dict[str, Any],
    region: str,
    grid: str,
    stride: int,
    max_length: int,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    tokenizer_path = _tokenizer_path(rows, bool(spec["is_policy"]))
    tok = _tokenizer(tokenizer_path)

    values: dict[str, list[float]] = {
        "map": [],
        "map_tie_aware": [],
        "chance_map": [],
        "norm_map": [],
        "norm_map_tie_aware": [],
        "mrr": [],
        "chance_mrr": [],
        "norm_mrr": [],
    }
    for k in HIT_KS:
        for window in WINDOWS:
            values[f"hit{k}_w{window}"] = []
            values[f"chance_hit{k}_w{window}"] = []
            values[f"norm_hit{k}_w{window}"] = []

    pred_fracs = []
    candidate_counts = []
    relevant_counts = []
    skipped = 0
    skipped_no_text = 0
    skipped_no_target = 0
    skipped_no_candidates = 0

    for row in rows:
        seq = row.get(spec["seq_key"], [])
        if not isinstance(seq, list) or len(seq) < 2:
            skipped += 1
            skipped_no_candidates += 1
            continue
        text = text_by_key.get(_row_key(row))
        if not isinstance(text, str):
            skipped += 1
            skipped_no_text += 1
            continue

        seq_len = len(seq)
        targets = _targets(row, bool(spec["is_policy"]), seq_len)
        if not targets:
            skipped += 1
            skipped_no_target += 1
            continue

        allowed_tokens = _allowed_token_positions(
            text,
            tok,
            region=region,
            seq_len=seq_len,
            max_length=max_length,
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
        if region != "all" and scores.size:
            mask = np.asarray([int(idx) in unit_allowed for idx in indices.tolist()], dtype=bool)
            scores = scores[mask]
            indices = indices[mask]
        if scores.size == 0:
            skipped += 1
            skipped_no_candidates += 1
            continue

        relevant = int(sum(1 for idx in indices.tolist() if int(idx) in set(unit_targets)))
        if relevant <= 0:
            skipped += 1
            skipped_no_target += 1
            continue

        candidate_counts.append(int(indices.shape[0]))
        relevant_counts.append(relevant)
        pred = int(indices[int(np.argmax(scores))])
        pred_fracs.append(float(pred / max(1, int(unit_seq.shape[0]))))

        for k in HIT_KS:
            for window in WINDOWS:
                unit_window = (
                    int(math.ceil(max(0, int(window)) / float(stride)))
                    if grid == "interval15"
                    else int(window)
                )
                hit = _hit_at_k(scores, indices, unit_targets, unit_window, k)
                chance = _hit_chance_at_k(indices, unit_targets, unit_window, k)
                values[f"hit{k}_w{window}"].append(hit)
                values[f"chance_hit{k}_w{window}"].append(chance)
                values[f"norm_hit{k}_w{window}"].append(_normalize(hit, chance))

        ap = _average_precision(scores, indices, unit_targets)
        ap_tie_aware = _tie_aware_average_precision(scores, indices, unit_targets)
        ap_chance = _expected_random_average_precision(int(indices.shape[0]), relevant)
        rr = _mrr(scores, indices, unit_targets)
        rr_chance = _expected_random_mrr(int(indices.shape[0]), relevant)
        values["map"].append(ap)
        values["map_tie_aware"].append(ap_tie_aware)
        values["chance_map"].append(ap_chance)
        values["norm_map"].append(_normalize(ap, ap_chance))
        values["norm_map_tie_aware"].append(_normalize(ap_tie_aware, ap_chance))
        values["mrr"].append(rr)
        values["chance_mrr"].append(rr_chance)
        values["norm_mrr"].append(_normalize(rr, rr_chance))

    metrics = {
        name: _mean_ci(vals, bootstrap_samples, bootstrap_alpha, bootstrap_seed)
        for name, vals in values.items()
    }
    return {
        "dataset_key": dataset["key"],
        "dataset": dataset["label"],
        "model": dataset["model"],
        "region": region,
        "grid": grid,
        "method_key": spec["key"],
        "signal": spec["signal"],
        "native_grid": spec["native_grid"],
        "path": str(spec["path"]),
        "n_input": len(rows),
        "n_scored": int(metrics["map_tie_aware"]["n"] or 0),
        "n_skipped": skipped,
        "n_skipped_no_text": skipped_no_text,
        "n_skipped_no_target": skipped_no_target,
        "n_skipped_no_candidates": skipped_no_candidates,
        "tokenizer_source": _tokenizer_source(tokenizer_path),
        "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else None,
        "relevant_count_mean": float(np.mean(relevant_counts)) if relevant_counts else None,
        "pred_fraction": {
            "median": float(np.median(pred_fracs)) if pred_fracs else None,
            "q25": float(np.percentile(pred_fracs, 25)) if pred_fracs else None,
            "q75": float(np.percentile(pred_fracs, 75)) if pred_fracs else None,
        },
        "metrics": metrics,
    }


def _fmt_pct(metric: dict[str, Any], *, signed: bool = False) -> str:
    mean = metric.get("mean")
    ci = metric.get("ci_halfwidth")
    if mean is None:
        return "-"
    sign = "+" if signed and float(mean) >= 0 else ""
    base = f"{sign}{100.0 * float(mean):.2f}"
    if ci is not None:
        base += f" +/- {100.0 * float(ci):.2f}"
    return base


def _fmt_delta(metric: dict[str, Any], chance: dict[str, Any]) -> str:
    if metric.get("mean") is None or chance.get("mean") is None:
        return "-"
    return f"{100.0 * (float(metric['mean']) - float(chance['mean'])):+.2f}"


def _compact_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in results:
        native_grid = row["native_grid"]
        if row["region"] != "think" or row["grid"] != native_grid:
            continue
        metrics = row["metrics"]
        out.append(
            {
                "Dataset": row["dataset"],
                "Model": row["model"],
                "Signal": row["signal"],
                "Grid": row["grid"],
                "N": str(row["n_scored"]),
                "Hit@7": _fmt_pct(metrics["hit1_w7"]),
                "Chance": _fmt_pct(metrics["chance_hit1_w7"]),
                "Delta": _fmt_delta(metrics["hit1_w7"], metrics["chance_hit1_w7"]),
                "MAP": _fmt_pct(metrics["map_tie_aware"]),
                "MAP Chance": _fmt_pct(metrics["chance_map"]),
                "Norm MAP": _fmt_pct(metrics["norm_map_tie_aware"], signed=True),
                "MRR": _fmt_pct(metrics["mrr"]),
            }
        )
    return out


def _full_rows(results: list[dict[str, Any]], *, region: str, grid: str) -> list[dict[str, str]]:
    out = []
    for row in results:
        if row["region"] != region or row["grid"] != grid:
            continue
        metrics = row["metrics"]
        out.append(
            {
                "Dataset": row["dataset"],
                "Signal": row["signal"],
                "N": str(row["n_scored"]),
                "Hit@1": _fmt_pct(metrics["hit1_w1"]),
                "Hit@3": _fmt_pct(metrics["hit1_w3"]),
                "Hit@5": _fmt_pct(metrics["hit1_w5"]),
                "Hit@7": _fmt_pct(metrics["hit1_w7"]),
                "Hit@7 Ch.": _fmt_pct(metrics["chance_hit1_w7"]),
                "MAP": _fmt_pct(metrics["map_tie_aware"]),
                "MAP Ch.": _fmt_pct(metrics["chance_map"]),
                "MRR": _fmt_pct(metrics["mrr"]),
            }
        )
    return out


def _markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No rows._\n"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "") for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Cross-Domain Natural Error Localisation",
        "",
        (
            "Metrics use LLM-labelled first invalid reasoning spans, filtered to "
            "examples with `label=invalid_step`. `Hit@7` is Hit@1 within a "
            "+/-7 token window on the selected grid; MAP is tie-aware AP so "
            "piecewise-constant interval scores do not benefit from arbitrary "
            "token-order tie breaking."
        ),
        "",
        "## Native Think-Only Summary",
        "",
        _markdown_table(payload["tables"]["native_think"]),
    ]
    for region in REGIONS:
        for grid in GRIDS:
            key = f"{region}_{grid}"
            rows = payload["tables"].get(key, [])
            if not rows:
                continue
            lines.extend(
                [
                    "",
                    f"## Region `{region}`, Grid `{grid}`",
                    "",
                    _markdown_table(rows),
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    missing = []
    for dataset in DATASETS:
        label_rows = _load_jsonl(Path(dataset["labels"]))
        label_keys = {_row_key(row) for row in label_rows}
        text_by_key = {
            _row_key(row): row.get("pert_text") or row.get("wrong_text")
            for row in label_rows
            if isinstance(row.get("pert_text") or row.get("wrong_text"), str)
        }
        for spec in _method_specs(dataset):
            path = Path(spec["path"])
            if not path.exists():
                missing.append(str(path))
                continue
            rows = [row for row in _load_jsonl(path) if _row_key(row) in label_keys]
            for region in args.regions:
                for grid in args.grids:
                    results.append(
                        _score_method(
                            rows=rows,
                            spec=spec,
                            text_by_key=text_by_key,
                            dataset=dataset,
                            region=str(region),
                            grid=str(grid),
                            stride=int(args.stride),
                            max_length=int(args.max_length),
                            bootstrap_samples=int(args.bootstrap_samples),
                            bootstrap_alpha=float(args.bootstrap_alpha),
                            bootstrap_seed=int(args.bootstrap_seed),
                        )
                    )

    tables: dict[str, list[dict[str, str]]] = {"native_think": _compact_rows(results)}
    for region in REGIONS:
        for grid in GRIDS:
            tables[f"{region}_{grid}"] = _full_rows(results, region=region, grid=grid)

    return {
        "datasets": [
            {
                "key": dataset["key"],
                "label": dataset["label"],
                "model": dataset["model"],
                "labels": str(dataset["labels"]),
            }
            for dataset in DATASETS
        ],
        "regions": list(args.regions),
        "grids": list(args.grids),
        "hit_ks": list(HIT_KS),
        "windows": list(WINDOWS),
        "stride": int(args.stride),
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "alpha": float(args.bootstrap_alpha),
            "seed": int(args.bootstrap_seed),
        },
        "missing_score_files": missing,
        "results": results,
        "tables": tables,
    }


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.output_dir / "scored_localisation_summary.json"
    output_md = args.output_md or args.output_dir / "scored_localisation_summary.md"
    payload = build_payload(args)
    _write_json(output_json, payload)
    _write_markdown(output_md, payload)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print("\nNative think-only summary:")
    print(_markdown_table(payload["tables"]["native_think"]))
    if payload["missing_score_files"]:
        print("Missing score files:")
        for path in payload["missing_score_files"]:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
