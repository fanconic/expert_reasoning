"""
plot_helpers.py

Helper utilities for reading eval_result.jsonl files, computing metrics,
creating plots, and saving outputs. Extracted from the original notebook-style
script and made reusable.
"""

from __future__ import annotations
import json
import os
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from sklearn.metrics import roc_auc_score
from src.utils.transformers_compat import configure_pytorch_transformers_runtime

configure_pytorch_transformers_runtime()

from transformers import AutoTokenizer
from collections import Counter

try:
    from src.plot_generators.token_viz import make_text_reward_image
except ModuleNotFoundError:
    from token_viz import make_text_reward_image

plt.style.use("bright")
plt.rcParams["font.family"] = "sans-serif"

import matplotlib.colors as mcolors

# Grab system default colours
c0 = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]  # usually blue
c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]  # usually orange/red

# Create a custom diverging cmap: negative = c1, positive = c0
CUSTOM_COLOR_MAP = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", [c1, "white", c0]
)

# -------------------------------
# Parsing / reward helpers
# -------------------------------
STRICT_FMT = re.compile(
    r"^<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$", flags=re.DOTALL
)
SOFT_FMT = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", flags=re.DOTALL)


def count_xml(text) -> float:
    """
    Calculate a score based on the presence and formatting of XML tags.

    Awards partial points for each correctly formatted tag and penalizes
    extra content after the closing </answer> tag.

    Args:
        text (str): The text to analyze for XML formatting.

    Returns:
        float: A score between 0.0 and 0.5 based on XML formatting quality.
    """
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.125
    if text.count("</think>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
        count -= len(text.split("</answer>")[-1]) * 0.001
    if text.count("</answer>") == 1:
        count += 0.125
        count -= (len(text.split("</answer>")[-1]) - 1) * 0.001
    return count


def strict_format_reward_func(response, **kwargs):
    return 0.5 if STRICT_FMT.match(response) else 0.0


def soft_format_reward_func(completions, **kwargs):
    responses = [c[0]["content"] for c in completions]
    return [0.5 if SOFT_FMT.search(r) else 0.0 for r in responses]


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# -------------------------------
# Metrics utilities
# -------------------------------


def _pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    if num_correct == 0 or k > num_samples:
        return 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def compute_pass_at_k(all_correct_flags: List[List[bool]], ks: Iterable[int]):
    totals = {k: 0.0 for k in ks}
    for flags in all_correct_flags:
        n = len(flags)
        m = sum(flags)
        for k in ks:
            totals[k] += _pass_at_k(m, n, k)
    num_problems = len(all_correct_flags)
    return {k: totals[k] / num_problems for k in ks}


def compute_success_at_k_from_scores(all_correct_flags, all_scores, ks):
    num_problems = len(all_correct_flags)
    totals = {k: 0.0 for k in ks}
    for flags, scores in zip(all_correct_flags, all_scores):
        scores = np.asarray(scores, dtype=float)
        flags = np.asarray(flags, dtype=bool)
        N = len(flags)
        order = np.argsort(scores)[::-1]
        for k in ks:
            if k > N:
                continue
            topk = order[:k]
            totals[k] += float(flags[topk].any())
    return {k: totals[k] / num_problems for k in ks}


def _softmax_scores(scores, temperature: float = 1.0):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.asarray([], dtype=float)

    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.ones(n, dtype=float) / n

    temp = max(float(temperature), 1e-8)
    shifted = scores[finite_mask] / temp
    shifted -= np.max(shifted)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)

    probs = np.zeros(n, dtype=float)
    if not np.isfinite(denom) or denom <= 0:
        probs[finite_mask] = 1.0 / finite_mask.sum()
        return probs

    probs[finite_mask] = exp_vals / denom
    return probs


def _pass_at_k_from_effective_mass(
    effective_num_correct: float, num_samples: int, k: int
) -> float:
    """
    Generalized pass@k using an effective (possibly fractional) number of correct samples.
    Reduces to exact Chen et al. pass@k when effective_num_correct is an integer.
    """
    if effective_num_correct <= 0.0 or k <= 0 or k > num_samples:
        return 0.0
    if effective_num_correct >= num_samples:
        return 1.0

    remaining = num_samples - effective_num_correct
    if remaining <= 0.0:
        return 1.0

    fail_prob = 1.0
    for j in range(k):
        fail_prob *= (remaining - j) / (num_samples - j)
    fail_prob = float(np.clip(fail_prob, 0.0, 1.0))
    return 1.0 - fail_prob


def compute_reward_weighted_pass_at_k_from_scores(
    all_correct_flags, all_scores, ks, temperature: float = 1.0
):
    """
    Reward-weighted pass@k with uniform-reward consistency:
      if all rewards are equal, this equals classical pass@k.

    We softmax scores to get probability mass on correct candidates, convert that
    mass to an effective number of correct samples, then apply generalized pass@k.
    """
    totals = {k: 0.0 for k in ks}
    num_problems = len(all_correct_flags)
    if num_problems == 0:
        return {k: 0.0 for k in ks}

    for flags, scores in zip(all_correct_flags, all_scores):
        flags = np.asarray(flags, dtype=bool)
        scores = np.asarray(scores, dtype=float)

        if len(flags) != len(scores):
            n = min(len(flags), len(scores))
            flags = flags[:n]
            scores = scores[:n]
        if len(flags) == 0:
            continue

        probs = _softmax_scores(scores, temperature=temperature)
        p_correct = float(np.sum(probs[flags])) if flags.any() else 0.0
        p_correct = float(np.clip(p_correct, 0.0, 1.0))
        n = len(flags)
        m_eff = n * p_correct

        for k in ks:
            if k <= 0:
                continue
            totals[k] += _pass_at_k_from_effective_mass(m_eff, n, k)

    return {k: totals[k] / num_problems for k in ks}


def bootstrap_ci(
    metric_fn, all_correct_flags, ks, all_scores=None, n_boot=1000, alpha=0.05, seed=42
):
    rng = np.random.default_rng(seed)
    n = len(all_correct_flags)
    bootstrapped = {k: [] for k in ks}
    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        flags_bs = [all_correct_flags[i] for i in idxs]
        if all_scores is not None:
            scores_bs = [all_scores[i] for i in idxs]
            metrics = metric_fn(flags_bs, scores_bs, ks)
        else:
            metrics = metric_fn(flags_bs, ks)
        for k in ks:
            bootstrapped[k].append(metrics[k])
    ci = {}
    for k in ks:
        lower = np.percentile(bootstrapped[k], 100 * alpha / 2)
        upper = np.percentile(bootstrapped[k], 100 * (1 - alpha / 2))
        ci[k] = (lower, upper)
    return ci


def compute_advantages(rewards, gamma=0.99, baseline=None):
    T = len(rewards)
    advantages = np.zeros(T)
    for t in range(T):
        discounted_sum = 0
        for s in range(t, T):
            discounted_sum += (gamma ** (s - t)) * rewards[s]
        if baseline is not None:
            advantages[t] = discounted_sum - baseline[t]
        else:
            advantages[t] = discounted_sum
    return advantages


def extract_flags(df: pd.DataFrame, num_generations: int = 16, disc: bool = True):
    all_correct_flags = []
    for i in range(0, len(df), num_generations):
        sub_df = df.iloc[i : i + num_generations]
        all_correct_flags.append(
            np.array(sub_df.correctness_reward_func == 2, dtype=int).tolist()
        )
    return all_correct_flags


def extract_flags_and_scores(df: pd.DataFrame, num_generations: int = 16):
    """
    Build aligned correctness flags and selector score lists per prompt.
    """
    grouped = _group_by_prompt(df, num_generations=num_generations)
    all_correct_flags, all_scores = [], []

    for sub_df in grouped:
        if sub_df.empty or "correctness_reward_func" not in sub_df.columns:
            continue

        flags = np.array(sub_df["correctness_reward_func"] == 2, dtype=int).tolist()
        if "selector" in sub_df.columns:
            scores = pd.to_numeric(sub_df["selector"], errors="coerce").tolist()
        else:
            # Fallback keeps shape aligned if selector is unavailable.
            scores = [0.0] * len(flags)

        n = min(len(flags), len(scores))
        if n == 0:
            continue
        all_correct_flags.append(flags[:n])
        all_scores.append(scores[:n])

    return all_correct_flags, all_scores


# -------------------------------
# IO + plotting orchestration
# -------------------------------


def discounted_mean(scores, gamma=0.9):
    """
    Calculates a weighted average where the last element has the highest weight (1.0),
    and previous elements decay by a factor of gamma. Handles NaNs.
    """
    # Ensure input is a numpy array
    scores = np.array(scores)

    # Create a mask for valid (non-NaN) values
    mask = ~np.isnan(scores)

    # If all values are NaN, return NaN
    if not np.any(mask):
        return np.nan

    # Generate weights: [gamma^(n-1), ..., gamma^1, 1]
    n = len(scores)
    weights = gamma ** np.arange(n)[::-1]

    # Apply mask to both scores and weights
    valid_scores = scores[mask]
    valid_weights = weights[mask]

    # Calculate weighted average
    return np.sum(valid_scores * valid_weights) / np.sum(valid_weights)


def _to_valid_float_array(values: Any) -> np.ndarray:
    if not isinstance(values, (list, np.ndarray)):
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _safe_softmax(values: np.ndarray, beta: float = 1.0) -> np.ndarray:
    if values.size == 0:
        return values
    scaled = beta * values
    scaled = scaled - np.max(scaled)
    exp_vals = np.exp(scaled)
    denom = np.sum(exp_vals)
    if not np.isfinite(denom) or denom <= 0:
        return np.ones_like(values, dtype=float) / len(values)
    return exp_vals / denom


def _map_span_to_reward_indices(
    answer_span: Any, token_count: Optional[int], reward_count: int
) -> Optional[Tuple[int, int]]:
    if reward_count <= 0:
        return None
    if not isinstance(answer_span, (tuple, list)) or len(answer_span) != 2:
        return None

    start_raw, end_raw = answer_span
    if start_raw is None or end_raw is None:
        return None

    try:
        start_raw = int(start_raw)
        end_raw = int(end_raw)
    except (TypeError, ValueError):
        return None

    if token_count is not None and token_count > 0:
        if end_raw < 0:
            end_raw = token_count + end_raw
        start = int(round((start_raw / token_count) * reward_count))
        end = int(round((end_raw / token_count) * reward_count))
    else:
        start = start_raw
        end = reward_count + end_raw if end_raw < 0 else end_raw

    start = max(0, min(start, reward_count))
    end = max(0, min(end, reward_count))
    if end <= start:
        return None
    return start, end


def aggregate_dense_rewards(
    rewards: Any,
    mode: str = "discounted_mean",
    *,
    gamma: float = 0.95,
    tail_k: int = 4,
    top_k: int = 4,
    softmax_beta: float = 2.0,
    power_p: float = 2.0,
    trim_frac: float = 0.1,
    answer_span: Any = None,
    token_count: Optional[int] = None,
    answer_weight: float = 2.0,
    fallback_tail_ratio: float = 0.2,
) -> float:
    """
    Aggregate dense per-token reward traces into a single scalar score.

    Notes:
      - `discounted_mean` is the historical selector used in this repo.
      - `answer_boost` upweights the answer segment when a span is available,
        otherwise it falls back to boosting the final tail portion.
    """
    arr = _to_valid_float_array(rewards)
    if arr.size == 0:
        return float("nan")

    mode_key = str(mode or "").strip().lower()
    if mode_key in {"discounted", "discounted_mean", "current"}:
        return float(discounted_mean(arr, gamma=gamma))
    if mode_key in {"mean", "avg", "average"}:
        return float(np.mean(arr))
    if mode_key in {"last", "final", "last_token"}:
        return float(arr[-1])
    if mode_key in {"tail_mean", "tail", "tail_k"}:
        k = max(1, min(int(tail_k), arr.size))
        return float(np.mean(arr[-k:]))
    if mode_key in {"topk_mean", "top_k_mean", "topk"}:
        k = max(1, min(int(top_k), arr.size))
        return float(np.mean(np.sort(arr)[-k:]))
    if mode_key in {"softmax_weighted", "softmax"}:
        w = _safe_softmax(arr, beta=float(softmax_beta))
        return float(np.sum(w * arr))
    if mode_key in {"power_mean", "power"}:
        p = max(float(power_p), 1e-6)
        min_val = float(np.min(arr))
        shift = -min_val + 1e-8 if min_val <= 0 else 0.0
        shifted = arr + shift
        return float((np.mean(shifted**p) ** (1.0 / p)) - shift)
    if mode_key in {"trimmed_mean", "trimmed"}:
        n = arr.size
        if n <= 2:
            return float(np.mean(arr))
        frac = float(np.clip(trim_frac, 0.0, 0.45))
        k = int(np.floor(n * frac))
        if k == 0 or (2 * k) >= n:
            return float(np.mean(arr))
        arr_sorted = np.sort(arr)
        return float(np.mean(arr_sorted[k : n - k]))
    if mode_key in {"answer_boost", "answer_weighted"}:
        n = arr.size
        weights = np.ones(n, dtype=float)

        span = _map_span_to_reward_indices(answer_span, token_count, n)
        if span is not None:
            s, e = span
            weights[s:e] *= float(answer_weight)
        else:
            tail_len = max(1, int(round(n * float(fallback_tail_ratio))))
            weights[-tail_len:] *= float(answer_weight)

        return float(np.sum(weights * arr) / np.sum(weights))

    raise ValueError(f"Unknown dense reward aggregation mode: {mode}")


SELECTOR_VARIANT_DISPLAY_ORDER: List[Tuple[str, str]] = [
    ("selector_variant_discounted", "Discounted Mean (gamma=0.95) [current]"),
    ("selector_variant_mean", "Uniform Mean"),
    ("selector_variant_last", "Last Token"),
    ("selector_variant_tail3", "Tail Mean (k=3)"),
    ("selector_variant_top3", "Top-3 Mean"),
    ("selector_variant_softmax2", "Softmax-Weighted (beta=2)"),
    ("selector_variant_power2", "Power Mean (p=2)"),
    ("selector_variant_trimmed10", "Trimmed Mean (10%)"),
    ("selector_variant_answer_boost2", "Answer-Boosted Mean (x2)"),
]


def _resolve_selector_column(
    selector_mode: str, *, answer_only: bool
) -> Tuple[str, str]:
    mode_key = str(selector_mode or "auto").strip().lower()
    if mode_key in {"auto", "default"}:
        col = (
            "selector_variant_discounted"
            if answer_only
            else "selector_variant_mean"
        )
        return col, f"{mode_key}->{col}"

    aliases = {
        "discounted": "selector_variant_discounted",
        "discounted_mean": "selector_variant_discounted",
        "current": "selector_variant_discounted",
        "mean": "selector_variant_mean",
        "avg": "selector_variant_mean",
        "average": "selector_variant_mean",
        "last": "selector_variant_last",
        "last_token": "selector_variant_last",
        "tail3": "selector_variant_tail3",
        "tail_3": "selector_variant_tail3",
        "top3": "selector_variant_top3",
        "top_3": "selector_variant_top3",
        "softmax2": "selector_variant_softmax2",
        "softmax_2": "selector_variant_softmax2",
        "power2": "selector_variant_power2",
        "power_2": "selector_variant_power2",
        "trimmed10": "selector_variant_trimmed10",
        "trimmed_10": "selector_variant_trimmed10",
        "answer_boost": "selector_variant_answer_boost2",
        "answer_boost2": "selector_variant_answer_boost2",
        "answer": "selector_variant_answer_boost2",
    }
    resolved = aliases.get(mode_key, mode_key)
    if not resolved.startswith("selector_variant_"):
        resolved = f"selector_variant_{resolved}"
    return resolved, selector_mode


def _compute_selector_variant_columns(df: pd.DataFrame, gamma: float = 0.95) -> pd.DataFrame:
    if "reward_model_score_np" not in df.columns:
        return df

    df = df.copy()
    rewards_list = df["reward_model_score_np"].tolist()

    df["selector_variant_discounted"] = [
        aggregate_dense_rewards(x, mode="discounted_mean", gamma=gamma)
        for x in rewards_list
    ]
    df["selector_variant_mean"] = [
        aggregate_dense_rewards(x, mode="mean") for x in rewards_list
    ]
    df["selector_variant_last"] = [
        aggregate_dense_rewards(x, mode="last") for x in rewards_list
    ]
    df["selector_variant_tail3"] = [
        aggregate_dense_rewards(x, mode="tail_mean", tail_k=3) for x in rewards_list
    ]
    df["selector_variant_top3"] = [
        aggregate_dense_rewards(x, mode="topk_mean", top_k=3) for x in rewards_list
    ]
    df["selector_variant_softmax2"] = [
        aggregate_dense_rewards(x, mode="softmax_weighted", softmax_beta=2.0)
        for x in rewards_list
    ]
    df["selector_variant_power2"] = [
        aggregate_dense_rewards(x, mode="power_mean", power_p=2.0) for x in rewards_list
    ]
    df["selector_variant_trimmed10"] = [
        aggregate_dense_rewards(x, mode="trimmed_mean", trim_frac=0.10)
        for x in rewards_list
    ]

    if "answer_positions" in df.columns and "response_token" in df.columns:
        answer_spans = df["answer_positions"].tolist()
        token_lens = df["response_token"].apply(
            lambda x: len(x) if isinstance(x, list) else None
        ).tolist()
    else:
        answer_spans = [None] * len(df)
        token_lens = [None] * len(df)

    df["selector_variant_answer_boost2"] = [
        aggregate_dense_rewards(
            rewards,
            mode="answer_boost",
            answer_span=span,
            token_count=tok_len,
            answer_weight=2.0,
        )
        for rewards, span, tok_len in zip(rewards_list, answer_spans, token_lens)
    ]
    return df


_TOKENIZER_CACHE = {}
MMLU_PRO_DATASET_ROOT = Path("/mnt/pdata/caf83/data/expert_reasoning/mmlu_pro")
MMLU_PRO_SPLITS = ("train", "eval", "test")
_MMLU_CATEGORY_LOOKUP_CACHE: Dict[Path, Dict[str, str]] = {}


def get_tokenizer(model_path: str):
    """Loads the tokenizer once per worker process and caches it."""
    if model_path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _TOKENIZER_CACHE[model_path]


def _normalize_question_for_lookup(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def _extract_question_from_prompt(prompt: Any) -> Optional[str]:
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return None

    if isinstance(prompt, dict):
        content = prompt.get("content")
        return content if isinstance(content, str) else None

    if isinstance(prompt, str):
        return prompt

    return None


def _get_mmlu_category_lookup(
    dataset_root: Path = MMLU_PRO_DATASET_ROOT,
) -> Dict[str, str]:
    dataset_root = Path(dataset_root)
    if dataset_root in _MMLU_CATEGORY_LOOKUP_CACHE:
        return _MMLU_CATEGORY_LOOKUP_CACHE[dataset_root]

    lookup: Dict[str, str] = {}
    for split in MMLU_PRO_SPLITS:
        split_path = dataset_root / f"{split}.jsonl"
        if not split_path.exists():
            continue

        with split_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question = row.get("question")
                category = row.get("category")
                if not isinstance(question, str) or not isinstance(category, str):
                    continue

                normalized_question = _normalize_question_for_lookup(question)
                if normalized_question and normalized_question not in lookup:
                    lookup[normalized_question] = category

    if not lookup:
        print(f"[WARNING] Could not build MMLU-Pro category lookup from {dataset_root}")

    _MMLU_CATEGORY_LOOKUP_CACHE[dataset_root] = lookup
    return lookup


def _attach_mmlu_category(df: pd.DataFrame) -> pd.DataFrame:
    if "prompt" not in df.columns:
        df = df.copy()
        df["mmlu_category"] = pd.NA
        return df

    lookup = _get_mmlu_category_lookup()
    if not lookup:
        df = df.copy()
        df["mmlu_category"] = pd.NA
        return df

    def _lookup_category(prompt: Any):
        question = _extract_question_from_prompt(prompt)
        if not isinstance(question, str):
            return pd.NA

        normalized_question = _normalize_question_for_lookup(question)
        if not normalized_question:
            return pd.NA

        return lookup.get(normalized_question, pd.NA)

    df = df.copy()
    df["mmlu_category"] = df["prompt"].apply(_lookup_category)
    return df


def _resolve_logprobs_path(jsonl_path: str | Path) -> Path:
    """
    Resolve the sidecar logprobs jsonl for a given eval results file.

    Supports both legacy naming:
      - eval_results_logprobs.jsonl
    and run-specific naming:
      - eval_results_logprobs_<same suffix as eval_results_*.jsonl>
    """
    source_path = Path(jsonl_path)
    candidates = []

    name = source_path.name
    if name.startswith("eval_results_"):
        suffix = name[len("eval_results_") :]
        candidates.append(source_path.with_name(f"eval_results_logprobs_{suffix}"))

    candidates.append(source_path.parent / "eval_results_logprobs.jsonl")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the first candidate for a useful missing-file warning path.
    return candidates[0]


def read_and_enhance(
    jsonl_path: str,
    gamma: float = 0.95,
    answer_only: bool = False,
    selector_mode: str = "auto",
) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)

    if "icml_mmlu" in str(jsonl_path).lower():
        df = _attach_mmlu_category(df)

    # ==========================================
    # 1. LOAD AND MERGE LOG PROBS
    # ==========================================
    logprobs_path = _resolve_logprobs_path(jsonl_path)
    if logprobs_path.exists():
        df_logprobs = pd.read_json(logprobs_path, lines=True)
        if "policy_log_probs" in df_logprobs.columns:
            df["policy_log_probs"] = df_logprobs["policy_log_probs"]
        else:
            print(f"[WARNING] 'policy_log_probs' column not found in {logprobs_path}")
    else:
        print(f"[WARNING] Logprobs file missing: {logprobs_path}")

    # ==========================================
    # 2. TOKENIZATION & ALIGNMENT
    # ==========================================
    if "qwen" in str(jsonl_path) and "response_token" not in df.columns:
        tokeniser = get_tokenizer("unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit")
        df = df.copy()
        df["response_token_ids"] = df.apply(
            lambda x: tokeniser(x["generation"]["content"] + tokeniser.eos_token)[
                "input_ids"
            ],
            axis=1,
        )
        df["response_token"] = df.apply(
            lambda x: tokeniser.convert_ids_to_tokens(x["response_token_ids"]), axis=1
        )

        # Shift both rewards and logprobs for Qwen
        if "reward_model_score" in df.columns:
            df["reward_model_score"] = df["reward_model_score"].apply(
                lambda x: [x[0]] + x if isinstance(x, list) and len(x) > 0 else x
            )
        if "policy_log_probs" in df.columns:
            df["policy_log_probs"] = df["policy_log_probs"].apply(
                lambda x: [x[0]] + x if isinstance(x, list) and len(x) > 0 else x
            )

    elif "llama" in str(jsonl_path) and "response_token" not in df.columns:
        tokeniser = get_tokenizer("unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit")
        df = df.copy()
        df["response_token_ids"] = df.apply(
            lambda x: tokeniser(x["generation"]["content"] + tokeniser.eos_token)[
                "input_ids"
            ][1:],
            axis=1,
        )
        df["response_token"] = df.apply(
            lambda x: tokeniser.convert_ids_to_tokens(x["response_token_ids"]), axis=1
        )
    else:
        print(
            f"[WARNING] `llama` or `qwen` not found in {jsonl_path}, skipping tokenization."
        )

    # ==========================================
    # 3. METRIC PROCESSING (Rewards & Logprobs)
    # ==========================================

    # Process Reward Scores
    if "reward_model_score" in df.columns:
        df["reward_model_score_np"] = df["reward_model_score"].apply(
            lambda x: (
                (np.array(x, dtype=float))[~np.isnan(np.array(x, dtype=float))]
                if isinstance(x, (list, np.ndarray))
                else np.array([])
            )
        )
        df["mean_rewards"] = df["reward_model_score_np"].apply(
            lambda x: np.nanmean(x) if len(x) > 0 else np.nan
        )

    # Process Policy Log Probs (Mirrored)
    if "policy_log_probs" in df.columns:
        df["policy_log_probs_np"] = df["policy_log_probs"].apply(
            lambda x: (
                (np.array(x, dtype=float))[~np.isnan(np.array(x, dtype=float))]
                if isinstance(x, (list, np.ndarray))
                else np.array([])
            )
        )
        df["mean_log_probs"] = df["policy_log_probs_np"].apply(
            lambda x: np.nanmean(x) if len(x) > 0 else np.nan
        )
        df["sum_log_probs"] = df["policy_log_probs_np"].apply(
            lambda x: np.nansum(x) if len(x) > 0 else np.nan
        )

    # ==========================================
    # 4. EXTRACTIONS & SELECTORS
    # ==========================================
    if "generation" in df.columns:
        df["strict_format_reward_func"] = df.generation.apply(
            lambda x: strict_format_reward_func(x["content"])
        )
        df["xmlcount_reward_func"] = df.generation.apply(
            lambda x: count_xml(x["content"])
        )

    if "response_token" in df.columns:
        df["answer_positions"] = df["response_token"].apply(
            lambda x: (
                (x.index("answer"), -4)
                if "answer" in x and x.index("answer") < len(x) - 4
                else (-10, -4)
            )
        )

    # Keep all selector variants for later benchmarking.
    if "reward_model_score_np" in df.columns:
        df = _compute_selector_variant_columns(df, gamma=gamma)

        selected_col, selected_mode = _resolve_selector_column(
            selector_mode, answer_only=answer_only
        )
        if selected_col not in df.columns:
            fallback = (
                "selector_variant_discounted"
                if answer_only
                else "selector_variant_mean"
            )
            print(
                f"[WARNING] selector_mode='{selected_mode}' resolved to '{selected_col}', "
                f"which is unavailable. Falling back to '{fallback}'."
            )
            selected_col = fallback

        df["selector"] = pd.to_numeric(df[selected_col], errors="coerce")
        df["selector_source"] = selected_col
    elif "mean_rewards" in df.columns:
        df["selector"] = pd.to_numeric(df["mean_rewards"], errors="coerce")
        df["selector_source"] = "mean_rewards_fallback"

    if "policy_log_probs_np" in df.columns:
        if answer_only:
            df["selector_logprobs"] = df["policy_log_probs_np"].apply(
                lambda x: discounted_mean(x, gamma=gamma)
            )
            df["selector_logprobs_source"] = f"discounted_mean_gamma_{gamma}"
        else:
            df["selector_logprobs"] = pd.to_numeric(
                df["mean_log_probs"], errors="coerce"
            )
            df["selector_logprobs_source"] = "mean_log_probs"
    return df


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_latex_table_txt(
    results: Dict, cis: Dict, ks: Iterable[int], out_file: str | Path
):
    """
    Write a LaTeX table fragment (4 columns for k in {1,3,5,10}).

    Keys expected in `results`/`cis`:
      - "Outcome Sup."        (GRPO row content)
      - "Exp. Reas. (ours)"   (AIRL row content)
      - "SFT"                 (SFT row content)
    """

    def _fmt_row(vals_label: str) -> str:
        return (
            f"{results[vals_label][1]:.4f} [{cis[vals_label][1][0]:.4f}, {cis[vals_label][1][1]:.4f}] & "
            f"{results[vals_label][3]:.4f} [{cis[vals_label][3][0]:.4f}, {cis[vals_label][3][1]:.4f}] & "
            f"{results[vals_label][5]:.4f} [{cis[vals_label][5][0]:.4f}, {cis[vals_label][5][1]:.4f}] & "
            f"{results[vals_label][10]:.4f} [{cis[vals_label][10][0]:.4f}, {cis[vals_label][10][1]:.4f}] \\\\"
        )

    lines = []
    lines.append("& \\rowcolor{gray!20}\\textcolor{gray!90}{GRPO}")
    lines.append(
        "                & \\textcolor{gray!90}{" + _fmt_row("Outcome Sup.") + "}"
    )
    lines.append("& AIRL (ours)    & " + _fmt_row("Exp. Reas. (ours)"))
    lines.append("& SFT            & " + _fmt_row("SFT"))

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))


def save_latex_table_txt_reranking(
    results: Dict, cis: Dict, ks: Iterable[int], out_file: str | Path
):
    """
    Write a LaTeX table fragment (4 columns for k in {1,3,5,10}).

    Keys expected in `results`/`cis`:
      - "Random Rerankning"
      - "Reasoning Reranking"
    """

    def _fmt_row(vals_label: str) -> str:
        return (
            f"{results[vals_label][1]:.4f} [{cis[vals_label][1][0]:.4f}, {cis[vals_label][1][1]:.4f}] & "
            f"{results[vals_label][3]:.4f} [{cis[vals_label][3][0]:.4f}, {cis[vals_label][3][1]:.4f}] & "
            f"{results[vals_label][5]:.4f} [{cis[vals_label][5][0]:.4f}, {cis[vals_label][5][1]:.4f}] & "
            f"{results[vals_label][10]:.4f} [{cis[vals_label][10][0]:.4f}, {cis[vals_label][10][1]:.4f}] \\\\"
        )

    lines = []
    lines.append("Random Reranking & " + _fmt_row("random"))
    lines.append("Reasoning Reranking & " + _fmt_row("reward"))
    lines.append("Length Reranking & " + _fmt_row("heuristic"))
    lines.append("Log Probability & " + _fmt_row("logprobs"))
    lines.append("Majority Voting & " + _fmt_row("majority"))
    lines.append("Weighted Majority & " + _fmt_row("majority_weighted"))
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))


def compute_selector_variant_success_results(
    df: pd.DataFrame,
    ks: Iterable[int],
    num_generations: int = 16,
    selector_columns: Optional[List[str]] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate selector aggregators on success@k|N (Best-of-N reranking objective).
    Returns a mapping: selector_column -> {k -> metric}.
    """
    if df.empty or "correctness_reward_func" not in df.columns:
        return {}

    df_small = df.copy()
    if "generation_idx" in df_small.columns:
        df_small = df_small[df_small["generation_idx"] < num_generations].copy()
    if df_small.empty:
        return {}

    if selector_columns is None:
        selector_columns = [
            col for col, _ in SELECTOR_VARIANT_DISPLAY_ORDER if col in df_small.columns
        ]
    if not selector_columns:
        return {}

    all_correct_flags: List[List[int]] = []
    all_scores_by_selector: Dict[str, List[List[float]]] = {
        col: [] for col in selector_columns
    }

    grouped = _group_by_prompt(df_small, num_generations=num_generations)
    for sub_df in grouped:
        if sub_df.empty or "correctness_reward_func" not in sub_df.columns:
            continue
        flags = np.array(sub_df["correctness_reward_func"] == 2, dtype=int).tolist()
        n = len(flags)
        if n == 0:
            continue

        all_correct_flags.append(flags)
        for col in selector_columns:
            if col not in sub_df.columns:
                all_scores_by_selector[col].append([0.0] * n)
                continue
            score_arr = pd.to_numeric(sub_df[col], errors="coerce").to_numpy(dtype=float)
            if score_arr.size < n:
                score_arr = np.pad(
                    score_arr,
                    (0, n - score_arr.size),
                    mode="constant",
                    constant_values=-np.inf,
                )
            elif score_arr.size > n:
                score_arr = score_arr[:n]

            score_arr = np.nan_to_num(
                score_arr, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf
            )
            all_scores_by_selector[col].append(score_arr.tolist())

    if not all_correct_flags:
        return {}

    results: Dict[str, Dict[int, float]] = {}
    for col in selector_columns:
        scores = all_scores_by_selector[col]
        if len(scores) != len(all_correct_flags):
            continue
        results[col] = compute_success_at_k_from_scores(all_correct_flags, scores, ks)
    return results


def save_selector_variant_table_txt(
    results: Dict[str, Dict[int, float]],
    ks: Iterable[int],
    out_file: str | Path,
) -> None:
    """
    Save selector-aggregator benchmark (success@k|N) as a LaTeX-ready table.
    """
    if not results:
        return

    display_map = {k: v for k, v in SELECTOR_VARIANT_DISPLAY_ORDER}
    sort_keys = sorted(results, key=lambda key: results[key].get(1, -1.0), reverse=True)

    baseline_key = (
        "selector_variant_discounted"
        if "selector_variant_discounted" in results
        else sort_keys[0]
    )
    baseline_p1 = results[baseline_key].get(1, float("nan"))

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Selector Aggregation & pass@1 & pass@3 & pass@5 & pass@10 & $\Delta$pass@1 \\",
        r"\midrule",
    ]

    for key in sort_keys:
        label = display_map.get(key, key.replace("selector_variant_", ""))
        p1 = results[key].get(1, float("nan"))
        p3 = results[key].get(3, float("nan"))
        p5 = results[key].get(5, float("nan"))
        p10 = results[key].get(10, float("nan"))
        delta = p1 - baseline_p1 if np.isfinite(p1) and np.isfinite(baseline_p1) else np.nan
        lines.append(
            f"{_latex_escape(label)} & {p1:.4f} & {p3:.4f} & {p5:.4f} & {p10:.4f} & {delta:+.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"% Baseline for delta: {baseline_key}",
        ]
    )

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))


def _prompt_to_key(prompt: Any) -> str:
    if isinstance(prompt, (list, dict)):
        return str(prompt)
    return str(prompt)


def _group_by_prompt(df: pd.DataFrame, num_generations: int = 16):
    if df.empty:
        return []

    if "prompt" in df.columns:
        df_local = df.copy()
        df_local["_prompt_key"] = df_local["prompt"].apply(_prompt_to_key)
        grouped = [
            sub_df.copy() for _, sub_df in df_local.groupby("_prompt_key", sort=False)
        ]
        for sub_df in grouped:
            if "_prompt_key" in sub_df.columns:
                sub_df.drop(columns=["_prompt_key"], inplace=True)
        return grouped

    groups = []
    for i in range(0, len(df), num_generations):
        groups.append(df.iloc[i : i + num_generations].copy())
    return groups


def _extract_flags_grouped(df: pd.DataFrame, num_generations: int = 16):
    grouped = _group_by_prompt(df, num_generations=num_generations)
    all_correct_flags = []
    for sub_df in grouped:
        if sub_df.empty or "correctness_reward_func" not in sub_df.columns:
            continue
        all_correct_flags.append(
            np.array(sub_df["correctness_reward_func"] == 2, dtype=int).tolist()
        )
    return all_correct_flags


def _available_mmlu_categories(*dfs: pd.DataFrame) -> List[str]:
    categories = set()
    for df in dfs:
        if "mmlu_category" not in df.columns:
            continue
        categories.update(df["mmlu_category"].dropna().astype(str).tolist())
    return sorted(categories)


def _latex_escape(text: str) -> str:
    escaped = str(text)
    escaped = escaped.replace("\\", r"\textbackslash{}")
    escaped = escaped.replace("&", r"\&")
    escaped = escaped.replace("%", r"\%")
    escaped = escaped.replace("$", r"\$")
    escaped = escaped.replace("#", r"\#")
    escaped = escaped.replace("_", r"\_")
    escaped = escaped.replace("{", r"\{")
    escaped = escaped.replace("}", r"\}")
    escaped = escaped.replace("~", r"\textasciitilde{}")
    escaped = escaped.replace("^", r"\textasciicircum{}")
    return escaped


def _format_pass_row_no_ci(values: Dict[int, float], ks: Iterable[int]) -> str:
    return " & ".join(f"{values[k]:.4f}" for k in ks)


def _compute_reranking_results_no_ci(
    df: pd.DataFrame, ks: Iterable[int], num_generations: int = 16
) -> Optional[Dict[str, Dict[int, float]]]:
    if df.empty:
        return None
    if "correctness_reward_func" not in df.columns or "selector" not in df.columns:
        return None

    df_small = df.copy()
    if "generation_idx" in df_small.columns:
        df_small = df_small[df_small["generation_idx"] < num_generations].copy()
    if df_small.empty:
        return None

    if "prompt" not in df_small.columns or "generation" not in df_small.columns:
        return None

    df_small["isolated_prompt"] = df_small["prompt"].apply(
        lambda x: _extract_question_from_prompt(x) or _prompt_to_key(x)
    )
    df_small["extracted_answer"] = df_small["generation"].apply(
        lambda x: (
            extract_xml_answer(x["content"])
            if isinstance(x, dict) and isinstance(x.get("content"), str)
            else ""
        )
    )
    df_small = add_answer_confidence(
        df_small, prompt_col="isolated_prompt", answer_col="extracted_answer"
    )
    df_small = add_weighted_answer_confidence(
        df_small,
        prompt_col="isolated_prompt",
        answer_col="extracted_answer",
        reward_col="selector",
    )
    df_small["length_heuristic"] = df_small["generation"].apply(
        lambda x: (
            -len(x["content"])
            if isinstance(x, dict) and isinstance(x.get("content"), str)
            else 0
        )
    )
    df_small["_prompt_key"] = df_small["prompt"].apply(_prompt_to_key)

    all_correct_flags = []
    all_scores = []
    all_dummy_scores = []
    all_scores_heuristic = []
    all_scores_logprobs = []
    all_scores_majority = []
    all_scores_majority_weighted = []

    for _, sub_df in df_small.groupby("_prompt_key", sort=False):
        flags = np.array(sub_df["correctness_reward_func"] == 2, dtype=int).tolist()
        if not flags:
            continue

        all_correct_flags.append(flags)
        all_scores.append(sub_df["selector"].tolist())
        all_dummy_scores.append([0.0] * len(flags))
        all_scores_heuristic.append(sub_df["length_heuristic"].tolist())
        if "selector_logprobs" in sub_df.columns:
            all_scores_logprobs.append(sub_df["selector_logprobs"].tolist())
        else:
            all_scores_logprobs.append(sub_df["majority_confidence"].tolist())
        all_scores_majority.append(sub_df["majority_confidence"].tolist())
        all_scores_majority_weighted.append(sub_df["weighted_confidence"].tolist())

    if not all_correct_flags:
        return None

    return {
        "reward": compute_success_at_k_from_scores(all_correct_flags, all_scores, ks),
        "random": compute_success_at_k_from_scores(
            all_correct_flags, all_dummy_scores, ks
        ),
        "heuristic": compute_success_at_k_from_scores(
            all_correct_flags, all_scores_heuristic, ks
        ),
        "logprobs": compute_success_at_k_from_scores(
            all_correct_flags, all_scores_logprobs, ks
        ),
        "majority": compute_success_at_k_from_scores(
            all_correct_flags, all_scores_majority, ks
        ),
        "majority_weighted": compute_success_at_k_from_scores(
            all_correct_flags, all_scores_majority_weighted, ks
        ),
    }


def save_latex_table_txt_mmlu_pass_by_category_no_ci(
    df_airl: pd.DataFrame,
    df_sft: pd.DataFrame,
    df_grpo: pd.DataFrame,
    ks: Iterable[int],
    out_file: str | Path,
    num_generations: int = 16,
):
    categories = _available_mmlu_categories(df_airl, df_sft, df_grpo)
    if not categories:
        return

    model_to_df = {
        "Outcome Sup.": df_grpo,
        "AIRL (ours)": df_airl,
        "SFT": df_sft,
    }
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Category & Method & pass@1 & pass@3 & pass@5 & pass@10 \\",
        r"\midrule",
    ]

    for cat_idx, category in enumerate(categories):
        cat_rows = []
        for method, df in model_to_df.items():
            if "mmlu_category" not in df.columns:
                continue
            df_cat = df[df["mmlu_category"] == category]
            flags = _extract_flags_grouped(df_cat, num_generations=num_generations)
            if not flags:
                continue

            values = compute_pass_at_k(flags, ks)
            cat_rows.append(
                f"{_latex_escape(category)} & {_latex_escape(method)} & "
                + _format_pass_row_no_ci(values, ks)
                + r" \\"
            )

        lines.extend(cat_rows)
        if cat_rows and cat_idx < len(categories) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))


def save_latex_table_txt_mmlu_reranking_by_category_no_ci(
    df_airl: pd.DataFrame,
    ks: Iterable[int],
    out_file: str | Path,
    num_generations: int = 16,
):
    categories = _available_mmlu_categories(df_airl)
    if not categories:
        return

    row_order = [
        ("Random Reranking", "random"),
        ("Reasoning Reranking", "reward"),
        ("Length Reranking", "heuristic"),
        ("Log Probability", "logprobs"),
        ("Majority Voting", "majority"),
        ("Weighted Majority", "majority_weighted"),
    ]

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Category & Reranking Method & pass@1 & pass@3 & pass@5 & pass@10 \\",
        r"\midrule",
    ]

    for cat_idx, category in enumerate(categories):
        if "mmlu_category" not in df_airl.columns:
            continue
        df_cat = df_airl[df_airl["mmlu_category"] == category]
        results = _compute_reranking_results_no_ci(
            df_cat, ks, num_generations=num_generations
        )
        if not results:
            continue

        for label, key in row_order:
            if key not in results:
                continue
            lines.append(
                f"{_latex_escape(category)} & {_latex_escape(label)} & "
                + _format_pass_row_no_ci(results[key], ks)
                + r" \\"
            )
        if cat_idx < len(categories) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))


def print_latex_table(results: Dict, cis: Dict, ks: Iterable[int]) -> None:
    """
    Print the exact LaTeX fragment to stdout, so you can copy/paste into your paper.
    """

    def _fmt_row(vals_label: str) -> str:
        return (
            f"{results[vals_label][1]:.2f} [{cis[vals_label][1][0]:.2f}, {cis[vals_label][1][1]:.2f}] & "
            f"{results[vals_label][3]:.2f} [{cis[vals_label][3][0]:.2f}, {cis[vals_label][3][1]:.2f}] & "
            f"{results[vals_label][5]:.2f} [{cis[vals_label][5][0]:.2f}, {cis[vals_label][5][1]:.2f}] & "
            f"{results[vals_label][10]:.2f} [{cis[vals_label][10][0]:.2f}, {cis[vals_label][10][1]:.2f}] \\\\"
        )

    print("& \\rowcolor{gray!20}\\textcolor{gray!90}{GRPO}")
    print("                & \\textcolor{gray!90}{" + _fmt_row("Outcome Sup.") + "}")
    print("& AIRL (ours)    & " + _fmt_row("Exp. Reas. (ours)"))
    print("& SFT            & " + _fmt_row("SFT"))


def compute_pass_results_ci(datasets: Dict[str, List[List[bool]]], ks: Iterable[int]):
    """
    Return (results, cis) dictionaries used for pass@k tables/plots.
    """
    results, cis = {}, {}
    for label, flags in datasets.items():
        res = compute_pass_at_k(flags, ks)
        ci = bootstrap_ci(compute_pass_at_k, flags, ks)
        results[label] = res
        cis[label] = ci
    return results, cis


def compute_reward_weighted_pass_results_ci(
    datasets_with_scores: Dict[str, Dict[str, List[List[float]]]], ks: Iterable[int]
):
    """
    Return (results, cis) for reward-weighted pass@k tables.
    """
    results, cis = {}, {}
    for label, payload in datasets_with_scores.items():
        flags = payload["flags"]
        scores = payload["scores"]
        if len(flags) == 0:
            results[label] = {k: 0.0 for k in ks}
            cis[label] = {k: (0.0, 0.0) for k in ks}
            continue
        res = compute_reward_weighted_pass_at_k_from_scores(flags, scores, ks)
        ci = bootstrap_ci(
            compute_reward_weighted_pass_at_k_from_scores,
            flags,
            ks,
            all_scores=scores,
        )
        results[label] = res
        cis[label] = ci
    return results, cis


def plot_pass_at_k(
    datasets: Dict[str, List[List[bool]]],
    ks: Iterable[int],
    out_path: str | Path,
    title: str = "pass@k comparison",
):
    results = {}
    cis = {}
    for label, flags in datasets.items():
        res = compute_pass_at_k(flags, ks)
        ci = bootstrap_ci(compute_pass_at_k, flags, ks)
        results[label] = res
        cis[label] = ci

    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = prop_cycle.by_key()["color"] if prop_cycle else [None] * 3
    styles = {
        "Outcome Sup.": {
            "color": colors[2] if len(colors) > 2 else None,
            "marker": "x",
            "linestyle": "--",
        },
        "Exp. Reas. (ours)": {
            "color": colors[0] if colors else None,
            "marker": "x",
            "linestyle": "--",
        },
        "SFT": {
            "color": colors[1] if len(colors) > 1 else None,
            "marker": "x",
            "linestyle": "--",
        },
    }

    plt.figure(figsize=(6, 3))
    for label in results:
        means = [results[label][k] for k in ks]
        ci = [cis[label][k] for k in ks]
        lower = [m - c[0] for m, c in zip(means, ci)]
        upper = [c[1] - m for m, c in zip(means, ci)]
        style = styles.get(label, {"color": None, "marker": "x", "linestyle": "--"})
        plt.errorbar(
            ks,
            means,
            yerr=[lower, upper],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=4,
            markersize=6,
        )
    plt.xlabel("k")
    plt.ylabel("pass@k")
    # plt.title(title)
    plt.legend()
    plt.grid()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def add_weighted_answer_confidence(
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    answer_col: str = "extracted_answer",
    reward_col: str = "selector",  # The column containing your scalar reward
) -> pd.DataFrame:
    """
    Returns the original dataframe with a 'weighted_confidence' column.
    Uses a softmax over the reward scores per prompt to assign a positive voting weight
    to each generation, then sums those weights for identical answers.
    """
    df_out = df.copy()

    # 1. Create a hashable version of the prompt for grouping
    df_out["_hashable_prompt"] = df_out[prompt_col].apply(
        lambda x: str(x) if isinstance(x, (list, dict)) else x
    )

    # Helper function to compute softmax safely (subtracting max for numerical stability)
    def calculate_softmax(series):
        # Drop NaNs temporarily for the math, fill with -inf so they get 0 weight
        s_clean = series.fillna(-np.inf)
        e_x = np.exp(s_clean - np.max(s_clean))
        return e_x / e_x.sum()

    # 2. Convert raw rewards into positive voting weights (probabilities) per prompt
    df_out["reward_weight"] = df_out.groupby("_hashable_prompt")[reward_col].transform(
        calculate_softmax
    )

    # 3. Sum the weights for each specific answer per prompt
    df_out["weighted_confidence"] = df_out.groupby(["_hashable_prompt", answer_col])[
        "reward_weight"
    ].transform("sum")

    # Fill NaNs with 0.0 (in case the answer itself was missing/un-parsable)
    df_out["weighted_confidence"] = df_out["weighted_confidence"].fillna(0.0)

    # Clean up temporary columns
    df_out = df_out.drop(columns=["_hashable_prompt", "reward_weight"])

    return df_out


def add_answer_confidence(
    df: pd.DataFrame, prompt_col: str = "prompt", answer_col: str = "extracted_answer"
) -> pd.DataFrame:
    """
    Returns the original dataframe with a new 'majority_confidence' column.
    The confidence represents how often the answer in that specific row
    appeared among all generations for that same prompt.
    """
    df_out = df.copy()

    # 1. Create a hashable version of the prompt for grouping
    df_out["_hashable_prompt"] = df_out[prompt_col].apply(
        lambda x: str(x) if isinstance(x, (list, dict)) else x
    )

    # 2. Count how many times each specific answer appears per prompt
    # Grouping by both prompt AND answer gives us the frequency of that exact response
    answer_counts = df_out.groupby(["_hashable_prompt", answer_col])[
        answer_col
    ].transform("count")

    # 3. Count the total number of valid (non-null) answers per prompt
    total_answers = df_out.groupby("_hashable_prompt")[answer_col].transform("count")

    # 4. Calculate the confidence (frequency ratio) for the row's answer
    df_out["majority_confidence"] = answer_counts / total_answers

    # Fill NaNs with 0.0 (in case the answer itself was missing/un-parsable)
    df_out["majority_confidence"] = df_out["majority_confidence"].fillna(0.0)

    # Clean up the temporary grouping column
    df_out = df_out.drop(columns=["_hashable_prompt"])

    return df_out


def plot_success_at_k_given(
    df: pd.DataFrame,
    ks: Iterable[int],
    num_generations: int,
    out_path: str | Path,
    title: str,
):

    for num_gen in num_generations:
        df_small = df[df.generation_idx < num_gen].copy().reset_index()

        # Extract flags + scores
        all_correct_flags, all_scores = [], []
        for i in range(0, len(df_small), num_gen):
            sub_df = df_small.iloc[i : i + num_gen]
            all_correct_flags.append(
                np.array(sub_df.correctness_reward_func == 2, dtype=int).tolist()
            )
            all_scores.append(sub_df["selector"].tolist())

        all_dummy_scores = [[0.0] * num_gen for _ in range(len(all_correct_flags))]

        all_scores_heuristic = []
        all_scores_logprobs = []
        all_scores_majority = []
        all_scores_majority_weighted = []

        df_small["isolated_prompt"] = df_small["prompt"].apply(
            lambda x: x[1]["content"]
        )
        df_small["extracted_answer"] = df_small["generation"].apply(
            lambda x: extract_xml_answer(x["content"])
        )
        df_small = add_answer_confidence(
            df_small, prompt_col="isolated_prompt", answer_col="extracted_answer"
        )
        df_small = add_weighted_answer_confidence(
            df_small,
            prompt_col="isolated_prompt",
            answer_col="extracted_answer",
            reward_col="selector",
        )  # Combine confidence with reward score
        df_small["length_heuristic"] = df_small["generation"].apply(
            lambda x: -len(x["content"])
        )

        for i in range(0, len(df_small), num_gen):
            sub_df = df_small.iloc[i : i + num_gen]
            all_scores_heuristic.append(sub_df["length_heuristic"].tolist())
            if "selector_logprobs" in df.columns:
                all_scores_logprobs.append(sub_df["selector_logprobs"].tolist())
            else:
                all_scores_logprobs.append(sub_df["majority_confidence"].tolist())
            all_scores_majority.append(sub_df["majority_confidence"].tolist())
            all_scores_majority_weighted.append(sub_df["weighted_confidence"].tolist())

        results_given = compute_success_at_k_from_scores(
            all_correct_flags, all_scores, ks
        )
        cis_given = bootstrap_ci(
            compute_success_at_k_from_scores,
            all_correct_flags,
            ks,
            all_scores=all_scores,
        )

        results_uniform = compute_success_at_k_from_scores(
            all_correct_flags, all_dummy_scores, ks
        )
        cis_uniform = bootstrap_ci(
            compute_success_at_k_from_scores,
            all_correct_flags,
            ks,
            all_scores=all_dummy_scores,
        )

        results_heuristic = compute_success_at_k_from_scores(
            all_correct_flags, all_scores_heuristic, ks
        )
        cis_heuristic = bootstrap_ci(
            compute_success_at_k_from_scores,
            all_correct_flags,
            ks,
            all_scores=all_scores_heuristic,
        )

        results_logprobs = compute_success_at_k_from_scores(
            all_correct_flags, all_scores_logprobs, ks
        )
        cis_logprobs = bootstrap_ci(
            compute_success_at_k_from_scores,
            all_correct_flags,
            ks,
            all_scores=all_scores_logprobs,
        )

        results_majority = compute_success_at_k_from_scores(
            all_correct_flags, all_scores_majority, ks
        )
        cis_majority = bootstrap_ci(
            compute_success_at_k_from_scores,
            all_correct_flags,
            ks,
            all_scores=all_scores_majority,
        )

        results_majority_weighted = compute_success_at_k_from_scores(
            all_correct_flags, all_scores_majority_weighted, ks
        )
        cis_majority_weighted = bootstrap_ci(
            compute_success_at_k_from_scores,
            all_correct_flags,
            ks,
            all_scores=all_scores_majority_weighted,
        )

        results = {
            "reward": results_given,
            "random": results_uniform,
            "heuristic": results_heuristic,
            "logprobs": results_logprobs,
            "majority": results_majority,
            "majority_weighted": results_majority_weighted,
        }
        cis = {
            "reward": cis_given,
            "random": cis_uniform,
            "heuristic": cis_heuristic,
            "logprobs": cis_logprobs,
            "majority": cis_majority,
            "majority_weighted": cis_majority_weighted,
        }
        save_latex_table_txt_reranking(
            results,
            cis,
            ks,
            Path(out_path) / f"pass_at_k_table_reranking_{num_gen}.txt",
        )

        prop_cycle = plt.rcParams.get("axes.prop_cycle")
        colors = prop_cycle.by_key()["color"] if prop_cycle else [None, None]
        styles = {
            "Reward Reranker": {
                "color": colors[0] if colors else None,
                "marker": "x",
                "linestyle": "--",
            },
            "Random Ranking": {
                "color": colors[1] if len(colors) > 1 else None,
                "marker": "x",
                "linestyle": "--",
            },
        }

        plt.figure(figsize=(6, 3))
        for label, (results_model, cis_model) in {
            "Reward Reranker": (results_given, cis_given),
            "Random Ranking": (results_uniform, cis_uniform),
            # "Length Reranker": (results_heuristic, cis_heuristic),
        }.items():
            means = [results_model[k] for k in ks]
            ci = [cis_model[k] for k in ks]
            lower = [m - c[0] for m, c in zip(means, ci)]
            upper = [c[1] - m for m, c in zip(means, ci)]
            style = styles[label]
            plt.errorbar(
                ks,
                means,
                yerr=[lower, upper],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                capsize=4,
                markersize=6,
            )

        plt.xlabel("k")
        plt.ylabel(rf"pass@k$\mid${num_gen}")
        plt.title(title)
        plt.legend()
        plt.grid()
        out_path = Path(out_path)
        ensure_dir(out_path)
        plt.savefig(out_path / f"pass_atkN_expert_{num_gen}.pdf", bbox_inches="tight")
        plt.close()


def plot_reward_distributions(
    df: pd.DataFrame, out_pdf: str | Path, out_pdf_discounted: str | Path
):
    import scipy.stats as stats

    correct = df[df.correctness_reward_func == 2].mean_rewards
    wrong = df[df.correctness_reward_func == 0].mean_rewards

    t_stat, p_value = stats.ttest_ind(correct, wrong, equal_var=False)

    plt.figure(figsize=(6, 3))
    sns.histplot(
        wrong,
        label="Wrong Answer",
        kde=True,
        stat="probability",
        bins=50,
        color="C1",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    sns.histplot(
        correct,
        label="Correct Answer",
        kde=True,
        stat="probability",
        bins=50,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    plt.legend()
    plt.xlabel("Mean Rewards")
    plt.ylabel("Probability")
    # plt.title("Distribution of Rewards based on Correctness")
    p_text = "$p < 0.001$" if p_value < 0.001 else f"p = {p_value:.3f}"
    text = f"t = {t_stat:.2f}, {p_text}"
    plt.text(
        0.03,
        0.78,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ensure_dir(Path(out_pdf).parent)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_rewards_vs_discounted(df: pd.DataFrame, out_pdf: str | Path):
    # Pick a reasonable example: near-zero mean but correct
    idx = df[(abs(df["mean_rewards"]) < 0.01) & (df["correctness_reward_func"] == 2)][
        "mean_rewards"
    ].idxmax()
    rewards = df.loc[idx, "reward_model_score_np"]
    discounted_rewards = df.loc[idx, "reward_model_score_np_discounted"]

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(range(len(rewards))), y=rewards, color="C0")
    # plt.title("Raw Rewards")
    plt.xlabel("Token Timestep")
    plt.ylabel("Reward")
    plt.xticks(rotation=90)

    plt.subplot(1, 2, 2)
    sns.barplot(
        x=list(range(len(discounted_rewards))), y=discounted_rewards, color="C1"
    )
    # plt.title("Discounted Rewards")
    plt.xlabel("Timestep")
    plt.ylabel("Discounted Reward")
    plt.xticks(rotation=90)

    plt.tight_layout()
    ensure_dir(Path(out_pdf).parent)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_formatting_distributions(
    df: pd.DataFrame, out_pdf: str | Path, out_pdf_discounted: str | Path
):
    import scipy.stats as stats

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df[df.strict_format_reward_func == 0].selector,
        label="Wrong Format",
        kde=True,
        stat="probability",
        bins=50,
        color="C1",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    sns.histplot(
        df[df.strict_format_reward_func == 0.5].selector,
        label="Correct Format",
        kde=True,
        stat="probability",
        bins=50,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    correct = df[df.strict_format_reward_func == 0.5].selector
    wrong = df[df.strict_format_reward_func == 0.0].selector
    t_stat, p_value = stats.ttest_ind(correct, wrong, equal_var=False)
    plt.legend()
    plt.xlabel("Mean Discounted Rewards")
    plt.ylabel("Probability")
    # plt.title("Distribution of Rewards based on Formatting")
    p_text = "$p < 0.001$" if p_value < 0.001 else f"p = {p_value:.3f}"
    text = f"t = {t_stat:.2f}, {p_text}"
    plt.text(
        0.02,
        0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ensure_dir(Path(out_pdf).parent)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_reward_correlations(df: pd.DataFrame, out_pdf: str | Path):
    reward_cols = [
        "selector",
        # "selector_discounted",
        "xmlcount_reward_func",
        "strict_format_reward_func",
        # "int_reward_func",
        "correctness_reward_func",
    ]
    rename_map = {
        "selector": "Rewards",
        # "selector_discounted": "Rewards\n(Discounted)",
        "xmlcount_reward_func": "XML Count",
        "strict_format_reward_func": "Strict Format",
        # "int_reward_func": "Integer",
        "correctness_reward_func": "Correctness",
    }
    corr_matrix = df[reward_cols].corr()
    corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=CUSTOM_COLOR_MAP,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        linewidths=0.5,
        square=True,
    )
    # plt.title("Correlation Matrix of GRPO Reward Functions with Reward Model", fontsize=14, pad=20)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    ensure_dir(Path(out_pdf).parent)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        # Find indices in the current bin
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if np.any(mask):
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            bin_weight = np.mean(mask)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece


def _prepare_calibration_arrays(
    labels: List[int], scores: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=float).reshape(-1)
    y_score = np.asarray(scores, dtype=float).reshape(-1)
    n = min(len(y_true), len(y_score))
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    y_true = y_true[:n]
    y_score = y_score[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if not np.any(mask):
        return np.array([], dtype=int), np.array([], dtype=float)

    y_true = (y_true[mask] > 0).astype(int)
    y_score = y_score[mask].astype(float)
    return y_true, y_score


def compute_calibration_metrics(
    labels: List[int], scores: List[float]
) -> Dict[str, float]:
    """
    Computes AUROC and ECE. Scores are converted to probabilities via sigmoid.
    """
    y_true, y_score = _prepare_calibration_arrays(labels, scores)
    if y_true.size == 0:
        return {"AUROC": float("nan"), "ECE": float("nan")}

    # Convert logits to probabilities; clip for numerical stability.
    y_prob = 1 / (1 + np.exp(-np.clip(y_score, -80.0, 80.0)))
    if np.unique(y_true).size < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))

    return {
        "AUROC": auroc,
        "ECE": float(compute_ece(y_true, y_prob)),
    }


def bootstrap_calibration_ci(
    labels: List[int],
    scores: List[float],
    n_boot: int = 1000,
    alpha: int = 0.05,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    y_true, y_score = _prepare_calibration_arrays(labels, scores)
    n = len(y_true)
    if n == 0:
        return {
            "AUROC": (float("nan"), float("nan"), float("nan")),
            "ECE": (float("nan"), float("nan"), float("nan")),
        }

    boot_results = {"AUROC": [], "ECE": []}
    full_metrics = compute_calibration_metrics(y_true.tolist(), y_score.tolist())

    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        labels_bs = y_true[idxs].tolist()
        scores_bs = y_score[idxs].tolist()
        metrics = compute_calibration_metrics(labels_bs, scores_bs)
        if np.isfinite(metrics["AUROC"]):
            boot_results["AUROC"].append(metrics["AUROC"])
        if np.isfinite(metrics["ECE"]):
            boot_results["ECE"].append(metrics["ECE"])

    ci = {}
    for metric in ["AUROC", "ECE"]:
        values = boot_results[metric]
        if len(values) == 0:
            fallback = full_metrics.get(metric, float("nan"))
            if np.isfinite(fallback):
                ci[metric] = (fallback, fallback, fallback)
            else:
                ci[metric] = (float("nan"), float("nan"), float("nan"))
            continue

        lower = float(np.percentile(values, 100 * alpha / 2))
        upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
        mean = float(np.mean(values))
        ci[metric] = (mean, lower, upper)
    return ci


def save_calibration_table_txt(all_metrics: Dict[str, Dict], out_file: str | Path):
    """
    Writes a LaTeX table for AUROC and ECE.
    all_metrics should be: { "Model Name": { "AUROC": (mean, l, u), "ECE": (mean, l, u) } }
    """
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & AUROC $\uparrow$ & ECE $\downarrow$ \\",
        r"\midrule",
    ]

    for label, metrics in all_metrics.items():
        auroc_fmt = f"{metrics['AUROC'][0]:.4f} [{metrics['AUROC'][1]:.4f}, {metrics['AUROC'][2]:.4f}]"
        ece_fmt = f"{metrics['ECE'][0]:.4f} [{metrics['ECE'][1]:.4f}, {metrics['ECE'][2]:.4f}]"
        lines.append(f"{label} & {auroc_fmt} & {ece_fmt} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))
    print(f"Calibration table saved to {out_file}")


def run_calibration_analysis(df_dict: Dict[str, pd.DataFrame], out_dir: Path):
    all_calibration_results = {}

    for name, df in df_dict.items():
        if "correctness_reward_func" not in df.columns or "selector" not in df.columns:
            all_calibration_results[name] = {
                "AUROC": (float("nan"), float("nan"), float("nan")),
                "ECE": (float("nan"), float("nan"), float("nan")),
            }
            continue

        # 1. Prepare labels (1 for correct, 0 for wrong)
        labels = (df["correctness_reward_func"] == 2).astype(int).tolist()

        # 2. Use your 'selector' (discounted mean) as the aggregate logit
        scores = df["selector"].tolist()

        # 3. Compute bootstrapped metrics
        ci_results = bootstrap_calibration_ci(labels, scores)
        all_calibration_results[name] = ci_results

    save_calibration_table_txt(
        all_calibration_results, out_dir / "calibration_metrics_table.txt"
    )


# -------------------------------
# Orchestrator to run everything for one experiment trio
# -------------------------------


def run_all_plots(
    df_airl: pd.DataFrame,
    df_sft: pd.DataFrame,
    df_grpo: pd.DataFrame,
    out_dir: str | Path,
    num_generations: int = 16,
    reranking_generations: list[int] | None = None,
    make_token_figs: bool = True,
):
    out_dir = ensure_dir(out_dir)

    ks = [1, 3, 5, 10]
    datasets = {
        "Outcome Sup.": extract_flags(df_grpo, num_generations),
        "Exp. Reas. (ours)": extract_flags(df_airl, num_generations),
        "SFT": extract_flags(df_sft, num_generations),
    }
    # NEW: compute + print + save LaTeX table fragment
    results, cis = compute_pass_results_ci(datasets, ks)
    print_latex_table(results, cis, ks)  # for direct copy/paste in your terminal
    save_latex_table_txt(results, cis, ks, Path(out_dir) / "pass_at_k_table.txt")

    weighted_datasets = {}
    for label, df in {
        "Outcome Sup.": df_grpo,
        "Exp. Reas. (ours)": df_airl,
        "SFT": df_sft,
    }.items():
        flags, scores = extract_flags_and_scores(df, num_generations=num_generations)
        weighted_datasets[label] = {"flags": flags, "scores": scores}

    weighted_results, weighted_cis = compute_reward_weighted_pass_results_ci(
        weighted_datasets, ks
    )
    save_latex_table_txt(
        weighted_results,
        weighted_cis,
        ks,
        Path(out_dir) / "pass_at_k_table_reward_weighted.txt",
    )

    selector_variant_results = compute_selector_variant_success_results(
        df_airl,
        ks,
        num_generations=num_generations,
    )
    save_selector_variant_table_txt(
        selector_variant_results,
        ks,
        Path(out_dir) / f"pass_at_k_table_selector_variants_{num_generations}.txt",
    )

    if _available_mmlu_categories(df_airl, df_sft, df_grpo):
        save_latex_table_txt_mmlu_pass_by_category_no_ci(
            df_airl,
            df_sft,
            df_grpo,
            ks,
            Path(out_dir) / "pass_at_k_table_by_category.txt",
            num_generations=num_generations,
        )
        save_latex_table_txt_mmlu_reranking_by_category_no_ci(
            df_airl,
            ks,
            Path(out_dir) / "pass_at_k_table_reranking_by_category.txt",
            num_generations=num_generations,
        )

    plot_pass_at_k(
        datasets, ks, out_dir / "pass_at_k_all.pdf", title="pass@k comparison"
    )

    # Create a mapping for the calibration function
    df_dict = {"Outcome Sup.": df_grpo, "Exp. Reas. (ours)": df_airl, "SFT": df_sft}

    # Run the calibration analysis (AUROC/ECE)
    run_calibration_analysis(df_dict, out_dir)

    # success@k|N for AIRL (expert reasoning)

    requested_reranking_generations = reranking_generations or [num_generations]
    valid_reranking_generations = []
    for g in requested_reranking_generations:
        if not isinstance(g, int):
            continue
        if g <= 0 or g > num_generations:
            continue
        if g in valid_reranking_generations:
            continue
        valid_reranking_generations.append(g)
    if not valid_reranking_generations:
        valid_reranking_generations = [num_generations]

    plot_success_at_k_given(
        df_airl,
        ks,
        valid_reranking_generations,
        out_dir,
        title=r"Expert Reasoning: pass@k$\mid$N comparison",
    )

    # distributions by correctness (AIRL)
    plot_reward_distributions(
        df_airl,
        out_dir / "correctness_reward_distribution.pdf",
        out_dir / "correctness_reward_distribution_discounted.pdf",
    )

    # raw vs discounted
    # plot_rewards_vs_discounted(df_airl, out_dir / "rewards_vs_discounted.pdf")

    # formatting distributions
    plot_formatting_distributions(
        df_airl,
        out_dir / "format_rewards.pdf",
        out_dir / "format_rewards_discounted.pdf",
    )

    # correlation heatmap
    plot_reward_correlations(df_airl, out_dir / "reward_correlation_matrix.pdf")

    # Token-based dense reward visualisations (best-effort; requires tokenizer + fields)
    if make_token_figs:
        colour_map = CUSTOM_COLOR_MAP
        discs = [False]

        for disc in discs:
            reward_score_name = (
                "reward_model_score_np_discounted" if disc else "reward_model_score_np"
            )
            postfix = "discounted" if disc else "raw"
            mean_name = "mean_rewards_discounted" if disc else "mean_rewards"
            if "response_token" in df_airl.columns:
                plt.rcParams["text.usetex"] = False
                # 1. Calculate Means
                # Note: Keeping your logic where 'wrong' mean is based on 0,
                # but sampling pool is based on != 2.
                correct_mean = df_airl[df_airl["correctness_reward_func"] == 2][
                    mean_name
                ].mean()
                wrong_mean = df_airl[df_airl["correctness_reward_func"] == 0][
                    mean_name
                ].mean()
                overall_mean = df_airl[mean_name].mean()
                # 2. Standardise rewards (Vectorized is faster than .apply)
                df_airl["prompt_idx"] = np.arange(len(df_airl)) // 16
                df_airl["reward_model_standard"] = (
                    df_airl[reward_score_name] - overall_mean
                )

                # 1. Find row index of Correct answer with HIGHEST 'selector'
                # idxmax returns the index label where the max value is found
                pos_series = (
                    df_airl[df_airl["correctness_reward_func"] == 2]
                    .groupby("prompt_idx")["selector"]
                    .idxmax()
                )

                # 2. Find row index of Wrong answer with LOWEST 'selector'
                # idxmin returns the index label where the min value is found
                neg_series = (
                    df_airl[df_airl["correctness_reward_func"] == 0]
                    .groupby("prompt_idx")["selector"]
                    .idxmin()
                )

                # 3. Merge the two series on 'prompt_idx'
                # This aligns them and drops groups that don't have both a correct and wrong answer
                aligned_pairs = pd.merge(
                    pos_series, neg_series, on="prompt_idx", suffixes=("_pos", "_neg")
                )

                # 4. Extract the aligned lists of indices
                positive_indices = aligned_pairs["selector_pos"].tolist()[:5]
                negative_indices = aligned_pairs["selector_neg"].tolist()[:5]

                # # --- Positive Indices ---
                # # Filter first: Correctness == 2 AND Strict Format == 0.5
                # pos_subset = df_airl[
                #     (df_airl["correctness_reward_func"] == 2) #&
                #     #(df_airl["strict_format_reward_func"] == 0.5)
                # ]
                # # Find the 5 points with the smallest absolute difference from correct_mean
                # positive_indices = (pos_subset[mean_name] - correct_mean).abs().nsmallest(5).index

                # # --- Negative Indices ---
                # # Filter first: Correctness != 2 AND Strict Format == 0.5
                # neg_subset = df_airl[
                #     (df_airl["correctness_reward_func"] != 2) #&
                #     #(df_airl["strict_format_reward_func"] == 0.5)
                # ]
                # # Find the 5 points with the smallest absolute difference from wrong_mean
                # negative_indices = (neg_subset[mean_name] - wrong_mean).abs().nsmallest(5).index
                all_indices = np.concatenate([positive_indices, negative_indices])
                df_airl["reward_model_max"] = df_airl["reward_model_standard"].apply(
                    lambda x: max(x)
                )
                df_airl["reward_model_min"] = df_airl["reward_model_standard"].apply(
                    lambda x: min(x)
                )
                max_value = df_airl.loc[all_indices, "reward_model_max"].max()
                min_value = df_airl.loc[all_indices, "reward_model_min"].min()

                for i, idx in enumerate(positive_indices):
                    tokens = df_airl.loc[idx, "response_token"]
                    scores = df_airl.loc[idx, "reward_model_standard"]
                    question = df_airl.loc[idx, "prompt"][1]["content"]
                    make_text_reward_image(
                        tokens,
                        scores,
                        out_dir / f"dense_rewards_{postfix}/true_{i}.pdf",
                        cmap_name=colour_map,
                        prompt_text=question,
                        font_size=18,
                        dpi=300,
                        max_width_px=4000,
                        max_val=max_value,
                        min_val=min_value,
                    )

                for i, idx in enumerate(negative_indices):
                    tokens = df_airl.loc[idx, "response_token"]
                    scores = df_airl.loc[idx, "reward_model_standard"]
                    question = df_airl.loc[idx, "prompt"][1]["content"]
                    make_text_reward_image(
                        tokens,
                        scores,
                        out_dir / f"dense_rewards_{postfix}/wrong_{i}.pdf",
                        cmap_name=colour_map,
                        prompt_text=question,
                        font_size=18,
                        dpi=300,
                        max_width_px=4000,
                        max_val=max_value,
                        min_val=min_value,
                    )

    # return path for reference
    return Path(out_dir)
