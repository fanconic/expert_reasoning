"""GSM8K process-sensitivity experiment with expert traces and reasoning-only perturbations.

This script:
1) Loads GSM8K expert demonstrations (`gsm8k_kd`, split configurable).
2) Builds clean/perturbed pairs where perturbations are applied to reasoning only.
3) Scores traces with the learned reward model (dense or sparse).
4) Reports:
   - Pairwise win-rate / margin
   - Monotonic degradation across perturbation severity
   - Dense localization metrics (LocalDrop/FarDrop/Gap, Hit@K)

It is designed for quick smoke checks first via `--max-examples`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.transformers_compat import (
    configure_pytorch_transformers_runtime,
    ensure_transformers_cache_alias,
)

configure_pytorch_transformers_runtime()

import numpy as np
import torch
from omegaconf import OmegaConf
from unsloth import FastLanguageModel
from peft import PeftModel

ensure_transformers_cache_alias()

from trl.trainer.grpo_trainer import apply_chat_template

from src.data.dataset import get_dataset
from src.eval.eval_mode_utils import (
    MODE_PREGENERATED_POLICY,
    MODE_PREGENERATED_POLICY_AND_REWARD,
    resolve_pregenerated_jsonl_path,
)
from src.rewards.reward_functions import eval_correctness_gsm8k
from src.rewards.perturbations import PERTURB_FN_MAP
from src.utils.utils import set_seed


os.environ.setdefault("UNSLOTH_COMPILE_OVERWRITE", "0")

ANSWER_RE = re.compile(r"(<answer>\s*)(.*?)(\s*</answer>)", flags=re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"(<think>\s*)(.*?)(\s*</think>)", flags=re.DOTALL | re.IGNORECASE)


@dataclass
class PairRecord:
    prompt_idx: int
    severity: int
    variant_idx: int
    prompt: list
    answer: str
    clean_trace_source: str
    clean_generation_idx: int | None
    clean_input_reward_score: float | None
    clean_text: str
    pert_text: str
    perturb_fns: list[str]
    changed_token_positions: list[int]
    clean_score_seq: list[float] | None = None
    pert_score_seq: list[float] | None = None
    clean_score_agg: float | None = None
    pert_score_agg: float | None = None
    margin: float | None = None
    win: int | None = None
    local_drop: float | None = None
    far_drop: float | None = None
    localization_gap: float | None = None
    hit_at: dict[str, int] | None = None
    hit_at_random: dict[str, float] | None = None
    hit_at_norm: dict[str, float] | None = None
    local_unit_count: int | None = None
    total_unit_count: int | None = None
    local_unit_ratio: float | None = None
    localization_mode: str | None = None
    # Stabilized localization metrics on normalized/smoothed traces.
    ldm: float | None = None
    ldm_random: float | None = None
    ldm_norm: float | None = None
    onset_lag: float | None = None
    onset_idx: int | None = None
    onset_threshold_used: float | None = None
    false_alarm: int | None = None
    clean_correct: bool | None = None
    pert_correct: bool | None = None
    answer_unchanged: bool | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_eval.yaml",
        help="Base config yaml (used for model defaults).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint dir that contains reward adapters in <checkpoint-dir>/reward_model. "
        "Defaults to model.name from config.",
    )
    parser.add_argument(
        "--reward-name",
        type=str,
        default=None,
        help="Base reward model name/path. Defaults to model.reward_name from config.",
    )
    parser.add_argument(
        "--reward-lora-rank",
        type=int,
        default=None,
        help="Override model.reward_lora_rank from config.",
    )
    parser.add_argument(
        "--reward-gpu-memory-utilization",
        type=float,
        default=None,
        help="Override model.reward_gpu_memory_utilization from config.",
    )
    parser.add_argument(
        "--dense-reward-mode",
        type=str,
        default=None,
        choices=["sparse", "full", "partial", "partial_fixed"],
        help="Override model.dense_rewards mode from config.",
    )
    parser.add_argument(
        "--dense-partial-fixed-n",
        type=int,
        default=None,
        help="Override model.dense_partial_fixed_n (stride for partial_fixed dense rewards).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override model.load_in_4bit from config.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--no-system", action="store_true")
    parser.add_argument(
        "--trace-source",
        type=str,
        default="expert",
        choices=["expert", "pregenerated"],
        help="Source of clean traces: expert targets (default) or pregenerated jsonl generations.",
    )
    parser.add_argument(
        "--pregenerated-mode",
        type=str,
        default=MODE_PREGENERATED_POLICY_AND_REWARD,
        choices=[MODE_PREGENERATED_POLICY, MODE_PREGENERATED_POLICY_AND_REWARD],
        help="Path-resolution mode used when trace-source=pregenerated and no explicit jsonl path is provided.",
    )
    parser.add_argument(
        "--pregenerated-jsonl-path",
        type=str,
        default=None,
        help="Explicit pregenerated jsonl path. If unset, resolved like evaluate.py via mode/source-dir/candidates.",
    )
    parser.add_argument(
        "--pregenerated-source-dir",
        type=str,
        default=None,
        help="Override source directory used for pregenerated path resolution.",
    )
    parser.add_argument(
        "--pregenerated-candidates",
        nargs="+",
        default=None,
        help="Candidate filenames used for pregenerated path resolution (evaluate.py-style).",
    )
    parser.add_argument(
        "--pregenerated-pick",
        type=str,
        default="generation_idx",
        choices=["first", "generation_idx", "max_reward_model_score", "random"],
        help="Which pregenerated trace to pick per prompt when multiple generations exist.",
    )
    parser.add_argument(
        "--pregenerated-generation-idx",
        type=int,
        default=0,
        help="Preferred generation_idx when pregenerated-pick=generation_idx.",
    )
    parser.add_argument(
        "--clean-correct-policy",
        type=str,
        default="require",
        choices=["require", "prefer", "ignore"],
        help=(
            "How to enforce clean-trace correctness before perturbation: "
            "'require' keeps only correct traces (skips prompt otherwise), "
            "'prefer' uses correct traces when available, "
            "'ignore' does no pre-filtering."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=32,
        help="Small subset size for quick smoke checks.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in dataset before taking max-examples.",
    )
    parser.add_argument(
        "--max-severity",
        type=int,
        default=3,
        help="Create perturbation levels 1..max-severity (ops per level = severity).",
    )
    parser.add_argument(
        "--variants-per-severity",
        type=int,
        default=1,
        help="How many perturbed variants to sample per severity and prompt.",
    )
    parser.add_argument(
        "--perturb-fns",
        nargs="+",
        default=["flip_operator_in_one_step", "corrupt_numbers"],
        help="Names from src/rewards/perturbations.py PERTURB_FN_MAP.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "last", "discounted_mean"],
        help="How to aggregate token scores into one scalar.",
    )
    parser.add_argument(
        "--discount-gamma",
        type=float,
        default=0.95,
        help="Gamma for discounted_mean aggregation.",
    )
    parser.add_argument(
        "--local-window",
        type=int,
        default=10,
        help="Token window +/-W around changed positions for localization metrics.",
    )
    parser.add_argument(
        "--hit-ks",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="K values for Hit@K localization metric.",
    )
    parser.add_argument(
        "--max-micro-batch",
        type=int,
        default=8,
        help="Micro-batch for reward scoring.",
    )
    parser.add_argument(
        "--zscore-bins",
        type=int,
        default=20,
        help="Number of relative-position bins for position-aware normalization.",
    )
    parser.add_argument(
        "--zscore-mode",
        type=str,
        default="robust",
        choices=["robust", "standard"],
        help="Normalization stats per bin: robust=median/MAD, standard=mean/std.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Centered moving-average smoothing window on normalized traces.",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=1.0,
        help="Base threshold for sustained-divergence onset on normalized delta.",
    )
    parser.add_argument(
        "--onset-persistence",
        type=int,
        default=3,
        help="Consecutive units above threshold required to mark divergence onset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gsm8k_process_sensitivity",
    )
    return parser.parse_args()


def _extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    return m.group(2).strip()


def _replace_answer(text: str, new_answer: str) -> str:
    m = ANSWER_RE.search(text)
    if not m:
        return text
    return text[: m.start(2)] + new_answer + text[m.end(2) :]


def _extract_think(text: str) -> str | None:
    m = THINK_RE.search(text)
    if not m:
        return None
    return m.group(2)


def _get_prompt_text(prompt_messages: list[dict]) -> str:
    if not prompt_messages:
        return ""
    # In these datasets, the user message is usually last.
    for msg in reversed(prompt_messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return prompt_messages[-1].get("content", "")


def _prompt_key(prompt_messages) -> str | None:
    if prompt_messages is None:
        return None
    try:
        return json.dumps(
            prompt_messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        return None


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _extract_generation_content(generation_field) -> str | None:
    if generation_field is None:
        return None
    if isinstance(generation_field, dict):
        content = generation_field.get("content", None)
        return content if isinstance(content, str) else None
    if isinstance(generation_field, str):
        return generation_field
    return None


def is_gsm8k_correct_trace(text: str, answer: str) -> bool:
    return bool(eval_correctness_gsm8k([{"content": text}], answer)[0])


def _scalar_from_reward_score(score_field) -> float:
    if score_field is None:
        return float("nan")
    if isinstance(score_field, (int, float)):
        return float(score_field)
    if isinstance(score_field, list):
        vals = []
        for x in score_field:
            if isinstance(x, (int, float)):
                vals.append(float(x))
        return float(np.mean(vals)) if vals else float("nan")
    return float("nan")


def load_pregenerated_indices(jsonl_path: str):
    by_prompt_key: dict[str, list[dict]] = {}
    by_user_text: dict[str, list[dict]] = {}

    total_rows = 0
    kept_rows = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            total_rows += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            prompt = obj.get("prompt", None)
            generation_content = _extract_generation_content(obj.get("generation", None))
            if not generation_content:
                continue

            row = {
                "prompt": prompt,
                "content": generation_content,
                "generation_idx": _safe_int(obj.get("generation_idx", None)),
                "reward_model_score_scalar": _scalar_from_reward_score(obj.get("reward_model_score", None)),
                "row_idx": kept_rows,
            }
            kept_rows += 1

            key = _prompt_key(prompt)
            if key is not None:
                by_prompt_key.setdefault(key, []).append(row)

            if isinstance(prompt, list):
                user_text = _get_prompt_text(prompt)
                if user_text:
                    by_user_text.setdefault(user_text, []).append(row)

    def _sort_key(item: dict):
        gidx = item.get("generation_idx", None)
        return (
            gidx is None,
            gidx if gidx is not None else 10**9,
            item.get("row_idx", 10**9),
        )

    for vals in by_prompt_key.values():
        vals.sort(key=_sort_key)
    for vals in by_user_text.values():
        vals.sort(key=_sort_key)

    stats = {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "n_prompt_keys": len(by_prompt_key),
        "n_user_text_keys": len(by_user_text),
    }
    return by_prompt_key, by_user_text, stats


def choose_pregenerated_candidate(
    candidates: Sequence[dict],
    pick: str,
    preferred_generation_idx: int,
    rng: random.Random,
) -> dict | None:
    if not candidates:
        return None

    if pick == "first":
        return candidates[0]

    if pick == "generation_idx":
        matched = [c for c in candidates if c.get("generation_idx", None) == preferred_generation_idx]
        return matched[0] if matched else candidates[0]

    if pick == "max_reward_model_score":
        finite = []
        for c in candidates:
            score = c.get("reward_model_score_scalar", float("nan"))
            if isinstance(score, (int, float)) and not math.isnan(float(score)):
                finite.append(c)
        if finite:
            return max(finite, key=lambda c: float(c.get("reward_model_score_scalar", float("-inf"))))
        return candidates[0]

    if pick == "random":
        return rng.choice(list(candidates))

    return candidates[0]


def _apply_single_perturbation(fn: Callable, text: str, prompt_text: str) -> str:
    """
    Call perturbation functions with different signatures robustly.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    if "text" in params:
        kwargs = {"text": text}
        if "prompt" in params:
            kwargs["prompt"] = prompt_text
        if "promtp" in params:
            kwargs["promtp"] = prompt_text
        if "question" in params:
            kwargs["question"] = prompt_text
        return fn(**kwargs)

    # Fallbacks for positional signatures.
    try:
        return fn(prompt_text, text)
    except Exception:
        return fn(text)


def perturb_reasoning_only(
    clean_text: str,
    prompt_text: str,
    perturb_fn_names: Sequence[str],
    fn_map: Dict[str, Callable],
    num_ops: int,
    rng: random.Random,
    max_retries_per_op: int = 8,
) -> tuple[str, list[str]]:
    """
    Apply perturbations while forcing answer invariance.
    """
    text = clean_text
    original_answer = _extract_answer(clean_text)
    applied = []

    for _ in range(num_ops):
        name = rng.choice(list(perturb_fn_names))
        fn = fn_map[name]

        changed = False
        before_text = text
        before_think = _extract_think(before_text)

        for _try in range(max_retries_per_op):
            candidate = _apply_single_perturbation(fn, before_text, prompt_text)
            if original_answer is not None:
                candidate = _replace_answer(candidate, original_answer)

            after_think = _extract_think(candidate)
            if after_think is not None and before_think is not None and after_think != before_think:
                text = candidate
                changed = True
                break

        if changed:
            applied.append(name)
        else:
            # Keep going, but note no-op.
            applied.append(f"{name}(no_op)")

    # Final hard guarantee answer is unchanged if answer block exists.
    if original_answer is not None:
        text = _replace_answer(text, original_answer)
    return text, applied


def every_n_tokens_mask(full_batch, base_completion_mask, n: int):
    token_indices = base_completion_mask.long().cumsum(dim=1)
    every_n_mask = (token_indices % n == 0) & base_completion_mask
    last_indices = token_indices.argmax(dim=1)
    bs = base_completion_mask.shape[0]
    device = base_completion_mask.device
    every_n_mask[torch.arange(bs, device=device), last_indices] |= base_completion_mask.any(dim=1)
    return every_n_mask


_BOUNDARY_TOKEN_DECODE_CACHE = {}


def sentence_boundary_mask(reward_tokenizer, full_batch, base_completion_mask, device):
    input_ids = full_batch["input_ids"]
    bs, _ = input_ids.shape
    boundary_mask = torch.zeros_like(base_completion_mask, dtype=torch.bool, device=device)

    explicit_boundaries = sorted(
        [
            "</think>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
            "####",
            "\r\n\r\n",
            "\n\n\n",
            "\n\n",
            ".\n",
            "!\n",
            "?\n",
            ";\n",
            ":\n",
            "\n- ",
            "\n* ",
            "\n• ",
            "\n1.",
            "\n2.",
            "\n3.",
            "\n4.",
            "\n5.",
            "\n6.",
            "\n7.",
            "\n8.",
            "\n9.",
            "\n10.",
        ],
        key=len,
        reverse=True,
    )
    max_explicit_len = max(len(x) for x in explicit_boundaries)
    suffix_window = max(96, max_explicit_len + 48)

    def decode_one(tok_id: int) -> str:
        if tok_id not in _BOUNDARY_TOKEN_DECODE_CACHE:
            _BOUNDARY_TOKEN_DECODE_CACHE[tok_id] = reward_tokenizer.decode(
                [tok_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        return _BOUNDARY_TOKEN_DECODE_CACHE[tok_id]

    def ends_boundary(s: str) -> bool:
        return any(s.endswith(x) for x in explicit_boundaries)

    for b in range(bs):
        positions = torch.nonzero(base_completion_mask[b].bool(), as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        pieces = [decode_one(int(input_ids[b, p].item())) for p in positions.tolist()]

        suffix = ""
        for i, pos in enumerate(positions.tolist()):
            piece = pieces[i]
            suffix += piece
            if len(suffix) > suffix_window:
                suffix = suffix[-suffix_window:]
            if ends_boundary(suffix):
                boundary_mask[b, pos] = True
        boundary_mask[b, int(positions[-1].item())] = True

    boundary_mask &= base_completion_mask.bool()
    return boundary_mask


def backfill_rewards(rewards, mask):
    bsz, seq_len = rewards.shape
    indices = torch.arange(seq_len, device=rewards.device).expand(bsz, seq_len)
    masked_indices = torch.where(mask.bool(), indices, torch.tensor(seq_len, device=rewards.device))
    next_valid_index = torch.cummin(masked_indices.flip(1), dim=1)[0].flip(1)
    next_valid_index = next_valid_index.clamp(max=seq_len - 1).long()
    return torch.gather(rewards, 1, next_valid_index)


@torch.no_grad()
def score_with_reward_model(
    reward_model,
    reward_tokenizer,
    prompts_msgs,
    decoded_per_prompt,
    dense_reward=False,
    max_length=512,
    micro_batch=16,
    clip_reward_model=False,
    reward_lb=-5.0,
    reward_ub=5.0,
    dense_partial_fixed_n=10,
):
    FastLanguageModel.for_inference(reward_model)
    device = next(reward_model.parameters()).device

    texts = []
    completion_texts = []
    for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
        for c in completions:
            content = c if isinstance(c, str) else c.get("content", "")
            msgs = p_msgs + [{"role": "assistant", "content": content}]
            texts.append(apply_chat_template({"messages": msgs}, reward_tokenizer)["text"])
            # Keep completion-length accounting aligned with the chat-templated full text.
            # Appending EOS here can shift completion start indices when template output
            # does not end with an explicit EOS token.
            completion_texts.append(content)

    if not texts:
        return [[] for _ in prompts_msgs]

    global_tokens = reward_tokenizer(
        completion_texts,
        return_attention_mask=True,
        add_special_tokens=False,
        padding=False,
    )
    seq_len = min(max(len(t) for t in global_tokens["input_ids"]), max_length)

    all_logits = []
    for i in range(0, len(texts), micro_batch):
        batch_texts = texts[i : i + micro_batch]
        batch_completion_texts = completion_texts[i : i + micro_batch]

        batch_inputs = reward_tokenizer(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding_side="right",
        ).to(device)

        batch_completions = reward_tokenizer(
            text=batch_completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).to(device)

        reward_outputs = reward_model(**batch_inputs)
        reward_logits = reward_outputs.logits.squeeze(-1)
        current_batch_max_len = batch_inputs["input_ids"].shape[1]

        if not dense_reward:
            reward_logits = reward_logits.unsqueeze(1).expand(-1, current_batch_max_len)

        if clip_reward_model:
            reward_logits = torch.clamp(reward_logits, min=reward_lb, max=reward_ub)

        completion_lens = batch_completions["attention_mask"].sum(dim=1).long()
        full_lens = batch_inputs["attention_mask"].sum(dim=1).long()
        start_indices = (full_lens - completion_lens).clamp(min=0)
        gather_indices = start_indices[:, None] + torch.arange(seq_len, device=device)[None, :]
        gather_indices_safe = gather_indices.clamp(max=current_batch_max_len - 1)
        reward_comp = reward_logits.gather(1, gather_indices_safe)

        if dense_reward in ["partial", "partial_fixed"]:
            if dense_reward == "partial":
                step_mask = sentence_boundary_mask(
                    reward_tokenizer,
                    batch_inputs,
                    batch_inputs["attention_mask"],
                    device,
                )
            else:
                step_mask = every_n_tokens_mask(
                    batch_inputs,
                    batch_inputs["attention_mask"],
                    dense_partial_fixed_n,
                )
            step_mask = step_mask.gather(1, gather_indices_safe)
            reward_comp = backfill_rewards(reward_comp, step_mask)

        output_mask = torch.arange(seq_len, device=device)[None, :] < completion_lens[:, None]
        reward_comp[~output_mask] = float("nan")
        all_logits.append(reward_comp.detach().float().cpu())

    bsz = len(prompts_msgs)
    return np.concatenate(all_logits, axis=0).reshape(bsz, -1, seq_len)


def _rankdata(values: np.ndarray) -> np.ndarray:
    # Minimal rank implementation (ties get arbitrary but deterministic order).
    return np.argsort(np.argsort(values)).astype(np.float64)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if x_np.size < 2 or y_np.size < 2:
        return float("nan")
    xr = _rankdata(x_np)
    yr = _rankdata(y_np)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = np.sqrt((xr * xr).sum() * (yr * yr).sum())
    if denom == 0:
        return float("nan")
    return float((xr * yr).sum() / denom)


def aggregate_score(score_seq: Sequence[float], mode: str, gamma: float) -> float:
    arr = np.asarray(score_seq, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if mode == "mean":
        return float(arr.mean())
    if mode == "last":
        return float(arr[-1])
    if mode == "discounted_mean":
        # Highest weight on latest token.
        t = arr.shape[0]
        weights = np.array([gamma ** (t - 1 - i) for i in range(t)], dtype=np.float64)
        denom = float(weights.sum()) if weights.sum() > 0 else 1.0
        return float(np.dot(arr, weights) / denom)
    raise ValueError(f"Unknown aggregate mode: {mode}")


def fit_positionwise_normalizer(
    clean_seqs: Sequence[Sequence[float]],
    n_bins: int,
    mode: str,
    eps: float = 1e-6,
) -> dict:
    n_bins = max(2, int(n_bins))
    per_bin: list[list[float]] = [[] for _ in range(n_bins)]
    global_vals: list[float] = []

    for seq in clean_seqs:
        arr = np.asarray(seq, dtype=np.float64)
        if arr.size == 0:
            continue
        n = arr.shape[0]
        for i, v in enumerate(arr):
            u = (i + 0.5) / float(n)
            b = min(n_bins - 1, int(u * n_bins))
            per_bin[b].append(float(v))
            global_vals.append(float(v))

    if not global_vals:
        # Fallback to identity normalization.
        return {
            "n_bins": n_bins,
            "mode": mode,
            "mu": np.zeros(n_bins, dtype=np.float64),
            "sigma": np.ones(n_bins, dtype=np.float64),
            "global_mu": 0.0,
            "global_sigma": 1.0,
            "eps": float(eps),
        }

    g = np.asarray(global_vals, dtype=np.float64)
    if mode == "robust":
        global_mu = float(np.median(g))
        global_sigma = float(1.4826 * np.median(np.abs(g - global_mu)))
    else:
        global_mu = float(np.mean(g))
        global_sigma = float(np.std(g))
    if not np.isfinite(global_sigma) or global_sigma < eps:
        global_sigma = 1.0

    mu = np.zeros(n_bins, dtype=np.float64)
    sigma = np.ones(n_bins, dtype=np.float64)
    for b in range(n_bins):
        vals = np.asarray(per_bin[b], dtype=np.float64)
        if vals.size == 0:
            mu[b] = global_mu
            sigma[b] = global_sigma
            continue
        if mode == "robust":
            m = float(np.median(vals))
            s = float(1.4826 * np.median(np.abs(vals - m)))
        else:
            m = float(np.mean(vals))
            s = float(np.std(vals))
        if (not np.isfinite(s)) or s < eps:
            s = global_sigma
        mu[b] = m
        sigma[b] = s

    return {
        "n_bins": n_bins,
        "mode": mode,
        "mu": mu,
        "sigma": sigma,
        "global_mu": global_mu,
        "global_sigma": global_sigma,
        "eps": float(eps),
    }


def apply_positionwise_normalizer(
    seq: Sequence[float],
    normalizer: dict,
    smooth_window: int,
) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float64)
    if arr.size == 0:
        return arr

    n_bins = int(normalizer["n_bins"])
    mu = np.asarray(normalizer["mu"], dtype=np.float64)
    sigma = np.asarray(normalizer["sigma"], dtype=np.float64)
    eps = float(normalizer.get("eps", 1e-6))

    n = arr.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        u = (i + 0.5) / float(n)
        b = min(n_bins - 1, int(u * n_bins))
        out[i] = (arr[i] - mu[b]) / max(sigma[b], eps)

    w = int(max(1, smooth_window))
    if w > 1 and out.shape[0] > 1:
        kernel = np.ones(w, dtype=np.float64) / float(w)
        out = np.convolve(out, kernel, mode="same")
    return out


def stabilized_localization_stats(
    clean_seq: Sequence[float],
    pert_seq: Sequence[float],
    changed_positions: Sequence[int],
    window: int,
    onset_threshold: float,
    onset_persistence: int,
    bucket_size: int | None = None,
) -> tuple[float, float, float, float, int | None, float, int]:
    def _sustained_first(
        arr: np.ndarray,
        thr: float,
        length: int,
        start: int,
        end_exclusive: int | None = None,
    ) -> int | None:
        n_idx = arr.shape[0]
        if n_idx == 0:
            return None
        L = max(1, int(length))
        lo = max(0, int(start))
        hi = n_idx if end_exclusive is None else max(0, min(int(end_exclusive), n_idx))
        if hi - lo < L:
            return None
        for i in range(lo, hi - L + 1):
            if np.all(arr[i : i + L] > thr):
                return i
        return None

    c = np.asarray(clean_seq, dtype=np.float64)
    p = np.asarray(pert_seq, dtype=np.float64)
    n = min(c.shape[0], p.shape[0])
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), None, float("nan"), 0

    delta = c[:n] - p[:n]
    changed = sorted({int(idx) for idx in changed_positions if 0 <= int(idx) < n})
    if not changed:
        return float("nan"), float("nan"), float("nan"), float("nan"), None, float("nan"), 0

    if bucket_size is not None and bucket_size > 1:
        n_buckets = int(math.ceil(n / float(bucket_size)))
        unit_delta = np.zeros(n_buckets, dtype=np.float64)
        for b in range(n_buckets):
            s = b * bucket_size
            e = min(n, (b + 1) * bucket_size)
            unit_delta[b] = float(delta[s:e].mean())
        changed_units = sorted({idx // bucket_size for idx in changed})
        unit_window = int(math.ceil(max(0, window) / float(bucket_size)))
    else:
        unit_delta = delta
        changed_units = changed
        unit_window = int(max(0, window))

    n_units = unit_delta.shape[0]
    if n_units == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), None, float("nan"), 0

    positions = np.arange(n_units)
    local_mask = np.zeros(n_units, dtype=bool)
    for idx in changed_units:
        local_mask |= np.abs(positions - idx) <= unit_window

    pos_mass = np.maximum(unit_delta, 0.0)
    total_pos_mass = float(pos_mass.sum())
    local_pos_mass = float(pos_mass[local_mask].sum()) if local_mask.any() else 0.0

    ldm = float(local_pos_mass / total_pos_mass) if total_pos_mass > 0 else float("nan")
    chance = float(local_mask.mean()) if n_units > 0 else float("nan")
    denom = 1.0 - chance
    ldm_norm = (
        float((ldm - chance) / denom)
        if (np.isfinite(ldm) and np.isfinite(chance) and denom > 1e-12)
        else float("nan")
    )

    perturb_start = int(min(changed_units))
    pre = unit_delta[:perturb_start]
    if pre.size >= 2:
        pre_mu = float(np.mean(pre))
        pre_std = float(np.std(pre))
        dynamic_thr = pre_mu + pre_std
    else:
        dynamic_thr = float(onset_threshold)
    thr = float(max(float(onset_threshold), dynamic_thr))

    L = max(1, int(onset_persistence))
    onset_idx = _sustained_first(unit_delta, thr, L, start=perturb_start)
    onset_lag = float(onset_idx - perturb_start) if onset_idx is not None else float("nan")
    false_alarm = int(_sustained_first(unit_delta, thr, L, start=0, end_exclusive=perturb_start) is not None)

    return ldm, chance, ldm_norm, onset_lag, onset_idx, thr, false_alarm


def changed_token_positions(
    reward_tokenizer,
    prompt_messages: list[dict] | None,
    clean_text: str,
    pert_text: str,
    max_length: int,
) -> list[int]:
    # Compute changed positions in the same token space used by reward scoring.
    # This avoids alignment artifacts from completion-only tokenization.
    if prompt_messages is not None:
        clean_full = apply_chat_template(
            {"messages": prompt_messages + [{"role": "assistant", "content": clean_text}]},
            reward_tokenizer,
        )["text"]
        pert_full = apply_chat_template(
            {"messages": prompt_messages + [{"role": "assistant", "content": pert_text}]},
            reward_tokenizer,
        )["text"]

        c_full_ids = reward_tokenizer(
            clean_full,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        p_full_ids = reward_tokenizer(
            pert_full,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        c_comp_ids = reward_tokenizer(
            clean_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        p_comp_ids = reward_tokenizer(
            pert_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        c_start = max(0, len(c_full_ids) - len(c_comp_ids))
        p_start = max(0, len(p_full_ids) - len(p_comp_ids))
        c_ids = c_full_ids[c_start:]
        p_ids = p_full_ids[p_start:]
    else:
        c_ids = reward_tokenizer(
            clean_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        p_ids = reward_tokenizer(
            pert_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

    sm = SequenceMatcher(a=c_ids, b=p_ids, autojunk=False)
    changed = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed.extend(range(j1, j2))
    # Dedup while preserving order.
    out = []
    seen = set()
    for pos in changed:
        if pos not in seen:
            out.append(pos)
            seen.add(pos)

    # Ensure earliest mismatch is captured (SequenceMatcher may align past it).
    m = min(len(c_ids), len(p_ids))
    first_mismatch = None
    for i in range(m):
        if c_ids[i] != p_ids[i]:
            first_mismatch = i
            break
    if first_mismatch is None and len(c_ids) != len(p_ids):
        first_mismatch = m
    if first_mismatch is not None and first_mismatch not in seen:
        out.insert(0, first_mismatch)
    return out


def localization_stats(
    clean_seq: Sequence[float],
    pert_seq: Sequence[float],
    changed_positions: Sequence[int],
    window: int,
    hit_ks: Sequence[int],
    bucket_size: int | None = None,
) -> tuple[float, float, float, dict[str, int], dict[str, float], dict[str, float], int, int]:
    def _compute_from_delta(
        delta_arr: np.ndarray,
        changed_idx: list[int],
        window_idx: int,
        ks: Sequence[int],
    ) -> tuple[float, float, float, dict[str, int], dict[str, float], dict[str, float], int, int]:
        n_idx = delta_arr.shape[0]
        if n_idx == 0:
            return (
                float("nan"),
                float("nan"),
                float("nan"),
                {str(k): 0 for k in ks},
                {str(k): float("nan") for k in ks},
                {str(k): float("nan") for k in ks},
                0,
                0,
            )

        changed_idx = [idx for idx in changed_idx if 0 <= idx < n_idx]
        if not changed_idx:
            return (
                float("nan"),
                float("nan"),
                float("nan"),
                {str(k): 0 for k in ks},
                {str(k): float("nan") for k in ks},
                {str(k): float("nan") for k in ks},
                0,
                int(n_idx),
            )

        positions = np.arange(n_idx)
        local_mask = np.zeros(n_idx, dtype=bool)
        for idx in changed_idx:
            local_mask |= np.abs(positions - idx) <= window_idx

        local_drop = float(delta_arr[local_mask].mean()) if local_mask.any() else float("nan")
        far_drop = float(delta_arr[~local_mask].mean()) if (~local_mask).any() else float("nan")
        gap = (
            float(local_drop - far_drop)
            if (not np.isnan(local_drop) and not np.isnan(far_drop))
            else float("nan")
        )

        hits: dict[str, int] = {}
        hit_random: dict[str, float] = {}
        hit_norm: dict[str, float] = {}
        sorted_idx = np.argsort(delta_arr)[::-1]
        changed_np = np.asarray(changed_idx)
        local_count = int(local_mask.sum())
        for k in ks:
            top = sorted_idx[: min(k, n_idx)]
            hit = 0
            for idx in top:
                if np.any(np.abs(changed_np - idx) <= window_idx):
                    hit = 1
                    break
            hits[str(k)] = hit

            k_eff = min(max(int(k), 0), int(n_idx))
            if k_eff == 0 or local_count == 0:
                chance = 0.0
            elif local_count >= int(n_idx):
                chance = 1.0
            else:
                chance = 1.0 - (
                    math.comb(int(n_idx) - local_count, k_eff)
                    / math.comb(int(n_idx), k_eff)
                )

            hit_random[str(k)] = float(chance)
            denom = 1.0 - float(chance)
            if denom <= 1e-12:
                hit_norm[str(k)] = float("nan")
            else:
                hit_norm[str(k)] = float((float(hit) - float(chance)) / denom)

        return local_drop, far_drop, gap, hits, hit_random, hit_norm, local_count, int(n_idx)

    c = np.asarray(clean_seq, dtype=np.float64)
    p = np.asarray(pert_seq, dtype=np.float64)
    n = min(c.shape[0], p.shape[0])
    if n == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            {str(k): 0 for k in hit_ks},
            {str(k): float("nan") for k in hit_ks},
            {str(k): float("nan") for k in hit_ks},
            0,
            0,
        )

    delta = c[:n] - p[:n]
    changed = [idx for idx in changed_positions if 0 <= idx < n]
    if bucket_size is not None and bucket_size > 1:
        # partial_fixed-style localization: evaluate on the scoring grid, not raw tokens.
        n_buckets = int(math.ceil(n / bucket_size))
        delta_bucket = np.zeros(n_buckets, dtype=np.float64)
        for b in range(n_buckets):
            s = b * bucket_size
            e = min(n, (b + 1) * bucket_size)
            delta_bucket[b] = float(delta[s:e].mean())

        changed_bucket = sorted({idx // bucket_size for idx in changed})
        window_bucket = int(math.ceil(max(0, window) / float(bucket_size)))
        return _compute_from_delta(delta_bucket, changed_bucket, window_bucket, hit_ks)

    # Default token-space localization.
    return _compute_from_delta(delta, changed, int(max(0, window)), hit_ks)


def load_reward_model_and_tokenizer(cfg):
    """
    Load only reward model + tokenizer from an AIRL checkpoint.
    """
    reward_model_name = cfg.model.reward_name
    max_seq_length = cfg.model.max_prompt_length + cfg.model.max_completion_length
    load_in_4bit = cfg.model.load_in_4bit
    reward_lora_rank = cfg.model.reward_lora_rank
    reward_gpu_memory_utilization = cfg.model.reward_gpu_memory_utilization

    if not cfg.model.dense_rewards:
        reward_model, reward_tokenizer = FastLanguageModel.from_pretrained(
            model_name=reward_model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=False,
            max_lora_rank=reward_lora_rank,
            gpu_memory_utilization=reward_gpu_memory_utilization,
            num_labels=1,
        )
    else:
        reward_model, reward_tokenizer = FastLanguageModel.from_pretrained(
            model_name=reward_model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=False,
            max_lora_rank=reward_lora_rank,
            gpu_memory_utilization=reward_gpu_memory_utilization,
        )
        try:
            hidden_size = reward_model.config.hidden_size
        except Exception:
            hidden_size = reward_model.config.text_config.hidden_size
        reward_model.lm_head = torch.nn.Linear(
            in_features=hidden_size,
            out_features=1,
            bias=False,
            device="cuda",
        )
        reward_model.config.num_labels = 1

    adapter_dir = os.path.join(cfg.model.name, "reward_model")
    reward_model = PeftModel.from_pretrained(
        reward_model,
        adapter_dir,
        is_trainable=False,
    )
    if hasattr(reward_model, "gradient_checkpointing_disable"):
        reward_model.gradient_checkpointing_disable()
    if hasattr(reward_model, "config"):
        reward_model.config.use_cache = False
    reward_model.eval()
    return reward_model, reward_tokenizer


def summarise(
    records: list[PairRecord],
    max_severity: int,
    hit_ks: list[int],
    localization_mode: str,
    local_window_tokens: int,
    partial_fixed_stride: int | None,
    zscore_bins: int,
    zscore_mode: str,
    smooth_window: int,
    onset_threshold: float,
    onset_persistence: int,
) -> dict:
    margins = [r.margin for r in records if r.margin is not None and not math.isnan(r.margin)]
    wins = [r.win for r in records if r.win is not None]
    ties = [1 for r in records if r.margin == 0]

    overall = {
        "n_pairs": len(records),
        "win_rate": float(np.mean(wins)) if wins else float("nan"),
        "mean_margin": float(np.mean(margins)) if margins else float("nan"),
        "median_margin": float(np.median(margins)) if margins else float("nan"),
        "tie_rate": float(len(ties) / len(records)) if records else float("nan"),
        "same_answer_rate": float(
            np.mean([1 if r.answer_unchanged else 0 for r in records]) if records else float("nan")
        ),
        "clean_correct_rate": float(
            np.mean([1 if r.clean_correct else 0 for r in records]) if records else float("nan")
        ),
        "pert_correct_rate": float(
            np.mean([1 if r.pert_correct else 0 for r in records]) if records else float("nan")
        ),
    }

    by_severity = {}
    for s in range(1, max_severity + 1):
        subset = [r for r in records if r.severity == s]
        if not subset:
            continue
        sub_margins = [r.margin for r in subset if r.margin is not None and not math.isnan(r.margin)]
        sub_wins = [r.win for r in subset if r.win is not None]
        by_severity[str(s)] = {
            "n_pairs": len(subset),
            "win_rate": float(np.mean(sub_wins)) if sub_wins else float("nan"),
            "mean_margin": float(np.mean(sub_margins)) if sub_margins else float("nan"),
            "median_margin": float(np.median(sub_margins)) if sub_margins else float("nan"),
        }

    # Monotonicity by prompt: mean perturbed score should decrease with severity.
    per_prompt: dict[int, dict[int, list[float]]] = {}
    for r in records:
        per_prompt.setdefault(r.prompt_idx, {}).setdefault(r.severity, []).append(float(r.pert_score_agg))

    rhos = []
    nonincreasing_flags = []
    for _pidx, sev_to_scores in per_prompt.items():
        ordered = []
        sev_list = []
        for s in sorted(sev_to_scores.keys()):
            vals = [v for v in sev_to_scores[s] if not math.isnan(v)]
            if vals:
                sev_list.append(float(s))
                ordered.append(float(np.mean(vals)))
        if len(ordered) >= 2:
            rhos.append(spearman_rho(sev_list, ordered))
            nonincreasing_flags.append(int(all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1))))

    # Localization aggregates.
    loc_subset = [r for r in records if r.localization_gap is not None and not math.isnan(r.localization_gap)]
    if loc_subset:
        hit_summary = {}
        hit_random_summary = {}
        hit_norm_summary = {}
        for k in hit_ks:
            vals = [r.hit_at.get(str(k), 0) for r in loc_subset if r.hit_at is not None]
            hit_summary[str(k)] = float(np.mean(vals)) if vals else float("nan")
            rand_vals = [
                float(r.hit_at_random.get(str(k), float("nan")))
                for r in loc_subset
                if r.hit_at_random is not None and not math.isnan(float(r.hit_at_random.get(str(k), float("nan"))))
            ]
            hit_random_summary[str(k)] = float(np.mean(rand_vals)) if rand_vals else float("nan")
            norm_vals = [
                float(r.hit_at_norm.get(str(k), float("nan")))
                for r in loc_subset
                if r.hit_at_norm is not None and not math.isnan(float(r.hit_at_norm.get(str(k), float("nan"))))
            ]
            hit_norm_summary[str(k)] = float(np.mean(norm_vals)) if norm_vals else float("nan")

        local_ratio_vals = [
            float(r.local_unit_ratio)
            for r in loc_subset
            if r.local_unit_ratio is not None and not math.isnan(float(r.local_unit_ratio))
        ]
        localization = {
            "n_pairs": len(loc_subset),
            "mode": localization_mode,
            "local_window_tokens": int(local_window_tokens),
            "partial_fixed_stride": partial_fixed_stride,
            "local_window_units": (
                int(math.ceil(local_window_tokens / float(partial_fixed_stride)))
                if (localization_mode == "bucket" and partial_fixed_stride is not None and partial_fixed_stride > 0)
                else int(local_window_tokens)
            ),
            "unit_name": "bucket" if localization_mode == "bucket" else "token",
            "mean_local_drop": float(np.mean([r.local_drop for r in loc_subset])),
            "mean_far_drop": float(np.mean([r.far_drop for r in loc_subset])),
            "mean_gap": float(np.mean([r.localization_gap for r in loc_subset])),
            "hit_at_k": hit_summary,
            "hit_at_k_random": hit_random_summary,
            "hit_at_k_norm": hit_norm_summary,
            "mean_local_unit_ratio": float(np.mean(local_ratio_vals)) if local_ratio_vals else float("nan"),
        }
    else:
        localization = {
            "n_pairs": 0,
            "mode": localization_mode,
            "local_window_tokens": int(local_window_tokens),
            "partial_fixed_stride": partial_fixed_stride,
            "local_window_units": (
                int(math.ceil(local_window_tokens / float(partial_fixed_stride)))
                if (localization_mode == "bucket" and partial_fixed_stride is not None and partial_fixed_stride > 0)
                else int(local_window_tokens)
            ),
            "unit_name": "bucket" if localization_mode == "bucket" else "token",
            "mean_local_drop": float("nan"),
            "mean_far_drop": float("nan"),
            "mean_gap": float("nan"),
            "hit_at_k": {str(k): float("nan") for k in hit_ks},
            "hit_at_k_random": {str(k): float("nan") for k in hit_ks},
            "hit_at_k_norm": {str(k): float("nan") for k in hit_ks},
            "mean_local_unit_ratio": float("nan"),
        }

    stable_subset = [r for r in records if r.ldm_norm is not None and not math.isnan(r.ldm_norm)]
    if stable_subset:
        ldm_vals = [float(r.ldm) for r in stable_subset if r.ldm is not None and not math.isnan(float(r.ldm))]
        ldm_rand_vals = [
            float(r.ldm_random)
            for r in stable_subset
            if r.ldm_random is not None and not math.isnan(float(r.ldm_random))
        ]
        ldm_norm_vals = [
            float(r.ldm_norm)
            for r in stable_subset
            if r.ldm_norm is not None and not math.isnan(float(r.ldm_norm))
        ]
        onset_vals = [
            float(r.onset_lag)
            for r in stable_subset
            if r.onset_lag is not None and not math.isnan(float(r.onset_lag))
        ]
        false_alarm_vals = [
            int(r.false_alarm)
            for r in stable_subset
            if r.false_alarm is not None
        ]
        stabilized = {
            "n_pairs": len(stable_subset),
            "zscore_bins": int(zscore_bins),
            "zscore_mode": zscore_mode,
            "smooth_window": int(smooth_window),
            "onset_threshold": float(onset_threshold),
            "onset_persistence": int(onset_persistence),
            "mean_ldm": float(np.mean(ldm_vals)) if ldm_vals else float("nan"),
            "mean_ldm_random": float(np.mean(ldm_rand_vals)) if ldm_rand_vals else float("nan"),
            "mean_ldm_norm": float(np.mean(ldm_norm_vals)) if ldm_norm_vals else float("nan"),
            "mean_onset_lag": float(np.mean(onset_vals)) if onset_vals else float("nan"),
            "median_onset_lag": float(np.median(onset_vals)) if onset_vals else float("nan"),
            "onset_detect_rate": float(len(onset_vals) / len(stable_subset)) if stable_subset else float("nan"),
            "false_alarm_rate": float(np.mean(false_alarm_vals)) if false_alarm_vals else float("nan"),
        }
    else:
        stabilized = {
            "n_pairs": 0,
            "zscore_bins": int(zscore_bins),
            "zscore_mode": zscore_mode,
            "smooth_window": int(smooth_window),
            "onset_threshold": float(onset_threshold),
            "onset_persistence": int(onset_persistence),
            "mean_ldm": float("nan"),
            "mean_ldm_random": float("nan"),
            "mean_ldm_norm": float("nan"),
            "mean_onset_lag": float("nan"),
            "median_onset_lag": float("nan"),
            "onset_detect_rate": float("nan"),
            "false_alarm_rate": float("nan"),
        }

    return {
        "overall": overall,
        "by_severity": by_severity,
        "monotonicity": {
            "n_prompts": len(per_prompt),
            "mean_spearman_rho": float(np.nanmean(rhos)) if rhos else float("nan"),
            "nonincreasing_rate": float(np.mean(nonincreasing_flags)) if nonincreasing_flags else float("nan"),
        },
        "localization": localization,
        "stabilized_localization": stabilized,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    cfg = OmegaConf.load(args.config)
    cfg.airl = True
    if args.checkpoint_dir is not None:
        cfg.model.name = args.checkpoint_dir
    if args.reward_name is not None:
        cfg.model.reward_name = args.reward_name
    if args.reward_lora_rank is not None:
        cfg.model.reward_lora_rank = args.reward_lora_rank
    if args.reward_gpu_memory_utilization is not None:
        cfg.model.reward_gpu_memory_utilization = args.reward_gpu_memory_utilization
    if args.load_in_4bit is not None:
        cfg.model.load_in_4bit = args.load_in_4bit
    if args.dense_reward_mode is not None:
        if args.dense_reward_mode == "sparse":
            cfg.model.dense_rewards = False
        elif args.dense_reward_mode == "full":
            cfg.model.dense_rewards = True
        else:
            cfg.model.dense_rewards = args.dense_reward_mode
    if args.dense_partial_fixed_n is not None:
        cfg.model.dense_partial_fixed_n = int(args.dense_partial_fixed_n)

    if not cfg.model.get("reward_lora_rank"):
        cfg.model.reward_lora_rank = 32
    if not cfg.model.get("reward_gpu_memory_utilization"):
        cfg.model.reward_gpu_memory_utilization = 0.2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fn_map = {}
    for name in args.perturb_fns:
        if name not in PERTURB_FN_MAP:
            raise ValueError(f"Unknown perturbation function: {name}. Available: {sorted(PERTURB_FN_MAP.keys())}")
        fn_map[name] = PERTURB_FN_MAP[name]

    dataset = get_dataset(
        "gsm8k_kd",
        split=args.split,
        ratio=1.0,
        no_system=args.no_system,
    )
    total = len(dataset)
    start = max(0, int(args.start_index))
    end = min(total, start + int(args.max_examples))
    if start >= end:
        raise ValueError(f"Empty slice: start={start}, end={end}, total={total}")
    subset = dataset.select(range(start, end))
    print(f"Loaded gsm8k_kd split={args.split}: total={total}, selected={len(subset)} ({start}:{end})")

    pregenerated_jsonl_path = None
    pregenerated_by_prompt_key = None
    pregenerated_by_user_text = None
    pregenerated_stats = None
    if args.trace_source == "pregenerated":
        pregenerated_jsonl_path = resolve_pregenerated_jsonl_path(
            mode=args.pregenerated_mode,
            model_name=cfg.model.name,
            policy_name=getattr(cfg.model, "policy_name", None),
            explicit_path=args.pregenerated_jsonl_path,
            source_dir_override=args.pregenerated_source_dir,
            candidate_filenames=args.pregenerated_candidates,
        )
        print(f"Loading pregenerated traces from: {pregenerated_jsonl_path}")
        (
            pregenerated_by_prompt_key,
            pregenerated_by_user_text,
            pregenerated_stats,
        ) = load_pregenerated_indices(pregenerated_jsonl_path)
        print(
            "Loaded pregenerated index: "
            f"rows={pregenerated_stats['kept_rows']}/{pregenerated_stats['total_rows']} | "
            f"prompt_keys={pregenerated_stats['n_prompt_keys']} | "
            f"user_keys={pregenerated_stats['n_user_text_keys']}"
        )

    reward_model, reward_tokenizer = load_reward_model_and_tokenizer(cfg)
    max_length = int(cfg.model.max_prompt_length + cfg.model.max_completion_length)

    # Build per-prompt completion groups: [clean, pert_s1_v1, pert_s2_v1, ...].
    prompts_for_scoring = []
    completions_for_scoring = []
    pair_records: list[PairRecord] = []
    pair_index_map: list[tuple[int, int]] = []  # (record_idx, completion_group_local_idx)
    skipped_missing_pregenerated = 0
    skipped_invalid_clean_trace = 0
    skipped_no_correct_clean = 0
    used_incorrect_clean_fallback = 0
    used_user_text_fallback = 0

    for row in subset:
        prompt = row["prompt"]
        answer = str(row["answer"])
        prompt_text = _get_prompt_text(prompt)

        clean_text = None
        clean_trace_source = args.trace_source
        clean_generation_idx = None
        clean_input_reward_score = None

        if args.trace_source == "expert":
            clean_text = row["target"]
            clean_trace_source = "expert"
            if args.clean_correct_policy == "require" and not is_gsm8k_correct_trace(clean_text, answer):
                skipped_no_correct_clean += 1
                continue
            if args.clean_correct_policy == "prefer" and not is_gsm8k_correct_trace(clean_text, answer):
                used_incorrect_clean_fallback += 1
        else:
            key = _prompt_key(prompt)
            candidates = pregenerated_by_prompt_key.get(key, []) if key is not None else []
            if not candidates:
                fallback = pregenerated_by_user_text.get(prompt_text, [])
                if fallback:
                    used_user_text_fallback += 1
                candidates = fallback

            selected_candidates = list(candidates)
            if args.clean_correct_policy in {"require", "prefer"}:
                correct_candidates = [
                    c for c in candidates if is_gsm8k_correct_trace(c["content"], answer)
                ]
                if correct_candidates:
                    selected_candidates = correct_candidates
                elif args.clean_correct_policy == "require":
                    skipped_no_correct_clean += 1
                    continue
                else:
                    used_incorrect_clean_fallback += 1

            chosen = choose_pregenerated_candidate(
                candidates=selected_candidates,
                pick=args.pregenerated_pick,
                preferred_generation_idx=args.pregenerated_generation_idx,
                rng=rng,
            )
            if chosen is None:
                skipped_missing_pregenerated += 1
                continue
            clean_text = chosen["content"]
            clean_trace_source = "pregenerated"
            clean_generation_idx = chosen.get("generation_idx", None)
            clean_input_reward_score = chosen.get("reward_model_score_scalar", None)

        # Require valid structured traces for process perturbations.
        if _extract_think(clean_text) is None or _extract_answer(clean_text) is None:
            skipped_invalid_clean_trace += 1
            continue

        score_prompt_idx = len(prompts_for_scoring)
        group = [{"content": clean_text}]
        prompts_for_scoring.append(prompt)
        base_local_idx = 1

        for severity in range(1, args.max_severity + 1):
            for v_idx in range(args.variants_per_severity):
                pert_text, applied = perturb_reasoning_only(
                    clean_text=clean_text,
                    prompt_text=prompt_text,
                    perturb_fn_names=args.perturb_fns,
                    fn_map=fn_map,
                    num_ops=severity,
                    rng=rng,
                )
                changed_pos = changed_token_positions(
                    reward_tokenizer=reward_tokenizer,
                    prompt_messages=prompt,
                    clean_text=clean_text,
                    pert_text=pert_text,
                    max_length=max_length,
                )
                rec = PairRecord(
                    prompt_idx=score_prompt_idx,
                    severity=severity,
                    variant_idx=v_idx,
                    prompt=prompt,
                    answer=answer,
                    clean_trace_source=clean_trace_source,
                    clean_generation_idx=clean_generation_idx,
                    clean_input_reward_score=clean_input_reward_score,
                    clean_text=clean_text,
                    pert_text=pert_text,
                    perturb_fns=applied,
                    changed_token_positions=changed_pos,
                )
                pair_records.append(rec)
                pair_index_map.append((len(pair_records) - 1, base_local_idx))
                group.append({"content": pert_text})
                base_local_idx += 1

        completions_for_scoring.append(group)

    if not prompts_for_scoring:
        raise ValueError(
            "No valid traces available after filtering. "
            "Check pregenerated alignment or trace formatting (<think>/<answer> tags)."
        )

    print(
        "Prepared prompts for scoring: "
        f"{len(prompts_for_scoring)} "
        f"(skipped_missing_pregenerated={skipped_missing_pregenerated}, "
        f"skipped_no_correct_clean={skipped_no_correct_clean}, "
        f"used_incorrect_clean_fallback={used_incorrect_clean_fallback}, "
        f"skipped_invalid_clean_trace={skipped_invalid_clean_trace}, "
        f"user_text_fallback_matches={used_user_text_fallback})"
    )

    dense_reward = cfg.model.dense_rewards
    partial_fixed_stride = int(getattr(cfg.model, "dense_partial_fixed_n", 10))
    localization_mode = (
        "bucket"
        if isinstance(dense_reward, str) and dense_reward == "partial_fixed"
        else "token"
    )
    scores = score_with_reward_model(
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        prompts_msgs=prompts_for_scoring,
        decoded_per_prompt=completions_for_scoring,
        dense_reward=dense_reward,
        max_length=max_length,
        micro_batch=args.max_micro_batch,
        clip_reward_model=bool(getattr(cfg.model, "clip_reward_model", False)),
        reward_lb=float(getattr(cfg.model, "reward_lb", -5.0)),
        reward_ub=float(getattr(cfg.model, "reward_ub", 5.0)),
        dense_partial_fixed_n=int(getattr(cfg.model, "dense_partial_fixed_n", 10)),
    )

    # Position-aware normalization reference built from clean traces.
    clean_reference_seqs = []
    for pidx in range(len(prompts_for_scoring)):
        clean_ref = np.asarray(scores[pidx][0], dtype=np.float64)
        clean_ref = clean_ref[~np.isnan(clean_ref)]
        clean_reference_seqs.append(clean_ref.tolist())
    normalizer = fit_positionwise_normalizer(
        clean_seqs=clean_reference_seqs,
        n_bins=args.zscore_bins,
        mode=args.zscore_mode,
    )

    # Fill per-pair stats from scores.
    for rec_idx, local_idx in pair_index_map:
        rec = pair_records[rec_idx]
        prompt_scores = scores[rec.prompt_idx]
        clean_seq = np.asarray(prompt_scores[0], dtype=np.float64)
        pert_seq = np.asarray(prompt_scores[local_idx], dtype=np.float64)
        clean_seq = clean_seq[~np.isnan(clean_seq)]
        pert_seq = pert_seq[~np.isnan(pert_seq)]

        clean_agg = aggregate_score(clean_seq, args.aggregate, args.discount_gamma)
        pert_agg = aggregate_score(pert_seq, args.aggregate, args.discount_gamma)
        margin = float(clean_agg - pert_agg)

        rec.clean_score_seq = clean_seq.tolist()
        rec.pert_score_seq = pert_seq.tolist()
        rec.clean_score_agg = clean_agg
        rec.pert_score_agg = pert_agg
        rec.margin = margin
        rec.win = int(margin > 0.0)

        # Answer invariance + correctness checks.
        rec.answer_unchanged = (_extract_answer(rec.clean_text) == _extract_answer(rec.pert_text))
        rec.clean_correct = bool(eval_correctness_gsm8k([{"content": rec.clean_text}], rec.answer)[0])
        rec.pert_correct = bool(eval_correctness_gsm8k([{"content": rec.pert_text}], rec.answer)[0])

        local_drop, far_drop, gap, hits, hits_random, hits_norm, local_units, total_units = localization_stats(
            clean_seq=rec.clean_score_seq,
            pert_seq=rec.pert_score_seq,
            changed_positions=rec.changed_token_positions,
            window=args.local_window,
            hit_ks=args.hit_ks,
            bucket_size=partial_fixed_stride if localization_mode == "bucket" else None,
        )
        rec.local_drop = local_drop
        rec.far_drop = far_drop
        rec.localization_gap = gap
        rec.hit_at = hits
        rec.hit_at_random = hits_random
        rec.hit_at_norm = hits_norm
        rec.local_unit_count = int(local_units)
        rec.total_unit_count = int(total_units)
        rec.local_unit_ratio = (
            float(local_units) / float(total_units) if total_units and total_units > 0 else float("nan")
        )
        rec.localization_mode = localization_mode

        clean_seq_z = apply_positionwise_normalizer(
            seq=rec.clean_score_seq,
            normalizer=normalizer,
            smooth_window=args.smooth_window,
        )
        pert_seq_z = apply_positionwise_normalizer(
            seq=rec.pert_score_seq,
            normalizer=normalizer,
            smooth_window=args.smooth_window,
        )
        (
            ldm,
            ldm_random,
            ldm_norm,
            onset_lag,
            onset_idx,
            onset_thr_used,
            false_alarm,
        ) = stabilized_localization_stats(
            clean_seq=clean_seq_z,
            pert_seq=pert_seq_z,
            changed_positions=rec.changed_token_positions,
            window=args.local_window,
            onset_threshold=args.onset_threshold,
            onset_persistence=args.onset_persistence,
            bucket_size=partial_fixed_stride if localization_mode == "bucket" else None,
        )
        rec.ldm = ldm
        rec.ldm_random = ldm_random
        rec.ldm_norm = ldm_norm
        rec.onset_lag = onset_lag
        rec.onset_idx = onset_idx
        rec.onset_threshold_used = onset_thr_used
        rec.false_alarm = false_alarm

    summary = summarise(
        pair_records,
        args.max_severity,
        args.hit_ks,
        localization_mode=localization_mode,
        local_window_tokens=args.local_window,
        partial_fixed_stride=partial_fixed_stride if localization_mode == "bucket" else None,
        zscore_bins=args.zscore_bins,
        zscore_mode=args.zscore_mode,
        smooth_window=args.smooth_window,
        onset_threshold=args.onset_threshold,
        onset_persistence=args.onset_persistence,
    )

    # Persist artifacts.
    summary_path = output_dir / "summary.json"
    detail_path = output_dir / "pair_details.jsonl"
    run_config_path = output_dir / "run_config.json"

    with open(run_config_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "model_name": cfg.model.name,
                "reward_name": cfg.model.reward_name,
                "dense_rewards": cfg.model.dense_rewards,
                "localization_mode": localization_mode,
                "partial_fixed_stride": partial_fixed_stride if localization_mode == "bucket" else None,
                "max_length": max_length,
                "dataset_slice": {"split": args.split, "start": start, "end": end, "n": len(subset)},
                "perturb_fns": args.perturb_fns,
                "trace_source": args.trace_source,
                "pregenerated_mode": args.pregenerated_mode if args.trace_source == "pregenerated" else None,
                "pregenerated_jsonl_path": pregenerated_jsonl_path,
                "pregenerated_pick": args.pregenerated_pick if args.trace_source == "pregenerated" else None,
                "pregenerated_generation_idx": (
                    args.pregenerated_generation_idx if args.trace_source == "pregenerated" else None
                ),
                "clean_correct_policy": args.clean_correct_policy,
                "pregenerated_index_stats": pregenerated_stats,
                "build_stats": {
                    "prepared_prompts": len(prompts_for_scoring),
                    "skipped_missing_pregenerated": skipped_missing_pregenerated,
                    "skipped_no_correct_clean": skipped_no_correct_clean,
                    "used_incorrect_clean_fallback": used_incorrect_clean_fallback,
                    "skipped_invalid_clean_trace": skipped_invalid_clean_trace,
                    "used_user_text_fallback": used_user_text_fallback,
                },
            },
            f,
            indent=2,
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(detail_path, "w") as f:
        for rec in pair_records:
            f.write(json.dumps(rec.__dict__) + "\n")

    # Console summary.
    print("\n=== Process Sensitivity Summary ===")
    print(f"Pairs: {summary['overall']['n_pairs']}")
    print(f"Win-rate: {summary['overall']['win_rate']:.4f}")
    print(f"Mean margin: {summary['overall']['mean_margin']:.4f}")
    print(f"Median margin: {summary['overall']['median_margin']:.4f}")
    print(f"Same-answer rate: {summary['overall']['same_answer_rate']:.4f}")
    print(f"Monotonic mean Spearman rho: {summary['monotonicity']['mean_spearman_rho']:.4f}")
    print(f"Monotonic non-increasing rate: {summary['monotonicity']['nonincreasing_rate']:.4f}")
    print(
        "Localization mode: "
        f"{summary['localization']['mode']} "
        f"(window={summary['localization']['local_window_units']} {summary['localization']['unit_name']}s)"
    )
    print(f"Localization mean gap: {summary['localization']['mean_gap']:.4f}")
    for k in args.hit_ks:
        print(
            f"Hit@{k}: {summary['localization']['hit_at_k'][str(k)]:.4f} "
            f"| rand={summary['localization']['hit_at_k_random'][str(k)]:.4f} "
            f"| norm={summary['localization']['hit_at_k_norm'][str(k)]:.4f}"
        )
    print("Stabilized localization:")
    print(
        f"  LDM: {summary['stabilized_localization']['mean_ldm']:.4f} "
        f"| rand={summary['stabilized_localization']['mean_ldm_random']:.4f} "
        f"| norm={summary['stabilized_localization']['mean_ldm_norm']:.4f}"
    )
    print(
        f"  Onset lag: mean={summary['stabilized_localization']['mean_onset_lag']:.4f}, "
        f"median={summary['stabilized_localization']['median_onset_lag']:.4f}, "
        f"detect_rate={summary['stabilized_localization']['onset_detect_rate']:.4f}, "
        f"false_alarm_rate={summary['stabilized_localization']['false_alarm_rate']:.4f}"
    )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved details: {detail_path}")
    print(f"Saved config:  {run_config_path}")


if __name__ == "__main__":
    main()
