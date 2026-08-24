"""Generator-token localisation baselines for saved GSM8K perturbation pairs.

This utility reads an existing localisation run's ``pair_details.jsonl`` and
scores the exact saved ``pert_text`` traces with a policy/SFT model. It writes
token log-probability and token entropy sequences, then applies the same
single-point localisation protocol used by ``table_localisation.py``:

* log-probability: predict the largest downward step in the perturbed trace;
* entropy: predict the largest upward step in the perturbed trace;
* random: analytic chance for a uniformly random token location.

Changed positions are recomputed in the policy tokenizer space, which is the
right alignment for generator-side token baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("UNSLOTH_COMPILE_OVERWRITE", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel
from trl.trainer.grpo_trainer import apply_chat_template


DEFAULT_POLICY_MODEL = "/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model"
DEFAULT_WINDOWS = [1, 7]
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class ScoreItem:
    key: str
    prompt: list[dict[str, Any]]
    text: str


@dataclass
class TokenScores:
    probs: list[float]
    log_probs: list[float]
    entropies: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Localisation run directory containing pair_details.jsonl.",
    )
    source.add_argument(
        "--pair-details",
        type=Path,
        help="Explicit pair_details.jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/policy_token_baselines.",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default=DEFAULT_POLICY_MODEL,
        help="SFT/policy model path used for generator-side token scores.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum full chat sequence length. Defaults to run_config max_length or 1124.",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=4,
        help="Number of unique traces per forward pass.",
    )
    parser.add_argument(
        "--entropy-token-chunk-size",
        type=int,
        default=32,
        help="Completion-token chunk size for entropy reductions over the vocab.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the policy model in 4-bit.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.55,
        help="Unsloth GPU memory utilization hint.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=256,
        help="Maximum LoRA rank for loading Unsloth adapters.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
        help="Hit@1@W windows to report.",
    )
    parser.add_argument(
        "--target-position-source",
        type=str,
        default="diff",
        choices=["diff", "target_char_span", "target_char_start", "step_first_diff"],
        help=(
            "How to define the target localisation positions. 'diff' recomputes changed "
            "tokens from clean/perturbed traces; 'target_char_span' uses a saved "
            "target_char_span/perturbed_step span in the perturbed trace; 'target_char_start' "
            "uses the first token of that span; 'step_first_diff' uses the first perturbed-step "
            "token that differs from original_step. Span modes use diff fallback."
        ),
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=1,
        help="Only score rows with this perturbation severity. Ignored with --all-severities.",
    )
    parser.add_argument(
        "--all-severities",
        action="store_true",
        help="Score all rows instead of the Table-5 default severity==1 subset.",
    )
    parser.add_argument(
        "--require-table-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only rows that the existing Table-5 raw localisation script can score: "
            "nonempty changed_token_positions and pert_score_seq length >= 2."
        ),
    )
    parser.add_argument(
        "--score-clean",
        action="store_true",
        help="Also store clean trace log-probability/entropy sequences.",
    )
    parser.add_argument(
        "--append-eos",
        action="store_true",
        help="Append EOS when measuring completion length, matching evaluate.py sidecars.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include prompt/clean_text/pert_text in the output jsonl.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
        help="Bootstrap samples for CI half-widths.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=BOOTSTRAP_ALPHA,
        help="Bootstrap alpha; 0.05 gives a 95%% CI.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help="Bootstrap RNG seed.",
    )
    return parser.parse_args()


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


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _item_key(prompt: list[dict[str, Any]], text: str) -> str:
    return json.dumps(
        {"prompt": prompt, "text": text},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _completion_text(text: str, tokenizer, append_eos: bool) -> str:
    if append_eos and tokenizer.eos_token:
        return text + tokenizer.eos_token
    return text


def changed_token_positions_policy(
    tokenizer,
    prompt_messages: list[dict[str, Any]],
    clean_text: str,
    pert_text: str,
    max_length: int,
) -> list[int]:
    """Changed perturbed-token positions in the policy tokenizer space."""

    clean_full = apply_chat_template(
        {"messages": prompt_messages + [{"role": "assistant", "content": clean_text}]},
        tokenizer,
    )["text"]
    pert_full = apply_chat_template(
        {"messages": prompt_messages + [{"role": "assistant", "content": pert_text}]},
        tokenizer,
    )["text"]

    c_full_ids = tokenizer(
        clean_full,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    p_full_ids = tokenizer(
        pert_full,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    c_comp_ids = tokenizer(
        clean_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    p_comp_ids = tokenizer(
        pert_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    c_start = max(0, len(c_full_ids) - len(c_comp_ids))
    p_start = max(0, len(p_full_ids) - len(p_comp_ids))
    c_ids = c_full_ids[c_start:]
    p_ids = p_full_ids[p_start:]

    sm = SequenceMatcher(a=c_ids, b=p_ids, autojunk=False)
    changed: list[int] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed.extend(range(j1, j2))

    out: list[int] = []
    seen: set[int] = set()
    for pos in changed:
        if pos not in seen:
            out.append(pos)
            seen.add(pos)

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


def _row_target_char_span(row: dict[str, Any]) -> list[int] | None:
    span = row.get("target_char_span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        start = _to_int(span[0])
        end = _to_int(span[1])
        if start is not None and end is not None and end > start:
            return [int(start), int(end)]

    pert_text = row.get("pert_text")
    step = row.get("perturbed_step")
    if isinstance(pert_text, str) and isinstance(step, str) and step:
        start = pert_text.find(step)
        if start >= 0:
            return [int(start), int(start + len(step))]
    return None


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), int(m.start()), int(m.end())) for m in re.finditer(r"\w+|[^\w\s]", text or "")]


def _row_step_first_diff_char_span(row: dict[str, Any]) -> list[int] | None:
    pert_text = row.get("pert_text")
    original_step = row.get("original_step")
    perturbed_step = row.get("perturbed_step")
    full_step_span = _row_target_char_span(row)
    if (
        not isinstance(pert_text, str)
        or not isinstance(original_step, str)
        or not isinstance(perturbed_step, str)
        or full_step_span is None
    ):
        return None

    old_tokens = _token_spans(original_step)
    new_tokens = _token_spans(perturbed_step)
    if not new_tokens:
        return None

    sm = SequenceMatcher(
        a=[tok for tok, _start, _end in old_tokens],
        b=[tok for tok, _start, _end in new_tokens],
        autojunk=False,
    )
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if j1 < j2:
            _tok, start, end = new_tokens[j1]
        else:
            fallback_idx = min(j1, len(new_tokens) - 1)
            _tok, start, end = new_tokens[fallback_idx]
        return [int(full_step_span[0] + start), int(full_step_span[0] + end)]
    return [int(full_step_span[0]), int(full_step_span[0] + max(1, new_tokens[0][2] - new_tokens[0][1]))]


def token_positions_from_char_span(
    tokenizer,
    text: str,
    char_span: Sequence[int] | None,
    max_length: int,
) -> list[int]:
    if char_span is None or len(char_span) != 2:
        return []
    start = _to_int(char_span[0])
    end = _to_int(char_span[1])
    if start is None or end is None or end <= start:
        return []

    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    except Exception:
        return []

    offsets = enc.get("offset_mapping", [])
    positions: list[int] = []
    for idx, offset in enumerate(offsets):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            continue
        tok_start = _to_int(offset[0])
        tok_end = _to_int(offset[1])
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            continue
        if tok_end > start and tok_start < end:
            positions.append(int(idx))
    return positions


def first_token_position_from_char_span(
    tokenizer,
    text: str,
    char_span: Sequence[int] | None,
    max_length: int,
) -> list[int]:
    positions = token_positions_from_char_span(
        tokenizer=tokenizer,
        text=text,
        char_span=char_span,
        max_length=max_length,
    )
    return positions[:1]


def load_policy_model(
    policy_model: str,
    max_length: int,
    load_in_4bit: bool,
    max_lora_rank: int,
    gpu_memory_utilization: float,
):
    adapter_config_path = Path(policy_model) / "adapter_config.json"
    adapter_dir: str | None = None
    model_name = policy_model
    if adapter_config_path.exists():
        adapter_cfg = _load_json(adapter_config_path)
        base_model = adapter_cfg.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}")
        model_name = str(base_model)
        adapter_dir = policy_model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=load_in_4bit,
        fast_inference=False,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def score_policy_items(
    model,
    tokenizer,
    items: Sequence[ScoreItem],
    max_length: int,
    micro_batch: int,
    entropy_token_chunk_size: int,
    append_eos: bool,
) -> dict[str, TokenScores]:
    device = next(model.parameters()).device
    out: dict[str, TokenScores] = {}
    if not items:
        return out

    for i in tqdm(range(0, len(items), micro_batch), desc="Scoring policy tokens"):
        batch_items = list(items[i : i + micro_batch])
        batch_texts = []
        batch_completion_texts = []
        for item in batch_items:
            msgs = item.prompt + [{"role": "assistant", "content": item.text}]
            batch_texts.append(apply_chat_template({"messages": msgs}, tokenizer)["text"])
            batch_completion_texts.append(
                _completion_text(item.text, tokenizer, append_eos=append_eos)
            )

        batch_inputs = tokenizer(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding_side="right",
        ).to(device)
        batch_completions = tokenizer(
            text=batch_completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**batch_inputs)
        logits = outputs.logits
        completion_lens = batch_completions["attention_mask"].sum(dim=1).long()
        full_lens = batch_inputs["attention_mask"].sum(dim=1).long()
        start_indices = (full_lens - completion_lens).clamp(min=0)

        for b, item in enumerate(batch_items):
            comp_len = int(completion_lens[b].item())
            start_idx = max(int(start_indices[b].item()) - 1, 0)
            end_idx = min(start_idx + comp_len, batch_inputs["input_ids"].size(1) - 1)
            actual_len = max(0, end_idx - start_idx)

            token_log_probs: list[float] = []
            token_probs: list[float] = []
            token_entropies: list[float] = []
            if actual_len > 0:
                seq_labels = batch_inputs["input_ids"][b, start_idx + 1 : end_idx + 1]
                chunk = max(1, int(entropy_token_chunk_size))
                for s in range(0, actual_len, chunk):
                    e = min(actual_len, s + chunk)
                    seq_logits = logits[b, start_idx + s : start_idx + e, :].float()
                    labels = seq_labels[s:e]
                    log_probs = torch.log_softmax(seq_logits, dim=-1)
                    picked = log_probs.gather(1, labels[:, None]).squeeze(1)
                    entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
                    token_log_probs.extend(picked.detach().cpu().tolist())
                    token_probs.extend(picked.exp().detach().cpu().tolist())
                    token_entropies.extend(entropies.detach().cpu().tolist())

            out[item.key] = TokenScores(
                probs=[float(x) for x in token_probs],
                log_probs=[float(x) for x in token_log_probs],
                entropies=[float(x) for x in token_entropies],
            )

        del outputs
        torch.cuda.empty_cache()

    return out


def _prediction_index(seq: Sequence[float], detector: str) -> int | None:
    if len(seq) < 2:
        return None
    arr = np.asarray(seq, dtype=np.float64)
    if not np.isfinite(arr).all():
        finite = np.isfinite(arr)
        if finite.sum() < 2:
            return None
        arr = arr[finite]
    if detector == "largest_drop":
        scores = np.maximum(0.0, arr[:-1] - arr[1:])
    elif detector == "largest_spike":
        scores = np.maximum(0.0, arr[1:] - arr[:-1])
    else:
        raise ValueError(f"Unknown detector: {detector}")
    return int(np.argmax(scores)) + 1


def _hit_and_chance(
    pred_idx: int | None,
    changed_positions: Sequence[int],
    seq_len: int,
    window: int,
) -> tuple[int | None, float]:
    changed = sorted({int(c) for c in changed_positions if 0 <= int(c) < seq_len})
    if seq_len <= 0 or not changed:
        return None, float("nan")

    local_mask = np.zeros(seq_len, dtype=bool)
    w = int(max(0, window))
    for c in changed:
        lo = max(0, c - w)
        hi = min(seq_len - 1, c + w)
        local_mask[lo : hi + 1] = True
    chance = float(local_mask.mean())

    if pred_idx is None:
        return None, chance
    hit = int(any(abs(pred_idx - c) <= w for c in changed))
    return hit, chance


def _mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _bootstrap_ci_halfwidth(
    values: Sequence[float],
    samples: int,
    alpha: float,
    seed: int,
) -> float | None:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    n = int(vals.shape[0])
    if n == 0:
        return None
    if n == 1:
        return 0.0
    rng = np.random.default_rng(int(seed))
    n_boot = max(100, int(samples))
    alpha = min(max(float(alpha), 1e-6), 0.5)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float((hi - lo) / 2.0)


def _summarize_metric(
    rows: Sequence[dict[str, Any]],
    key: str,
    windows: Sequence[int],
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for w in windows:
        vals = []
        for row in rows:
            value = row.get(key, {}).get(str(w), None)
            if value is not None:
                vals.append(float(value))
        summary[str(w)] = {
            "mean": _mean(vals),
            "ci_halfwidth": _bootstrap_ci_halfwidth(
                vals,
                samples=bootstrap_samples,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            ),
            "n": len(vals),
        }
    return summary


def _resolve_pair_details(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    if args.run_dir is not None:
        run_dir = args.run_dir
        pair_details = run_dir / "pair_details.jsonl"
    else:
        pair_details = args.pair_details
        run_dir = pair_details.parent

    if pair_details is None or not pair_details.exists():
        raise FileNotFoundError(f"pair_details.jsonl not found: {pair_details}")

    run_config_path = run_dir / "run_config.json"
    run_config = _load_json(run_config_path) if run_config_path.exists() else {}
    return pair_details, run_dir, run_config


def _default_max_length(args: argparse.Namespace, run_config: dict[str, Any]) -> int:
    if args.max_length is not None:
        return int(args.max_length)
    cfg_len = run_config.get("max_length") if isinstance(run_config, dict) else None
    if cfg_len is not None:
        return int(cfg_len)
    return 1124


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _table_valid_row(row: dict[str, Any]) -> bool:
    pert_seq = row.get("pert_score_seq", [])
    changed_positions = row.get("changed_token_positions", [])
    if not isinstance(pert_seq, list) or len(pert_seq) < 2:
        return False
    if not isinstance(changed_positions, list):
        return False
    changed = {
        int(c)
        for c in changed_positions
        if _to_int(c) is not None and 0 <= int(c) < len(pert_seq)
    }
    return bool(changed)


def _filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_severities:
        selected = rows
    else:
        selected = [row for row in rows if int(row.get("severity", -1)) == int(args.severity)]
    selected = [
        row
        for row in selected
        if isinstance(row.get("clean_text"), str) and isinstance(row.get("pert_text"), str)
    ]
    if bool(args.require_table_valid):
        selected = [row for row in selected if _table_valid_row(row)]
    selected = selected[int(args.start_index) :]
    if int(args.max_examples) > 0:
        selected = selected[: int(args.max_examples)]
    return selected


def main() -> None:
    args = parse_args()
    pair_details, run_dir, run_config = _resolve_pair_details(args)
    output_dir = args.output_dir or (run_dir / "policy_token_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    max_length = _default_max_length(args, run_config)
    rows_all = _load_jsonl(pair_details)
    rows = _filter_rows(rows_all, args)
    if not rows:
        raise ValueError("No rows selected for scoring.")

    print(f"Pair details: {pair_details}")
    print(f"Selected rows: {len(rows)} / {len(rows_all)}")
    print(f"Policy model: {args.policy_model}")
    print(f"Max length: {max_length}")

    model, tokenizer = load_policy_model(
        policy_model=args.policy_model,
        max_length=max_length,
        load_in_4bit=bool(args.load_in_4bit),
        max_lora_rank=int(args.max_lora_rank),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )

    items_by_key: dict[str, ScoreItem] = {}
    for row in rows:
        prompt = row["prompt"]
        pert_text = row["pert_text"]
        pert_key = _item_key(prompt, pert_text)
        items_by_key.setdefault(pert_key, ScoreItem(key=pert_key, prompt=prompt, text=pert_text))
        if args.score_clean:
            clean_text = row["clean_text"]
            clean_key = _item_key(prompt, clean_text)
            items_by_key.setdefault(
                clean_key,
                ScoreItem(key=clean_key, prompt=prompt, text=clean_text),
            )

    scores = score_policy_items(
        model=model,
        tokenizer=tokenizer,
        items=list(items_by_key.values()),
        max_length=max_length,
        micro_batch=int(args.micro_batch),
        entropy_token_chunk_size=int(args.entropy_token_chunk_size),
        append_eos=bool(args.append_eos),
    )

    detail_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Building baseline details"):
        prompt = row["prompt"]
        pert_key = _item_key(prompt, row["pert_text"])
        pert_scores = scores[pert_key]
        diff_changed = changed_token_positions_policy(
            tokenizer=tokenizer,
            prompt_messages=prompt,
            clean_text=row["clean_text"],
            pert_text=row["pert_text"],
            max_length=max_length,
        )
        target_span = _row_target_char_span(row)
        target_position_source = "diff"
        changed = diff_changed
        target_error_span = None
        if args.target_position_source in {"target_char_span", "target_char_start", "step_first_diff"}:
            if args.target_position_source == "step_first_diff":
                target_error_span = _row_step_first_diff_char_span(row)
                char_span_for_tokens = target_error_span
                span_changed = token_positions_from_char_span(
                    tokenizer=tokenizer,
                    text=row["pert_text"],
                    char_span=char_span_for_tokens,
                    max_length=max_length,
                )
            elif args.target_position_source == "target_char_start":
                char_span_for_tokens = target_span
                span_changed = first_token_position_from_char_span(
                    tokenizer=tokenizer,
                    text=row["pert_text"],
                    char_span=char_span_for_tokens,
                    max_length=max_length,
                )
            else:
                char_span_for_tokens = target_span
                span_changed = token_positions_from_char_span(
                    tokenizer=tokenizer,
                    text=row["pert_text"],
                    char_span=char_span_for_tokens,
                    max_length=max_length,
                )
            if span_changed:
                changed = span_changed
                target_position_source = args.target_position_source
            else:
                target_position_source = "diff_fallback"
        seq_len = len(pert_scores.log_probs)

        prob_pred = _prediction_index(pert_scores.probs, detector="largest_drop")
        logprob_pred = _prediction_index(pert_scores.log_probs, detector="largest_drop")
        entropy_pred = _prediction_index(pert_scores.entropies, detector="largest_spike")

        prob_hit: dict[str, int | None] = {}
        logprob_hit: dict[str, int | None] = {}
        entropy_hit: dict[str, int | None] = {}
        random_hit: dict[str, float] = {}
        for window in args.windows:
            h_prob, _ = _hit_and_chance(prob_pred, changed, seq_len, int(window))
            h_lp, chance = _hit_and_chance(logprob_pred, changed, seq_len, int(window))
            h_ent, _ = _hit_and_chance(entropy_pred, changed, seq_len, int(window))
            prob_hit[str(window)] = h_prob
            logprob_hit[str(window)] = h_lp
            entropy_hit[str(window)] = h_ent
            random_hit[str(window)] = chance

        out_row: dict[str, Any] = {
            "prompt_idx": row.get("prompt_idx"),
            "severity": row.get("severity"),
            "variant_idx": row.get("variant_idx"),
            "clean_generation_idx": row.get("clean_generation_idx"),
            "perturb_fns": row.get("perturb_fns"),
            "source_pair_details": str(pair_details),
            "policy_model": args.policy_model,
            "target_position_source": target_position_source,
            "target_char_span": target_span,
            "target_error_char_span": target_error_span,
            "target_perturbed_step": row.get("perturbed_step"),
            "policy_token_seq_len": seq_len,
            "policy_changed_token_positions": changed,
            "policy_diff_changed_token_positions": diff_changed,
            "reward_token_changed_token_positions": row.get("changed_token_positions"),
            "pert_policy_probs": pert_scores.probs,
            "pert_policy_log_probs": pert_scores.log_probs,
            "pert_policy_entropies": pert_scores.entropies,
            "prob_detector": "largest_drop",
            "logprob_detector": "largest_drop",
            "entropy_detector": "largest_spike",
            "prob_pred_idx": prob_pred,
            "logprob_pred_idx": logprob_pred,
            "entropy_pred_idx": entropy_pred,
            "prob_hit1_at_window": prob_hit,
            "logprob_hit1_at_window": logprob_hit,
            "entropy_hit1_at_window": entropy_hit,
            "random_hit1_at_window": random_hit,
        }
        if args.score_clean:
            clean_key = _item_key(prompt, row["clean_text"])
            clean_scores = scores[clean_key]
            out_row["clean_policy_probs"] = clean_scores.probs
            out_row["clean_policy_log_probs"] = clean_scores.log_probs
            out_row["clean_policy_entropies"] = clean_scores.entropies
        if args.include_text:
            out_row["prompt"] = prompt
            out_row["clean_text"] = row["clean_text"]
            out_row["pert_text"] = row["pert_text"]
        detail_rows.append(out_row)

    detail_path = output_dir / "policy_token_baselines.jsonl"
    _write_jsonl(detail_path, detail_rows)

    summary = {
        "source_pair_details": str(pair_details),
        "source_run_dir": str(run_dir),
        "policy_model": args.policy_model,
        "max_length": max_length,
        "append_eos": bool(args.append_eos),
        "n_rows_total": len(rows_all),
        "n_rows_scored": len(rows),
        "n_unique_traces_scored": len(items_by_key),
        "severity_filter": None if args.all_severities else int(args.severity),
        "require_table_valid": bool(args.require_table_valid),
        "target_position_source": args.target_position_source,
        "windows": [int(w) for w in args.windows],
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "alpha": float(args.bootstrap_alpha),
            "seed": int(args.bootstrap_seed),
        },
        "metrics": {
            "prob_largest_drop": _summarize_metric(
                detail_rows,
                key="prob_hit1_at_window",
                windows=args.windows,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            ),
            "logprob_largest_drop": _summarize_metric(
                detail_rows,
                key="logprob_hit1_at_window",
                windows=args.windows,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            ),
            "entropy_largest_spike": _summarize_metric(
                detail_rows,
                key="entropy_hit1_at_window",
                windows=args.windows,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            ),
            "random_location": _summarize_metric(
                detail_rows,
                key="random_hit1_at_window",
                windows=args.windows,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            ),
        },
    }
    summary_path = output_dir / "policy_token_baselines_summary.json"
    _write_json(summary_path, summary)

    print(f"Wrote details: {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print("Summary:")
    for metric_name, by_window in summary["metrics"].items():
        for window, vals in by_window.items():
            mean = vals["mean"]
            ci = vals["ci_halfwidth"]
            if mean is None:
                print(f"  {metric_name} Hit@1@{window}: n/a")
            else:
                print(f"  {metric_name} Hit@1@{window}: {mean:.4f} +/- {ci:.4f}")


if __name__ == "__main__":
    main()
