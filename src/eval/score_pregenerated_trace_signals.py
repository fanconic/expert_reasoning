"""Score a pregenerated trace JSONL with reward or policy token signals.

The script is intentionally narrower than ``evaluate.py``: it reads rows that
already contain ``prompt`` and ``generation.content`` fields and writes the same
rows augmented with either reward-model traces or policy log-probability and
entropy traces. This avoids loading the policy model when only a reward model is
needed, which is useful for one-GPU cross-domain scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("UNSLOTH_COMPILE_OVERWRITE", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.transformers_compat import (  # noqa: E402
    configure_pytorch_transformers_runtime,
    ensure_transformers_cache_alias,
)

configure_pytorch_transformers_runtime()

from unsloth import FastLanguageModel  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from peft import PeftModel  # noqa: E402
from tqdm import tqdm  # noqa: E402
from trl.trainer.grpo_trainer import apply_chat_template  # noqa: E402

from src.models.model_module import (  # noqa: E402
    _model_hidden_size,
    attach_airl_segment_h_head,
    attach_airl_segment_heads,
    load_airl_segment_h_head,
)
from src.training.airl_segment_utils import (  # noqa: E402
    fixed_interval_boundary_mask,
    gather_token_positions,
    segment_layout_from_boundaries,
)
from src.utils.utils import set_seed  # noqa: E402


DEFAULT_TRACE_FILE = Path(
    "/mnt/pdata/caf83/neurips2026/medicine/outputs/qwen7b_sft/best_model/"
    "eval_results_medicine_qwen7b_sft_t0p5.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--mode", choices=["reward", "policy"], required=True)
    parser.add_argument("--score-label", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1124)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append to output-file instead of overwriting; summary is recomputed from the full file.",
    )
    parser.add_argument(
        "--include-original-reward-model-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep any existing reward_model_score field from the input row.",
    )

    reward = parser.add_argument_group("reward scoring")
    reward.add_argument("--checkpoint-dir", type=Path, default=None)
    reward.add_argument("--config", type=Path, default=None)
    reward.add_argument("--reward-name", type=str, default=None)
    reward.add_argument("--reward-lora-rank", type=int, default=None)
    reward.add_argument("--reward-gpu-memory-utilization", type=float, default=0.2)
    reward.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    reward.add_argument(
        "--reward-variant",
        choices=["dense", "interval", "sparse", "partial_fixed"],
        default="dense",
    )
    reward.add_argument("--segment-tokens", type=int, default=15)
    reward.add_argument("--reward-mode", type=str, default=None)
    reward.add_argument("--airl-gamma", type=float, default=None)
    reward.add_argument("--lambda-shape", type=float, default=None)
    reward.add_argument("--clip-reward-model", action=argparse.BooleanOptionalAction, default=True)
    reward.add_argument("--reward-lb", type=float, default=-5.0)
    reward.add_argument("--reward-ub", type=float, default=5.0)

    policy = parser.add_argument_group("policy scoring")
    policy.add_argument("--policy-model", type=Path, default=None)
    policy.add_argument("--policy-gpu-memory-utilization", type=float, default=0.2)
    policy.add_argument("--max-lora-rank", type=int, default=256)
    policy.add_argument("--entropy-token-chunk-size", type=int, default=8)
    policy.add_argument("--append-eos", action="store_true")

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


def _row_generation_text(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if isinstance(generation, dict):
        value = generation.get("content", "")
    else:
        value = generation or ""
    return str(value)


def _prompt_key(row: dict[str, Any]) -> str:
    return json.dumps(row.get("prompt"), sort_keys=True, ensure_ascii=False)


def _row_correct(row: dict[str, Any]) -> float | None:
    for key in ("correctness_reward_func", "correct", "is_correct"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            value_f = float(value)
            return 1.0 if value_f > 0.0 else 0.0
        except Exception:
            return None
    return None


def _iter_batches(rows: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch_size = max(1, int(batch_size))
    for i in range(0, len(rows), batch_size):
        yield list(rows[i : i + batch_size])


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean_or_none(values: Sequence[float]) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _sum_or_none(values: Sequence[float]) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(arr.sum())


def _last_or_none(values: Sequence[float]) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(arr[-1])


def _std(values: Sequence[float]) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(arr.std())


def _every_n_tokens_mask_from_completion_mask(
    completion_mask: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Mark every n-th completion token and always include the final token."""
    n = max(1, int(n))
    completion_mask = completion_mask.bool()
    token_indices = completion_mask.long().cumsum(dim=1)
    every_n_mask = (token_indices % n == 0) & completion_mask

    if bool(completion_mask.any().item()):
        positions = torch.arange(
            completion_mask.size(1),
            device=completion_mask.device,
        ).unsqueeze(0)
        last_indices = (positions * completion_mask.long()).max(dim=1).values
        rows = torch.nonzero(completion_mask.any(dim=1), as_tuple=False).flatten()
        every_n_mask[rows, last_indices[rows]] = True
    return every_n_mask


def _backfill_rewards(rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Copy each marked interval reward backward until the previous mark."""
    batch_size, seq_len = rewards.shape
    indices = torch.arange(seq_len, device=rewards.device).expand(batch_size, seq_len)
    sentinel = torch.full_like(indices, seq_len)
    masked_indices = torch.where(mask.bool(), indices, sentinel)
    next_valid_index = torch.cummin(masked_indices.flip(1), dim=1)[0].flip(1)
    next_valid_index = next_valid_index.clamp(max=seq_len - 1).long()
    return torch.gather(rewards, 1, next_valid_index)


def _load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    with (adapter_dir / "adapter_config.json").open("r") as f:
        return json.load(f)


def _resolve_adapter_dir(checkpoint_dir: Path) -> Path:
    if (checkpoint_dir / "reward_model" / "adapter_config.json").exists():
        return checkpoint_dir / "reward_model"
    if (checkpoint_dir / "adapter_config.json").exists():
        return checkpoint_dir
    raise FileNotFoundError(
        f"Could not find reward adapter under {checkpoint_dir} or {checkpoint_dir / 'reward_model'}."
    )


def _load_yaml_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return cfg if isinstance(cfg, dict) else {}


def _model_cfg(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    return model if isinstance(model, dict) else {}


def _find_model_head(model, names: str | Sequence[str]):
    if isinstance(names, str):
        names = [names]

    def safe_getattr(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            return None

    seen: set[int] = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        for name in names:
            found = safe_getattr(obj, name)
            if found is not None:
                return found
        for child_name in ("base_model", "model", "module"):
            child = safe_getattr(obj, child_name)
            if child is not None:
                stack.append(child)

    if hasattr(model, "named_modules"):
        suffixes = tuple(f".{name}" for name in names)
        for module_name, module in model.named_modules():
            if module_name in names or module_name.endswith(suffixes):
                return module
    return None


def _apply_scalar_head(head, hidden_states: torch.Tensor) -> torch.Tensor:
    head_param = next(head.parameters(), None)
    if head_param is not None:
        hidden_states = hidden_states.to(dtype=head_param.dtype)
    out = head(hidden_states)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "logits"):
        out = out.logits
    if out.ndim >= 1 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out.float()


def _attach_scalar_lm_head(model):
    hidden_size = _model_hidden_size(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.lm_head = torch.nn.Linear(
        in_features=hidden_size,
        out_features=1,
        bias=False,
        device=device,
        dtype=dtype,
    )
    model.config.num_labels = 1
    return model


def load_reward_model(args: argparse.Namespace):
    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required for --mode reward.")

    checkpoint_dir = Path(args.checkpoint_dir)
    adapter_dir = _resolve_adapter_dir(checkpoint_dir)
    adapter_cfg = _load_adapter_config(adapter_dir)
    default_config = checkpoint_dir / "evaluation_config.yaml"
    config = _load_yaml_if_exists(args.config or default_config)
    model_cfg = _model_cfg(config)

    reward_name = (
        args.reward_name
        or model_cfg.get("reward_name")
        or adapter_cfg.get("base_model_name_or_path")
        or "Qwen/Qwen2.5-7B-Instruct"
    )
    reward_lora_rank = int(
        args.reward_lora_rank
        or model_cfg.get("reward_lora_rank")
        or adapter_cfg.get("r")
        or 256
    )
    max_length = int(
        args.max_length
        or (
            int(model_cfg.get("max_prompt_length", 300))
            + int(model_cfg.get("max_completion_length", 824))
        )
    )
    variant = args.reward_variant
    critic_type = str(model_cfg.get("critic_type", "standard"))
    if variant == "interval":
        critic_type = "airl_segment"
    reward_mode = args.reward_mode or str(model_cfg.get("reward_mode", "mean_g"))
    airl_gamma = float(args.airl_gamma if args.airl_gamma is not None else model_cfg.get("airl_gamma", 1.0))
    lambda_shape = float(
        args.lambda_shape if args.lambda_shape is not None else model_cfg.get("lambda_shape", 1.0)
    )
    segment_tokens = int(args.segment_tokens or model_cfg.get("segment_tokens", 15))
    dense_partial_fixed_n = int(
        model_cfg.get("dense_partial_fixed_n")
        or args.segment_tokens
        or segment_tokens
        or 15
    )

    use_scalar_cls = variant in {"sparse"} and critic_type != "airl_segment"
    model_kwargs = dict(
        model_name=reward_name,
        max_seq_length=max_length,
        load_in_4bit=bool(args.load_in_4bit),
        fast_inference=False,
        max_lora_rank=reward_lora_rank,
        gpu_memory_utilization=float(args.reward_gpu_memory_utilization),
    )
    if use_scalar_cls:
        model_kwargs["num_labels"] = 1
    reward_model, reward_tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)

    if critic_type == "airl_segment":
        reward_model = attach_airl_segment_heads(reward_model)
    elif variant != "sparse":
        reward_model = _attach_scalar_lm_head(reward_model)

    reward_model = PeftModel.from_pretrained(
        reward_model,
        str(adapter_dir),
        is_trainable=False,
    )
    if critic_type == "airl_segment":
        reward_model = attach_airl_segment_h_head(reward_model)
        reward_model = load_airl_segment_h_head(reward_model, str(adapter_dir))

    if hasattr(reward_model, "gradient_checkpointing_disable"):
        reward_model.gradient_checkpointing_disable()
    if hasattr(reward_model, "config"):
        reward_model.config.use_cache = False
    FastLanguageModel.for_inference(reward_model)
    reward_model.eval()

    meta = {
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_dir": str(adapter_dir),
        "reward_name": str(reward_name),
        "reward_variant": variant,
        "critic_type": critic_type,
        "reward_mode": reward_mode,
        "segment_tokens": segment_tokens,
        "dense_partial_fixed_n": dense_partial_fixed_n,
        "airl_gamma": airl_gamma,
        "lambda_shape": lambda_shape,
        "max_length": max_length,
        "load_in_4bit": bool(args.load_in_4bit),
    }
    return reward_model, reward_tokenizer, meta


def load_policy_model(args: argparse.Namespace):
    if args.policy_model is None:
        raise ValueError("--policy-model is required for --mode policy.")
    policy_model = Path(args.policy_model)
    adapter_dir = None
    model_name = str(policy_model)
    if (policy_model / "adapter_config.json").exists():
        adapter_cfg = _load_adapter_config(policy_model)
        model_name = str(adapter_cfg.get("base_model_name_or_path") or model_name)
        adapter_dir = policy_model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=int(args.max_length),
        load_in_4bit=bool(args.load_in_4bit),
        fast_inference=False,
        max_lora_rank=int(args.max_lora_rank),
        gpu_memory_utilization=float(args.policy_gpu_memory_utilization),
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    FastLanguageModel.for_inference(model)
    model.eval()
    meta = {
        "policy_model": str(policy_model),
        "base_model_name": str(model_name),
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "max_length": int(args.max_length),
        "load_in_4bit": bool(args.load_in_4bit),
    }
    return model, tokenizer, meta


def _texts_for_rows(rows: Sequence[dict[str, Any]], tokenizer, append_eos: bool = False):
    full_texts = []
    completion_texts = []
    for row in rows:
        content = _row_generation_text(row)
        msgs = row["prompt"] + [{"role": "assistant", "content": content}]
        full_texts.append(apply_chat_template({"messages": msgs}, tokenizer)["text"])
        if append_eos and tokenizer.eos_token:
            content = content + tokenizer.eos_token
        completion_texts.append(content)
    return full_texts, completion_texts


@torch.inference_mode()
def score_reward_batch(
    model,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    full_texts, completion_texts = _texts_for_rows(rows, tokenizer, append_eos=False)
    batch_inputs = tokenizer(
        text=full_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=int(meta["max_length"]),
        padding_side="right",
    ).to(device)
    batch_completions = tokenizer(
        text=completion_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=int(meta["max_length"]),
    ).to(device)
    completion_lens = batch_completions["attention_mask"].sum(dim=1).long()
    full_lens = batch_inputs["attention_mask"].sum(dim=1).long()
    start_indices = (full_lens - completion_lens).clamp(min=0)
    batch_width = int(batch_inputs["input_ids"].shape[1])

    out_rows: list[dict[str, Any]] = []
    if meta["critic_type"] == "airl_segment":
        outputs = model(**batch_inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1]
        idx = torch.arange(batch_width, device=device).unsqueeze(0)
        completion_mask = batch_inputs["attention_mask"].bool() & (
            idx >= start_indices.unsqueeze(1)
        )
        boundary_mask = fixed_interval_boundary_mask(
            completion_mask,
            int(meta["segment_tokens"]),
        )
        layout = segment_layout_from_boundaries(boundary_mask, completion_mask)
        g_head = _find_model_head(model, ["lm_head", "classifier", "score", "g_head"])
        h_head = _find_model_head(model, "h_head")
        if g_head is None or h_head is None:
            raise RuntimeError("AIRL segment scoring requires scalar g/lm and h_head modules.")

        end_hidden = gather_token_positions(hidden_states, layout.ends)
        prev_hidden = gather_token_positions(hidden_states, layout.prev_indices)
        next_hidden = gather_token_positions(hidden_states, layout.next_indices)
        g = _apply_scalar_head(g_head, end_hidden)
        h_prev = _apply_scalar_head(h_head, prev_hidden)
        h_next = _apply_scalar_head(h_head, next_hidden)
        shape = float(meta["airl_gamma"]) * h_next - h_prev
        f = g + shape
        valid_mask = layout.valid_mask
        reward_mode = str(meta["reward_mode"])
        if reward_mode == "mean_g":
            segment_values = g
        elif reward_mode in {"standard", "mean_f"}:
            segment_values = f
        elif reward_mode == "mean_g_plus_shape":
            segment_values = g + float(meta["lambda_shape"]) * shape
        else:
            raise ValueError(f"Unknown AIRL segment reward_mode: {reward_mode}")
        if bool(args.clip_reward_model):
            segment_values = torch.clamp(
                segment_values,
                min=float(args.reward_lb),
                max=float(args.reward_ub),
            )

        for b, row in enumerate(rows):
            comp_len = int(completion_lens[b].item())
            start_idx = int(start_indices[b].item())
            actual_len = max(0, min(comp_len, batch_width - start_idx))
            values = np.full(actual_len, np.nan, dtype=np.float64)
            segments: list[dict[str, Any]] = []
            for s in range(layout.max_segments):
                if not bool(valid_mask[b, s].item()):
                    continue
                abs_start = int(layout.starts[b, s].item())
                abs_end = int(layout.ends[b, s].item())
                local_start = max(0, abs_start - start_idx)
                local_end = min(actual_len - 1, abs_end - start_idx)
                if local_start > local_end:
                    continue
                g_val = float(g[b, s].detach().cpu().item())
                shape_val = float(shape[b, s].detach().cpu().item())
                f_val = float(f[b, s].detach().cpu().item())
                value = float(segment_values[b, s].detach().cpu().item())
                values[local_start : local_end + 1] = value
                segments.append(
                    {
                        "start": int(local_start),
                        "end": int(local_end),
                        "value": value,
                        "g": g_val,
                        "shape": shape_val,
                        "f": f_val,
                    }
                )
            seq = values[np.isfinite(values)].tolist()
            seg_vals = [seg["value"] for seg in segments]
            out_rows.append(
                _augment_reward_row(
                    row,
                    args,
                    meta,
                    seq=seq,
                    scalar=_mean_or_none(seg_vals),
                    extra={"reward_segments": segments},
                )
            )
    else:
        outputs = model(**batch_inputs)
        reward_logits = outputs.logits
        if isinstance(reward_logits, (tuple, list)):
            reward_logits = reward_logits[0]
        if reward_logits.ndim == 3 and reward_logits.shape[-1] == 1:
            reward_logits = reward_logits.squeeze(-1)
        if reward_logits.ndim == 1:
            reward_logits = reward_logits.unsqueeze(1).expand(-1, batch_width)
        if bool(args.clip_reward_model):
            reward_logits = torch.clamp(
                reward_logits,
                min=float(args.reward_lb),
                max=float(args.reward_ub),
            )
        if str(meta["reward_variant"]) == "partial_fixed":
            idx = torch.arange(batch_width, device=device).unsqueeze(0)
            completion_mask = batch_inputs["attention_mask"].bool() & (
                idx >= start_indices.unsqueeze(1)
            )
            boundary_mask = _every_n_tokens_mask_from_completion_mask(
                completion_mask,
                int(meta["dense_partial_fixed_n"]),
            )
            reward_logits = _backfill_rewards(reward_logits, boundary_mask)

        for b, row in enumerate(rows):
            comp_len = int(completion_lens[b].item())
            start_idx = int(start_indices[b].item())
            actual_len = max(0, min(comp_len, batch_width - start_idx))
            if actual_len > 0:
                seq_tensor = reward_logits[b, start_idx : start_idx + actual_len]
                seq = [float(x) for x in seq_tensor.detach().float().cpu().tolist()]
            else:
                seq = []
            out_rows.append(
                _augment_reward_row(
                    row,
                    args,
                    meta,
                    seq=seq,
                    scalar=_mean_or_none(seq),
                    extra={},
                )
            )

    del batch_inputs, batch_completions
    if "outputs" in locals():
        del outputs
    torch.cuda.empty_cache()
    return out_rows


def _base_output_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    if not args.include_original_reward_model_score:
        out.pop("reward_model_score", None)
    out["score_label"] = args.score_label
    out["source_trace_file"] = str(args.trace_file)
    return out


def _augment_reward_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    meta: dict[str, Any],
    seq: Sequence[float],
    scalar: float | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    out = _base_output_row(row, args)
    out.update(
        {
            "score_model_type": "reward",
            "reward_checkpoint_dir": meta["checkpoint_dir"],
            "reward_adapter_dir": meta["adapter_dir"],
            "reward_name": meta["reward_name"],
            "reward_variant": meta["reward_variant"],
            "critic_type": meta["critic_type"],
            "reward_mode": meta["reward_mode"],
            "reward_model_score": [float(x) for x in seq],
            "reward_score_scalar": scalar,
            "reward_score_mean_tokens": _mean_or_none(seq),
            "reward_score_sum_tokens": _sum_or_none(seq),
            "reward_score_last": _last_or_none(seq),
            "reward_score_num_tokens": len(seq),
        }
    )
    out.update(extra)
    return out


@torch.inference_mode()
def score_policy_batch(
    model,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    full_texts, completion_texts = _texts_for_rows(
        rows,
        tokenizer,
        append_eos=bool(args.append_eos),
    )
    batch_inputs = tokenizer(
        text=full_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=int(args.max_length),
        padding_side="right",
    ).to(device)
    batch_completions = tokenizer(
        text=completion_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=int(args.max_length),
    ).to(device)

    outputs = model(**batch_inputs)
    logits = outputs.logits
    completion_lens = batch_completions["attention_mask"].sum(dim=1).long()
    full_lens = batch_inputs["attention_mask"].sum(dim=1).long()
    start_indices = (full_lens - completion_lens).clamp(min=0)

    out_rows: list[dict[str, Any]] = []
    chunk = max(1, int(args.entropy_token_chunk_size))
    for b, row in enumerate(rows):
        comp_len = int(completion_lens[b].item())
        start_idx = max(int(start_indices[b].item()) - 1, 0)
        end_idx = min(start_idx + comp_len, batch_inputs["input_ids"].size(1) - 1)
        actual_len = max(0, end_idx - start_idx)
        seq_log_probs: list[float] = []
        seq_entropies: list[float] = []
        if actual_len > 0:
            labels = batch_inputs["input_ids"][b, start_idx + 1 : end_idx + 1]
            for s in range(0, actual_len, chunk):
                e = min(actual_len, s + chunk)
                seq_logits = logits[b, start_idx + s : start_idx + e, :].float()
                seq_labels = labels[s:e]
                log_probs = torch.log_softmax(seq_logits, dim=-1)
                picked = log_probs.gather(1, seq_labels[:, None]).squeeze(1)
                entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
                seq_log_probs.extend(float(x) for x in picked.detach().cpu().tolist())
                seq_entropies.extend(float(x) for x in entropies.detach().cpu().tolist())

        out = _base_output_row(row, args)
        out.update(
            {
                "score_model_type": "policy",
                "policy_model": meta["policy_model"],
                "policy_base_model_name": meta["base_model_name"],
                "policy_log_probs": seq_log_probs,
                "policy_entropies": seq_entropies,
                "policy_logprob_mean": _mean_or_none(seq_log_probs),
                "policy_logprob_sum": _sum_or_none(seq_log_probs),
                "policy_entropy_mean": _mean_or_none(seq_entropies),
                "policy_num_tokens": len(seq_log_probs),
            }
        )
        out_rows.append(out)

    del outputs, batch_inputs, batch_completions
    torch.cuda.empty_cache()
    return out_rows


def _summarize_output_rows(rows: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_prompts": len({_prompt_key(row) for row in rows}),
        "input_correctness_mean": _mean_or_none(
            [c for row in rows if (c := _row_correct(row)) is not None]
        ),
    }

    def grouped_best(metric_key: str, higher_is_better: bool) -> float | None:
        groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            score = row.get(metric_key)
            correct = _row_correct(row)
            if score is None or correct is None:
                continue
            try:
                score_f = float(score)
                correct_f = float(correct)
            except Exception:
                continue
            if math.isfinite(score_f):
                groups[_prompt_key(row)].append((score_f, correct_f))
        chosen = []
        for vals in groups.values():
            if not vals:
                continue
            best = max(vals, key=lambda x: x[0]) if higher_is_better else min(vals, key=lambda x: x[0])
            chosen.append(best[1])
        return _mean_or_none(chosen)

    if mode == "reward":
        scalars = [row.get("reward_score_scalar") for row in rows]
        summary["reward_score_scalar_mean"] = _mean_or_none([x for x in scalars if x is not None])
        summary["reward_score_scalar_std"] = _std([x for x in scalars if x is not None])
        summary["rerank_acc_by_reward_scalar"] = grouped_best(
            "reward_score_scalar",
            higher_is_better=True,
        )
    else:
        lp_mean = [row.get("policy_logprob_mean") for row in rows]
        lp_sum = [row.get("policy_logprob_sum") for row in rows]
        ent_mean = [row.get("policy_entropy_mean") for row in rows]
        summary["policy_logprob_mean_mean"] = _mean_or_none([x for x in lp_mean if x is not None])
        summary["policy_logprob_sum_mean"] = _mean_or_none([x for x in lp_sum if x is not None])
        summary["policy_entropy_mean_mean"] = _mean_or_none([x for x in ent_mean if x is not None])
        summary["rerank_acc_by_logprob_mean"] = grouped_best(
            "policy_logprob_mean",
            higher_is_better=True,
        )
        summary["rerank_acc_by_logprob_sum"] = grouped_best(
            "policy_logprob_sum",
            higher_is_better=True,
        )
        summary["rerank_acc_by_entropy_mean_low"] = grouped_best(
            "policy_entropy_mean",
            higher_is_better=False,
        )
    return summary


def _summary_projection(row: dict[str, Any], mode: str) -> dict[str, Any]:
    out = {
        "prompt": row.get("prompt"),
        "correctness_reward_func": row.get("correctness_reward_func"),
        "correct": row.get("correct"),
        "is_correct": row.get("is_correct"),
    }
    if mode == "reward":
        out["reward_score_scalar"] = row.get("reward_score_scalar")
    else:
        out["policy_logprob_mean"] = row.get("policy_logprob_mean")
        out["policy_logprob_sum"] = row.get("policy_logprob_sum")
        out["policy_entropy_mean"] = row.get("policy_entropy_mean")
    return out


def _load_summary_projections(path: Path, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(_summary_projection(json.loads(raw), mode))
    return rows


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_file.with_suffix(args.output_file.suffix + ".summary.json")

    rows_all = _load_jsonl(args.trace_file)
    rows = rows_all[int(args.start_index) :]
    if int(args.max_examples) > 0:
        rows = rows[: int(args.max_examples)]
    if not rows:
        raise ValueError("No trace rows selected for scoring.")

    print(f"Trace file: {args.trace_file}")
    print(f"Rows selected: {len(rows)} / {len(rows_all)}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_file}")
    print(f"Append output: {bool(args.append_output)}")

    if args.mode == "reward":
        model, tokenizer, meta = load_reward_model(args)
        score_batch = lambda batch: score_reward_batch(model, tokenizer, batch, meta, args)
    else:
        model, tokenizer, meta = load_policy_model(args)
        score_batch = lambda batch: score_policy_batch(model, tokenizer, batch, meta, args)

    output_rows_for_summary: list[dict[str, Any]] = []
    write_mode = "a" if args.append_output else "w"
    with args.output_file.open(write_mode) as f:
        for batch in tqdm(
            _iter_batches(rows, int(args.micro_batch)),
            total=math.ceil(len(rows) / max(1, int(args.micro_batch))),
            desc=f"Scoring {args.score_label}",
        ):
            scored_rows = score_batch(batch)
            for out in scored_rows:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            output_rows_for_summary.extend(
                _summary_projection(out, args.mode) for out in scored_rows
            )

    if args.append_output:
        output_rows_for_summary = _load_summary_projections(args.output_file, args.mode)

    summary = {
        "trace_file": str(args.trace_file),
        "output_file": str(args.output_file),
        "score_label": args.score_label,
        "mode": args.mode,
        "metadata": meta,
        "metrics": _summarize_output_rows(output_rows_for_summary, args.mode),
    }
    _write_json(summary_file, summary)

    print(f"Wrote scored rows: {args.output_file}")
    print(f"Wrote summary: {summary_file}")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
