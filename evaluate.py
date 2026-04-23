# evaluate.py
import os
import inspect

os.environ["UNSLOTH_COMPILE_OVERWRITE"] = "0"
from unsloth import FastLanguageModel
from src.models.model_module import (
    load_model_and_tokenizer,
    irl_load_model_and_tokenizer,
)
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.utils import set_seed, save_results_to_jsonl
from src.data.dataset import get_dataset
from src.rewards.reward_functions import (
    xmlcount_reward_func,
    gsm8k_correctness_reward_func,
    countdown_correctness_function,
    medical_correctness_reward_func,
    scienceqa_correctness_reward_func,
    mmlu_correctness_reward_func,
    eval_correctness_gsm8k,
    eval_correctness_countdown,
    eval_correctness_medical,
    eval_correctness_scienceqa,
    eval_correctness_mmlu,
)
import torch
import numpy as np
import pandas as pd
from src.eval.eval_module import (
    compute_pass_at_k,
    compute_reward_weighted_pass_at_k_from_scores,
    compute_success_at_k_from_scores,
    compute_oracle_at_1_from_N,
)
from src.eval.eval_mode_utils import (
    MODE_AIME,
    MODE_GENERATE,
    MODE_PREGENERATED_POLICY_AND_REWARD,
    canonical_eval_mode,
    default_output_filename,
    eval_mode_uses_pregenerated,
    resolve_pregenerated_jsonl_path,
)
from vllm import SamplingParams
import vllm.envs as vllm_envs
import wandb
from trl.trainer.grpo_trainer import maybe_apply_chat_template, apply_chat_template

# --- NEW IMPORTS FOR GUIDANCE ---
import copy
import re

wandb.login()


def _vllm_supports_request_logits_processors() -> bool:
    """
    Returns whether this vLLM runtime supports request-level logits processors
    in SamplingParams.
    """
    if vllm_envs.VLLM_USE_V1:
        return False
    try:
        from vllm.engine.llm_engine import LLMEngine

        src = inspect.getsource(LLMEngine.add_request)
        if "Logits processors are not supported in multi-step decoding" in src:
            return False
    except Exception:
        # If introspection fails, avoid hard-blocking; let vLLM runtime decide.
        pass
    return True


class TopKRewardLogitsProcessor:
    def __init__(
        self,
        reward_model,
        reward_tokenizer,
        policy_tokenizer,
        alpha=1.0,
        k=10,
        device="cuda",
        dense_reward=False,
        reward_discount_gamma=0.95,
    ):
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.policy_tokenizer = policy_tokenizer
        self.alpha = alpha
        self.k = k
        self.device = device
        self.dense_reward = dense_reward
        self.reward_discount_gamma = reward_discount_gamma
        self._policy_piece_cache = {}

    def _decode_policy_piece(self, tok_id: int) -> str:
        if tok_id not in self._policy_piece_cache:
            self._policy_piece_cache[tok_id] = self.policy_tokenizer.decode(
                [tok_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        return self._policy_piece_cache[tok_id]

    def __call__(self, prompt_tokens_ids, generated_tokens_ids, logits):
        """
        Implements Algorithm 1: Reward-Augmented Decoding
        1. Get Top-K tokens from Policy (logits).
        2. Compute Rewards only for those K tokens.
        3. Reweight and return.
        """
        # 1. Identify Top-K candidates from the Policy Model
        # logits shape: [vocab_size]
        top_k_scores, top_k_indices = torch.topk(logits, self.k)

        # 2. Prepare inputs for the Reward Model
        # vLLM logits are in policy tokenizer space, so we decode in policy space
        # and re-tokenize in reward-tokenizer space for correctness.
        base_seq = list(prompt_tokens_ids) + list(generated_tokens_ids)
        base_text = self.policy_tokenizer.decode(
            base_seq,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        candidate_texts = [
            base_text + self._decode_policy_piece(int(token_idx.item()))
            for token_idx in top_k_indices
        ]
        inputs = _tokenize_for_reward_model(
            self.reward_tokenizer,
            candidate_texts,
            self.device,
            max_length=getattr(self.reward_model, "max_seq_length", None),
        )

        # 3. Compute Rewards (Batched for Efficiency)
        with torch.no_grad():
            # Run the RM on the K candidates
            output = self.reward_model(**inputs)
            rm_scores = _extract_rm_scores(
                output,
                attention_mask=inputs.get("attention_mask"),
                dense_reward=self.dense_reward,
                reward_discount_gamma=self.reward_discount_gamma,
            )
            rm_scores = rm_scores.to(dtype=top_k_scores.dtype, device=top_k_scores.device)

        new_logits = torch.full_like(logits, float("-inf"))
        guided_scores = top_k_scores + (self.alpha * rm_scores)
        new_logits.scatter_(0, top_k_indices, guided_scores)

        return new_logits


def _discounted_token_mean(logits, attention_mask=None, gamma=0.95):
    """
    Compute discounted mean across sequence positions with newest token weighted
    highest: w_t = gamma^(T-1-t), where t increases left->right.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")

    gamma = float(gamma)
    B, T = logits.shape
    device = logits.device

    # Highest weight on most recent token (right-most / last valid position).
    powers = torch.arange(T - 1, -1, -1, device=device, dtype=torch.float32)
    base = torch.full((T,), gamma, device=device, dtype=torch.float32)
    pos_weights = torch.pow(base, powers).unsqueeze(0)  # [1, T]

    if attention_mask is None:
        weights = pos_weights.expand(B, -1)
    else:
        mask = attention_mask.to(device=device, dtype=torch.float32)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2:
            raise ValueError(
                f"Expected 2D attention_mask, got shape {tuple(mask.shape)}"
            )
        if mask.shape[1] != T:
            if mask.shape[1] > T:
                # Keep the most recent suffix aligned with truncated logits.
                mask = mask[:, -T:]
            else:
                # Pad on the right; extra logits positions receive zero weight.
                pad = torch.zeros(
                    (mask.shape[0], T - mask.shape[1]),
                    dtype=mask.dtype,
                    device=mask.device,
                )
                mask = torch.cat([mask, pad], dim=1)
        weights = pos_weights * mask

    logits_f = logits.to(dtype=torch.float32)
    numer = (logits_f * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1e-12)
    return numer / denom


def _extract_rm_scores(
    rm_out, attention_mask=None, dense_reward=False, reward_discount_gamma=0.95
):
    """
    Extract one scalar reward per sequence from reward model output.
    For dense outputs:
      - dense_reward=False: use last valid token score.
      - dense_reward=True: use discounted mean with highest weight on most recent
        token.
    """
    logits = rm_out.logits if hasattr(rm_out, "logits") else rm_out

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)

    if logits.ndim == 1:
        return logits

    if logits.ndim == 2:
        T = logits.shape[1]
        if T <= 0:
            return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)

        mask = attention_mask
        if mask is not None:
            mask = mask.to(device=logits.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.ndim != 2:
                raise ValueError(
                    f"Expected 2D attention_mask, got shape {tuple(mask.shape)}"
                )
            if mask.shape[1] != T:
                if mask.shape[1] > T:
                    # Keep the right-most mask suffix to match truncated logits.
                    mask = mask[:, -T:]
                else:
                    pad = torch.zeros(
                        (mask.shape[0], T - mask.shape[1]),
                        dtype=mask.dtype,
                        device=mask.device,
                    )
                    mask = torch.cat([mask, pad], dim=1)

        if dense_reward:
            return _discounted_token_mean(
                logits, attention_mask=mask, gamma=reward_discount_gamma
            )
        if mask is None:
            return logits[:, -1]
        last_idx = mask.long().sum(dim=1).clamp(min=1) - 1
        last_idx = last_idx.clamp(min=0, max=T - 1)
        return logits.gather(1, last_idx.unsqueeze(1)).squeeze(1)

    # Conservative fallback for unusual output shapes.
    flat = logits.reshape(logits.shape[0], -1)
    return flat.mean(dim=1)


def _tokenize_for_reward_model(tokenizer, texts, device, max_length=None):
    """
    Tokenize for reward-model scoring while preserving the latest suffix when
    truncation is needed (important for chunk guidance).
    """
    if max_length is not None:
        max_length = int(max_length)
        # Reserve one token of slack to avoid occasional off-by-one shape issues
        # in some model/tokenizer combinations near the context limit.
        max_length = max(8, max_length - 1)

    original_truncation_side = getattr(tokenizer, "truncation_side", None)
    try:
        if max_length is not None and original_truncation_side is not None:
            tokenizer.truncation_side = "left"

        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False,
            padding_side="right",
        )
    finally:
        if original_truncation_side is not None:
            tokenizer.truncation_side = original_truncation_side

    return batch.to(device)


# ==========================================
# 2. Chunk-Level Guidance (Step Search)
# ==========================================
def generate_with_chunk_guidance(
    model,
    reward_model,
    reward_tokenizer,
    policy_tokenizer,
    prompts_text,
    sampling_params,
    step_size=5,
    n_candidates=4,
    max_tokens=None,
    policy_max_length=None,
    rm_max_length=None,
    dense_reward=False,
    reward_discount_gamma=0.95,
    lora_req=None,
):
    """
    Performs generation by stepping 'step_size' tokens at a time,
    generating 'n_candidates', and selecting the best one via Reward Model.
    """
    # Initialize current generation with prompts
    current_gens = list(prompts_text)
    num_sequences = len(current_gens)
    if num_sequences == 0:
        return current_gens

    # We iterate until we hit max length (simplified loop)
    # Note: vLLM is most efficient with batching, this manual loop
    # splits the batch logic somewhat.

    if max_tokens is None:
        max_tokens = sampling_params.max_tokens

    # Safety cap for policy context length to avoid vLLM prompt-overflow errors.
    if policy_max_length is None:
        policy_max_length = getattr(model, "max_seq_length", None)
    if policy_max_length is not None:
        # Small safety margin for tokenizer/engine off-by-few differences.
        policy_max_length = max(8, int(policy_max_length) - 4)

    # Per-sequence remaining generation budget in tokens.
    if policy_max_length is not None and policy_tokenizer is not None:
        prompt_tok = policy_tokenizer(
            text=current_gens,
            return_attention_mask=True,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        prompt_lens = [len(ids) for ids in prompt_tok["input_ids"]]
        remaining_budget = [
            max(0, min(int(max_tokens), policy_max_length - prompt_len))
            for prompt_len in prompt_lens
        ]
    else:
        remaining_budget = [int(max_tokens)] * num_sequences

    while True:
        active_indices = [i for i, rem in enumerate(remaining_budget) if rem > 0]
        if not active_indices:
            break

        step_tokens = min(step_size, min(remaining_budget[i] for i in active_indices))
        if step_tokens <= 0:
            break

        # 1. Generate Candidates for the next step
        step_params = copy.deepcopy(sampling_params)
        step_params.max_tokens = int(step_tokens)
        step_params.n = n_candidates

        active_texts = [current_gens[i] for i in active_indices]

        # model.fast_generate typically returns a list of RequestOutputs
        outputs = model.fast_generate(
            active_texts,
            sampling_params=step_params,
            use_tqdm=False,
            lora_request=lora_req,
        )

        # 2. Build one global RM batch for all candidates in this step.
        flat_full_candidates = []
        grouped_candidates = []
        for i, out in enumerate(outputs):
            cand_texts = [o.text for o in out.outputs]
            grouped_candidates.append(cand_texts)
            parent_text = active_texts[i]
            for cand in cand_texts:
                flat_full_candidates.append(parent_text + cand)

        if not flat_full_candidates:
            break

        inputs = _tokenize_for_reward_model(
            reward_tokenizer,
            flat_full_candidates,
            reward_model.device,
            max_length=rm_max_length,
        )

        with torch.no_grad():
            rm_out = reward_model(**inputs)
            rm_scores = _extract_rm_scores(
                rm_out,
                attention_mask=inputs.get("attention_mask"),
                dense_reward=dense_reward,
                reward_discount_gamma=reward_discount_gamma,
            ).detach()

        # 3. Select best candidate for each parent sequence.
        rm_scores_cpu = rm_scores.float().cpu().numpy()
        offset = 0
        any_nonempty_extension = False
        for i, cand_texts in enumerate(grouped_candidates):
            global_idx = active_indices[i]
            parent_text = current_gens[global_idx]
            num_cands = len(cand_texts)
            if num_cands == 0:
                remaining_budget[global_idx] = 0
                continue

            local_scores = rm_scores_cpu[offset : offset + num_cands]
            best_idx = int(np.argmax(local_scores))
            best_extension = cand_texts[best_idx]
            current_gens[global_idx] = parent_text + best_extension

            all_empty = all(c == "" for c in cand_texts)
            if all_empty:
                remaining_budget[global_idx] = 0
            else:
                remaining_budget[global_idx] = max(
                    0, remaining_budget[global_idx] - int(step_tokens)
                )

            if best_extension != "":
                any_nonempty_extension = True
            offset += num_cands

        if not any_nonempty_extension:
            break

    return current_gens


# Module-level cache
_BOUNDARY_TOKEN_DECODE_CACHE = {}


def sentence_boundary_mask(reward_tokenizer, full_batch, base_completion_mask, device):
    """
    Robust step-boundary detector for process reward modelling.

    Args:
        full_batch: dict with key "input_ids" -> LongTensor [bs, L]
        base_completion_mask: Bool/0-1 tensor [bs, L], True only on assistant completion tokens
        reward_tokenizer: HuggingFace tokenizer used to decode token pieces

    Returns:
        boundary_mask: Bool tensor [bs, L]
    """
    global _BOUNDARY_TOKEN_DECODE_CACHE

    input_ids = full_batch["input_ids"]
    bs, L = input_ids.shape

    boundary_mask = torch.zeros((bs, L), dtype=torch.bool, device=device)

    explicit_boundaries = [
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
    ]
    explicit_boundaries = sorted(explicit_boundaries, key=len, reverse=True)

    max_explicit_len = max(len(x) for x in explicit_boundaries)
    suffix_window = max(96, max_explicit_len + 48)

    _abbr = {
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "cf.",
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "no.",
        "fig.",
        "eq.",
        "sec.",
        "resp.",
    }

    _wrapper_tags = {
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
        "<reasoning>",
        "</reasoning>",
    }

    def decode_one(tok_id: int) -> str:
        if tok_id not in _BOUNDARY_TOKEN_DECODE_CACHE:
            _BOUNDARY_TOKEN_DECODE_CACHE[tok_id] = reward_tokenizer.decode(
                [tok_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        return _BOUNDARY_TOKEN_DECODE_CACHE[tok_id]

    def _last_nonspace_char(s: str):
        for ch in reversed(s):
            if not ch.isspace():
                return ch
        return None

    def _strip_trailing_space(s: str) -> str:
        return s.rstrip(" \t")

    def _looks_like_abbreviation(s: str) -> bool:
        s = _strip_trailing_space(s).lower()

        m = re.search(r"([a-z]{1,10}\.)$", s)
        if m and m.group(1) in _abbr:
            return True

        m2 = re.search(r"([a-z]\.[a-z]\.)$", s)
        if m2 and m2.group(1) in _abbr:
            return True

        return False

    def _piece_is_only_layout(piece: str) -> bool:
        return piece.strip() == ""

    def _normalise_visible_text(s: str) -> str:
        x = s
        for tag in _wrapper_tags:
            x = x.replace(tag, "")
        x = re.sub(r"<\|[^>]+?\|>", "", x)
        x = re.sub(r"\s+", "", x)
        return x

    def _starts_with_digit(piece: str) -> bool:
        if piece is None:
            return False
        m = re.match(r"^[ \t\r\n]*([0-9])", piece)
        return m is not None

    def _is_explicit_boundary(s: str) -> bool:
        return any(s.endswith(x) for x in explicit_boundaries)

    def _is_sentence_punct_boundary(
        s: str, just_added_piece: str, next_piece: str
    ) -> bool:
        if just_added_piece != "" and just_added_piece.strip(" \t") == "":
            return False

        s = _strip_trailing_space(s)
        if not s:
            return False

        last = s[-1]
        if last not in ".!?;:":
            return False

        if last == "." and _looks_like_abbreviation(s):
            return False

        # Avoid splitting on decimal points like 90.2
        if last == "." and _starts_with_digit(next_piece):
            return False

        if last in "!?":
            return True

        if last in ";:":
            return True

        return True

    def _is_newline_boundary(s: str, just_added_piece: str) -> bool:
        if "\n" not in just_added_piece and "\r" not in just_added_piece:
            return False

        if not s.endswith("\n"):
            return False

        if s.endswith("\n\n"):
            return True

        prefix = s[:-1]
        ch = _last_nonspace_char(prefix)
        if ch is None:
            return False

        if ch in ".!?;:)":
            return True

        if prefix.endswith("</think>") or prefix.endswith("####"):
            return True

        return False

    def _ends_reasoning_step(s: str, just_added_piece: str, next_piece: str) -> bool:
        if _is_explicit_boundary(s):
            return True

        if _is_newline_boundary(s, just_added_piece):
            return True

        if _is_sentence_punct_boundary(s, just_added_piece, next_piece):
            return True

        return False

    for b in range(bs):
        completion_positions = torch.nonzero(
            base_completion_mask[b].bool(), as_tuple=False
        ).flatten()

        if completion_positions.numel() == 0:
            continue

        completion_positions_list = completion_positions.tolist()
        decoded_pieces = [
            decode_one(int(input_ids[b, pos].item()))
            for pos in completion_positions_list
        ]

        suffix = ""
        seen_meaningful_content = False
        prev_was_boundary = False

        for i, pos in enumerate(completion_positions_list):
            piece = decoded_pieces[i]
            next_piece = decoded_pieces[i + 1] if i + 1 < len(decoded_pieces) else None

            suffix += piece
            if len(suffix) > suffix_window:
                suffix = suffix[-suffix_window:]

            if not seen_meaningful_content and _normalise_visible_text(suffix) != "":
                seen_meaningful_content = True

            is_boundary = _ends_reasoning_step(suffix, piece, next_piece)

            if is_boundary and not seen_meaningful_content:
                is_boundary = False

            if is_boundary and prev_was_boundary and _piece_is_only_layout(piece):
                is_boundary = False

            if is_boundary:
                boundary_mask[b, pos] = True
                prev_was_boundary = True
            else:
                prev_was_boundary = False

        # Always include the final completion token so the last segment gets a reward
        boundary_mask[b, int(completion_positions[-1].item())] = True

    boundary_mask &= base_completion_mask.bool()
    return boundary_mask


def every_n_tokens_mask(full_batch, base_completion_mask, n: int):
    """
    Returns mask [bs, L] that is True every n tokens within the completion
    and always True at the final completion token.
    """
    input_ids = full_batch["input_ids"]  # [bs, L]
    bs, L = input_ids.shape
    device = input_ids.device

    # Count token positions within the completion (1-based)
    token_indices = base_completion_mask.long().cumsum(dim=1)  # [bs, L]

    # Mark every n-th token (ignore positions outside completion)
    every_n_mask = (token_indices % n == 0) & base_completion_mask

    # Find last completion token index per batch
    last_indices = token_indices.argmax(dim=1)  # [bs]

    # Ensure last completion token is always included
    every_n_mask[
        torch.arange(bs, device=device), last_indices
    ] |= base_completion_mask.any(dim=1)
    return every_n_mask


def backfill_rewards(rewards, mask):
    B, T = rewards.shape
    indices = torch.arange(T, device=rewards.device).expand(B, T)
    masked_indices = torch.where(
        mask.bool(), indices, torch.tensor(T, device=rewards.device)
    )
    next_valid_index = torch.cummin(masked_indices.flip(1), dim=1)[0].flip(1)
    next_valid_index = next_valid_index.clamp(max=T - 1).long()
    result = torch.gather(rewards, 1, next_valid_index)

    return result


@torch.no_grad()
def score_with_policy_model(
    policy_model,
    policy_tokenizer,
    prompts_msgs,
    decoded_per_prompt,
    max_length=512,
    micro_batch=16,
):
    FastLanguageModel.for_inference(policy_model)

    device = next(policy_model.parameters()).device

    # 1. Flatten prompts and completions into a single list of strings
    texts = []
    completion_texts = []

    for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
        for c in completions:
            content = c if isinstance(c, str) else c.get("content", "")
            msgs = p_msgs + [{"role": "assistant", "content": content}]
            texts.append(
                apply_chat_template({"messages": msgs}, policy_tokenizer)["text"]
            )

            comp_text = content + (policy_tokenizer.eos_token or "")
            completion_texts.append(comp_text)

    if not texts:
        return [[] for _ in prompts_msgs]

    # Calculate global max seq_len strictly for the final output array shape
    global_tokens = policy_tokenizer(
        completion_texts,
        return_attention_mask=True,
        add_special_tokens=False,
        padding=False,
    )
    seq_len = min(max(len(t) for t in global_tokens["input_ids"]), max_length)

    all_log_probs = []

    # 2. Dynamic batching loop
    for i in range(0, len(texts), micro_batch):
        batch_texts = texts[i : i + micro_batch]
        batch_completion_texts = completion_texts[i : i + micro_batch]

        batch_inputs = policy_tokenizer(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding_side="right",
        ).to(device)

        batch_completions = policy_tokenizer(
            text=batch_completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.inference_mode():
            outputs = policy_model(**batch_inputs)

            # Find where the completions start
            completion_lens = batch_completions["attention_mask"].sum(dim=1).long()
            full_lens = batch_inputs["attention_mask"].sum(dim=1).long()
            start_indices = (full_lens - completion_lens).clamp(min=0)

            current_micro_batch_size = batch_inputs["input_ids"].size(0)
            batch_res = torch.full(
                (current_micro_batch_size, seq_len), float("nan"), device=device
            )

            # Process log probs sequence-by-sequence to avoid large temporary tensors
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

            for b in range(current_micro_batch_size):
                comp_len = completion_lens[b].item()
                # The logit at index i predicts the token at index i+1
                start_idx = max(start_indices[b].item() - 1, 0)

                # Prevent out-of-bounds if the sequence hit max_length
                end_idx = min(
                    start_idx + comp_len, batch_inputs["input_ids"].size(1) - 1
                )
                actual_len = end_idx - start_idx

                if actual_len > 0:
                    seq_logits = outputs.logits[b, start_idx:end_idx, :]
                    seq_labels = batch_inputs["input_ids"][
                        b, start_idx + 1 : end_idx + 1
                    ]

                    # Cross entropy gives -log(p). Invert it to get log(p).
                    seq_log_probs = -loss_fct(seq_logits, seq_labels)

                    copy_len = min(actual_len, seq_len)
                    batch_res[b, :copy_len] = seq_log_probs[:copy_len]

            all_log_probs.append(batch_res.cpu().numpy())

            del outputs
            torch.cuda.empty_cache()

    B = len(prompts_msgs)
    final_log_probs = np.concatenate(all_log_probs, axis=0).reshape(B, -1, seq_len)
    return final_log_probs


# ==========================================
# 3. Helper for Scoring (Existing)
# ==========================================
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
    # --- Optimization 1: Enable Unsloth Inference Kernels ---
    FastLanguageModel.for_inference(reward_model)

    device = next(reward_model.parameters()).device

    # 1. Flatten prompts and completions into a single list of strings
    texts = []
    completion_texts = []

    for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
        for c in completions:
            content = c if isinstance(c, str) else c.get("content", "")
            # Build full text
            msgs = p_msgs + [{"role": "assistant", "content": content}]
            texts.append(
                apply_chat_template({"messages": msgs}, reward_tokenizer)["text"]
            )

            # Build completion text for length calculation
            comp_text = content + (reward_tokenizer.eos_token or "")
            completion_texts.append(comp_text)

    if not texts:
        return [[] for _ in prompts_msgs]

    # Calculate global max seq_len strictly for the final output array shape
    # We batch this strictly for CPU side length checking
    global_tokens = reward_tokenizer(
        completion_texts,
        return_attention_mask=True,
        add_special_tokens=False,
        padding=False,
    )
    # The max length of any completion in the dataset (clamped to max_length limit)
    seq_len = min(max(len(t) for t in global_tokens["input_ids"]), max_length)

    new_logits = []

    # --- Optimization 2: Dynamic Batching Loop ---
    for i in range(0, len(texts), micro_batch):
        batch_texts = texts[i : i + micro_batch]
        batch_completion_texts = completion_texts[i : i + micro_batch]

        # Tokenize ONLY this batch with padding=True (Dynamic Padding)
        # This makes the tensor width = length of longest sequence in THIS batch, not 512.
        batch_inputs = reward_tokenizer(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding_side="right",
        ).to(device)

        # Tokenize completions just for length calculations
        batch_completions = reward_tokenizer(
            text=batch_completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.inference_mode():
            # Model Forward Pass
            # logits shape: [micro_batch, dynamic_seq_len] or [micro_batch]
            reward_outputs = reward_model(**batch_inputs)
            reward_logits = reward_outputs.logits.squeeze(-1)

            current_batch_max_len = batch_inputs["input_ids"].shape[1]

            # Handle Non-Dense (Scalar) Rewards
            if not dense_reward:
                # --- Optimization 3: Use expand instead of repeat (Memory View) ---
                reward_logits = reward_logits.unsqueeze(1).expand(
                    -1, current_batch_max_len
                )

            if clip_reward_model:
                reward_logits = torch.clamp(reward_logits, min=reward_lb, max=reward_ub)

            # Calculate indices
            completion_lens = batch_completions["attention_mask"].sum(dim=1).long()
            full_lens = batch_inputs["attention_mask"].sum(dim=1).long()
            start_indices = (full_lens - completion_lens).clamp(min=0)

            # Generate gather indices
            # We must clamp to seq_len because the final output expects fixed width
            gather_indices = (
                start_indices[:, None] + torch.arange(seq_len, device=device)[None, :]
            )

            # Important: Clamp indices to the current batch's dynamic width to avoid out-of-bounds
            gather_indices_safe = gather_indices.clamp(max=current_batch_max_len - 1)

            # Gather
            reward_comp = reward_logits.gather(1, gather_indices_safe)

            # --- Handle Dense Partial Logic (Optional) ---
            if dense_reward in ["partial", "partial_fixed"]:
                # Note: These masks need to be regenerated for the dynamic batch shape
                if dense_reward == "partial":
                    end_of_thought_mask = sentence_boundary_mask(
                        reward_tokenizer,
                        batch_inputs,
                        batch_inputs["attention_mask"],
                        device,
                    )

                    # # --- DEBUGGING VISUALIZATION START ---
                    # # Check only the first sample in the micro-batch
                    # sample_ids = batch_inputs["input_ids"][0]
                    # sample_tokens = reward_tokenizer.convert_ids_to_tokens(sample_ids)
                    # sample_mask = end_of_thought_mask[0]

                    # print("\n" + "="*50)
                    # print("DEBUG: Reward Model Token Alignment")
                    # print(f"BOS Token: {reward_tokenizer.bos_token} (ID: {reward_tokenizer.bos_token_id})")
                    # print("-" * 50)

                    # for idx, (token, is_boundary) in enumerate(zip(sample_tokens, sample_mask)):
                    #     # Only print non-padding tokens for clarity
                    #     if token == reward_tokenizer.pad_token:
                    #         continue
                    #     boundary_marker = " [STEP END] <---" if is_boundary else ""
                    #     print(f"Token {idx:3}: '{token:15}' {boundary_marker}")
                    # print("="*50 + "\n")

                    # # --- DEBUGGING VISUALIZATION END ---

                else:
                    end_of_thought_mask = every_n_tokens_mask(
                        batch_inputs,
                        batch_inputs["attention_mask"],
                        dense_partial_fixed_n,
                    )

                # We need to act carefully here because gather_indices might be wider than the dynamic batch
                # But since we clamped gather_indices_safe, it is valid for gathering.
                end_of_thought_mask = end_of_thought_mask.gather(1, gather_indices_safe)
                reward_comp = backfill_rewards(reward_comp, end_of_thought_mask)

            # Apply NaN mask for padding/invalid
            # import IPython; IPython.embed(); exit()
            output_mask = (
                torch.arange(seq_len, device=device)[None, :] < completion_lens[:, None]
            )
            reward_comp[~output_mask] = float("nan")

            out_cpu = reward_comp.detach().float().cpu()
            new_logits.append(out_cpu)

    B = len(prompts_msgs)
    all_scores = np.concatenate(new_logits, axis=0).reshape(B, -1, seq_len)
    return all_scores


def _resolve_eval_functions(dataset_name: str):
    if dataset_name in {"gsm8k", "gsm8k_kd"} or "aime" in dataset_name:
        reward_fns = [
            ("xmlcount_reward_func", xmlcount_reward_func),
            ("correctness_reward_func", gsm8k_correctness_reward_func),
        ]
        eval_correctness = eval_correctness_gsm8k
    elif dataset_name in {"countdown", "countdown_kd"}:
        reward_fns = [("correctness_reward_func", countdown_correctness_function)]
        eval_correctness = eval_correctness_countdown
    elif dataset_name in {"medreason", "medreason_kd"}:
        reward_fns = [("correctness_reward_func", medical_correctness_reward_func)]
        eval_correctness = eval_correctness_medical
    elif dataset_name in {"science", "science_kd"}:
        reward_fns = [("correctness_reward_func", scienceqa_correctness_reward_func)]
        eval_correctness = eval_correctness_scienceqa
    elif dataset_name in {"mmlu", "mmlu_kd"}:
        reward_fns = [("correctness_reward_func", mmlu_correctness_reward_func)]
        eval_correctness = eval_correctness_mmlu
    elif dataset_name in {"medical", "medical_kd"}:
        reward_fns = [("correctness_reward_func", medical_correctness_reward_func)]
        eval_correctness = eval_correctness_medical
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return reward_fns, eval_correctness


def _cfg_bool(cfg_eval: DictConfig, key: str, default: bool) -> bool:
    value = getattr(cfg_eval, key, None)
    return default if value is None else bool(value)


def _zero_scores_like(completions):
    return [[[0]] * len(c) for c in completions]


@hydra.main(config_path="configs", config_name="config_eval", version_base="1.3")
def main(cfg: DictConfig):
    print("Evaluation configuration:\n", OmegaConf.to_yaml(cfg))

    os.makedirs(cfg.model.name, exist_ok=True)
    config_save_path = os.path.join(cfg.model.name, "evaluation_config.yaml")
    with open(config_save_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Configuration saved to: {config_save_path}")

    set_seed(cfg.seed)

    eval_mode = canonical_eval_mode(getattr(cfg.eval, "mode", MODE_GENERATE))
    use_pregenerated = eval_mode_uses_pregenerated(eval_mode)
    guidance_method = getattr(cfg, "guidance", {}).get("method", "none")
    guidance_requires_reward_model = (
        (not use_pregenerated) and guidance_method in {"topk", "chunk"}
    )
    compute_policy_log_probs = _cfg_bool(
        cfg.eval, "compute_policy_log_probs", use_pregenerated
    )
    default_compute_reward = cfg.airl and eval_mode in {
        MODE_GENERATE,
        MODE_AIME,
        MODE_PREGENERATED_POLICY_AND_REWARD,
    }
    compute_reward_model_scores = _cfg_bool(
        cfg.eval,
        "compute_reward_model_scores",
        default_compute_reward,
    )
    print(
        "Evaluation mode: "
        f"{eval_mode} | pregenerated={use_pregenerated} | "
        f"policy_log_probs={compute_policy_log_probs} | "
        f"reward_model_scores={compute_reward_model_scores}"
    )

    reward_fns, eval_correctness = _resolve_eval_functions(cfg.dataset.name)

    if cfg.eval.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=f"eval_{cfg.wandb.run_name}-cp{cfg.model.checkpoint}",
        )

    no_system = getattr(cfg.dataset, "no_system", False)
    dataset = get_dataset(
        cfg.dataset.name, split=cfg.dataset.split, ratio=1, no_system=no_system
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda examples: examples,
    )

    model = None
    tokenizer = None
    policy_tokenizer = None
    reward_model = None
    reward_tokenizer = None
    lora_req = None

    if cfg.airl:
        model, reward_model, tokenizer, reward_tokenizer = irl_load_model_and_tokenizer(
            cfg, pretrained=True
        )
        model.eval()
        policy_tokenizer = tokenizer

        if compute_reward_model_scores or guidance_requires_reward_model:
            reward_model.eval()
            if guidance_requires_reward_model:
                FastLanguageModel.for_inference(reward_model)
        else:
            del reward_model
            reward_model = None
            torch.cuda.empty_cache()
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        model.eval()
        policy_tokenizer = tokenizer

    if guidance_requires_reward_model and not cfg.airl:
        raise ValueError(
            "Reward-guided generation requires AIRL components. "
            "Set `airl=true` to load reward model/tokenizer."
        )

    if (compute_reward_model_scores or guidance_requires_reward_model) and reward_model is None:
        raise ValueError(
            "Reward model is required (for scoring and/or guidance), but none is loaded. "
            "Set `airl=true` or disable reward-model-dependent features."
        )

    # Generation parameters
    n = cfg.sampling.n_samples
    sampling_params = SamplingParams(
        n=n,
        seed=cfg.seed,
        max_tokens=cfg.model.max_completion_length,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
    )

    if not use_pregenerated:
        if hasattr(model, "load_lora"):
            lora_req = model.load_lora(cfg.model.name, load_tensors=True)

        if guidance_method == "topk" and cfg.airl:
            if not _vllm_supports_request_logits_processors():
                raise ValueError(
                    "guidance.method=topk is unsupported by this vLLM runtime: "
                    "per-request logits processors are unavailable "
                    "(V1 limitation and/or V0 multi-step limitation). "
                    "Use guidance.method=chunk instead."
                )
            print(
                "--- ACTIVATING REWARD-AUGMENTED DECODING "
                f"(Top-K={getattr(cfg.guidance, 'k', 5)}) ---"
            )
            rw_processor = TopKRewardLogitsProcessor(
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                policy_tokenizer=policy_tokenizer,
                alpha=getattr(cfg.guidance, "alpha", 1.0),
                k=getattr(cfg.guidance, "k", 5),
                device=next(reward_model.parameters()).device,
                dense_reward=bool(getattr(cfg.model, "dense_rewards", False)),
                reward_discount_gamma=getattr(cfg.guidance, "dense_reward_gamma", 0.95),
            )
            sampling_params.logits_processors = [rw_processor]
        elif guidance_method == "chunk" and cfg.airl:
            print("--- ACTIVATING CHUNK-LEVEL GUIDANCE ---")
        else:
            print("--- STANDARD GENERATION (No Active Guidance) ---")

    pregenerated_df = None
    if use_pregenerated:
        explicit_jsonl = getattr(cfg.eval, "pregenerated_jsonl_path", None)
        source_dir = getattr(cfg.eval, "pregenerated_source_dir", None)
        candidates = getattr(cfg.eval, "pregenerated_candidates", None)
        candidates = list(candidates) if candidates is not None else None

        pregenerated_jsonl_path = resolve_pregenerated_jsonl_path(
            mode=eval_mode,
            model_name=cfg.model.name,
            policy_name=getattr(cfg.model, "policy_name", None),
            explicit_path=explicit_jsonl,
            source_dir_override=source_dir,
            candidate_filenames=candidates,
        )
        print(f"Loading pregenerated completions from: {pregenerated_jsonl_path}")
        pregenerated_df = pd.read_json(pregenerated_jsonl_path, lines=True)

    # Metrics storage
    all_correct_flags = []
    all_reward_scores = []
    sums = {name: 0.0 for name, _ in reward_fns}
    sum_sqs = {name: 0.0 for name, _ in reward_fns}
    count = 0
    all_results = []

    bs = cfg.eval.per_device_eval_batch_size
    batch_counter = 0
    for batch in tqdm(loader):
        prompts = [b["prompt"] for b in batch]
        answers = [b["answer"] for b in batch]

        if use_pregenerated:
            start = batch_counter * bs * n
            end = start + len(batch) * n
            sub_df = pregenerated_df.iloc[start:end].copy().reset_index(drop=True)
            expected_rows = len(batch) * n
            if len(sub_df) != expected_rows:
                raise ValueError(
                    f"Pregenerated jsonl has {len(sub_df)} rows for batch {batch_counter}, "
                    f"expected {expected_rows}. start={start}, end={end}."
                )
            if len(sub_df) > 0:
                assert (
                    sub_df.iloc[0]["prompt"][1]["content"]
                    == batch[0]["prompt"][1]["content"]
                ), "DataLoader and JSONL are misaligned!"
                assert (
                    sub_df.iloc[-1]["prompt"][1]["content"]
                    == batch[-1]["prompt"][1]["content"]
                ), "DataLoader and JSONL are misaligned!"

            gens = sub_df["generation"].tolist()
            completions = [gens[i : i + n] for i in range(0, len(gens), n)]
            batch_counter += 1
        else:
            prompts_text = [
                maybe_apply_chat_template({"prompt": p}, tokenizer)["prompt"]
                for p in prompts
            ]
            if guidance_method == "chunk" and cfg.airl:
                generated_texts = generate_with_chunk_guidance(
                    model=model,
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                    policy_tokenizer=policy_tokenizer,
                    prompts_text=[p for p in prompts_text for _ in range(n)],
                    sampling_params=sampling_params,
                    step_size=getattr(cfg.guidance, "step_size", 10),
                    n_candidates=getattr(cfg.guidance, "n_candidates", 4),
                    max_tokens=cfg.model.max_completion_length,
                    policy_max_length=cfg.model.max_prompt_length
                    + cfg.model.max_completion_length,
                    rm_max_length=cfg.model.max_prompt_length
                    + cfg.model.max_completion_length,
                    dense_reward=bool(getattr(cfg.model, "dense_rewards", False)),
                    reward_discount_gamma=getattr(cfg.guidance, "dense_reward_gamma", 0.95),
                    lora_req=lora_req,
                )
                grouped_generated_texts = [
                    generated_texts[i * n : (i + 1) * n]
                    for i in range(len(prompts_text))
                ]
                completions = [
                    [{"content": t[len(prompts_text[i]) :]} for t in grouped_generated_texts[i]]
                    for i in range(len(prompts_text))
                ]
            else:
                outputs = model.fast_generate(
                    prompts_text,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    lora_request=lora_req,
                )
                gens = [[out.outputs[i].text for i in range(n)] for out in outputs]
                completions = [[{"content": g[i]} for i in range(n)] for g in gens]

        batch_rewards = []
        for prompt, completion, answer in zip(prompts, completions, answers):
            batch_rewards_list = []
            for c in completion:
                rewards = {}
                for name, fn in reward_fns:
                    rewards[name] = float(
                        np.mean(
                            fn(prompts=[prompt], completions=[[c]], answer=[answer])
                        )
                    )
                batch_rewards_list.append(rewards)
            batch_rewards.append(batch_rewards_list)

        if compute_reward_model_scores:
            batch_scores = score_with_reward_model(
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                prompts_msgs=prompts,
                decoded_per_prompt=completions,
                dense_reward=cfg.model.dense_rewards,
                max_length=cfg.model.max_prompt_length
                + cfg.model.max_completion_length,
                micro_batch=cfg.eval.max_micro_batch,
                clip_reward_model=cfg.model.clip_reward_model,
                reward_lb=cfg.model.reward_lb,
                reward_ub=cfg.model.reward_ub,
                dense_partial_fixed_n=cfg.model.dense_partial_fixed_n,
            )
        else:
            batch_scores = _zero_scores_like(completions)

        if compute_policy_log_probs:
            batch_log_probs = score_with_policy_model(
                policy_model=model,
                policy_tokenizer=policy_tokenizer,
                prompts_msgs=prompts,
                decoded_per_prompt=completions,
                max_length=cfg.model.max_prompt_length
                + cfg.model.max_completion_length,
                micro_batch=cfg.eval.max_micro_batch,
            )
        else:
            batch_log_probs = None

        # Store results
        for p_idx, (prompt, generations, scores, rewards) in enumerate(
            zip(prompts, completions, batch_scores, batch_rewards)
        ):
            log_probs = batch_log_probs[p_idx] if batch_log_probs is not None else None
            for gen_idx, (generation, score, rews) in enumerate(
                zip(generations, scores, rewards)
            ):
                result = {
                    "prompt": prompt,
                    "generation": generation,
                    "generation_idx": gen_idx,
                    "reward_model_score": (
                        score[~np.isnan(score)].tolist()
                        if isinstance(score, np.ndarray) and score.ndim > 0
                        else score
                    ),
                }
                if log_probs is not None:
                    log_prob = log_probs[gen_idx]
                    result["policy_log_probs"] = (
                        log_prob[~np.isnan(log_prob)].tolist()
                        if isinstance(log_prob, np.ndarray) and log_prob.ndim > 0
                        else log_prob
                    )
                result = result | rews
                all_results.append(result)
            all_reward_scores.append(np.nanmean(scores, axis=1).tolist())

        for completion, answer in zip(completions, answers):
            correct_flags = eval_correctness(completions=completion, answer=answer)
            all_correct_flags.append(correct_flags)

            for name, fn in reward_fns:
                batch_score = float(
                    np.mean(
                        fn(prompts=prompts, completions=completions, answer=answers)
                    )
                )
                sums[name] += batch_score
                sum_sqs[name] += batch_score**2
            count += 1

    # Metrics computation
    pass_at_k = compute_pass_at_k(all_correct_flags, cfg.eval.ks)
    reward_weighted_pass_at_k = compute_reward_weighted_pass_at_k_from_scores(
        all_correct_flags, all_reward_scores, cfg.eval.ks
    )
    success_at_k = compute_success_at_k_from_scores(
        all_correct_flags, all_reward_scores, cfg.eval.ks
    )
    oracle_at_1 = compute_oracle_at_1_from_N(all_correct_flags)

    print("\n--- Final metrics ---")
    for k, v in pass_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/pass@{k}": v})
        print(f"pass@{k}: {v:.4f}")

    for k, v in reward_weighted_pass_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/reward_weighted_pass@{k}|N={n}": v})
        print(f"reward_weighted_pass@{k}|N={n}: {v:.4f}")

    for k, v in success_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/success@{k}|N={n}": v})
        print(f"success@{k}|N={n}: {v:.4f}")

    if cfg.eval.report_to == "wandb":
        wandb.log({"test/oracle@1|N": oracle_at_1})
    print(f"oracle@1|N={n}: {oracle_at_1:.4f}")

    metrics_mean = {
        f"test/rewards/{name}/mean": sums[name] / count for name, _ in reward_fns
    }
    metrics_std = {
        f"test/rewards/{name}/std": np.sqrt(
            (sum_sqs[name] / count) - (sums[name] / count) ** 2
        )
        for name, _ in reward_fns
    }

    metrics = {**metrics_mean, **metrics_std}
    print("\n--- Final Rewards ---")
    if cfg.eval.report_to == "wandb":
        wandb.log(metrics)
    for name, _ in reward_fns:
        print(
            f"{name} mean: {metrics[f'test/rewards/{name}/mean']:.2f}, "
            f"std: {metrics[f'test/rewards/{name}/std']:.2f}"
        )

    output_override = getattr(cfg.eval, "output_file", None)
    if output_override:
        output_file = output_override
    else:
        output_file = (
            f"{cfg.model.name}/{default_output_filename(eval_mode, cfg.dataset.name)}"
        )
    save_results_to_jsonl(output_file, all_results)
    print(f"\nSaved evaluation results to {output_file}")


if __name__ == "__main__":
    main()
