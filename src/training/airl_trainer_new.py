"""
AIRLTrainer – Adversarial Inverse‑RL trainer compatible with Hugging Face TRL
-----------------------------------------------------------------------------
This implementation mixes the Group‑Relative Policy Optimisation (GRPO) ideas
already present in `trl.GRPOTrainer` with the adversarial inverse‑reinforcement–
learning formulation of AIRL (Fu et al., ICLR 2018).

Key features
============
• **Two models** – a policy (`AutoModelForCausalLM`) *and* a discriminator / reward
model (`AutoModelForSequenceClassification` with `num_labels=1`).
• **Joint optimisation**  – in each gradient‑accumulation cycle we
1️⃣ generate *K* candidate reasoning traces with the policy,
2️⃣ update the reward model to classify expert vs. policy traces (binary‑CE),
3️⃣ compute shaped rewards  \( r̂ = \log D − \log (1−D) \) for the same batch and
    apply GRPO on the policy.
• **Expert demonstrations** – pass a separate `expert_dataset` (or a
column in the main dataset marked with `"is_expert"`). These traces can come
from a larger teacher model (knowledge‑distillation setting).
• **Drop‑in replacement** – follows the high‑level API of
`trl.GRPOTrainer`; accepts the same `GRPOConfig`, supports PEFT / DeepSpeed /
FSDP, wandb logging, etc.
"""

from __future__ import annotations

# Standard library imports
from collections import defaultdict
import math
from typing import Any, Dict, List, Optional, Union, Callable
import os

# Third-party imports
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset

from accelerate import logging
from accelerate.utils import gather, gather_object
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    Trainer,
    is_wandb_available
)
from trl import GRPOTrainer
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available
from trl.data_utils import apply_chat_template, is_conversational
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available, is_liger_kernel_available
from trl.trainer.grpo_trainer import nanstd, nanmin, nanmax
from trl.trainer.utils import (
    pad,
)
import copy
from src.training.reward_model_utils import (
    tokenize_examples,
    dedup_token_batch,
    prepare_reward_batch,
)
from tqdm import tqdm

# Add this import for threadpool parallelism

# Local imports
from src.config.irl_config import IRLConfig

# Conditional imports
if is_vllm_available():
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


# Type aliases ---------------------------------------------------------------
RewardModelType = Union[str, PreTrainedModel]
PolicyModelType = Union[str, PreTrainedModel]

if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb


logger = logging.get_logger(__name__)

def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given prompt tensor, it returns all the left padded tokens in that sequence. so [pad, pad, pad, cat] = 3 tokens
    """
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")

    prompt_section = input_ids[:, :-logits_to_keep]
    padding_mask = (prompt_section == pad_token_id)
    pad_token_counts = padding_mask.sum(dim=1)
    return pad_token_counts

def create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given that we have a sequence, [p,p,p,c,c,c,pad,pad,pad]

    Where p are extra prompt tokens we got from slicing the torch tensor, c is completion tokens
    and pad are pad tokens, this function would make a completion mask that would 0 out the pad
    and p tokens. so in this example [0,0,0,1,1,1,0,0,0]
    """
    batch_size, completion_len = completion_input_ids.shape
    device = completion_input_ids.device
    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt
    indices = torch.arange(completion_len, device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)
    non_padding_mask = (completion_input_ids != pad_token_id)
    final_mask = shift_mask & non_padding_mask
    return final_mask

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="none"):
    """
    inputs: logits
    targets: labels (0 or 1)
    """
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = torch.exp(-bce_loss) # Prevents nans
    focal_term = (1 - pt) ** gamma
    
    # Alpha balancing (optional, but good for imbalanced SFT vs Expert batches)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * focal_term * bce_loss
    else:
        loss = focal_term * bce_loss
        
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def pad_to_attention_layout(x: torch.Tensor, new_mask: torch.Tensor, pad_value: float) -> torch.Tensor:
    """
    Args:
        x: [B, L] tensor (right-padded)
        new_mask: [B, L + C] tensor (mixed left/right padding)
        pad_value: Value to fill padding with (default 0)
    """
    B, L = x.shape
    total_len = new_mask.shape[1]
    if L > total_len:
        x = x[:, :total_len]
        L = total_len
        
    C = total_len - L
    new_x = torch.full((B, total_len), fill_value=pad_value, dtype=x.dtype, device=x.device)
    is_left_padded = new_mask[:, 0] == 0

    # If left padded (standard HF), data goes to the right (C:)
    if is_left_padded.any():
        new_x[is_left_padded, C:] = x[is_left_padded]
        
    # If right padded (unsloth/packed), data goes to the left (:L)
    if (~is_left_padded).any():
        new_x[~is_left_padded, :L] = x[~is_left_padded]
        
    return new_x


def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Moves all padding tokens in each sequence of a batch to the right.
    """
    mask = (tensor != pad_id)
    # Must do stable=True since binary mark is unordered
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor


def extract_expert_targets(self, inputs: list[dict]) -> list[Optional[str]]:
    """Try a few common keys; returns None when no expert target exists."""
    out: list[Optional[str]] = []
    for ex in inputs:
        t = None
        if "target" in ex and ex["target"] is not None:
            t = ex["target"]
        out.append(t)
    return out


def build_texts(prompts: list, completions: list, reward_tok, is_chat: bool) -> list[str]:
    """Build discriminator inputs exactly like your reward path (chat-template or plain concat)."""
    if is_chat:
        full_messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
        return [apply_chat_template(x, reward_tok)["text"] for x in full_messages]
    else:
        return [p + c for p, c in zip(prompts, completions)]

# ---------------------------------------------------------------------------
class AIRLTrainer(GRPOTrainer):
    """Adversarial IRL trainer using the AIRL discriminator‑style reward.

    Parameters
    ----------
    policy_model:
        Causal‑LM that is optimised with GRPO.
    reward_model:
        Sequence‑classification model *or* HF Hub id. Final layer must output a
        single logit per sequence. The trainer applies a sigmoid internally.
    expert_dataset:
        Dataset of expert reasoning traces (dicts with *prompt* & *completion*)
        used as positive examples for the discriminator.
    args:
        Standard `GRPOConfig` (inherits from `transformers.TrainingArguments`).
    reward_tokenizer:
        Optional tokenizer to use for the reward model. If *None* we load it
        from `reward_model` (if string) or fall back to `policy_tokenizer`.
    callbacks, optimizers, peft_config:
        Forwarded to the base Trainer.
    """

    _tag_names = ["trl", "airl"]

    def __init__(
        self,
        policy_model: PolicyModelType,
        reward_model: RewardModelType,
        args: IRLConfig,
        reward_funcs: Optional[List[Callable]] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        policy_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        reward_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers=(None, None),
    ) -> None:

        self.reward_model = reward_model
        # Tokenizers --------------------------------------------------------------------
        self.policy_tokenizer = policy_tokenizer
        self.reward_tokenizer = reward_tokenizer

        # AIRL specific arguments
        self.use_outcome_rewards = args.use_outcome_rewards
        self.reward_updates_per_policy_step = args.reward_updates_per_policy_step
        self.max_micro_batch = args.max_micro_batch

        # Internal buffers --------------------------------------------------------------
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # Reward functions --------------------------------------------------------------
        self.dense_rewards = args.dense_rewards
        self.dense_gamma = args.dense_gamma
        self.advantage_calculation = args.advantage_calculation
        self.add_expert_to_policy_optim = args.add_expert_to_policy_optim
        self.add_expert_to_policy_balanced = args.add_expert_to_policy_balanced
        self.classifier_loss = args.classifier_loss
        self.normalise_rewards = args.normalise_rewards
        self.expert_error_rate = args.expert_error_rate

        if reward_funcs is None:
            reward_funcs = [self.reward_model]
        if not isinstance(reward_funcs, list):
            reward_funcs = [self.reward_model, reward_funcs]
        else:
            reward_funcs = [self.reward_model] + reward_funcs

        # Reward processing class --------------------------------------------------------------
        if reward_processing_classes is None:
            reward_processing_classes = [self.reward_tokenizer] + [None] * (len(reward_funcs) - 1)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [self.reward_tokenizer, reward_processing_classes]
        else:
            reward_processing_classes = [self.reward_tokenizer] + reward_processing_classes
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        # ---- Prepare backend  ---------------------------------------------------------
        super().__init__(
            model=policy_model,  # base Trainer still expects a *single* model; we wrap reward manually
            args=args,
            reward_funcs=reward_funcs,
            reward_processing_classes=reward_processing_classes,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.policy_tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reward optimiser (separate from policy) --------------------------------------
        tmp_args = copy.copy(args)
        opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(tmp_args)
        opt_kwargs["lr"] = args.reward_learning_rate
        opt_kwargs["weight_decay"] = getattr(args, "reward_weight_decay", opt_kwargs.get("weight_decay", 0.0))
        self.reward_optimizer = opt_cls(self.reward_model.parameters(), **opt_kwargs)
        self.reward_optimizer.zero_grad()
        self.reward_model.to(self.accelerator.device)  # harmless if already on device
        self.reward_model, self.reward_optimizer = self.accelerator.prepare(
            self.reward_model, self.reward_optimizer
        )

        if not self.use_outcome_rewards: # Only the reward model is used for training
            self.reward_weights = torch.zeros_like(self.reward_weights, dtype=torch.float32)
            self.reward_weights[0] = 1.0  

        self.eps = getattr(args, "disc_label_smoothing", 0.0)
        self.disc_temperature = getattr(args, "disc_temperature", 1.0)
        self.clip_reward_model = getattr(args, "clip_reward_model", False)
        self.reward_lb = getattr(args, "reward_lb", -10.0)
        self.reward_ub = getattr(args, "reward_ub", 10.0)
        self.response_only = getattr(args, "response_only", False)

        # Negatives from perturbed expert reasonings
        self.neg_perturb_fns = getattr(args, "neg_perturb_fns", None)  # List[Callable[[str], str]] or None
        self.num_neg_perturbations_per_expert = getattr(args, "num_neg_perturbations_per_expert", 1)
        self.neg_sample_weight = getattr(args, "neg_sample_weight", 1.0)  # weight in BCE
        self.disc_pairwise_margin = getattr(args, "disc_pairwise_margin", 0.0)  # >0 to enable pairwise hinge
        self.neg_label_smoothing = getattr(args, "neg_label_smoothing", None)  # defaults to self.eps if None
        
        self.num_generations_with_expert = self.num_generations + 1 if self.add_expert_to_policy_optim else self.num_generations
        self.reward_warmup_steps = self.args.reward_warmup_steps
        self.warmup_done = False
        self.max_length = self.args.max_prompt_length + self.args.max_completion_length

    # -----------------------------------------------------------------------
    # Core overwrites overrides
    # -----------------------------------------------------------------------$
    def train(self, *args, **kwargs):
        if self.reward_warmup_steps > 0 and not self.warmup_done:
            self._warmup_discriminator()
            self.warmup_done = True
        return super().train(*args, **kwargs)


    @profiling_decorator
    def _update_reward_model_step(
        self,
        neg_prompts: list[Dict[str: str]],
        neg_completions: list[str],
        pos_prompts: list[Dict[str: str]],
        pos_completions: list[str],
        *,
        do_step: bool = True,
        log_prefix: str = "disc",
        is_chat: bool = False,
    ) -> dict[str, float]:
        """
        One discriminator update (or forward-only if do_step=False).

        Typical regime:
        len(pos_*) = B
        len(neg_*) = B * K  (K = self.num_generations)
        We do NOT repeat positives. Instead we balance BCE with weights.

        Pairwise margin (optional) is computed per-prompt by reshaping negatives to (B, K)
        and comparing each positive to an aggregate negative (mean/max).
        """
        device = self.accelerator.device

        B = len(pos_completions)
        if B == 0:
            raise ValueError("Empty pos_completions in _update_reward_model_step.")
        if len(pos_prompts) != B:
            raise ValueError(f"pos_prompts length mismatch: {len(pos_prompts)=} vs {B=}.")

        N = len(neg_completions)
        if N == 0:
            raise ValueError("Empty neg_completions in _update_reward_model_step.")
        if N % B != 0:
            raise ValueError(f"Expected len(neg_completions) to be a multiple of B. Got {N=} and {B=}.")
        K = N // B
        
        pos_texts = build_texts(pos_prompts, pos_completions, self.reward_tokenizer, is_chat=is_chat)
        neg_texts = build_texts(neg_prompts, neg_completions, self.reward_tokenizer, is_chat=is_chat)
        pos_w, neg_w = K, 1.0

        # ---- Microbatch sizing:
        ga = int(getattr(self.args, "gradient_accumulation_steps", 1))
        total = B + N
        default_micro = math.ceil(total / max(1, ga))
        micro_bs = int(getattr(self, "max_micro_batch", default_micro))
        micro_bs = max(1, micro_bs)

        # ----- Accumulate losses as *sums* (no graph) for logging
        logits_pos_all = []
        logits_neg_all = []

        if do_step:
            self.reward_optimizer.zero_grad()

        def _tok(texts: list[str]):
            return self.reward_tokenizer(
                text=texts, return_tensors="pt", padding="max_length", padding_side="right",
                add_special_tokens=False,truncation=True, max_length=self.max_length,
            ).to(device)
            
        def _prompt_only_texts(prompts_chunk):
            empty = [[{"role": "assistant", "content": ""}]] * len(prompts_chunk)
            return build_texts(prompts_chunk, empty, self.reward_tokenizer, is_chat=is_chat)

        def _completion_mask(full_batch, prompts_chunk):
            """
            full_batch: tokenised prompt+completion
            prompts_chunk: corresponding prompts (same length)
            Returns bool mask [bs, L] that is True ONLY for completion tokens (excludes prompt + pad).
            """
            prompt_batch = _tok(_prompt_only_texts(prompts_chunk))
            prompt_lens = prompt_batch["attention_mask"].sum(dim=1)          # [bs]
            attn = full_batch["attention_mask"].bool()                       # [bs, L]
            L = attn.size(1)
            idx = torch.arange(L, device=device).unsqueeze(0)                # [1, L]
            return attn & (idx >= prompt_lens.unsqueeze(1))                  # [bs, L]
        
        def _sentence_boundary_mask(full_batch, prompts_chunk, base_completion_mask):
            """
            Returns mask [bs, L] that is True ONLY at sentence boundaries within completions.
            Sentence boundaries are: '.', '\n', or '.\n' sequences.
            """
            input_ids = full_batch["input_ids"]  # [bs, L]
            bs, L = input_ids.shape
            
            # Get token IDs for sentence boundaries
            period_ids = self.reward_tokenizer.encode(".", add_special_tokens=False)
            newline_ids = self.reward_tokenizer.encode("\n", add_special_tokens=False)
            period_id = period_ids[0] if period_ids else -1
            newline_id = newline_ids[0] if newline_ids else -1
            
            # Vectorized: find all periods and newlines in completion region
            is_period = (input_ids == period_id) & base_completion_mask  # [bs, L]
            is_newline = (input_ids == newline_id) & base_completion_mask  # [bs, L]
            
            # Detect .\n pattern: period followed by newline
            is_newline_after_period = is_period[:, :-1] & is_newline[:, 1:]  # [bs, L-1]
            is_newline_after_period = F.pad(is_newline_after_period, (1, 0), value=False)  # [bs, L]
            
            # Mark boundaries: newlines OR (periods that aren't followed by newlines)
            boundary_mask = is_newline | (is_period & ~is_newline_after_period)
            boundary_mask = torch.ones_like(boundary_mask) # TODO: ATTNETION, make modular
            
            # Always include last completion token
            last_indices = base_completion_mask.long().cumsum(dim=1).argmax(dim=1)  # [bs]
            boundary_mask[torch.arange(bs, device=device), last_indices] |= base_completion_mask.any(dim=1)
            
            return boundary_mask

        # ------------------------------------------------------------------
        # PASS 1: count total valid completion tokens (only matters if dense)
        # -----------------------------------------------------------------
        if self.dense_rewards:
            T_pos = torch.tensor(0.0, device=device)
            T_neg = torch.tensor(0.0, device=device)

            for i in range(0, B, micro_bs):
                j = min(i + micro_bs, B)
                batch_pos = _tok(pos_texts[i:j])
                base_mask = _completion_mask(batch_pos, pos_prompts[i:j])
                mask = _sentence_boundary_mask(batch_pos, pos_prompts[i:j], base_mask)
                T_pos += mask.sum().to(T_pos.dtype)

            for i in range(0, N, micro_bs):
                j = min(i + micro_bs, N)
                batch_neg = _tok(neg_texts[i:j])
                base_mask = _completion_mask(batch_neg, neg_prompts[i:j])
                mask = _sentence_boundary_mask(batch_neg, neg_prompts[i:j], base_mask)
                T_neg += mask.sum().to(T_neg.dtype)

        # ------------------------------------------------------------------
        # PASS 2: forward/backward microbatched with EXACT global denominators
        # ------------------------------------------------------------------
        if do_step:
            self.reward_optimizer.zero_grad()

        # Logging accumulators (exact means)
        pos_sum = torch.tensor(0.0, device=device)
        neg_sum = torch.tensor(0.0, device=device)
        pos_cnt = torch.tensor(0.0, device=device)  # either B (seq) or T_pos (dense) depending on logits dim
        neg_cnt = torch.tensor(0.0, device=device)

        logits_pos_all = []
        logits_neg_all = []

        # ---- POS
        for i in range(0, B, micro_bs):
            j = min(i + micro_bs, B)
            batch_pos = _tok(pos_texts[i:j])
            logits_pos = self.reward_model(**batch_pos).logits[..., 0]
            y_pos = torch.rand_like(logits_pos) * self.eps + (1.0 - self.eps)

            if self.dense_rewards:
                base_mask = _completion_mask(batch_pos, pos_prompts[i:j])
                mask = _sentence_boundary_mask(batch_pos, pos_prompts[i:j], base_mask)
                
                # No ramp weighting for sentence boundaries - treat all equally
                loss_elt = focal_loss(logits_pos, y_pos, gamma=2.0, reduction="none")
                loss_sum = (loss_elt * mask.to(loss_elt.dtype)).sum()
                cnt = mask.sum().to(loss_sum.dtype).clamp_min(1)

                pos_sum += loss_sum.detach()
                pos_cnt += cnt.detach()
                logits_pos_all.append(logits_pos.detach()[mask])
                denom = T_pos.clamp_min(1.0)
            else:
                loss_sum = F.binary_cross_entropy_with_logits(logits_pos, y_pos, reduction="sum")
                cnt = torch.tensor(float(logits_pos.numel()), device=device)

                pos_sum += loss_sum.detach()
                pos_cnt += cnt.detach()
                logits_pos_all.append(logits_pos.detach().view(-1))
                denom = torch.tensor(float(B), device=device)

            if do_step:
                scale = (pos_w / (pos_w + neg_w)) * (1.0 / denom)
                self.accelerator.backward(loss_sum * scale)

        # ---- NEG
        for i in range(0, N, micro_bs):
            j = min(i + micro_bs, N)
            batch_neg = _tok(neg_texts[i:j])
            logits_neg = self.reward_model(**batch_neg).logits[..., 0]
            y_neg = torch.rand_like(logits_neg) * self.eps
            
            if self.dense_rewards:
                base_mask = _completion_mask(batch_neg, neg_prompts[i:j])
                mask = _sentence_boundary_mask(batch_neg, neg_prompts[i:j], base_mask)
                
                # No ramp weighting for sentence boundaries - treat all equally
                loss_elt = focal_loss(logits_neg, y_neg, gamma=2.0, reduction="none")
                loss_sum = (loss_elt * mask.to(loss_elt.dtype)).sum()
                cnt = mask.sum().to(loss_sum.dtype).clamp_min(1)

                neg_sum += loss_sum.detach()
                neg_cnt += cnt.detach()
                logits_neg_all.append(logits_neg.detach()[mask])
                denom = T_neg.clamp_min(1.0)
            else:
                loss_sum = F.binary_cross_entropy_with_logits(logits_neg, y_neg, reduction="sum")
                cnt = torch.tensor(float(logits_neg.numel()), device=device)

                neg_sum += loss_sum.detach()
                neg_cnt += cnt.detach()
                logits_neg_all.append(logits_neg.detach().view(-1))
                denom = torch.tensor(float(N), device=device)

            if do_step:
                scale = (neg_w / (pos_w + neg_w)) * (1.0 / denom)
                self.accelerator.backward(loss_sum * scale)

        if do_step:
            reward_max_grad_norm = getattr(self.args, "max_grad_norm", None)
            if reward_max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(self.reward_model.parameters(), float(reward_max_grad_norm))
            self.reward_optimizer.step()

        # Metrics (exact means + accuracies)
        bce_pos = (pos_sum / pos_cnt.clamp_min(1.0))
        bce_neg = (neg_sum / neg_cnt.clamp_min(1.0))
        loss = (pos_w * bce_pos + neg_w * bce_neg) / (pos_w + neg_w)

        logits_pos_det = torch.cat(logits_pos_all, dim=0)
        logits_neg_det = torch.cat(logits_neg_all, dim=0)

        p_pos = torch.sigmoid(logits_pos_det)
        p_neg = torch.sigmoid(logits_neg_det)
        acc_pos = (p_pos >= 0.5).float().mean()
        acc_neg = (p_neg < 0.5).float().mean()
        acc = p_pos.size(0) / (p_pos.size(0) + p_neg.size(0)) * acc_pos + p_neg.size(0) / (p_pos.size(0) + p_neg.size(0)) * acc_neg

        def gmean(x: torch.Tensor) -> float:
            return self.accelerator.gather(x.detach()).mean().item()

        return {
            f"{log_prefix}/loss": gmean(loss),
            f"{log_prefix}/bce_pos": gmean(bce_pos),
            f"{log_prefix}/bce_neg": gmean(bce_neg),
            f"{log_prefix}/acc": gmean(acc),
            f"{log_prefix}/acc_pos": gmean(acc_pos),
            f"{log_prefix}/acc_neg": gmean(acc_neg),
            f"{log_prefix}/logit_pos_mean": gmean(logits_pos_det.mean()),
            f"{log_prefix}/logit_neg_mean": gmean(logits_neg_det.mean()),
            f"{log_prefix}/prob_pos_mean": gmean(p_pos.mean()),
            f"{log_prefix}/prob_neg_mean": gmean(p_neg.mean()),
        }


    def _warmup_discriminator(self):
        if self.reward_warmup_steps <= 0:
            return

        policy_was_training = self.model.training
        reward_was_training = self.reward_model.training
        self.model.eval()
        self.reward_model.train()

        dataloader = self.get_train_dataloader()
        iterator = iter(dataloader)

        for step_idx in tqdm(range(self.reward_warmup_steps), desc="Warming up Reward Model"):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            neg_prompts = [x["prompt"] for x in batch]

            images = None
            if "images" in batch[0]:
                images = [x.get("images") for x in batch]
            elif "image" in batch[0]:
                images = [[x.get("image")] if x.get("image") is not None else None for x in batch]
            if images is not None and all(img_list == [] for img_list in images):
                images = None

            # Efficient negatives (uses vLLM if GRPOTrainer configured that way)
            with torch.no_grad():
                _, completion_ids_list, _, _, _ = self._generate(neg_prompts, images)

            neg_completions = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)
            neg_completions = [[{"role": "assistant", "content": c}] for c in neg_completions]
            pos_prompts = [batch[i]["prompt"] for i in range(0, len(batch), self.num_generations)]
            pos_completions = [batch[i]["target"] for i in range(0, len(batch), self.num_generations)]
            pos_completions = [[{"role": "assistant", "content": c + self.reward_tokenizer.eos_token}] for c in pos_completions]
            
            is_chat = is_conversational(batch[0])
            metrics = self._update_reward_model_step(
                neg_prompts, neg_completions, pos_prompts, pos_completions, do_step=True, log_prefix="reward_warmup", is_chat=is_chat
            )

            if (step_idx + 1) % self.args.logging_steps == 0 or (step_idx + 1) == self.reward_warmup_steps:
                metrics["reward_warmup/step"] = step_idx + 1
                self.log(metrics)
                if self.accelerator.is_main_process:
                    logger.info(
                        f"[reward warmup] step {step_idx+1}/{self.reward_warmup_steps} | "
                        f"loss={metrics['reward_warmup/loss']:.4f} "
                        f"acc={metrics['reward_warmup/acc']:.2f} | POS: "
                        f"loss={metrics['reward_warmup/bce_pos']:.4f} "
                        f"acc={metrics['reward_warmup/acc_pos']:.2f} | NEG: "
                        f"loss={metrics['reward_warmup/bce_neg']:.4f} "
                        f"acc={metrics['reward_warmup/acc_neg']:.2f}"
                    )

        if policy_was_training: self.model.train()
        if not reward_was_training: self.reward_model.eval()


    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        
        # 1. ESTABLISH GROUND TRUTH (The Policy's Output)
        # We trust the Policy's output lengths as the absolute truth.
        completion_lens = torch.tensor([len(x) for x in completion_ids_list], device=device, dtype=torch.long)
        seq_len = completion_lens.max().item() if self.dense_rewards else 1
        
        # Shape: (Batch, Num_Funcs, Seq_Len)
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), seq_len, device=device)

        # 2. PREPARE KWARGS (For outcome rewards)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        # 3. GLOBAL MASK (Valid for all branches)
        # True where indices < completion_len, False where padding.
        if self.dense_rewards:
            output_mask = torch.arange(seq_len, device=device)[None, :] < completion_lens[:, None]

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                
                # === BRANCH A: NEURAL REWARD MODEL ===
                if isinstance(reward_func, nn.Module):
                    # OPTIMIZATION: We no longer need 'prompt_texts'. We only need the full conversation.
                    full_texts = build_texts(prompts, completions, reward_processing_class, is_conversational(inputs[0]))

                    # Tokenize (Padding side = Right) ATTENTION always right padded
                    reward_inputs = reward_processing_class(
                        text=full_texts, return_tensors="pt", padding="max_length", padding_side="right", 
                        add_special_tokens=False, truncation=True, max_length=self.max_length
                    ).to(device)

                    with torch.inference_mode():
                        if self.dense_rewards:
                            # logits: (B, Full_Len)
                            reward_logits = reward_func(**reward_inputs).logits[:, :, 0] / self.disc_temperature
                            if self.clip_reward_model:
                                reward_logits = torch.clamp(reward_logits, self.reward_lb, self.reward_ub)
                            full_lens = reward_inputs["attention_mask"].sum(dim=1).long()
                            start_indices = (full_lens - completion_lens).clamp(min=0)
                            gather_indices = start_indices[:, None] + torch.arange(seq_len, device=device)[None, :]
                            gather_indices = gather_indices.clamp(max=reward_logits.size(1) - 1)
                            reward_comp = reward_logits.gather(1, gather_indices)
                            reward_comp[~output_mask] = float('nan')
                            rewards_per_func[:, i, :] = reward_comp
                        else:
                            # Sequence-level reward (last token)
                            reward_val = reward_func(**reward_inputs).logits[:, 0] / self.disc_temperature
                            if self.clip_reward_model:
                                reward_val = torch.clamp(reward_val, self.reward_lb, self.reward_ub)
                            rewards_per_func[:, i, 0] = reward_val

                # === BRANCH B: OUTCOME / VERIFIABLE REWARD ===
                else:
                    output_rewards = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output_rewards = [r if r is not None else torch.nan for r in output_rewards]
                    output_tensor = torch.tensor(output_rewards, dtype=torch.float32, device=device)

                    if self.dense_rewards:
                        # Expand (B) -> (B, L)
                        expanded_rewards = output_tensor.unsqueeze(1).repeat(1, seq_len)
                        masked_rewards = expanded_rewards.clone()
                        masked_rewards[~output_mask] = float('nan')
                        rewards_per_func[:, i, :] = masked_rewards
                    else:
                        rewards_per_func[:, i, 0] = output_tensor

        if not self.dense_rewards:
            rewards_per_func = rewards_per_func.squeeze(-1)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func


    def _advantage_calculation(self, rewards_per_func, device, prompts):
        """
        Calculates the advantages, normalizing the rewards per group. Handles sparse and dense rewards
        """
        # Apply weights to each reward function's output and sum
        if self.dense_rewards and self.advantage_calculation == "discounted_dense":
            B, F, L = rewards_per_func.shape
            weights = self.reward_weights.to(device)

            # Outcome rewards first (standard GRPO style)
            if weights[1:].any():
                outcome_rewards = (rewards_per_func[:, 1:, :] * weights[1:].view(1, -1, 1)).sum(dim=1)  # Shape: (B, L)
                outcome_rewards = outcome_rewards.nanmean(dim=1) # (B)
                mean_grouped_sparse_rewards = outcome_rewards.view(-1, self.num_generations_with_expert).mean(dim=1)
                mean_grouped_sparse_rewards = mean_grouped_sparse_rewards.repeat_interleave(self.num_generations_with_expert, dim=0)
                outcome_advantages = outcome_rewards - mean_grouped_sparse_rewards
                std_rewards_sparse = outcome_rewards.view(-1, self.num_generations_with_expert).std(dim=1)
                std_rewards_sparse = std_rewards_sparse.repeat_interleave(self.num_generations_with_expert, dim=0)
                is_std_zero_sparse = torch.isclose(std_rewards_sparse, torch.zeros_like(std_rewards_sparse))
                outcome_advantages = outcome_advantages / (std_rewards_sparse + 1e-4)
            else:
                outcome_advantages = torch.zeros(B, device=device)

            # Dense discounted rewards
            # Construct a matrix M where M[k, t] = gamma^(k-t) for k >= t
            dense_rewards = rewards_per_func[:, 0, :] * weights[0]  # Shape: (B, L)
            mean_rewards_dense = dense_rewards.nanmean(1)
            mean_grouped_dense_rewards = mean_rewards_dense.view(-1, self.num_generations_with_expert).mean(dim=1) 
            mean_grouped_dense_rewards = mean_grouped_dense_rewards.repeat_interleave(self.num_generations_with_expert, dim=0)
            std_rewards_dense = mean_rewards_dense.view(-1, self.num_generations_with_expert).std(dim=1)
            std_rewards_dense = std_rewards_dense.repeat_interleave(self.num_generations_with_expert, dim=0)
            is_std_zero_dense = torch.isclose(std_rewards_dense, torch.zeros_like(std_rewards_dense))
            # Normalise dense rewards
            dense_advantages = (dense_rewards - mean_grouped_dense_rewards.unsqueeze(1)) / (std_rewards_dense.unsqueeze(1) + 1e-4)
            
            indices = torch.arange(L, device=device)
            diff = indices.view(-1, 1) - indices.view(1, -1)  # (L, 1) - (1, L) -> Matrix of (k - t)
            mask = diff >= 0

            # Use clamp(min=0) to ensure numerical stability for the upper triangle before masking
            discount_matrix = (self.dense_gamma ** diff.clamp(min=0)) * mask
            efficiency_mask = discount_matrix < 1e-4  # Shape: (L, L)
            discount_matrix = discount_matrix.masked_fill(efficiency_mask, 0.0)
            # Apply discounting: (B, L) @ (L, L) -> (B, L)
            nan_mask = torch.isnan(dense_advantages)
            dense_advantages = dense_advantages.nan_to_num(0.0)  # Replace NaNs with 0 for matrix multiplication
            discounted_dense_advantages = torch.matmul(dense_advantages, discount_matrix.to(dense_advantages.dtype))
            discounted_dense_advantages = discounted_dense_advantages.masked_fill(nan_mask, float('nan'))
            # 3. Combine
            advantages = outcome_advantages.unsqueeze(1) + discounted_dense_advantages  # Shape: (B, L)
            #advantages = advantages.nan_to_num(0.0)[:, :-1]
            all_process_advantages = advantages.nanmean(dim=1)  #(B)
            if weights[1:].any():
                mean_grouped_rewards = (mean_grouped_sparse_rewards + mean_grouped_dense_rewards) / 2
                std_rewards =  (std_rewards_sparse + std_rewards_dense) / 2
                is_std_zero = is_std_zero_sparse | is_std_zero_dense
            else:
                mean_grouped_rewards = mean_grouped_dense_rewards
                std_rewards = std_rewards_dense
                is_std_zero = is_std_zero_dense
            
        else:
            if self.dense_rewards and self.advantage_calculation == "average_dense":
                rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0).unsqueeze(2)).sum(dim=1) #(B, L)
                rewards = rewards.nanmean(dim=1)  #(B)
            else:  
                rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1) #(B)
            mean_grouped_rewards = rewards.view(-1, self.num_generations_with_expert).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations_with_expert, dim=0)
            advantages = rewards - mean_grouped_rewards

            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll still log group level std
                std_rewards = rewards.view(-1, self.num_generations_with_expert).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(self.num_generations_with_expert, dim=0)
            elif self.scale_rewards == "batch":
                # Compute global std
                std_rewards = rewards.std().expand_as(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
                
            all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        advantages = advantages[process_slice]

        return advantages, all_process_advantages, mean_grouped_rewards, std_rewards, is_std_zero


    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(prompts, images)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        # Add expert demonstrations to the batch if specified
        if self.add_expert_to_policy_optim:
            B, max_completion_length = completion_ids.shape
            expert_completions = [inputs[i]["target"] + self.processing_class.eos_token for i in range(0, B, self.num_generations)]
            expert_tokens = self.processing_class(
                text=expert_completions, return_tensors="pt", padding="max_length", padding_side="right", 
                add_special_tokens=False, max_length=max_completion_length, truncation=True
            )
            raw_expert_ids = self.processing_class(text=expert_completions, add_special_tokens=False).input_ids
            expert_completion_ids_list = [ids[:max_completion_length] for ids in raw_expert_ids]
            expert_completion_ids = expert_tokens["input_ids"].to(completion_ids.device).unsqueeze(1)
            expert_completion_mask = expert_tokens["attention_mask"].to(completion_ids.device).unsqueeze(1)
            
            # Add expert completions to the completion_ids and completion_mask
            completion_ids = completion_ids.view(-1, self.num_generations, max_completion_length)
            completion_mask = completion_mask.view(-1, self.num_generations, max_completion_length)
            completion_ids = torch.cat([completion_ids, expert_completion_ids], dim=1).view(-1, max_completion_length)
            completion_mask = torch.cat([completion_mask, expert_completion_mask], dim=1).view(-1, max_completion_length)
            
            # repeat the prompts accordingly
            max_prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_ids.view(-1, self.num_generations, max_prompt_length)
            prompt_mask = prompt_mask.view(-1, self.num_generations, max_prompt_length)
            prompt_ids = torch.cat([prompt_ids, prompt_ids[:,[0],:]], dim=1).view(-1, max_prompt_length)
            prompt_mask = torch.cat([prompt_mask, prompt_mask[:,[0],:]], dim=1).view(-1, max_prompt_length)
            
            # Same thing for prompts and inputs
            N = self.num_generations
            prompts = [x for i in range(0, B, N) for x in prompts[i:i+N] + [prompts[i]]]
            inputs = [x for i in range(0, B, N) for x in inputs[i:i+N] + [inputs[i]]]
            completion_ids_list = [x for i, extra in zip(range(0, B, N), expert_completion_ids_list) for x in completion_ids_list[i:i+N] + [extra]]
            
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        try:
            # TRL 0.23.1 and below path
            if not has_images:
                # Left pad prompt before calculation old and ref hidden states
                prompt_completion_ids = left_pack_padding(prompt_completion_ids, self.processing_class.pad_token_id)
            self.model.for_training()
        except:
            # TRL 0.24.0 and below path
            if images is None:
                # Left pad prompt before calculation old and ref hidden states
                # ATTENTION doing this, because of the unsloth implementation (version: 2025.12.9)
                left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(prompt_completion_ids, logits_to_keep, self.processing_class.pad_token_id)
                max_left_pad = max(left_pad_tokens_per_prompt).item()
                prompt_completion_ids = left_pack_padding(prompt_completion_ids, self.processing_class.pad_token_id)
                pseudo_completion_input_ids = prompt_completion_ids[:, -(logits_to_keep +max_left_pad):]
                pseudo_completion_mask = create_completion_attention_mask(
                    pseudo_completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, self.processing_class.pad_token_id
                ).to(attention_mask.dtype)      
        self.model.for_training()

        num_images = [len(img_list) for img_list in images] if images is not None else None

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
        
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction: # This leads to a bug
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        is_chat = is_conversational(inputs[0])
        if is_chat:
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
            
        # Update Reward Model: import for AIRL
        if self.add_expert_to_policy_optim:
            prompts_neg = [p for i, p in enumerate(prompts) if (i+1) % self.num_generations_with_expert != 0]
            completions_neg = [c for i, c in enumerate(completions) if (i+1) % self.num_generations_with_expert != 0]
            prompts_pos = [p for i, p in enumerate(prompts) if (i+1) % self.num_generations_with_expert == 0]
            completions_pos = [c for i, c in enumerate(completions) if (i+1) % self.num_generations_with_expert == 0]
        else:
            prompts_neg = prompts
            completions_neg = completions
            prompts_pos =  [inputs[i]["prompts"] for i in range(0, len(inputs), self.num_generations)]
            completions_pos = [
                [{"role": "assistant", "content": inputs[i]["targets"]}] for i in range(0, len(inputs), self.num_generations)
            ]

        if mode == "train":
            reward_metrics = self._update_reward_model_step(
                prompts_neg, completions_neg, prompts_pos, completions_pos, 
                do_step=True, log_prefix="reward", is_chat=is_chat
            )
        
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(reward_metrics)

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        advantages, all_process_advantages, mean_grouped_rewards, std_rewards, is_std_zero = self._advantage_calculation(
            rewards_per_func, device, prompts
        )
        
        # ATTENTION doing this, because of the unsloth implementation (version: 2025.12.9)
        if advantages.ndim == 2:
            advantages = pad_to_attention_layout(advantages, pseudo_completion_mask, float("nan"))
            advantages = advantages.nan_to_num(0.0)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())   
        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            if rewards_per_func.dim() == 3:
                self._logs["rewards"][name].extend(rewards_per_func[:, i].nanmean(1).tolist())
            else:   
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output


    def save_model(self, output_dir, _internal_call=True):
        """
        Save the policy (handled by super) AND the reward model (+ tokenizer).
        If the reward model is a PEFT PeftModel, this saves only the adapters.
        Otherwise, it saves the full reward model.
        """
        # First, let the base class save the policy (this already handles LoRA adapters).
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        if not self.accelerator.is_main_process:
            return

        output_dir = output_dir or self.args.output_dir
        reward_dir = os.path.join(output_dir, "reward_model")
        os.makedirs(reward_dir, exist_ok=True)

        # Unwrap reward model in case it's wrapped by accelerate/FS*DP etc.
        reward_model_unwrapped = self.accelerator.unwrap_model(self.reward_model)
        reward_model_unwrapped.save_pretrained(reward_dir, safe_serialization=True)

        # Save reward tokenizer if available (kept separate from policy tokenizer on purpose)
        if self.reward_tokenizer is not None:
            self.reward_tokenizer.save_pretrained(reward_dir)

    def save_state(self):
        """
        Extend Trainer.save_state to also save the reward optimizer state.
        """
        super().save_state()

        if not self.accelerator.is_main_process:
            return

        output_dir = output_dir or self.args.output_dir
        reward_dir = os.path.join(output_dir, "reward_model")
        os.makedirs(reward_dir, exist_ok=True)

        # Reward optimizer state dict (so we can resume properly)
        if getattr(self, "reward_optimizer", None) is not None:
            torch.save(
                self.reward_optimizer.state_dict(), reward_dir / "reward_optimizer.pt"
            )


# ---------------------------------------------------------------------------
__all__ = ["AIRLTrainer"]
