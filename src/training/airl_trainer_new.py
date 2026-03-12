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
import random

# Third-party imports
from peft import set_peft_model_state_dict
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
    is_wandb_available,
    get_scheduler
)
from trl import GRPOTrainer
from transformers.utils import is_peft_available
from trl.data_utils import apply_chat_template, is_conversational
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available, is_liger_kernel_available
from trl.trainer.grpo_trainer import nanstd, nanmin, nanmax, nullcontext
from trl.trainer.utils import (
    pad,
)
import numpy as np
import copy
from tqdm import tqdm
from collections import deque

# Add this import for threadpool parallelism

# Local imports
from src.config.irl_config import IRLConfig
from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothEfficientGRPO, grpo_compute_loss_slow, align_logprobs_with_mask

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


def grpo_accumulated_loss(
    trainer,
    input_ids,
    attention_mask,
    logits_to_keep,
    completion_mask,
    advantages,
    old_hidden_states,
    ref_hidden_states,
    n_chunks = -1,
    batch_max_left_pad = None,
    **kwargs,
):
    # All Unsloth Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape

    pixel_values = kwargs.get('pixel_values',None)
    image_grid_thw = kwargs.get('image_grid_thw',None)
    pixel_attention_mask = kwargs.get('pixel_attention_mask',None)
    image_sizes = kwargs.get('image_sizes',None)
    #delete this from kwargs so less issues
    sampling_per_token_logps = kwargs.pop("sampling_per_token_logps", None)
    kwargs["vllm_importance_sampling_cap"] = trainer.vllm_importance_sampling_cap if sampling_per_token_logps is not None else None
    kwargs["use_vllm"] = trainer.use_vllm
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1: n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors)-1)]

    if not hasattr(trainer, '_autocast_dtype'):
        trainer._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': trainer._autocast_dtype = None
    pass
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    lm_head = trainer.model.get_output_embeddings().weight

    if pixel_values is None:
        left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, trainer.processing_class.pad_token_id)
        
        #max_left_pad = max(left_pad_tokens_per_prompt).item()
        max_left_pad = batch_max_left_pad
        
        input_ids = left_pack_padding(input_ids, trainer.processing_class.pad_token_id)

        completion_input_ids = input_ids[:, -(logits_to_keep +max_left_pad):]

        completion_mask = create_completion_attention_mask(completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, trainer.processing_class.pad_token_id).to(attention_mask.dtype)
        #TODO given the completion mask here we need to, handle the left pad tokens so the sizes of completion
        #token or old logprobs are compatible with the importance sampling logprobs
        if trainer.use_vllm and sampling_per_token_logps is not None:
            sampling_per_token_logps = align_logprobs_with_mask(sampling_per_token_logps, completion_mask)
        attention_mask =  input_ids != trainer.processing_class.pad_token_id
        attention_mask = attention_mask.to(attention_mask.dtype)
    else:
        completion_input_ids = input_ids[:, -logits_to_keep:]

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False)

    # Do not move hidden_states from device 1 to device 0:
    for module in unwrapped_model.modules():
        if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "io_same_decice"):
            module._hf_hook.io_same_decice = False
    pass
    # Get autocaster
    if trainer._autocast_dtype is None:
        autocaster = nullcontext()
    else:
        autocaster = torch.amp.autocast(device_type = trainer.model.device.type, dtype = trainer._autocast_dtype)
    with autocaster:
        if pixel_values is None:
            new_hidden_states = unwrapped_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                pixel_attention_mask = pixel_attention_mask,
                image_sizes = image_sizes,
                # logits_to_keep = logits_to_keep + 1,
            ).logits

            #keep extra logit as we generated a new token
            new_hidden_states = new_hidden_states[:, -(logits_to_keep +max_left_pad+1): , :]
            if ref_hidden_states is not None:
                ref_hidden_states = ref_hidden_states[:, -(logits_to_keep +max_left_pad+1): , :]
            if old_hidden_states is not None:
                old_hidden_states = old_hidden_states[:, -(logits_to_keep +max_left_pad+1): , :]
        else:
            new_hidden_states = unwrapped_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                pixel_attention_mask = pixel_attention_mask,
                image_sizes = image_sizes,
                logits_to_keep = logits_to_keep + 1,
            ).logits
    loss, completion_length, mean_kl, delta, flat_is_ratio = UnslothEfficientGRPO.apply(
        new_hidden_states,
        old_hidden_states,
        ref_hidden_states,
        sampling_per_token_logps,
        lm_head,
        completion_input_ids,
        completion_mask,
        advantages,
        trainer.beta,
        trainer.accelerator.scaler,
        n_chunks,
        kwargs # pass kwargs as a dict
    )
    # Must force not returning hidden states but logits otherwise gibberish
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    return loss, completion_length, mean_kl, delta, flat_is_ratio

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

def build_texts(prompts: list, completions: list, reward_tok, is_chat: bool) -> list[str]:
    """Build discriminator inputs exactly like your reward path (chat-template or plain concat)."""
    if is_chat:
        full_messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
        return [apply_chat_template(x, reward_tok)["text"] for x in full_messages]
    else:
        return [p + c for p, c in zip(prompts, completions)]

def backfill_rewards(rewards, mask):
    B, T = rewards.shape
    indices = torch.arange(T, device=rewards.device).expand(B, T)
    masked_indices = torch.where(mask.bool(), indices, torch.tensor(T, device=rewards.device))
    next_valid_index = torch.cummin(masked_indices.flip(1), dim=1)[0].flip(1)
    next_valid_index = next_valid_index.clamp(max=T-1).long()
    result = torch.gather(rewards, 1, next_valid_index)
    
    return result

def switch_label_if_correct_func(
    prompts_neg: List[Any], 
    completions_neg: List[Any], 
    prompts_pos: List[Any],
    completions_pos: List[Any],
    correctness_func: Callable,
    answers: List[Any],
):
    correct_mask = correctness_func(prompts=None, completions=completions_neg, answer=answers)
    [prompts_pos.append(neg) for mask, neg in zip(correct_mask, prompts_neg) if mask]
    [completions_pos.append(neg) for mask, neg in zip(correct_mask, completions_neg) if mask]
    prompts_neg = [neg for mask, neg in zip(correct_mask, prompts_neg) if not mask]
    completions_neg = [neg for mask, neg in zip(correct_mask, completions_neg) if not mask]
    return prompts_neg, completions_neg, prompts_pos, completions_pos

def perturb_expert_completions(
    prompts_neg: List[Any], 
    completions_neg: List[Any], 
    prompts_pos: List[Any],
    completions_pos: List[Any],
    perturb_fns: List[Callable],
    n_perturbs: int
): 
    if not perturb_fns:
        return prompts_neg, completions_neg, prompts_pos, completions_pos
    for _ in range(n_perturbs):
        num_selected = random.choice(range(1, len(perturb_fns) + 1))
        selected_perturbs = random.sample(perturb_fns, k=num_selected)
        
        new_prompts, new_completions = [], []
        for prompt, expert_text in zip(prompts_pos, completions_pos):
            text = expert_text[0]["content"]
            for perturb_func in selected_perturbs:
                text = perturb_func(text=text)
            
            new_prompts.append(prompt)
            new_completions.append([{"role": "assistant", "content": text}])
        
        prompts_neg.extend(new_prompts)
        completions_neg.extend(new_completions)
    
    return prompts_neg, completions_neg, prompts_pos, completions_pos

def perturb_expert_completions_medical(
    prompts_neg: List[Any], 
    completions_neg: List[Any],
    prompts_pos: List[Any], 
    corrupted_reasonings: List[Any],
    corrupted_answers: List[Any],
    n_perturbs: int
): 

    corrupted_targets = []
    for reasonings, answers in zip(corrupted_reasonings, corrupted_answers):
        corrupted_targets_single = []
        for reasoning, answer in zip(reasonings, answers):
            corrupted_targets_single.append(f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>")
        corrupted_targets.append(corrupted_targets_single)  
    
    new_prompts, new_completions = [], []
    for prompt, corruptions in zip(prompts_pos, corrupted_targets):
        for i, text in enumerate(corruptions):
            if i+1==n_perturbs:
                break
            new_prompts.append(prompt)
            new_completions.append([{"role": "assistant", "content": text}])
    
    prompts_neg.extend(new_prompts)
    completions_neg.extend(new_completions)
    return prompts_neg, completions_neg

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
        self.eval_reward_buffer = defaultdict(list)

        # Reward functions --------------------------------------------------------------
        self.dense_rewards = args.dense_rewards
        self.dense_gamma = args.dense_gamma
        self.advantage_calculation = args.advantage_calculation
        self.add_expert_to_policy_optim = args.add_expert_to_policy_optim
        self.add_expert_to_policy_balanced = args.add_expert_to_policy_balanced
        self.classifier_loss = args.classifier_loss
        self.normalise_rewards = args.normalise_rewards
        self.expert_error_rate = args.expert_error_rate
        
        # This is used for later propagation in the unlsoth code, if we calculate dense rewards.
        self.batch_max_left_pad = None

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
        self.reward_warmup_steps = self.args.reward_warmup_steps
        self.warmup_done = False
        if args.warmup_reward_dir:
            self.load_reward_warmup_checkpoint(args.warmup_reward_dir)

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
        self.switch_label_if_correct = getattr(args, "switch_label_if_correct", False)
        self.neg_sample_weight = getattr(args, "neg_sample_weight", 1.0)  # weight in BCE
        self.disc_pairwise_margin = getattr(args, "disc_pairwise_margin", 0.0)  # >0 to enable pairwise hinge
        self.neg_label_smoothing = getattr(args, "neg_label_smoothing", None)  # defaults to self.eps if None
        
        self.num_generations_with_expert = self.num_generations + 1 if self.add_expert_to_policy_optim else self.num_generations 
        self.max_length = self.args.max_prompt_length + self.args.max_completion_length
        
        # Add a replay buffer
        self.buffer_size = getattr(args, "buffer_size", 0)
        if self.buffer_size > 0:
            self.neg_replay_buffer = deque(maxlen=self.buffer_size * self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps) 
            self.pos_replay_buffer = deque(maxlen=self.buffer_size * self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)  # Keep enough positives for balanced sampling

    # -----------------------------------------------------------------------
    # Core overwrites overrides
    # -----------------------------------------------------------------------$
    def train(self, *args, **kwargs):
        # Set the learning rate warm up as the same as the scheduler warmup steps
        num_policy_steps = self.args.max_steps
        total_reward_steps = self.reward_warmup_steps + num_policy_steps
        if self.reward_optimizer is not None:
            self.reward_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=self.reward_optimizer,
                num_warmup_steps=self.reward_warmup_steps, 
                num_training_steps=total_reward_steps,
            )

        # Run warmup of the reward model
        if self.reward_warmup_steps > 0 and not self.warmup_done:
            self._warmup_discriminator()
            self.warmup_done = True

        return super().train(*args, **kwargs)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.eval_reward_buffer.clear()
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
        if self.eval_reward_buffer:
            for key, values in self.eval_reward_buffer.items():
                if values:
                    final_mean = sum(values) / len(values)
                    final_key = f"{metric_key_prefix}_{key}"
                    metrics[final_key] = final_mean
        return metrics
    
    def _sentence_boundary_mask(self, full_batch, base_completion_mask):
        """
        Identifies step boundaries by scanning token content and IDs.
        Ensures rewards start strictly at the Assistant completion.
        """
        input_ids = full_batch["input_ids"]
        bs, L = input_ids.shape
        device = input_ids.device

        # 1. Define all strings that signify the end of a reasoning "step"
        boundary_strings = [
            ".", "!", "?", ";",      # Standard punctuation
            "\n", "\n\n", "\r\n",    # Newlines / Paragraph breaks
            ".\n", "!\n", "?\n",     # Punctuation + Newline (very common)
            "</think>",              # Reasoning closure (Qwen/DeepSeek style)
            "####",                  # GSM8K answer marker
            "\n1.", "\n2.", "\n-",   # Start of a new list item (implies previous step ended)
        ]

        # 2. Build an extensive Stop ID set (Cached for performance)
        if not hasattr(self, "_cached_stop_ids"):
            print("Initializing extensive boundary token set...")
            stop_ids = set()
            
            # Add special tokens directly
            if self.reward_tokenizer.eos_token_id is not None:
                stop_ids.add(self.reward_tokenizer.eos_token_id)

            # Scan the entire vocabulary for tokens containing or ending with boundaries
            # This catches merged tokens like "reasoning.\n" in Qwen/Llama
            vocab = self.reward_tokenizer.get_vocab()
            for t_str, t_id in vocab.items():
                # Clean BPE markers (Ġ for Llama, █ for Qwen) to see raw text
                clean_t = t_str.replace('Ġ', ' ').replace(' ', ' ')
                
                # Check if token ends with a boundary or is a reasoning tag
                if any(clean_t.endswith(s) for s in boundary_strings) or clean_t.strip() in boundary_strings:
                    stop_ids.add(t_id)
            
            self._cached_stop_ids = torch.tensor(list(stop_ids), device=device)

        # 3. Apply vectorized ID matching
        boundary_mask = torch.isin(input_ids, self._cached_stop_ids)

        # 4. STRICT ASSISTANT START ENFORCEMENT
        # base_completion_mask is 1 for Assistant tokens, 0 for User/System prompt tokens.
        # By ANDing them, we guarantee the reward model NEVER rewards the prompt.
        boundary_mask &= base_completion_mask.bool()

        # 5. Always include the very last token of the completion
        # This ensures the final answer segment gets its reward even if it doesn't end in "."
        last_indices = base_completion_mask.long().cumsum(dim=1).argmax(dim=1)
        has_completion = base_completion_mask.any(dim=1)
        boundary_mask[torch.arange(bs, device=device)[has_completion], last_indices[has_completion]] = True

        return boundary_mask
        
    def _every_n_tokens_mask(self, full_batch, base_completion_mask, n: int):
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
        every_n_mask[torch.arange(bs, device=device),last_indices] |= base_completion_mask.any(dim=1)
        return every_n_mask


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
        len(neg_*) = N
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
        K = N / B
        
        pos_texts = build_texts(pos_prompts, pos_completions, self.reward_tokenizer, is_chat=is_chat)
        neg_texts = build_texts(neg_prompts, neg_completions, self.reward_tokenizer, is_chat=is_chat)
        pos_w, neg_w = 1.0, 1.0

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
                if self.dense_rewards == "partial":
                    mask = self._sentence_boundary_mask(batch_pos, base_mask)
                elif self.dense_rewards == "partial_fixed":
                    mask = self._every_n_tokens_mask(batch_pos, base_mask, n=self.args.dense_partial_fixed_n)
                else:
                    mask = base_mask
                T_pos += mask.sum().to(T_pos.dtype)

            for i in range(0, N, micro_bs):
                j = min(i + micro_bs, N)
                batch_neg = _tok(neg_texts[i:j])
                base_mask = _completion_mask(batch_neg, neg_prompts[i:j])
                if self.dense_rewards == "partial":
                    mask = self._sentence_boundary_mask(batch_neg, base_mask)
                elif self.dense_rewards == "partial_fixed":
                    mask = self._every_n_tokens_mask(batch_neg, base_mask, n=self.args.dense_partial_fixed_n)
                else:
                    mask = base_mask
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
            y_pos = torch.ones_like(logits_pos) * (1.0 - self.eps)

            if self.dense_rewards in ["full", "partial"]:
                base_mask = _completion_mask(batch_pos, pos_prompts[i:j])
                if self.dense_rewards == "full":
                    mask = base_mask
                elif self.dense_rewards == "partial":
                    mask = self._sentence_boundary_mask(batch_pos, base_mask)
                elif self.dense_rewards == "partial_fixed":
                    mask = self._every_n_tokens_mask(batch_pos, base_mask, n=self.args.dense_partial_fixed_n)
                loss_elt = F.binary_cross_entropy_with_logits(logits_pos, y_pos, reduction="none")
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
            y_neg = torch.ones_like(logits_neg) * self.eps
            
            if self.dense_rewards in ["full", "partial"]:
                base_mask = _completion_mask(batch_neg, neg_prompts[i:j])
                if self.dense_rewards == "full":
                    mask = base_mask
                elif self.dense_rewards == "partial":
                    mask = self._sentence_boundary_mask(batch_neg, base_mask)
                elif self.dense_rewards == "partial_fixed":
                    mask = self._every_n_tokens_mask(batch_neg, base_mask, n=self.args.dense_partial_fixed_n)
                loss_elt = F.binary_cross_entropy_with_logits(logits_neg, y_neg, reduction="none")
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
            if hasattr(self, "reward_scheduler") and self.reward_scheduler is not None:
                self.reward_scheduler.step()

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
        
        current_reward_lr = self.reward_optimizer.param_groups[0]["lr"]

        return {
            f"{log_prefix}/loss": gmean(loss),
            f"{log_prefix}/bce_pos": gmean(bce_pos),
            f"{log_prefix}/lr": current_reward_lr,
            f"{log_prefix}/bce_neg": gmean(bce_neg),
            f"{log_prefix}/acc": gmean(acc),
            f"{log_prefix}/acc_pos": gmean(acc_pos),
            f"{log_prefix}/acc_neg": gmean(acc_neg),
            f"{log_prefix}/logit_pos_mean": gmean(logits_pos_det.mean()),
            f"{log_prefix}/logit_neg_mean": gmean(logits_neg_det.mean()),
            f"{log_prefix}/prob_pos_mean": gmean(p_pos.mean()),
            f"{log_prefix}/prob_neg_mean": gmean(p_neg.mean()),
            f"{log_prefix}/neg_pos_ratio": K,
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
            
            # medical corruptions (before label switch, as they are wrong guaranteed)
            if self.num_neg_perturbations_per_expert and "corrupted_reasonings" in batch[0].keys():
                corrupted_reasonings = [batch[i]["corrupted_reasonings"] for i in range(0, len(batch), self.num_generations)]
                corrupted_answers = [batch[i]["corrupted_answers"] for i in range(0, len(batch), self.num_generations)]
                neg_prompts, neg_completions = perturb_expert_completions_medical(
                    prompts_neg=neg_prompts, 
                    completions_neg=neg_completions,
                    prompts_pos=pos_prompts,
                    corrupted_reasonings=corrupted_reasonings,
                    corrupted_answers=corrupted_answers,
                    n_perturbs=self.num_neg_perturbations_per_expert,
                )
            
            # Check which negatives are correct and switch labels
            if self.switch_label_if_correct:
                neg_prompts, neg_completions, pos_prompts, pos_completions = switch_label_if_correct_func(
                    prompts_neg=neg_prompts, 
                    completions_neg=neg_completions, 
                    prompts_pos=pos_prompts,
                    completions_pos=pos_completions,
                    correctness_func=self.reward_funcs[-1],
                    answers=[x["answer"] for x in batch]
                )
                
            # Pertub texts and make them negative
            
            if self.num_neg_perturbations_per_expert and not "corrupted_reasonings" in batch[0].keys():
                neg_prompts, neg_completions, pos_prompts, pos_completions = perturb_expert_completions(
                    prompts_neg=neg_prompts, 
                    completions_neg=neg_completions, 
                    prompts_pos=pos_prompts,
                    completions_pos=pos_completions,
                    perturb_fns=self.neg_perturb_fns,
                    n_perturbs=self.num_neg_perturbations_per_expert,
                )
            
                
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


        # Save reward model after warmup
        reward_dir = os.path.join(self.args.output_dir, "reward_model_warmup")
        os.makedirs(reward_dir, exist_ok=True)

        # Unwrap reward model in case it's wrapped by accelerate/FS*DP etc.
        reward_model_unwrapped = self.accelerator.unwrap_model(self.reward_model)
        reward_model_unwrapped.save_pretrained(reward_dir, safe_serialization=True)

        # Save reward tokenizer if available (kept separate from policy tokenizer on purpose)
        if self.reward_tokenizer is not None:
            self.reward_tokenizer.save_pretrained(reward_dir)

        # Save the optimizer too
        if getattr(self, "reward_optimizer", None) is not None:
            torch.save(
                self.reward_optimizer.state_dict(), os.path.join(reward_dir,"reward_optimizer_warmup.pt")
            )
            
        if policy_was_training: self.model.train()
        if not reward_was_training: self.reward_model.eval()
        
        
    def load_reward_warmup_checkpoint(self, checkpoint_path: str):
        """
        Loads the reward model and optimizer state from a warmup checkpoint.
        Call this before train() to bypass the _warmup_discriminator step.
        """
        from safetensors.torch import load_file as safe_load_file
        
        if not os.path.isdir(checkpoint_path):
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist. Skipping load.")
            return

        logger.info(f"Loading reward model warmup checkpoint from {checkpoint_path}...")

        unwrapped_model = self.accelerator.unwrap_model(self.reward_model)
        is_peft = is_peft_available() and isinstance(unwrapped_model, PeftModel)

        if is_peft:
            logger.info("Detected PEFT model. Loading adapters...")
            adapters_weights = safe_load_file(os.path.join(checkpoint_path, "adapter_model.safetensors"))
            set_peft_model_state_dict(unwrapped_model, adapters_weights)
            del adapters_weights
            torch.cuda.empty_cache()
        else:
            model_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(model_path):
                state_dict = safe_load_file(model_path)
            else:
                model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                if os.path.exists(model_path):
                    state_dict = torch.load(model_path, map_location=self.accelerator.device)
                else:
                    raise FileNotFoundError(f"No model weights (safetensors/bin) found in {checkpoint_path}")
            
            unwrapped_model.load_state_dict(state_dict, strict=True)

        # --- 2. Load Optimizer State ---
        opt_path = os.path.join(checkpoint_path, "reward_optimizer_warmup.pt")
        if os.path.exists(opt_path):
            logger.info(f"Loading reward optimizer state from {opt_path}...")
            opt_state = torch.load(opt_path, map_location="cpu")
            self.reward_optimizer.load_state_dict(opt_state)
            del opt_state
            torch.cuda.empty_cache()
        else:
            logger.warning(f"No optimizer checkpoint found at {opt_path}. Optimizer starts fresh.")

        self.warmup_done = False
        self.reward_warmup_steps = 1 #Make one, otherwise there is OOM
        logger.info("Reward model loaded. Warmup phase will be skipped.")


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
                                reward_logits.clamp_(self.reward_lb, self.reward_ub)
                            full_lens = reward_inputs["attention_mask"].sum(dim=1).long()
                            start_indices = (full_lens - completion_lens).clamp(min=0)
                            gather_indices = start_indices[:, None] + torch.arange(seq_len, device=device)[None, :]
                            gather_indices = gather_indices.clamp(max=reward_logits.size(1) - 1)
                            reward_comp = reward_logits.gather(1, gather_indices)
                            reward_comp[~output_mask] = float('nan')
                            if self.dense_rewards=="partial":
                                end_of_thought_mask = self._sentence_boundary_mask(reward_inputs, reward_inputs["attention_mask"])
                                
                                # # --- DEBUGGING VISUALIZATION START ---
                                # if self.accelerator.is_main_process:
                                #     # Check only the first sample in the micro-batch
                                #     sample_ids = reward_inputs["input_ids"][0]
                                #     sample_tokens = self.reward_tokenizer.convert_ids_to_tokens(sample_ids)
                                #     sample_mask = end_of_thought_mask[0]
                                    
                                #     print("\n" + "="*50)
                                #     print("DEBUG: Reward Model Token Alignment")
                                #     print(f"BOS Token: {self.reward_tokenizer.bos_token} (ID: {self.reward_tokenizer.bos_token_id})")
                                #     print("-" * 50)
                                    
                                #     for idx, (token, is_boundary) in enumerate(zip(sample_tokens, sample_mask)):
                                #         # Only print non-padding tokens for clarity
                                #         if token == self.reward_tokenizer.pad_token:
                                #             continue
                                #         boundary_marker = " [STEP END] <---" if is_boundary else ""
                                #         print(f"Token {idx:3}: '{token:15}' {boundary_marker}")
                                #     print("="*50 + "\n")
                                    
                                # import IPython; IPython.embed(); exit()
                                # # --- DEBUGGING VISUALIZATION END ---
                                
                                end_of_thought_mask = end_of_thought_mask.gather(1, gather_indices)
                                reward_comp = backfill_rewards(reward_comp, end_of_thought_mask)
                            elif self.dense_rewards=="partial_fixed":
                                end_of_thought_mask = self._every_n_tokens_mask(reward_inputs, reward_inputs["attention_mask"], n=self.args.dense_partial_fixed_n)
                                end_of_thought_mask = end_of_thought_mask.gather(1, gather_indices)
                                reward_comp = backfill_rewards(reward_comp, end_of_thought_mask)
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
            #mean_rewards_dense = dense_rewards.nanmean(1)
            rew_mask = ~torch.isnan(dense_rewards)                  # True where valid
            last_idx = rew_mask.sum(dim=1) - 1                # index of last valid element in each row
            mean_rewards_dense = dense_rewards[torch.arange(dense_rewards.size(0)), last_idx]
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
        _was_training = self.model.training
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
                self.batch_max_left_pad = max_left_pad  # Hack to pass max_left_pad to the model forward in unsloth
                prompt_completion_ids = left_pack_padding(prompt_completion_ids, self.processing_class.pad_token_id)
                pseudo_completion_input_ids = prompt_completion_ids[:, -(logits_to_keep +max_left_pad):]
                pseudo_completion_mask = create_completion_attention_mask(
                    pseudo_completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, self.processing_class.pad_token_id
                ).to(attention_mask.dtype)
               # import IPython; IPython.embed(); exit()
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
            prompts_neg = prompts.copy()
            completions_neg = completions.copy()
            prompts_pos =  [inputs[i]["prompt"] for i in range(0, len(inputs), self.num_generations)]
            completions_pos = [
                [{"role": "assistant", "content": inputs[i]["target"]}] for i in range(0, len(inputs), self.num_generations)
            ]


        if mode == "train" and self.reward_updates_per_policy_step > 0:
            
            # medical corruptions (before label switch, as they are wrong guaranteed)
            if self.num_neg_perturbations_per_expert and "corrupted_reasonings" in inputs[0].keys():
                corrupted_reasonings = [inputs[i]["corrupted_reasonings"] for i in range(0, len(inputs), self.num_generations)]
                corrupted_answers = [inputs[i]["corrupted_answers"] for i in range(0, len(inputs), self.num_generations)]
                prompts_neg, completions_neg = perturb_expert_completions_medical(
                    prompts_neg=prompts_neg, 
                    completions_neg=completions_neg,
                    prompts_pos=prompts_pos, 
                    corrupted_reasonings=corrupted_reasonings,
                    corrupted_answers=corrupted_answers,
                    n_perturbs=self.num_neg_perturbations_per_expert,
                )
            
            if self.switch_label_if_correct:
                prompts_neg, completions_neg, prompts_pos, completions_pos = switch_label_if_correct_func(
                    prompts_neg=prompts_neg, 
                    completions_neg=completions_neg, 
                    prompts_pos=prompts_pos,
                    completions_pos=completions_pos,
                    correctness_func=self.reward_funcs[-1],
                    answers=[x["answer"] for i, x in enumerate(inputs) if (i+1) % self.num_generations_with_expert != 0]
                )

            # Pertub texts and make them negative
            if self.num_neg_perturbations_per_expert and not "corrupted_reasonings" in inputs[0].keys():
                prompts_neg, completions_neg, prompts_pos, completions_pos = perturb_expert_completions(
                    prompts_neg=prompts_neg, 
                    completions_neg=completions_neg, 
                    prompts_pos=prompts_pos,
                    completions_pos=completions_pos,
                    perturb_fns=self.neg_perturb_fns,
                    n_perturbs=self.num_neg_perturbations_per_expert,
                )
            
            # Store as list of dicts to keep pairs together
            if self.buffer_size > 0:
                for p, c in zip(prompts_neg, completions_neg):
                    self.neg_replay_buffer.append({"p": p, "c": c})
                for p, c in zip(prompts_pos, completions_pos):
                    self.pos_replay_buffer.append({"p": p, "c": c})
            
            for update_idx in range(self.reward_updates_per_policy_step):
                # Sample equally from both (e.g., 32 samples each)
                if self.buffer_size > 0:
                    batch_size = self.args.per_device_train_batch_size
                    num_to_sample = min(len(self.neg_replay_buffer), len(self.pos_replay_buffer), batch_size // 2)
                    
                    neg_samples = random.sample(self.neg_replay_buffer, k=num_to_sample)
                    pos_samples = random.sample(self.pos_replay_buffer, k=num_to_sample)
                    
                    # Reconstruct lists for the update step
                    sampled_neg_p = [s["p"] for s in neg_samples]
                    sampled_neg_c = [s["c"] for s in neg_samples]
                    sampled_pos_p = [s["p"] for s in pos_samples]
                    sampled_pos_c = [s["c"] for s in pos_samples]
                    
                    #import IPython; IPython.embed(); exit()
                else:
                    sampled_neg_p = prompts_neg
                    sampled_neg_c = completions_neg
                    sampled_pos_p = prompts_pos
                    sampled_pos_c = completions_pos
                
                reward_metrics = self._update_reward_model_step(
                    sampled_neg_p, sampled_neg_c, sampled_pos_p, sampled_pos_c, 
                    do_step=True, log_prefix="reward", is_chat=is_chat
                )
        
                if (self.state.global_step + 1) % self.args.logging_steps == 0 and update_idx == self.reward_updates_per_policy_step - 1:
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
        
        # Remove the experts from the reward metric logging 
        if self.add_expert_to_policy_optim:
            B = advantages.size(0)
            non_expert_mask = [(i+1) % self.num_generations_with_expert != 0 for i in range(B)]
            rewards_per_func = rewards_per_func[non_expert_mask]
            mean_grouped_rewards = mean_grouped_rewards[non_expert_mask] 
            std_rewards = std_rewards[non_expert_mask] 
            is_std_zero = is_std_zero[non_expert_mask] 
            prompts_text = [p for p, m in zip(prompts_text, non_expert_mask) if m]
            completions_text = [c for c, m in zip(completions_text, non_expert_mask) if m]
            all_process_advantages = all_process_advantages[non_expert_mask]
            
        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            if mode == "eval":
                self.eval_reward_buffer[f"rewards/{reward_func_name}/mean"].append(mean_rewards)
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
        if not _was_training:
            self.model.for_inference()
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

        output_dir = self.args.output_dir
        reward_dir = os.path.join(output_dir, "reward_model")
        os.makedirs(reward_dir, exist_ok=True)

        # Reward optimizer state dict (so we can resume properly)
        if getattr(self, "reward_optimizer", None) is not None:
            torch.save(
                self.reward_optimizer.state_dict(), reward_dir / "reward_optimizer.pt"
            )
            
    def compute_loss(
        self, model, inputs, return_outputs = False, num_items_in_batch = None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        pixel_values, image_grid_thw = (
            inputs.get("pixel_values", None),
            inputs.get("image_grid_thw", None),
        )
        pixel_attention_mask, image_sizes = (
            inputs.get("pixel_attention_mask", None),
            inputs.get("image_sizes", None),
        )
        num_items_in_batch = inputs.get("num_items_in_batch", None)
        sampling_per_token_logps = inputs.get("sampling_per_token_logps", None)
        current_gradient_accumulation_steps = self.current_gradient_accumulation_steps
        num_processes = self.accelerator.num_processes

        input_ids = torch.cat([prompt_ids, completion_ids], dim = 1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim = 1)
        # attention_mask = None
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep

        get_logps_func = (
            lambda model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size = None,
            compute_entropy = False,
            compute_efficient = False: self._get_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep, compute_efficient
            )
            if hasattr(self, "_get_per_token_logps")
            else self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                compute_entropy,
                compute_efficient,
            )[0]
        )  # logps

        per_token_logps = get_logps_func(
            model, input_ids, attention_mask, logits_to_keep, compute_efficient = True
        )
        # Compute the KL divergence between the model and the reference model
        # _prepare_inputs doesn't return reference log probs anymore. We need to calculate it ourselves.
        # https://github.com/huggingface/trl/blob/05bc43e960396581e458195b8388efe6b82cae1f/trl/trainer/grpo_trainer.py#L1328
        # if self.beta != 0.0:
        #     with torch.inference_mode(), model.disable_adapter():
        #         ref_per_token_logps = per_token_logps = get_logps_func(model, input_ids, attention_mask, logits_to_keep)
        # else:
        #     ref_per_token_logps = None
        ref_hidden_states = inputs.get("ref_per_token_logps", None)
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        old_hidden_states = inputs.get("old_per_token_logps", None)

        input_ids = input_ids[:, -logits_to_keep:]

        # Get logit softcapping and logit scale
        logit_softcapping = getattr(model.config, "final_logit_softcapping", 0)  # Gemma
        if logit_softcapping is None:
            logit_softcapping = 0
        logit_scale_multiply = getattr(model.config, "logit_scale", 0)  # Cohere
        if logit_scale_multiply is None:
            logit_scale_multiply = 0
        logit_scale_divide = getattr(model.config, "logits_scaling", 0)  # Granite
        if logit_scale_divide is None:
            logit_scale_divide = 0

        if per_token_logps is not None:
            if ref_hidden_states is not None:
                ref_hidden_states = ref_hidden_states[
                    :, :-1, :
                ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            if old_hidden_states is not None:
                old_hidden_states = old_hidden_states[
                    :, :-1, :
                ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            per_token_logps = per_token_logps[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            loss, completion_length, mean_kl, delta, flat_is_ratio = (
                grpo_compute_loss_slow(
                    ref_hidden_states,
                    per_token_logps,
                    old_hidden_states,
                    input_ids,
                    completion_mask,
                    self.beta,
                    advantages,
                    pixel_values = pixel_values,
                    image_grid_thw = image_grid_thw,
                    loss_type = self.args.loss_type,
                    importance_sampling_level = self.importance_sampling_level,
                    epsilon_low = self.epsilon_low,
                    epsilon_high = self.epsilon_high,
                    max_completion_length = self.args.max_completion_length,
                    delta = self.args.delta,
                    temperature = self.args.temperature,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    num_items_in_batch = num_items_in_batch,
                    current_gradient_accumulation_steps = current_gradient_accumulation_steps,
                    num_processes = num_processes,
                    sampling_per_token_logps = sampling_per_token_logps,
                )
            )
        else:
            if hasattr(self.args, "loss_type"):
                loss, completion_length, mean_kl, delta, flat_is_ratio = (
                    grpo_accumulated_loss(
                        trainer = self,
                        input_ids = _input_ids,
                        pixel_values = pixel_values,
                        image_grid_thw = image_grid_thw,
                        logits_to_keep = logits_to_keep,
                        completion_mask = completion_mask,
                        advantages = advantages,
                        old_hidden_states = old_hidden_states,
                        ref_hidden_states = ref_hidden_states,
                        n_chunks = self.args.unsloth_num_chunks,
                        batch_max_left_pad = self.batch_max_left_pad,
                        loss_type = self.args.loss_type,
                        importance_sampling_level = self.importance_sampling_level,
                        epsilon_low = self.epsilon_low,
                        epsilon_high = self.epsilon_high,
                        max_completion_length = self.args.max_completion_length,
                        delta = self.args.delta,
                        temperature = self.args.temperature,
                        logit_softcapping = logit_softcapping,
                        logit_scale_multiply = logit_scale_multiply,
                        logit_scale_divide = logit_scale_divide,
                        attention_mask = attention_mask,
                        num_items_in_batch = num_items_in_batch,
                        current_gradient_accumulation_steps = current_gradient_accumulation_steps,
                        num_processes = num_processes,
                        sampling_per_token_logps = sampling_per_token_logps,
                    )
                )
            else:
                # to ensure backwards compatibility with trl 0.15.2 and maybe even 0.17
                loss, completion_length, mean_kl = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_hidden_states = old_hidden_states,
                    ref_hidden_states = ref_hidden_states,
                    n_chunks = self.args.unsloth_num_chunks,
                    batch_max_left_pad = self.batch_max_left_pad,
                    temperature = self.args.temperature,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    attention_mask = attention_mask,
                )

        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())

        if self.use_vllm and delta is not None:
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                self.accelerator.gather(min_importance_sampling_ratio)
                .nan_to_num(nan = float("inf"))
                .min()
                .item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                self.accelerator.gather(max_importance_sampling_ratio)
                .nan_to_num(nan = float("-inf"))
                .max()
                .item()
            )

        return loss


# ---------------------------------------------------------------------------
__all__ = ["AIRLTrainer"]
