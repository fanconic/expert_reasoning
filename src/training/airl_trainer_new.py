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

def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Moves all padding tokens in each sequence of a batch to the right.
    """
    mask = (tensor != pad_id)
    # Must do stable=True since binary mark is unordered
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor


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


    # -----------------------------------------------------------------------
    # Core overwrites overrides
    # -----------------------------------------------------------------------
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
                    if is_conversational(inputs[0]):
                        full_messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        full_texts = [apply_chat_template(x, reward_processing_class)["text"] for x in full_messages]
                    else:
                        full_texts = [p + c for p, c in zip(prompts, completions)]

                    # Tokenize (Padding side = Right) ATTENTION always right padded
                    reward_inputs = reward_processing_class(
                        text=full_texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
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
                mean_grouped_rewards = outcome_rewards.view(-1, self.num_generations_with_expert).mean(dim=1)
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations_with_expert, dim=0)
                outcome_advantages = outcome_rewards - mean_grouped_rewards
                std_rewards = outcome_rewards.view(-1, self.num_generations_with_expert).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(self.num_generations_with_expert, dim=0)
                is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
                outcome_advantages = outcome_advantages / (std_rewards + 1e-4)
            else:
                outcome_advantages = torch.zeros(B, device=device)

            # Dense discounted rewards
            # Construct a matrix M where M[k, t] = gamma^(k-t) for k >= t
            dense_rewards = rewards_per_func[:, 0, :] * weights[0]  # Shape: (B, L)
            mean_rewards_dense = dense_rewards.nanmean(1)
            mean_grouped_rewards_dense = mean_rewards_dense.view(-1, self.num_generations_with_expert).mean(dim=1) 
            mean_grouped_rewards_dense = mean_grouped_rewards_dense.repeat_interleave(self.num_generations_with_expert, dim=0)
            std_rewards_dense = mean_rewards_dense.view(-1, self.num_generations_with_expert).std(dim=1)
            std_rewards_dense = std_rewards_dense.repeat_interleave(self.num_generations_with_expert, dim=0)
            is_std_zero_dense = torch.isclose(std_rewards_dense, torch.zeros_like(std_rewards_dense))

            dense_advantages = (dense_rewards - mean_grouped_rewards_dense.unsqueeze(1)) / (std_rewards_dense.unsqueeze(1) + 1e-4)
            
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
            all_process_advantages = advantages.clone()
            
            import IPython; IPython.embed()
            mean_grouped_rewards
            is_std_zero
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

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
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
            expert_completions = list(set([x["target"] + self.processing_class.eos_token for x in inputs]))
            expert_tokens = self.processing_class(
                text=expert_completions, return_tensors="pt", padding="max_length", padding_side="right", 
                add_special_tokens=False, max_length=max_completion_length, truncation=True
            )
            expert_completion_ids_list = self.processing_class(text=expert_completions, add_special_tokens=False).input_ids
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
                prompt_completion_ids = left_pack_padding(prompt_completion_ids, self.processing_class.pad_token_id)
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
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        import IPython; IPython.embed()
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        advantages, all_process_advantages, mean_grouped_rewards, std_rewards, is_std_zero = self._advantage_calculation(
            rewards_per_func, device, prompts
        )

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
