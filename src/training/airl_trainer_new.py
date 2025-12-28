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

                            # --- PRECISE ALIGNMENT (RIGHT-ALIGNED) ---
                            # 1. Get the end of the valid sequence
                            full_lens = reward_inputs["attention_mask"].sum(dim=1).long()

                            # 2. Calculate Start Index by subtracting Completion Length from End
                            # This mathematically skips the Prompt and the Header.
                            start_indices = (full_lens - completion_lens).clamp(min=0)

                            # 3. Create Gather Indices: [start, start+1, ... start+seq_len]
                            gather_indices = start_indices[:, None] + torch.arange(seq_len, device=device)[None, :]
                            gather_indices = gather_indices.clamp(max=reward_logits.size(1) - 1)

                            # 4. Gather and Mask
                            # We use 'output_mask' to zero out any padding/garbage
                            reward_comp = reward_logits.gather(1, gather_indices) * output_mask.to(reward_logits.dtype)
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
                        rewards_per_func[:, i, :] = expanded_rewards * output_mask.float()
                    else:
                        rewards_per_func[:, i, 0] = output_tensor

        if not self.dense_rewards:
            rewards_per_func = rewards_per_func.squeeze(-1)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

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
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
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

    # -----------------------------------------------------------------------
    # Reward model utilities
    # -----------------------------------------------------------------------

    def _update_reward_model(
        self,
        inputs: List[Dict[str, Any]],
        prompts: List[List[Dict[str, str]]],
        policy_completions: List[List[Dict]],
    ) -> torch.Tensor:
        """One discriminator step with expert positives, policy negatives, and (optional) perturbed expert negatives."""

        device = self.accelerator.device

        # Positives = expert completions
        expert_completions = [
            [{"role": "assistant", "content": element["target"]}] for element in inputs
        ]

        # Build perturbed negatives from experts
        _per_out = self._make_perturbed_completions(prompts, expert_completions)
        perturbed_completions = _per_out[0] if isinstance(_per_out, tuple) else _per_out

        # Tokenise and prepare all inputs
        expert_tokens, policy_tokens = tokenize_examples(
            prompts,
            expert_completions,
            policy_completions,
            self.reward_tokenizer,
            self.max_completion_length,
            device,
            self.response_only,
            self.dense_rewards,
        )

        # Deduplicate expert tokens and get multiplicities
        expert_tokens, pos_counts, _ = dedup_token_batch(expert_tokens)
        n_pos = expert_tokens["input_ids"].size(0)
        n_pol = policy_tokens["input_ids"].size(0)

        # Handle perturbed examples if present
        perturbed_tokens = None
        n_per = 0
        if perturbed_completions:
            _, perturbed_tokens = tokenize_examples(
                prompts[: len(perturbed_completions)],
                expert_completions[: len(perturbed_completions)],
                perturbed_completions,
                self.reward_tokenizer,
                self.max_completion_length,
                device,
                self.response_only,
                self.dense_rewards,
            )
            n_per = perturbed_tokens["input_ids"].size(0)

        # Prepare batched data for training
        batch, labels, weights = prepare_reward_batch(
            expert_tokens,
            policy_tokens,
            perturbed_tokens,
            n_pos,
            n_pol,
            n_per,
            device,
            self.eps,
            self.neg_label_smoothing,
            self.neg_sample_weight,
            self.num_neg_perturbations_per_expert,
            self.args.gradient_accumulation_steps,
        )

        # Training loop over gradient accumulation steps
        total_loss = self._train_reward_model_loop(
            batch, labels, weights, n_pos, n_pol, n_per, pos_counts
        )

        # Optimizer step if needed
        if (
            self.state.global_step % self.reward_updates_per_policy_step == 0
            and not self.standard_grpo
        ):
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
            self.reward_optimizer.step()
            self.reward_optimizer.zero_grad()

        return total_loss / self.args.gradient_accumulation_steps

    def _train_reward_model_loop(
        self,
        batch: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        weights: torch.Tensor,
        n_pos: int,
        n_pol: int,
        n_per: int,
        pos_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Training loop for reward model with gradient accumulation and micro-batching."""
        # Compute samples per accumulation step
        pos_per_step = n_pos // self.args.gradient_accumulation_steps
        pol_per_step = n_pol // self.args.gradient_accumulation_steps
        per_per_step = (
            n_per // self.args.gradient_accumulation_steps if n_per > 0 else 0
        )
        have_perturbed = n_per > 0

        total_loss = 0
        for step in range(self.args.gradient_accumulation_steps):
            # Get slices for this step
            pos_slice = slice(
                step * pos_per_step,
                (
                    (step + 1) * pos_per_step
                    if step < self.args.gradient_accumulation_steps - 1
                    else n_pos
                ),
            )
            pol_slice = slice(
                step * pol_per_step,
                (
                    (step + 1) * pol_per_step
                    if step < self.args.gradient_accumulation_steps - 1
                    else n_pol
                ),
            )

            # Prepare step data
            if have_perturbed:
                per_slice = slice(
                    step * per_per_step,
                    (
                        (step + 1) * per_per_step
                        if step < self.args.gradient_accumulation_steps - 1
                        else n_per
                    ),
                )
                step_batch = {
                    k: torch.cat(
                        [
                            batch[k][:n_pos][pos_slice],
                            batch[k][n_pos : n_pos + n_pol][pol_slice],
                            batch[k][n_pos + n_pol :][per_slice],
                        ],
                        dim=0,
                    )
                    for k in batch.keys()
                }
                step_labels = torch.cat(
                    [
                        labels[:n_pos][pos_slice],
                        labels[n_pos : n_pos + n_pol][pol_slice],
                        labels[n_pos + n_pol :][per_slice],
                    ]
                )
                step_weights = torch.cat(
                    [
                        weights[:n_pos][pos_slice],
                        weights[n_pos : n_pos + n_pol][pol_slice],
                        weights[n_pos + n_pol :][per_slice],
                    ]
                )
            else:
                step_batch = {
                    k: torch.cat(
                        [batch[k][:n_pos][pos_slice], batch[k][n_pos:][pol_slice]],
                        dim=0,
                    )
                    for k in batch.keys()
                }
                step_labels = torch.cat(
                    [labels[:n_pos][pos_slice], labels[n_pos:][pol_slice]]
                )
                step_weights = torch.cat(
                    [weights[:n_pos][pos_slice], weights[n_pos:][pol_slice]]
                )

            # Process micro-batches
            step_size = step_batch["input_ids"].size(0)
            num_micro_batches = (
                step_size + self.max_micro_batch - 1
            ) // self.max_micro_batch
            micro_batch_size = (step_size + num_micro_batches - 1) // num_micro_batches

            step_loss = 0
            for micro_idx in range(num_micro_batches):
                start_idx = micro_idx * micro_batch_size
                end_idx = min((micro_idx + 1) * micro_batch_size, step_size)

                micro_batch = {k: v[start_idx:end_idx] for k, v in step_batch.items()}
                micro_labels = step_labels[start_idx:end_idx]
                micro_weights = step_weights[start_idx:end_idx]

                if self.dense_rewards:
                    micro_labels = micro_labels.unsqueeze(1).repeat(
                        1, micro_batch["input_ids"].size(1)
                    )

                if self.classifier_loss == "bce":
                    # Forward pass with micro-batch
                    with self.accelerator.autocast():
                        micro_logits = self.reward_model(
                            input_ids=micro_batch["input_ids"],
                            attention_mask=micro_batch["attention_mask"],
                        ).logits.squeeze(-1)
                        micro_bce = F.binary_cross_entropy_with_logits(
                            micro_logits, micro_labels, reduction="none"
                        )
                        # Compute full loss but scale it down for proper gradient accumulation

                        if self.dense_rewards:
                            # Make sure only the tokens within the response mask contribute to the loss
                            masked_micro_bce = micro_bce.masked_fill(
                                micro_batch["response_mask"] == 0, 0.0
                            )
                            masked_micro_bce = masked_micro_bce / micro_batch[
                                "response_mask"
                            ].sum(1).unsqueeze(1)
                            micro_loss = (
                                masked_micro_bce * micro_weights.unsqueeze(1)
                            ).sum() / step_weights.sum()
                        else:
                            micro_loss = (
                                micro_bce * micro_weights
                            ).sum() / step_weights.sum()
                        scaled_loss = micro_loss / (
                            self.args.gradient_accumulation_steps * num_micro_batches
                        )

                    # Backward pass on scaled loss for proper gradient accumulation
                    scaled_loss.backward()
                    step_loss += micro_loss.detach()  # Track full loss for return value

                elif self.classifier_loss == "wgan":
                    with self.accelerator.autocast():
                        # Critic scores (no sigmoid) — shape: [B, T] if dense, else [B]
                        micro_scores = self.reward_model(
                            input_ids=micro_batch["input_ids"],
                            attention_mask=micro_batch["attention_mask"],
                        ).logits.squeeze(-1)

                        micro_labels = micro_labels.float()  # 1 = expert, 0 = policy

                        if self.dense_rewards:
                            # token mask for the generated response region
                            resp_mask = micro_batch["response_mask"].float()  # [B, T]
                            # per-token masks for real/fake
                            real_tok_mask = resp_mask * (
                                micro_labels
                                if micro_labels.dim() == 2
                                else micro_labels.unsqueeze(1)
                            )
                            fake_tok_mask = resp_mask * (
                                1.0
                                - (
                                    micro_labels
                                    if micro_labels.dim() == 2
                                    else micro_labels.unsqueeze(1)
                                )
                            )

                            # mean score over tokens (avoid div-by-zero)
                            real_count = real_tok_mask.sum().clamp_min(1.0)
                            fake_count = fake_tok_mask.sum().clamp_min(1.0)
                            real_mean = (
                                micro_scores * real_tok_mask
                            ).sum() / real_count
                            fake_mean = (
                                micro_scores * fake_tok_mask
                            ).sum() / fake_count

                            # WGAN critic loss: minimise (fake - real)
                            micro_loss = fake_mean - real_mean

                        else:
                            # sequence-level case
                            is_real = labels.view(-1) > 0.5
                            real_scores = micro_scores[is_real]
                            fake_scores = micro_scores[~is_real]

                            # guard against empty split in tiny micro-batches
                            real_mean = (
                                real_scores.mean()
                                if real_scores.numel() > 0
                                else micro_scores.new_zeros(())
                            )
                            fake_mean = (
                                fake_scores.mean()
                                if fake_scores.numel() > 0
                                else micro_scores.new_zeros(())
                            )

                            micro_loss = fake_mean - real_mean

                        # scale for gradient accumulation (keep your existing scaling)
                        scaled_loss = micro_loss / (
                            self.args.gradient_accumulation_steps * num_micro_batches
                        )

                    # Backward pass for critic
                    scaled_loss.backward()
                    step_loss += micro_loss.detach()

                else:
                    raise NotImplementedError(
                        f"Classifier loss function `{self.classifier_loss}` not implemented"
                    )

            total_loss += step_loss

        return total_loss

    # -----------------------------------------------------------------------
    def log(
        self, logs: Dict[str, float], start_time: Optional[float] = None
    ):  # noqa: D401
        # merge local metric buffer
        for k, vlist in self._metrics["train"].items():
            if vlist:
                logs[k] = sum(vlist) / len(vlist)
        self._metrics["train"].clear()
        super().log(logs, start_time)

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
