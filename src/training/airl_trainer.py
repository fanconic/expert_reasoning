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
from contextlib import nullcontext
import re
from typing import Any, Dict, List, Optional, Union, Callable
import warnings
import os
from pathlib import Path

# Third-party imports
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset

from accelerate.utils import broadcast_object_list, gather, gather_object
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    Trainer,
    is_wandb_available,
)
from trl import GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import nanstd
from trl.trainer.utils import (
    disable_dropout_in_model,
    pad,
)
import copy
import random
from src.training.reward_model_utils import tokenize_examples, dedup_token_batch, prepare_reward_batch

# Local imports
from src.config.irl_config import IRLConfig

# Conditional imports
if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb


# Type aliases ---------------------------------------------------------------
RewardModelType = Union[str, PreTrainedModel]
PolicyModelType = Union[str, PreTrainedModel]


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
        peft_config=None,
    ) -> None:

        # ---------- Load / init policy --------------------------------------------------
        if isinstance(policy_model, str):
            self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                policy_model, **(args.model_init_kwargs or {})
            )
        else:
            self.policy = policy_model

        # ---------- Reward model --------------------------------------------------------
        if isinstance(reward_model, str):
            self.reward_model: PreTrainedModel = (
                AutoModelForSequenceClassification.from_pretrained(
                    reward_model, num_labels=1, **(args.model_init_kwargs or {})
                )
            )
        else:
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
        self.standard_grpo = args.standard_grpo
        self.dense_rewards = args.dense_rewards
        
        if self.standard_grpo:
            self.use_outcome_rewards = True
            if not isinstance(reward_funcs, list):
                reward_funcs = [reward_funcs]
            else:
                reward_funcs = reward_funcs
        else:
            if reward_funcs is None:
                reward_funcs = [self.reward_model]
            if not isinstance(reward_funcs, list):
                reward_funcs = [self.reward_model, reward_funcs]
            else:
                reward_funcs = [self.reward_model] + reward_funcs

        # Reward processing class --------------------------------------------------------------
        if self.standard_grpo:
            self.use_outcome_rewards = True
            if reward_processing_classes is None:
                reward_processing_classes =  [None] * (len(reward_funcs))
            elif not isinstance(reward_processing_classes, list):
                reward_processing_classes = [reward_processing_classes]
            else:
                reward_processing_classes = reward_processing_classes
                
                if len(reward_processing_classes) != len(reward_funcs):
                    raise ValueError(
                        "The number of reward processing classes must match the number of reward functions."
                    )
        else:
            if reward_processing_classes is None:
                reward_processing_classes = [self.reward_tokenizer] + [None] * (
                    len(reward_funcs) - 1
                )
            elif not isinstance(reward_processing_classes, list):
                reward_processing_classes = [
                    self.reward_tokenizer,
                    reward_processing_classes,
                ]
            else:
                reward_processing_classes = [self.reward_tokenizer] + reward_processing_classes
                
                if len(reward_processing_classes) != len(reward_funcs):
                    raise ValueError(
                        "The number of reward processing classes must match the number of reward functions."
                    )

        # ---- Prepare backend  ---------------------------------------------------------
        super().__init__(
            model=self.policy,  # base Trainer still expects a *single* model; we wrap reward manually
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
        # Use reward-specific hyperparams
        opt_kwargs["lr"] = args.reward_learning_rate
        opt_kwargs["weight_decay"] = getattr(args, "reward_weight_decay", opt_kwargs.get("weight_decay", 0.0))
        self.reward_optimizer = opt_cls(self.reward_model.parameters(), **opt_kwargs)
        self.reward_optimizer.zero_grad()
        
        if not self.use_outcome_rewards:
            self.reward_weights = torch.zeros_like(
                self.reward_weights, dtype=torch.float32
            )
            self.reward_weights[0] = 1.0  # Only the reward model is used for training

        self.eps = getattr(args, "disc_label_smoothing", 0.1)
        self.disc_temperature = getattr(args, "disc_temperature", 1.0)
        self.clip_reward_model = getattr(args, "clip_reward_model", False)
        self.reward_lb = getattr(args, "reward_lb", -1.0)
        self.reward_ub = getattr(args, "reward_ub", 1.0)
        self.response_only = getattr(args, "response_only", False)
        
        # Negatives from perturbed expert reasonings
        self.neg_perturb_fns = getattr(args, "neg_perturb_fns", None)  # List[Callable[[str], str]] or None
        self.num_neg_perturbations_per_expert = getattr(args, "num_neg_perturbations_per_expert", 1)
        self.neg_sample_weight = getattr(args, "neg_sample_weight", 1.0)  # weight in BCE
        self.disc_pairwise_margin = getattr(args, "disc_pairwise_margin", 0.0)  # >0 to enable pairwise hinge
        self.neg_label_smoothing = getattr(args, "neg_label_smoothing", None)  # defaults to self.eps if None

    # -----------------------------------------------------------------------
    # Data utilities
    # -----------------------------------------------------------------------
    def _make_perturbed_completions(self, expert_completions: List[List[Dict]]) -> List[List[Dict]]:
        """
        expert_completions: list of chat-format completions
            [[{"role": "assistant", "content": "..."}], ...]
        Returns the same format, but perturbed.
        """
        if not self.neg_perturb_fns or self.num_neg_perturbations_per_expert == 0:
            return [], []
        fns = self.neg_perturb_fns if isinstance(self.neg_perturb_fns, list) else [self.neg_perturb_fns]
        perturbed, src_idx = [], []
        for i, comp in enumerate(expert_completions):
            base = comp[0]["content"]
            for _ in range(max(1, int(self.num_neg_perturbations_per_expert))):
                fn = random.choice(fns)
                try:
                    corrupted = fn(base)
                except Exception:
                    corrupted = base
                perturbed.append([{"role": "assistant", "content": corrupted}])
                src_idx.append(i)
        return perturbed, src_idx
    
    def _dedup_token_batch(
        self, tokens: Dict[str, torch.Tensor]
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Deduplicate rows in a token batch by (input_ids, attention_mask).
        Returns:
        - unique_tokens: same dict with only unique rows
        - counts: float tensor [n_unique] with multiplicity per unique row
        - inverse: long tensor [n_rows] mapping each original row -> unique idx
        """
        ids = tokens["input_ids"]
        mask = tokens["attention_mask"]

        key_to_uidx = {}
        unique_idx = []
        counts_py = []
        # Build inverse map on CPU to keep it light
        inv = torch.empty(ids.size(0), dtype=torch.long)

        for i in range(ids.size(0)):
            # bytes keys avoid huge tuples and are fast to hash
            key = (ids[i].detach().cpu().numpy().tobytes(),
                mask[i].detach().cpu().numpy().tobytes())
            j = key_to_uidx.get(key)
            if j is None:
                j = len(unique_idx)
                key_to_uidx[key] = j
                unique_idx.append(i)
                counts_py.append(1)
            else:
                counts_py[j] += 1
            inv[i] = j

        unique_tokens = {k: v[unique_idx] for k, v in tokens.items()}
        counts = torch.tensor(counts_py, dtype=torch.float32, device=ids.device)
        inverse = inv.to(ids.device)
        return unique_tokens, counts, inverse
        
    def _tokenise_examples(
        self, prompts, expert_completions, policy_completions, tokenizer
    ):
        """Tokenise list of {prompt, completion} dicts into model inputs."""
        if self.response_only:
            expert_messages = [{"messages": [{"role": "system", "content": ""}] + c} for c in expert_completions]
            expert_texts = [apply_chat_template(x, tokenizer)["text"] for x in expert_messages]
            policy_messages = [{"messages": [{"role": "system", "content": ""}] + c} for c in policy_completions]
            policy_texts = [apply_chat_template(x, tokenizer)["text"] for x in policy_messages]
        
        else:
            # Create texts
            expert_messages = [{"messages": p + c} for p, c in zip(prompts, expert_completions)]
            expert_texts = [apply_chat_template(x, tokenizer)["text"] for x in expert_messages]
            policy_messages = [{"messages": p + c} for p, c in zip(prompts, policy_completions)]
            policy_texts = [apply_chat_template(x, tokenizer)["text"] for x in policy_messages]

        # Tokenize
        expert_tokens = tokenizer(
            text=expert_texts,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right",
            max_length=self.max_completion_length,
            truncation=True
        )
        policy_tokens = tokenizer(
            text=policy_texts,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right",
            max_length=self.max_completion_length,
            truncation=True
        )

        # Put on right device
        expert_tokens = {
            k: v.to(self.accelerator.device) for k, v in expert_tokens.items()
        }
        policy_tokens = {
            k: v.to(self.accelerator.device) for k, v in policy_tokens.items()
        }
        return expert_tokens, policy_tokens


    #def _move_model_to_vllm(self, *args, **kwargs): return None
    
    # -----------------------------------------------------------------------
    # Core overwrites overrides
    # -----------------------------------------------------------------------
    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [
            key
            for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        ]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
            )
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c} for p, c in zip(prompts, completions)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )

                    # Create response mask that's 1 only for completion tokens
                    prompt_only_tokens = reward_processing_class(
                        text=[p[0]["content"] if is_conversational(inputs[0]) else p for p in prompts],
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    prompt_lengths = prompt_only_tokens["input_ids"].shape[1]
                    response_mask = torch.zeros_like(reward_inputs["attention_mask"])
                    response_mask[:, prompt_lengths:] = reward_inputs["attention_mask"][:, prompt_lengths:]
                    reward_inputs["response_mask"] = response_mask
                    reward_inputs = {k: v.to(self.accelerator.device) for k, v in reward_inputs.items()}
                    
                    # expand the rewards_per_func if dense rewards
                    if self.dense_rewards and rewards_per_func.dim() == 2:
                        seq_len = reward_inputs["input_ids"].shape[1]
                        rewards_per_func = rewards_per_func.unsqueeze(-1).expand(-1, -1, seq_len).contiguous()
                        
                    with torch.inference_mode():
                        # At this point we use the logits of the reward model as reward scores (IMPORTANT)
                        if not self.dense_rewards:
                            reward_from_model =  reward_func(**reward_inputs).logits[:, 0] / self.disc_temperature
                            if self.clip_reward_model:
                                reward_from_model = torch.clamp(reward_from_model, self.reward_lb, self.reward_ub)
                            rewards_per_func[:, i] = reward_from_model
                        else:
                            reward_from_model =  reward_func(
                                input_ids = reward_inputs["input_ids"],
                                attention_mask = reward_inputs["attention_mask"],
                                ).logits[:, :, 0] / self.disc_temperature
                            reward_from_model = reward_from_model * reward_inputs["response_mask"]
                            if self.clip_reward_model:
                                reward_from_model = torch.clamp(reward_from_model, self.reward_lb, self.reward_ub)
                            rewards_per_func[:, i, :] = reward_from_model
                else:
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    if not self.dense_rewards:
                        rewards_per_func[:, i] = torch.tensor(
                            output_reward_func, dtype=torch.float32, device=device
                        )
                    else:
                        seq_len = rewards_per_func.size(2)
                        rewards_per_func[:, i, :] = torch.tensor(
                            output_reward_func, dtype=torch.float32, device=device
                            ).unsqueeze(1).repeat(1,seq_len)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
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
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = {k: v.to(self.accelerator.device) for k, v in prompt_inputs.items()}
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            pad_token = self.processing_class.pad_token
            def strip_leading_tokens(text):
                while text.startswith(pad_token):
                    text = text.removeprefix(pad_token)
                return text

            if pad_token is not None:
                prompts_text = [
                    strip_leading_tokens(text) for text in prompts_text
                ]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        all_prompts_text, 
                        sampling_params=sampling_params, 
                        use_tqdm=False,
                        lora_request = self.model.load_lora('grpo_trainer_lora_model', load_tensors = True)
                    )

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # After the generations and before assigning the reward, we need to fit the classifier
        if mode == "train":
            classifier_loss = self._update_reward_model(inputs, prompts, completions)
            self._metrics[mode]["loss/classifier"].append(classifier_loss.item())

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )

        # Apply weights to each reward function's output and sum
        if not self.dense_rewards:
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)
            
             # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            
        else:
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0).unsqueeze(-1)
            ).nansum(dim=1)
            
             # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations, rewards.size(1)).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations, rewards.size(1)).std(dim=1)
            
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
       
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = (
            advantages.clone()
        )  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(
            agg_completion_lengths
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if (
            len(term_completion_lengths) == 0
        ):  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            if not self.dense_rewards:
                self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            else:
                self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].mean(-1).tolist())
        if not self.dense_rewards:
            self._textual_logs["advantages"].extend(all_process_advantages.tolist())
        else:
            self._textual_logs["advantages"].extend(all_process_advantages.mean(-1).tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

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
        _per_out = self._make_perturbed_completions(expert_completions)
        perturbed_completions = _per_out[0] if isinstance(_per_out, tuple) else _per_out

        # Tokenise and prepare all inputs
        expert_tokens, policy_tokens = tokenize_examples(
            prompts, expert_completions, policy_completions,
            self.reward_tokenizer, self.max_completion_length,
            device, self.response_only
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
                prompts[:len(perturbed_completions)],
                expert_completions[:len(perturbed_completions)],
                perturbed_completions,
                self.reward_tokenizer,
                self.max_completion_length,
                device,
                self.response_only
            )
            n_per = perturbed_tokens["input_ids"].size(0)

        # Prepare batched data for training
        batch, labels, weights = prepare_reward_batch(
            expert_tokens, policy_tokens, perturbed_tokens,
            n_pos, n_pol, n_per, device,
            self.eps, self.neg_label_smoothing,
            self.neg_sample_weight,
            self.num_neg_perturbations_per_expert,
            self.args.gradient_accumulation_steps
        )

        # Training loop over gradient accumulation steps
        total_loss = self._train_reward_model_loop(
            batch, labels, weights,
            n_pos, n_pol, n_per,
            pos_counts
        )

        # Optimizer step if needed
        if self.state.global_step % self.reward_updates_per_policy_step == 0 and not self.standard_grpo:
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
        per_per_step = n_per // self.args.gradient_accumulation_steps if n_per > 0 else 0
        have_perturbed = n_per > 0

        
        total_loss = 0
        for step in range(self.args.gradient_accumulation_steps):
            # Get slices for this step
            pos_slice = slice(
                step * pos_per_step,
                (step + 1) * pos_per_step if step < self.args.gradient_accumulation_steps - 1 else n_pos
            )
            pol_slice = slice(
                step * pol_per_step,
                (step + 1) * pol_per_step if step < self.args.gradient_accumulation_steps - 1 else n_pol
            )

            # Prepare step data
            if have_perturbed:
                per_slice = slice(
                    step * per_per_step,
                    (step + 1) * per_per_step if step < self.args.gradient_accumulation_steps - 1 else n_per
                )
                step_batch = {
                    k: torch.cat([
                        batch[k][:n_pos][pos_slice],
                        batch[k][n_pos:n_pos+n_pol][pol_slice],
                        batch[k][n_pos+n_pol:][per_slice]
                    ], dim=0)
                    for k in batch.keys()
                }
                step_labels = torch.cat([
                    labels[:n_pos][pos_slice],
                    labels[n_pos:n_pos+n_pol][pol_slice],
                    labels[n_pos+n_pol:][per_slice]
                ])
                step_weights = torch.cat([
                    weights[:n_pos][pos_slice],
                    weights[n_pos:n_pos+n_pol][pol_slice],
                    weights[n_pos+n_pol:][per_slice]
                ])
            else:
                step_batch = {
                    k: torch.cat([
                        batch[k][:n_pos][pos_slice],
                        batch[k][n_pos:][pol_slice]
                    ], dim=0)
                    for k in batch.keys()
                }
                step_labels = torch.cat([
                    labels[:n_pos][pos_slice],
                    labels[n_pos:][pol_slice]
                ])
                step_weights = torch.cat([
                    weights[:n_pos][pos_slice],
                    weights[n_pos:][pol_slice]
                ])

            # Process micro-batches
            step_size = step_batch["input_ids"].size(0)
            num_micro_batches = (step_size + self.max_micro_batch - 1) // self.max_micro_batch
            micro_batch_size = (step_size + num_micro_batches - 1) // num_micro_batches

            step_loss = 0
            for micro_idx in range(num_micro_batches):
                start_idx = micro_idx * micro_batch_size
                end_idx = min((micro_idx + 1) * micro_batch_size, step_size)
                
                micro_batch = {k: v[start_idx:end_idx] for k, v in step_batch.items()}
                micro_labels = step_labels[start_idx:end_idx]
                micro_weights = step_weights[start_idx:end_idx]

                if self.dense_rewards:
                    micro_labels = micro_labels.unsqueeze(1).repeat(1,micro_batch["input_ids"].size(1))
                
                # Forward pass with micro-batch
                with self.accelerator.autocast():
                    micro_logits = self.reward_model(
                        input_ids=micro_batch["input_ids"], 
                        attention_mask=micro_batch["attention_mask"]).logits.squeeze(-1)
                    micro_bce = F.binary_cross_entropy_with_logits(
                        micro_logits, micro_labels, reduction="none"
                    )
                    # Compute full loss but scale it down for proper gradient accumulation
                    
                    if self.dense_rewards:
                        # Make sure only the tokens within the response mask contribute to the loss
                        masked_micro_bce = micro_bce.masked_fill(micro_batch["response_mask"] == 0, 0.0)
                        masked_micro_bce = masked_micro_bce / micro_batch["response_mask"].sum(1).unsqueeze(1)
                        micro_loss = (masked_micro_bce * micro_weights.unsqueeze(1)).sum() / step_weights.sum()
                    else:
                        micro_loss = (micro_bce * micro_weights).sum() / step_weights.sum()
                    scaled_loss = micro_loss / (self.args.gradient_accumulation_steps * num_micro_batches)
                
                # Backward pass on scaled loss for proper gradient accumulation
                scaled_loss.backward()
                step_loss += micro_loss.detach()  # Track full loss for return value

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
            torch.save(self.reward_optimizer.state_dict(), reward_dir / "reward_optimizer.pt")


# ---------------------------------------------------------------------------
__all__ = ["AIRLTrainer"]
