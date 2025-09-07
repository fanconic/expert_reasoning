"""airl_config.py
~~~~~~~~~~~~~~~~~~
Configuration dataclass for the :class:`AIRLTrainer` (adversarial inverse‑reinforcement learning trainer).

It extends :class:`transformers.TrainingArguments` and merges the knobs you
already used for knowledge‑distillation‑based IRL with the additional switches
needed for AIRL – namely separate optimisation hyper‑parameters for the policy
and the reward / discriminator, the frequency with which each one is updated,
and the generation controls inherited from GRPO.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from trl.trainer import GRPOConfig

__all__ = ["AIRLConfig"]


@dataclass
class IRLConfig(GRPOConfig):
    r"""
    Configuration class for :class:`~trl.AIRLTrainer`.

    The class carries **all** the knobs expected by the trainer:

    * *Policy* optimisation (learning‑rate, batch size, etc.)
    * *Reward / discriminator* optimisation (ditto)
    * Generation hyper‑parameters reused from GRPO (temperature, top‑p …)
    * AIRL‑specific scheduling – how many discriminator steps per policy
      update, whether to add gradient‑penalty, etc.

    Only the parameters **specific to AIRL** are documented below – everything
    else comes from :class:`transformers.TrainingArguments` and behaves the
    same way.
    """

    # ---------------------------------------------------------------------
    # === Optimisers & schedulers ===
    policy_learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning‑rate for the policy model (AdamW)."},
    )
    reward_learning_rate: float = field(
        default=2e-5,
        metadata={
            "help": "Initial learning‑rate for the reward / discriminator model (AdamW)."
        },
    )

    # Per‑device batch sizes
    policy_per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Per‑device training batch size for the policy model."},
    )
    reward_per_device_train_batch_size: int = field(
        default=8,
        metadata={
            "help": "Per‑device training batch size for the reward / discriminator model."
        },
    )

    # Gradient accumulation
    policy_gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient‑accumulation steps for the policy optimiser."},
    )
    reward_gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient‑accumulation steps for the reward optimiser."},
    )

    # AIRL‑specific schedule: how many reward‑model updates per policy update
    reward_updates_per_policy_step: int = field(
        default=1,
        metadata={
            "help": "Number of discriminator (reward) optimisation steps per policy optimisation step."
        },
    )
    
    disc_label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "Label smoother for the discriminator"
        },
    )
    
    disc_temperature: float = field(
        default=1.0,
        metadata={
            "help": "temperature of the logit rewards"
        },
    )
    clip_reward_model: bool = field(
        default=False,
        metadata={
            "help": "Clip the rewards to the range [reward_lb, reward_ub]"
        },
    )
    reward_lb: float = field(
        default=-1.0,
        metadata={"help": "Lower bound for the rewards"},
    )
    reward_ub: float = field(
        default=1.0,
        metadata={"help": "Upper bound for the rewards"},
    )
    response_only: bool = field(
        default=False,
        metadata={
            "help": "If true, only the response tokens are used for the reward model loss"
        },
    )
    num_neg_perturbations_per_expert: int = field(
        default=0,
        metadata={
            "help": "Number of perturbations to apply to each expert trace (0 = no perturbations)."
        },
    )
    neg_perturb_fns: list[str] = field(
        default_factory=None,
        metadata={
            "help": "List of perturbation functions to apply to the expert traces."
        },
    )
    neg_sample_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for the perturbed negative samples in the reward model loss."
        },
    )
    disc_pairwise_margin: float = field(
        default=0.0,
        metadata={  
            "help": "Margin for the pairwise loss in the discriminator (0 = no margin)."
        },
    )

    use_outcome_rewards: bool = field(
        default=False,
        metadata={
            "help": "If reward_funcs is not None, use reward functions in addition to reward model. If false, but reward_funcs is not None use them only for validation metrics"
        },
    )
    standard_grpo: bool = field(
        default=False,
        metadata={
            "help": "Only trains with the verifiable rewards (no discriminator)."
        },
    )
    max_micro_batch: int = field(
        default=6,
        metadata={
            "help": "Maximum samples per micro-batch through reward model to avoid OOM."
        },
    )
    dense_rewards: bool = field(
        default=False,
        metadata={
            "help": "If true, use token-level (dense) rewards instead of sequence-level (sparse) rewards."
        },
    )
    # ------------------------------------------------------------------
    # === Generation / sampling (copied from GRPOConfig) ===
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Hard truncate the prompt to the last N tokens before generation (None = no limit)."
        },
    )
    max_completion_length: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate for each completion."},
    )
    num_generations: int = field(
        default=4,
        metadata={
            "help": "Number of reasoning traces sampled per prompt (G in GRPO/AIRL)."
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature."},
    )
    top_p: Optional[float] = field(
        default=0.9,
        metadata={"help": "Nucleus sampling probability threshold."},
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Top‑k sampling cutoff (None = disabled)."},
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum top‑p (a.k.a. Mirostat min‑p)."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty (1.0 = disabled)."},
    )

    # ------------------------------------------------------------------
    # === AIRL regularisation knobs ===
    gradient_penalty_coef: float = field(
        default=10.0,
        metadata={
            "help": "Coefficient for gradient‑penalty (R1 style) used during discriminator training."
        },
    )
    entropy_reg_coef: float = field(
        default=0.0,
        metadata={
            "help": "Optional entropy bonus added to the discriminator loss to discourage over‑confidence."
        },
    )

    # ------------------------------------------------------------------
    # === Misc ===
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X update steps (int) or ratio (float in [0,1)).",
        },
    )

    average_tokens_across_devices: bool = field(
        default=True,
        metadata={
            "help": "Average token counts across devices for precise loss normalisation.",
        },
    )

    # ------------------------------------------------------------------
    # === Model‑initialisation helpers ===
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword args forwarded to AutoModel*.from_pretrained when `model` is a string ID/path.",
        },
    )
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the tokenizer / Jinja chat template to use (if overriding the default).",
        },
    )

    # ------------------------------------------------------------------
    # === Data‑processing knobs (intact from your original IRLConfig) ===
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the column that contains text data in the dataset."},
    )
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Misc dataset‑preparation kwargs (currently only `skip_prepare_dataset`).",
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing."},
    )
    eos_token: Optional[str] = field(
        default=None,
        metadata={"help": "Override default EOS token (rarely needed)."},
    )
    stop_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token signalling end‑of‑response (if different from EOS)."},
    )
    stop_token_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "ID of the stop token (mutually exclusive with `stop_token`)."
        },
    )
    pad_token: Optional[str] = field(
        default=None,
        metadata={"help": "Override padding token."},
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Max total sequence length (prompt + completion when packing)."
        },
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Activate sequence packing."},
    )
    packing_strategy: str = field(
        default="ffd",
        metadata={"help": "Packing strategy: 'ffd' or 'wrapped'."},
    )
    padding_free: bool = field(
        default=False,
        metadata={"help": "Enable padding‑free forward pass (Flash‑attention‑2 only)."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": "Pad sequences so their length is a multiple of N (for Flash / XLA efficiency)."
        },
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to pack eval dataset (defaults to same as `packing`)."
        },
    )

    # ------------------------------------------------------------------
    # === Loss‑masking helpers ===
    completion_only_loss: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Compute loss only on completion tokens (for prompt‑completion datasets)."
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": "Compute loss only on assistant messages (for chat datasets)."
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Offload activations to CPU to save GPU memory (slower)."},
    )

    # ------------------------------------------------------------------
    # === Deprecated ===
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated – use `max_length` instead."},
    )

    # ----------------------------------------------------------------------------------
    # Internal plumbing @ init‑time
    # ----------------------------------------------------------------------------------
    def __post_init__(self):
        # bf16 default logic — mirror SFT/GRPO behaviour
        self.bf16 = not self.fp16 if self.bf16 is None else self.bf16

        super().__post_init__()

        # Deprecation shim
        if self.max_seq_length is not None:
            warnings.warn(
                "`max_seq_length` is deprecated and will be removed in v0.20.0; use `max_length` instead.",
                DeprecationWarning,
            )
            self.max_length = self.max_seq_length
