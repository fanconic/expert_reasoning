from typing import Tuple
import os

import torch
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)

try:
    # BitsAndBytes is optional; only needed for 4-bit loading
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


def irl_load_model_and_tokenizer_trl(
    config,
    pretrained=False,
    checkpoint=None
) -> Tuple[torch.nn.Module, torch.nn.Module, AutoTokenizer, AutoTokenizer]:
    """
    Load policy and reward models with separate LoRA adapters for adversarial IRL training.
    Both models can share the same base architecture but use different adapters for independent training.
    This function is intended for TRL / standard HF training loops and does NOT rely on Unsloth.

    Args:
        config: Configuration object with the following required fields:
            model.policy_name (str): Pretrained policy model name or local path
            model.reward_name (str): Pretrained reward model name or local path
            model.max_seq_length (int): Maximum sequence length for tokenization
            model.load_in_4bit (bool): Whether to load models in 4-bit quantization
            model.lora_rank (int): Default LoRA rank for both models
            model.policy_lora_rank (int, optional): Specific rank for policy model
            model.reward_lora_rank (int, optional): Specific rank for reward model
            model.use_gradient_checkpointing (bool, optional): Enable gradient checkpointing
            seed (int): Random seed for initialization

    Returns:
        Tuple containing:
            policy_model (torch.nn.Module): Causal LM for generating responses
            reward_model (torch.nn.Module): Model for reward/discriminator function
            policy_tokenizer (AutoTokenizer): Tokenizer for the policy model
            reward_tokenizer (AutoTokenizer): Tokenizer for the reward model

    Raises:
        ImportError: If bitsandbytes is not available when load_in_4bit=True
    """
    policy_model_name = config.model.policy_name
    reward_model_name = config.model.reward_name
    max_seq_length = config.model.max_prompt_length + config.model.max_completion_length
    load_in_4bit = config.model.load_in_4bit
    lora_rank = config.model.lora_rank
    random_state = config.seed
    use_grad_ckpt = config.model.use_gradient_checkpointing
    
    reward_model_class = AutoModelForTokenClassification if config.model.dense_rewards else AutoModelForSequenceClassification

    torch.manual_seed(random_state)

    def setup_tokenizer(model_name: str, max_length: int) -> AutoTokenizer:
        """Helper function to setup and configure a tokenizer with consistent settings"""
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
        tokenizer.model_max_length = max_length
        return tokenizer

    # --------------------------------------------------------------
    # Tokenizers
    policy_tokenizer = setup_tokenizer(policy_model_name, max_seq_length)
    reward_tokenizer = setup_tokenizer(reward_model_name, max_seq_length)

    # --------------------------------------------------------------
    # Quantization config for 4-bit loading
    quantization_config = None
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError(
                "bitsandbytes not available but load_in_4bit=True was requested."
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # --------------------------------------------------------------
    # LoRA adapter target modules
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    policy_lora_rank = getattr(config.model, "policy_lora_rank", lora_rank)
    reward_lora_rank = getattr(config.model, "reward_lora_rank", lora_rank)

    # --------------------------------------------------------------
    # Policy Model
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        use_cache=False if use_grad_ckpt else True,  # Set at config level
    )

    # If we added a pad token, resize embeddings.
    if len(policy_tokenizer) != policy_model.get_input_embeddings().weight.shape[0]:
        policy_model.resize_token_embeddings(len(policy_tokenizer))

    # ----- Prep for k-bit (QLoRA) training if quantized -----
    if load_in_4bit:
        policy_model = prepare_model_for_kbit_training(
            policy_model, use_gradient_checkpointing=use_grad_ckpt
        )
        
    if pretrained:
        policy_model = PeftModel.from_pretrained(policy_model, checkpoint, strict=True)
        print("Loaded Policy Model strictly")
    else:

        # Create policy model with its own LoRA adapter
        policy_lora_config = LoraConfig(
            r=policy_lora_rank,
            lora_alpha=policy_lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
        )
        policy_model = get_peft_model(policy_model, policy_lora_config)

        if hasattr(policy_model, "enable_input_require_grads"):
            policy_model.enable_input_require_grads()

    # --------------------------------------------------------------
    # Reward Model
    reward_model = reward_model_class.from_pretrained(
        reward_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        use_cache=False if use_grad_ckpt else True,  # Set at config level
        num_labels=1,
    )
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    if load_in_4bit:
        reward_model = prepare_model_for_kbit_training(
            reward_model, use_gradient_checkpointing=use_grad_ckpt
        )
        
    if pretrained:
        reward_checkpoint = os.path.join(checkpoint, "reward_model")
        reward_model = PeftModel.from_pretrained(reward_model, reward_checkpoint, strict=True)
        print("Loaded Reward Model strictly")
        
    else:

        # Add LoRA adapter to reward model with potentially different config
        reward_lora_config = LoraConfig(
            r=reward_lora_rank,
            lora_alpha=reward_lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.SEQ_CLS,  # Use sequence classification for reward model
            target_modules=target_modules,
            inference_mode=False,
        )
        reward_model = get_peft_model(reward_model, reward_lora_config)

    if hasattr(reward_model, "enable_input_require_grads"):
        reward_model.enable_input_require_grads()

    def configure_gradient_checkpointing(model: torch.nn.Module, enable: bool):
        """Helper function to configure gradient checkpointing for a model"""
        if enable:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            model.config.use_cache = False
            if hasattr(model, "base_model"):
                model.base_model.config.use_cache = False
        else:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            if hasattr(model, "base_model"):
                model.base_model.config.use_cache = True

    # --------------------------------------------------------------
    # Configure gradient checkpointing for both models
    configure_gradient_checkpointing(policy_model, use_grad_ckpt)
    configure_gradient_checkpointing(reward_model, use_grad_ckpt)

    return policy_model, reward_model, policy_tokenizer, reward_tokenizer
