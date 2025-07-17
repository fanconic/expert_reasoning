from unsloth import FastLanguageModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
try:
    # BitsAndBytes is optional; only needed for 4-bit loading
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

def load_model_and_tokenizer(config):
    """
    Load and configure a language model and its tokenizer using Unsloth's FastLanguageModel.

    Args:
        config: Configuration object containing model parameters including:
            - model.name: Name or path of the pretrained model
            - model.max_seq_length: Maximum sequence length for the model
            - model.load_in_4bit: Whether to load the model in 4-bit quantization
            - model.fast_inference: Whether to enable fast inference optimizations
            - model.lora_rank: Rank for LoRA adapters
            - model.gpu_memory_utilization: Target GPU memory utilization
            - random_state: Random seed for reproducibility

    Returns:
        tuple: (model, tokenizer) - The configured model with LoRA adapters and its tokenizer
    """
    model_name = config.model.name
    max_seq_length = config.model.max_seq_length
    load_in_4bit = config.model.load_in_4bit
    fast_inference = config.model.fast_inference
    lora_rank = config.model.lora_rank
    gpu_memory_utilization = config.model.gpu_memory_utilization
    random_state = config.seed

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
    )
    return model, tokenizer


def irl_load_model_and_tokenizer_trl(config):
    """
    Load policy and reward models with separate LoRA adapters for adversarial IRL training.
    Both models can share the same base architecture but use different adapters for independent training.

    This function is intended for TRL / standard HF training loops and does NOT rely on Unsloth.

    Expected config fields:
        config.model.name : str  -> pretrained model name or local path
        config.model.max_seq_length : int
        config.model.load_in_4bit : bool 
        config.model.lora_rank : int
        config.model.policy_lora_rank : int -> optional, specific rank for policy model
        config.model.reward_lora_rank : int -> optional, specific rank for reward model
        config.model.use_gradient_checkpointing : bool (optional; default True)
        config.seed : int

    Returns:
        tuple: (policy_model, reward_model, tokenizer)
            - policy_model: Causal LM for generating responses
            - reward_model: Model for reward/discriminator function
            - tokenizer: Shared tokenizer for both models
    """
    model_name = config.model.name
    max_seq_length = config.model.max_seq_length
    load_in_4bit = getattr(config.model, "load_in_4bit", False)
    lora_rank = config.model.lora_rank
    random_state = getattr(config, "seed", 42)
    use_grad_ckpt = True if getattr(config.model, "use_gradient_checkpointing", True) else False

    torch.manual_seed(random_state)

    # ----- Tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.model_max_length = max_seq_length

    quantization_config = None
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes not available but load_in_4bit=True was requested.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", 
        quantization_config=quantization_config,
    )

    # If we added a pad token, resize embeddings.
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # ----- Prep for k-bit (QLoRA) training if quantized -----
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Optional gradient checkpointing (saves memory; slight compute overhead)
    if use_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # Some models require this for proper grad ckpt w/ LoRA:
        model.config.use_cache = False

    # ----- LoRA config -----
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # Get specific LoRA ranks for policy and reward models
    policy_lora_rank = getattr(config.model, "policy_lora_rank", lora_rank)
    reward_lora_rank = getattr(config.model, "reward_lora_rank", lora_rank)

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
    policy_model = get_peft_model(model, policy_lora_config)
    
    # Create reward model starting from the same base model
    # Clone the model to have separate instance
    reward_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
    )
    
    if load_in_4bit:
        reward_model = prepare_model_for_kbit_training(reward_model)
    
    if use_grad_ckpt and hasattr(reward_model, "gradient_checkpointing_enable"):
        reward_model.gradient_checkpointing_enable()
        reward_model.config.use_cache = False

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
    
    # Add classification head to reward model
    reward_model.score = torch.nn.Linear(
        reward_model.config.hidden_size, 1, bias=False
    ).to(reward_model.device)
    
    return policy_model, reward_model, tokenizer