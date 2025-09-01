from unsloth import FastLanguageModel


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
    gpu_memory_utilization = config.model.policy_gpu_memory_utilization
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

def irl_load_model_and_tokenizer(config):
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
    # Policy model and tokenizer
    
    policy_model_name = config.model.policy_name
    reward_model_name = config.model.reward_name
    max_seq_length = config.model.max_seq_length
    load_in_4bit = config.model.load_in_4bit
    policy_lora_rank = config.model.policy_lora_rank
    reward_lora_rank = config.model.reward_lora_rank
    random_state = config.seed
    fast_inference = config.model.fast_inference
    lora_rank = config.model.lora_rank
    policy_gpu_memory_utilization = config.model.policy_gpu_memory_utilization
    reward_gpu_memory_utilization = config.model.reward_gpu_memory_utilization
    
    policy_model, policy_tokenizer = FastLanguageModel.from_pretrained(
        model_name=policy_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=policy_gpu_memory_utilization,
    )

    policy_model = FastLanguageModel.get_peft_model(
        policy_model,
        r=policy_lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=policy_lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
    )
    print("Policy model loaded.")
    
    
    # Reward model and tokenizer
    reward_model, reward_tokenizer = FastLanguageModel.from_pretrained(
        model_name=reward_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=reward_lora_rank,
        gpu_memory_utilization=reward_gpu_memory_utilization,
        num_labels = 1,
    )

    reward_model = FastLanguageModel.get_peft_model(
        reward_model,
        r=reward_lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=reward_lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
    )
    print("Reward model loaded.")
    
    return policy_model, reward_model, policy_tokenizer, reward_tokenizer