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
