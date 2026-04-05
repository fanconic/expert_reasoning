from trl import GRPOConfig, GRPOTrainer
from src.training.callbacks import SaveBestByMetricCallback

def run_grpo_training(
    model, 
    tokenizer, 
    train_dataset, 
    reward_funcs, 
    cfg, 
    val_dataset=None, 
    reward_processing_classes=None
):
    """
    Runs GRPO training and periodically evaluates on validation and test sets.

    Args:
        model, tokenizer: Loaded via unsloth.
        train_dataset: The training dataset.
        reward_funcs: List of reward functions.
        training_cfg: Training configuration (expects attributes like max_steps, etc.).
        val_dataset: Optional validation dataset.
    """
    
    if "gsm8k" in cfg.dataset.name or "aime" in cfg.dataset.name:
        dataset_name = "gsm8k"
    elif "medical" in cfg.dataset.name:
        dataset_name = "medical"
    elif "countdown" in cfg.dataset.name:
        dataset_name = "countdown"
    elif "science" in cfg.dataset.name:
        dataset_name = "scienceqa"
    elif "mmlu" in cfg.dataset.name:
        dataset_name = "mmlu"
    else:
        raise ValueError(f"Unknown dataset name in config: {cfg.dataset.name}")
    grpo_config = GRPOConfig(
        learning_rate=cfg.training.learning_rate,
        adam_beta1=cfg.training.adam_beta1,
        adam_beta2=cfg.training.adam_beta2,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.eval.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=cfg.training.num_generations,
        max_prompt_length=cfg.model.max_prompt_length,
        max_completion_length=cfg.model.max_completion_length,
        max_steps=cfg.training.max_steps,
        save_steps=cfg.eval.eval_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=cfg.training.report_to,
        output_dir=cfg.training.output_dir,
        use_vllm=cfg.model.fast_inference,
        do_eval=cfg.eval.do_eval,
        eval_strategy=cfg.eval.eval_strategy,
        eval_steps=cfg.eval.eval_steps,
        num_train_epochs=cfg.training.epochs,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="steps",  # or "epoch" or "no"
        save_total_limit=1,  # Keep only 2 checkpoints: best + final
        beta=getattr(cfg.training, "beta", 0.0),
    )

    # Instantiate the GRPOTrainer.
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_processing_classes=reward_processing_classes,
    )
    cb = SaveBestByMetricCallback(
            f"eval_rewards/{dataset_name}_correctness_reward_func/mean",
            cfg.training.output_dir, 
            greater_is_better=True
    )
    cb.trainer = trainer
    trainer.add_callback(cb)
        
    trainer.train()
    return trainer
