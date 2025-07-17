from trl import GRPOConfig, GRPOTrainer
import wandb


def run_grpo_training(
    model, tokenizer, train_dataset, reward_funcs, cfg, val_dataset=None
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
        max_prompt_length=None,
        max_completion_length=cfg.model.max_seq_length,
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
        log_completions = True,
        num_completions_to_print=2
    )

    # Instantiate the GRPOTrainer.
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    return trainer
