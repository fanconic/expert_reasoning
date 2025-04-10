from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def run_sft_training(model, tokenizer, train_dataset, cfg, val_dataset=None):
    """
    Runs SFT training and periodically evaluates on validation and test sets.

    Args:
        model, tokenizer: Loaded via unsloth.
        train_dataset: The training dataset.
        reward_funcs: List of reward functions.
        training_cfg: Training configuration (expects attributes like max_steps, etc.).
        val_dataset: Optional validation dataset.
    """

    sft_config = SFTConfig(
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
        max_steps=cfg.training.max_steps,
        save_steps=cfg.training.save_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=cfg.training.report_to,
        output_dir=cfg.training.output_dir,
        do_eval=cfg.eval.do_eval,
        eval_strategy=cfg.eval.eval_strategy,
        eval_steps=cfg.eval.eval_steps,
    )

    def formatting_prompt_func(example):
        # Use the tokenizer's chat template to format the messages
        formatted_prompt = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=False
        )
        return [formatted_prompt]

    response_template = "<think>\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Instantiate the SFTTrainer.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_prompt_func,
        data_collator=collator,
        dataset_num_proc=1,
    )

    trainer.train()
    return trainer
