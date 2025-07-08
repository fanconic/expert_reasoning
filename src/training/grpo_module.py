from trl import GRPOConfig, GRPOTrainer
import wandb


def evaluate_model(model, tokenizer, dataset, sampling_params, metric_fn):
    """
    Run inference on the dataset and compute averaged metrics.

    Args:
        model: The trained model.
        tokenizer: The corresponding tokenizer.
        dataset: A dataset (or similar iterable) of evaluation examples.
        sampling_params: Inference parameters (e.g., SamplingParams instance).
        metric_fn: A function that takes (ground_truth, prediction) and returns a dict of metrics.

    Returns:
        A dict containing the averaged metrics.
    """
    metrics = {}
    count = 0
    for example in dataset:
        # Prepare the prompt using the tokenizer.
        text = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        # Generate the output.
        output = (
            model.fast_generate(
                [text],
                sampling_params=sampling_params,
                lora_request=None,
            )[0]
            .outputs[0]
            .text
        )

        # Obtain metrics for this example.
        sample_metrics = metric_fn(example["answer"], output)
        # Accumulate the metrics.
        for key, value in sample_metrics.items():
            metrics[key] = metrics.get(key, 0.0) + value
        count += 1

    # Average each metric over the dataset.
    avg_metrics = {key: value / count for key, value in metrics.items()}
    return avg_metrics


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
    # Calculate max_completion_length.
    max_completion_length = cfg.model.max_seq_length - cfg.training.max_prompt_length

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
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=cfg.training.max_steps,
        save_steps=cfg.training.save_steps,
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
    trainer.evaluate()
    return trainer
