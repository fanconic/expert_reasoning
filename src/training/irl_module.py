from src.rewards.perturbations import PERTURB_FN_MAP
from src.config.irl_config import IRLConfig
from src.training.airl_trainer_new import AIRLTrainer
import os

def run_irl_training(
    policy_model,
    reward_model,
    policy_tokenizer,
    reward_tokenizer,
    train_dataset,
    reward_funcs,
    cfg,
    val_dataset=None,
    reward_processing_classes=None,
):
    if "gsm8k" in cfg.dataset.name:
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
    irl_config = IRLConfig(
        learning_rate=cfg.model.policy_learning_rate,
        reward_learning_rate=cfg.model.reward_learning_rate,
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
        num_generations=cfg.sampling.num_generations,
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
        use_outcome_rewards=cfg.model.use_outcome_rewards,
        reward_updates_per_policy_step=cfg.model.reward_updates_per_policy_step,
        disc_label_smoothing=cfg.model.disc_label_smoothing,
        disc_temperature=cfg.model.disc_temperature,
        clip_reward_model=cfg.model.clip_reward_model,
        reward_lb=cfg.model.reward_lb,
        reward_ub=cfg.model.reward_ub,
        response_only=cfg.model.response_only,
        num_neg_perturbations_per_expert=cfg.model.num_neg_perturbations_per_expert,
        neg_perturb_fns=[PERTURB_FN_MAP[name] for name in cfg.model.neg_perturb_fns],
        switch_label_if_correct=getattr(cfg.model, "switch_label_if_correct", False),
        neg_sample_weight=cfg.model.neg_sample_weight,
        disc_pairwise_margin=cfg.model.disc_pairwise_margin,
        standard_grpo=cfg.training.standard_grpo,
        mask_truncated_completions=False,
        max_micro_batch=cfg.training.max_micro_batch,
        dense_rewards="full" if cfg.model.dense_rewards==True else cfg.model.dense_rewards,
        advantage_calculation=cfg.model.advantage_calculation,
        dense_gamma=cfg.model.dense_gamma,
        add_expert_to_policy_optim=cfg.model.add_expert_to_policy_optim,
        add_expert_to_policy_balanced=cfg.model.add_expert_to_policy_balanced,
        classifier_loss=cfg.model.classifier_loss,
        normalise_rewards=getattr(cfg.model, "normalise_rewards", True),
        expert_error_rate=getattr(cfg.dataset, "expert_error_rate", 0.0),
        beta=getattr(cfg.training, "beta", 0.0),
        reward_warmup_steps=getattr(cfg.training, "reward_warmup_steps", 0),
        vllm_importance_sampling_correction=False, # set this one to false, else it leads to mismatch (https://github.com/huggingface/trl/issues/4205)
        save_strategy="steps",  # or "epoch" or "no"
        save_total_limit=1,  # Keep only 2 checkpoints: best + final
        load_best_model_at_end=True,  # Load best model when training ends
        metric_for_best_model=f"eval_rewards/{dataset_name}_correctness_reward_func/mean",  # Replace with your actual reward metric name
        greater_is_better=True,  # Set to False if lower is better for your metric
        warmup_reward_dir=getattr(cfg.model, "warmup_reward_dir", None),
        dense_partial_fixed_n=getattr(cfg.model, "dense_partial_fixed_n", None),
        buffer_size=getattr(cfg.training, "buffer_size", 0)
    )

    def formatting_prompt_func(examples):
        """
        For each example:
        1) format the system+user prompt with the chat template,
        2) append an assistant span around the target (<think>…</think><answer>…</answer>).
        """
        prompts = examples["prompt"]  # list of lists of messages
        targets = examples[
            "target"
        ]  # list of "<think>…</think><answer>…</answer>" strings
        texts = []
        for msgs, tgt in zip(prompts, targets):
            # 1) format system+user
            formatted_prompt = policy_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            # 2) wrap the reasoning+answer as the assistant reply
            assistant_block = "<|im_start|>assistant\n" f"{tgt}" "<|im_end|>"
            texts.append(formatted_prompt + assistant_block)
        return {"text": texts}

    train_dataset = train_dataset.map(
        formatting_prompt_func,
        batched=True,
    )
    val_dataset = val_dataset.map(
        formatting_prompt_func,
        batched=True,
    )

    trainer = AIRLTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        reward_funcs=reward_funcs,  # Shall only use them for logging to start (or potential sparse reward)
        policy_tokenizer=policy_tokenizer,
        reward_tokenizer=reward_tokenizer,
        args=irl_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_processing_classes=reward_processing_classes,
    )

    trainer.train()
    trainer.save_model(os.path.join(cfg.training.output_dir, "best_model"))
    return trainer
