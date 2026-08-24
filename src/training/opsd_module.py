import os

from trl import GRPOConfig

from src.training.callbacks import SaveBestByMetricCallback
from src.training.opsd_direct import DirectOPSDTrainer
from src.training.opsd_trainer import OPSDTrainer, opsd_reward_func


def run_opsd_training(
    model,
    tokenizer,
    train_dataset,
    cfg,
    val_dataset=None,
):
    opsd_cfg = getattr(cfg, "opsd", {})
    mode = str(getattr(opsd_cfg, "mode", "direct")).lower()
    if mode in {"direct", "non_rl", "weighted_nll", "token"}:
        trainer = DirectOPSDTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            cfg=cfg,
        )
        trainer.train()
        return trainer

    if mode not in {"grpo", "grpo_reward", "rl_reward"}:
        raise ValueError(f"Unknown OPSD mode: {mode}")

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
        save_strategy="steps",
        save_total_limit=1,
        beta=getattr(cfg.training, "beta", 0.0),
    )

    trainer = OPSDTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[opsd_reward_func],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_processing_classes=[None],
        opsd_reward_temperature=getattr(opsd_cfg, "reward_temperature", 1.0),
        opsd_reward_lb=getattr(opsd_cfg, "reward_lb", -5.0),
        opsd_reward_ub=getattr(opsd_cfg, "reward_ub", 5.0),
        opsd_max_micro_batch=getattr(opsd_cfg, "max_micro_batch", 2),
        opsd_log_first_batch=getattr(opsd_cfg, "log_first_batch", True),
    )

    cb = SaveBestByMetricCallback(
        "eval_rewards/opsd_reward_func/mean",
        cfg.training.output_dir,
        greater_is_better=True,
    )
    cb.trainer = trainer
    trainer.add_callback(cb)

    trainer.train()
    best_model_dir = os.path.join(cfg.training.output_dir, "best_model")
    if not os.path.exists(best_model_dir):
        trainer.save_model(best_model_dir)
    return trainer
