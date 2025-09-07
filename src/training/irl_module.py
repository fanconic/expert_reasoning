from vllm import SamplingParams
from src.rewards.perturbations import PERTURB_FN_MAP
from src.config.irl_config import IRLConfig
from src.training.airl_trainer import AIRLTrainer

from src.rewards.reward_functions import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)

def run_irl_training(
    policy_model,
    reward_model,
    policy_tokenizer,
    reward_tokenizer,
    train_dataset,
    reward_funcs,
    cfg,
    val_dataset=None,
    reward_processing_classes=None
):

    irl_config = IRLConfig(
        policy_learning_rate=cfg.model.policy_learning_rate,
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
        neg_sample_weight=cfg.model.neg_sample_weight,
        disc_pairwise_margin=cfg.model.disc_pairwise_margin,
        standard_grpo=cfg.training.standard_grpo,
        mask_truncated_completions=False,
        max_micro_batch=cfg.training.max_micro_batch,
        dense_rewards=cfg.model.dense_rewards
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
    trainer.evaluate()
    return trainer
