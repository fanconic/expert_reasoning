from vllm import SamplingParams
from src.training.callbacks import GenerationEvalCallback
from src.config.irl_config import IRLConfig
from src.training.irl_trainer import IRLTrainer
from src.training.airl_trainer import AIRLTrainer

from src.rewards.reward_functions import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)

reward_fns = [
    ("xmlcount_reward_func", xmlcount_reward_func),
    ("soft_format_reward_func", soft_format_reward_func),
    ("strict_format_reward_func", strict_format_reward_func),
    ("int_reward_func", int_reward_func),
    ("correctness_reward_func", correctness_reward_func),
]


def run_irl_training(
    policy_model,
    reward_model,
    policy_tokenizer,
    reward_tokenizer,
    train_dataset,
    reward_funcs,
    cfg,
    val_dataset=None,
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
        log_completions=True,
        num_completions_to_print=2,
    )

    # sampling params for generation
    sampling_params = SamplingParams(
        max_tokens=cfg.model.max_seq_length,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
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

    # create the callback
    gen_eval_cb = GenerationEvalCallback(
        val_dataset=val_dataset,
        tokenizer=policy_tokenizer,
        reward_fns=reward_fns,
        sampling_params=sampling_params,
        batch_size=cfg.eval.per_device_eval_batch_size,
        output_dir=cfg.training.output_dir,
    )

    trainer = AIRLTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        reward_funcs=reward_funcs, # Shall only use them for logging to start (or potential sparse reward)
        policy_tokenizer=policy_tokenizer,
        reward_tokenizer=reward_tokenizer,
        args=irl_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[gen_eval_cb],
    )

    trainer.train()
    trainer.evaluate()
    return trainer
