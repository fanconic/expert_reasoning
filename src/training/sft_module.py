from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer


from vllm import SamplingParams
from src.training.callbacks import GenerationEvalCallback


from src.rewards.reward_functions import (
    strict_format_reward_func,
    soft_format_reward_func,
    int_reward_func,
    xmlcount_reward_func,
    gsm8k_correctness_reward_func,
    countdown_correctness_function,
    medical_correctness_reward_func,
)

def run_sft_training(model, tokenizer, train_dataset, cfg, val_dataset=None):
    
    if cfg.dataset.name == "gsm8k" or cfg.dataset.name == "gsm8k_kd":
        reward_fns = [
            ("xmlcount_reward_func", xmlcount_reward_func),
            ("soft_format_reward_func", soft_format_reward_func),
            ("strict_format_reward_func", strict_format_reward_func),
            ("int_reward_func", int_reward_func),
            ("gsm8k_correctness_reward_func", gsm8k_correctness_reward_func),
        ]
    elif cfg.dataset.name == "countdown" or cfg.dataset.name == "countdown_kd":
        reward_fns = [
            ("xmlcount_reward_func", xmlcount_reward_func),
            ("soft_format_reward_func", soft_format_reward_func),
            ("strict_format_reward_func", strict_format_reward_func),
            ("countdown_correctness_reward_func", countdown_correctness_function),
        ]
    elif cfg.dataset.name == "medical" or cfg.dataset.name == "medical_kd":
        reward_fns = [
            ("xmlcount_reward_func", xmlcount_reward_func),
            ("soft_format_reward_func", soft_format_reward_func),
            ("strict_format_reward_func", strict_format_reward_func),
            ("medical_correctness_reward_func", medical_correctness_reward_func),
        ]
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")
    

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
        save_strategy="no",  # Setting this to no, because I save it on my custom call-back
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=cfg.training.report_to,
        output_dir=cfg.training.output_dir,
        do_eval=cfg.eval.do_eval,
        eval_strategy=cfg.eval.eval_strategy,
        eval_steps=cfg.eval.eval_steps,
        eval_accumulation_steps=cfg.eval.eval_accumulation_steps,
        prediction_loss_only=cfg.eval.prediction_loss_only,
        num_train_epochs=cfg.training.epochs,
        dataset_kwargs = {"skip_prepare_dataset": True},   # YES
    )

    # sampling params for generation
    sampling_params = SamplingParams(
        max_tokens=cfg.model.max_prompt_length + cfg.model.max_completion_length,
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
        targets = examples["target"]  # list of "<think>…</think><answer>…</answer>" strings
        new_promtps, completions = [], []
        for msgs, tgt in zip(prompts, targets):
            # 1) format system+user
            formatted_prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            completion = tgt + tokenizer.eos_token
            new_promtps.append(formatted_prompt)
            completions.append(completion)
        return {"prompt": new_promtps, "completion": completions}
    
    
    def tokenize_completion_only(examples, tokenizer, max_length):
        prompts = examples["prompt"]         # already string prompts (chat templated)
        completions = examples["completion"] # already string completions (target + eos)

        input_ids_list, attn_list, labels_list = [], [], []

        for p, c in zip(prompts, completions):
            p_ids = tokenizer(p, add_special_tokens=False).input_ids
            c_ids = tokenizer(c, add_special_tokens=False).input_ids

            input_ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + c_ids

            # truncate (keep the *end* so completion survives; or keep start—pick your preference)
            if len(input_ids) > max_length:
                # keep tail (often better if completion is at end)
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]

                # if we truncated into the prompt/completion boundary, we may have lost some prompt masking;
                # still fine because labels for prompt are -100 wherever prompt remains.

            attention_mask = [1] * len(input_ids)

            input_ids_list.append(input_ids)
            attn_list.append(attention_mask)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "attention_mask": attn_list, "labels": labels_list}

    max_len = cfg.model.max_prompt_length + cfg.model.max_completion_length

    train_dataset = train_dataset.map(formatting_prompt_func, batched=True)
    val_dataset   = val_dataset.map(formatting_prompt_func, batched=True)

    train_dataset = train_dataset.map(
        lambda ex: tokenize_completion_only(ex, tokenizer, max_len),
        batched=True,
    )

    val_dataset = val_dataset.map(
        lambda ex: tokenize_completion_only(ex, tokenizer, max_len),
        batched=True,
    )

    # create the callback
    gen_eval_cb = GenerationEvalCallback(
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        reward_fns=reward_fns,
        sampling_params=sampling_params,
        batch_size=cfg.eval.per_device_eval_batch_size,
        output_dir=cfg.training.output_dir,
        metric_for_best_model=reward_fns[-1][0]
    )
    
    def get_collator(tokenizer):
        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
    
    collator = get_collator(tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        dataset_text_field="text",
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_num_proc=1,
        callbacks=[gen_eval_cb],
        data_collator=collator,
    )

    trainer.train()
    return trainer
