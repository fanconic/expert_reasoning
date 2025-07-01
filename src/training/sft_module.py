import numpy as np
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import EvalPrediction

from src.rewards.reward_functions import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)

# 1) Our compute_metrics for HF Trainer

def run_sft_training(model, tokenizer, train_dataset, cfg, val_dataset=None):
    # … your existing SFTConfig, formatting_prompt_func, collator, etc. …
    
    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits, labels = eval_pred
        # a) get predicted token‐ids
        pred_ids = np.argmax(logits, axis=-1)
        # b) decode predictions & references to text
        decoded_preds  = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,   skip_special_tokens=True)
        # c) wrap into the format each reward fn expects
        completions = [[{"content": p}] for p in decoded_preds]
        answers     = decoded_labels

        # d) compute each reward over the whole batch
        out = {}
        out["xmlcount_reward_func"]      = float(np.mean(xmlcount_reward_func(completions)))
        out["soft_format_reward_func"]   = float(np.mean(soft_format_reward_func(completions)))
        out["strict_format_func"] = float(np.mean(strict_format_reward_func(completions)))
        out["int_reward_func"]    = float(np.mean(int_reward_func(completions)))
        out["correctness_reward_func"]   = float(np.mean(correctness_reward_func(None, completions, answers)))

        # e) (optional) also compute plain accuracy
        accs = [1.0 if p == a else 0.0
                for p, a in zip(decoded_preds, decoded_labels)]
        out["accuracy"] = float(np.mean(accs))

        return out


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

    def formatting_prompt_func(examples):
        """
        For each example:
        1) format the system+user prompt with the chat template,
        2) append an assistant span around the target (<think>…</think><answer>…</answer>).
        """
        prompts = examples["prompt"]    # list of lists of messages
        targets = examples["target"]    # list of "<think>…</think><answer>…</answer>" strings
        texts = []
        for msgs, tgt in zip(prompts, targets):
            # 1) format system+user
            formatted_prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False
            )
            # 2) wrap the reasoning+answer as the assistant reply
            assistant_block = (
                "<|im_start|>assistant\n"
                f"{tgt}"
                "<|im_end|>"
            )
            texts.append(formatted_prompt + assistant_block)
        return {"text": texts}


    response_template = "<|im_start|>assistant\n<think>\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
        
    train_dataset = train_dataset.map(formatting_prompt_func, batched = True,)
    val_dataset = val_dataset.map(formatting_prompt_func, batched = True,)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        dataset_text_field="text",
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,     # ← pass it here
        dataset_num_proc=1,
    )

    trainer.train()
    trainer.evaluate()
    return trainer