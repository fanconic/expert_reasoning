# evaluate.py
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from src.utils.utils import set_seed
from src.data.dataset import get_dataset  # same as training
from src.models.model_module_trl import irl_load_model_and_tokenizer_trl
from src.rewards.reward_functions import (
    strict_format_reward_func,
    soft_format_reward_func,
    answer_reward_function,
    eval_correctness_gsm8k,
    eval_correctness_countdown,
    eval_correctness_medical_o1,
    int_reward_func,
    xmlcount_reward_func,
    correctness_reward_func,
)
import torch
import numpy as np
from src.eval.eval_module import compute_pass_at_k, compute_success_at_k_from_scores, compute_oracle_at_1_from_N
import wandb
wandb.login()
from trl.trainer.grpo_trainer import maybe_apply_chat_template
import json
from trl.data_utils import apply_chat_template  # add this import at top

@torch.no_grad()
def score_with_reward_model(
    reward_model, reward_tokenizer, prompts_msgs, decoded_per_prompt, dense=False
):
    """
    Args:
        prompts_msgs:        list[list[dict]]  # your 'prompts' (chat messages)
        decoded_per_prompt:  list[list[str]]   # per prompt N completions (strings)
    Returns:
        all_scores: list[list[float]] aligned with decoded_per_prompt
    """
    device = next(reward_model.parameters()).device
    texts = []
    idx_slices = []  # [(start, end), ...] per prompt to split back
    start = 0
    for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
        for c in completions:
            msgs = p_msgs + [{"role": "assistant", "content": c}]
            texts.append(apply_chat_template({"messages": msgs}, reward_tokenizer)["text"])
        end = start + len(completions)
        idx_slices.append((start, end))
        start = end

    if len(texts) == 0:
        return [[] for _ in prompts_msgs]

    enc = reward_tokenizer(
        text=texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        padding_side="left" if dense else "right",
    ).to(device)

    with torch.autocast("cuda"):
        logits = reward_model(**enc).logits.squeeze(-1)  # [total]
        
    if dense:
        completion_texts = []
        for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
            for c in completions:
                completion_texts.append(c + reward_tokenizer.eos_token)

                
        response_mask = reward_tokenizer(
            text=completion_texts,
            return_tensors="pt",
            padding="max_length",
            add_special_tokens=False,
            padding_side="left",
            max_length=logits.size(1)
        ).to(device)["attention_mask"]
        
        logits = logits.masked_fill(response_mask == 0.0, np.nan)
    logits = logits.detach().float().cpu().tolist()

    # split back per prompt
    all_scores = []
    for s, e in idx_slices:
        all_scores.append(logits[s:e])
    return all_scores



def save_results_to_jsonl(filename, results):
    """Save evaluation results to a JSONL file."""
    with open(filename, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')


@hydra.main(config_path="configs", config_name="config_irl_eval", version_base="1.3")
def main(cfg: DictConfig):
    """
    Evaluate a trained model and compute pass@k on a dataset split.
    """

    print("Evaluation configuration:\n", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    
    
    if cfg.dataset.name == "gsm8k" or cfg.dataset.name == "gsm8k_kd":
        reward_fns = [
            ("xmlcount_reward_func", xmlcount_reward_func),
            ("soft_format_reward_func", soft_format_reward_func),
            ("strict_format_reward_func", strict_format_reward_func),
            ("int_reward_func", int_reward_func),
            ("correctness_reward_func", correctness_reward_func),
        ]
        eval_correctness = eval_correctness_gsm8k
    elif cfg.dataset.name == "countdown" or cfg.dataset.name == "countdown_kd":
        reward_fns = [
            ("xmlcount_reward_func", xmlcount_reward_func),
            ("soft_format_reward_func", soft_format_reward_func),
            ("strict_format_reward_func", strict_format_reward_func),
            ("answer_reward_function", answer_reward_function),
        ]
        eval_correctness = eval_correctness_countdown
    elif cfg.dataset.name == "medical" or cfg.dataset.name == "medical_kd":
        reward_fns = [
            ("xmlcount_reward_func", xmlcount_reward_func),
            ("soft_format_reward_func", soft_format_reward_func),
            ("strict_format_reward_func", strict_format_reward_func),
        ]
        eval_correctness = eval_correctness_medical_o1
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")
    

    # Initialize wandb
    if cfg.eval.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=f"eval_{cfg.wandb.run_name}-cp{cfg.model.checkpoint}",
        )

    # Load dataset and ceate loader
    dataset = get_dataset(cfg.dataset.name, split=cfg.dataset.split, ratio=1)
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda examples: examples,
    )

    # Load model and tokenizer
    model, reward_model, tokenizer, reward_tokenizer = irl_load_model_and_tokenizer_trl(
        cfg, pretrained=True, checkpoint=cfg.model.name
    )
    model.eval()
    reward_model.eval()

    # Generation loop
    all_correct_flags = []  # list[list[bool]]  (per-problem)
    all_reward_scores = []   # list[list[float]]  per problem
    
    n = cfg.sampling.n_samples
    sampling_params = {
        "max_new_tokens": int(cfg.model.max_completion_length),
        "temperature": float(cfg.sampling.temperature),
        "top_p": float(cfg.sampling.top_p),
        "do_sample": True,
        "num_return_sequences": n,
    }

    sums = {name: 0.0 for name, _ in reward_fns}
    sum_sqs = {name: 0.0 for name, _ in reward_fns}
    count = 0

    # Before generation loop
    all_results = []
    
    for batch in tqdm(loader):

        # each batch is a dict with “prompt” and “answer”
        prompts = [b["prompt"] for b in batch]  # list of lists of messages
        answers = [b["answer"] for b in batch]  # list of strings

        # turn each prompt into a single string ready for generation
        prompts_text = [
            maybe_apply_chat_template({"prompt": p}, tokenizer)["prompt"]
            for p in prompts
        ]
        
        enc = tokenizer(
                prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left"
            )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        input_lengths = enc["attention_mask"].size(1)

        # HF generate returns (batch * n) sequences; we regroup below
        gen_out = model.generate(
            **enc,
            **sampling_params,
        )

        # generate with vllm

        batch_size = len(prompts_text)
        assert gen_out.size(0) == batch_size * n, "Unexpected number of generated sequences."

        # Decode only the newly generated part per sequence
        # Map rows -> prompt index
        decoded_per_prompt = [[] for _ in range(batch_size)]
        for row_idx in range(gen_out.size(0)):
            prompt_idx = row_idx // n
            seq = gen_out[row_idx]
            # slice off the prompt tokens so we only keep the completion
            gen_only = seq[input_lengths:]
            text = tokenizer.decode(gen_only, skip_special_tokens=True)
            decoded_per_prompt[prompt_idx].append(text)

        # Build completions structure: list over batch of list[{"content": str}] of length n
        completions = [[{"content": t} for t in texts] for texts in decoded_per_prompt]


        # Collect reward function scores
        batch_rewards = []
        for prompt, completion, answer in zip(prompts, completions, answers):
            rewards = {}
            for name, fn in reward_fns:
                rewards[name] = float(np.mean(fn(prompts=[prompt], completions=[completion], answer=[answer])))
            batch_rewards.append(rewards)
            
        # Get reward model scores
        batch_scores = score_with_reward_model(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            prompts_msgs=prompts,
            decoded_per_prompt=decoded_per_prompt,
            dense=cfg.model.dense_rewards
        )
        
        # Store results for this batch - modified to create separate entries
        for prompt, generations, scores, rewards in zip(prompts, decoded_per_prompt, batch_scores, batch_rewards):
            for gen_idx, (generation, score) in enumerate(zip(generations, scores)):
                result = {
                    "prompt": prompt,
                    "generation": generation,
                    "generation_idx": gen_idx,
                    "reward_model_score": score,
                }
                # Add individual reward function scores
                for name, fn in reward_fns:
                    result[f"reward_{name}"] = rewards[name]
                all_results.append(result)

        for completion, answer in zip(completions, answers):
            correct_flags = eval_correctness(completions=completion, answer=answer)
            all_correct_flags.append(correct_flags)

            for name, fn in reward_fns:
                batch_score = float(
                    np.mean(
                        fn(prompts=prompts, completions=completions, answer=answers)
                    )
                )
                sums[name] += batch_score
                sum_sqs[name] += batch_score**2
            count += 1
            

    pass_at_k = compute_pass_at_k(all_correct_flags, cfg.eval.ks)
    
    success_at_k = compute_success_at_k_from_scores(
        all_correct_flags=all_correct_flags,
        all_scores=all_reward_scores,
        ks=cfg.eval.ks,
    )
    oracle_at_1 = compute_oracle_at_1_from_N(all_correct_flags)

    print("\n--- Final metrics ---")
    for k, v in pass_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/pass@{k}": v})
        print(f"pass@{k}: {v:.4f}")
        
    for k, v in success_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/success@{k}|N={n}": v})
        print(f"success@{k}|N={n}: {v:.4f}")

    if cfg.eval.report_to == "wandb":
        wandb.log({"test/oracle@1|N": oracle_at_1})
    print(f"oracle@1|N={n}: {oracle_at_1:.4f}")

    metrics_mean = {
        f"test/rewards/{name}/mean": sums[name] / count for name, _ in reward_fns
    }

    metrics_std = {
        f"test/rewards/{name}/std": np.sqrt(
            (sum_sqs[name] / count) - (sums[name] / count) ** 2
        )
        for name, _ in reward_fns
    }

    metrics = {**metrics_mean, **metrics_std}
    print("\n--- Final Rewards ---")
    if cfg.eval.report_to == "wandb":
        wandb.log(metrics)
    for name, _ in reward_fns:
        print(
            f"{name} mean: {metrics[f'test/rewards/{name}/mean']:.2f}, std: {metrics[f'test/rewards/{name}/std']:.2f}"
        )

    # Save results to JSONL
    output_file = f"{cfg.model.name}/eval_results.jsonl"
    save_results_to_jsonl(output_file, all_results)
    print(f"\nSaved evaluation results to {output_file}")


if __name__ == "__main__":
    main()
