# evaluate.py
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.utils import set_seed
from src.data.dataset import get_dataset  # same as training
from src.models.model_module import load_model_and_tokenizer
from src.rewards.reward_functions import (
    strict_format_reward_func,
    soft_format_reward_func,
    int_reward_func,
    xmlcount_reward_func,
    correctness_reward_func,
    answer_reward_function,
    eval_correctness_gsm8k,
    eval_correctness_countdown,
    eval_correctness_medical_o1
)
import numpy as np
from src.eval.eval_module import compute_pass_at_k, compute_success_at_k_from_scores, compute_oracle_at_1_from_N
from vllm import SamplingParams
import wandb
wandb.login()
from trl.trainer.grpo_trainer import maybe_apply_chat_template
import json

def save_results_to_jsonl(filename, results):
    """Save evaluation results to a JSONL file."""
    with open(filename, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

@hydra.main(config_path="configs", config_name="config_eval", version_base="1.3")
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
    model, tokenizer = load_model_and_tokenizer(cfg)
    model.eval()

    lora_req = model.load_lora(cfg.model.name, load_tensors=True)

    # Generation loop
    all_correct_flags = []  
    all_reward_scores = []
    n = cfg.sampling.n_samples
    sampling_params = SamplingParams(
        n=n,
        seed=cfg.seed,
        max_tokens=cfg.model.max_seq_length,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
    )

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

        # generate with vllm
        outputs = model.fast_generate(
            prompts_text,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_req,
        )

        gens = [[out.outputs[i].text for i in range(n)] for out in outputs]
        completions = [[{"content": g[i]} for i in range(n)] for g in gens]
        
        # Collect reward function scores
        batch_rewards = []
        for prompt, completion, answer in zip(prompts, completions, answers):
            rewards = {}
            for name, fn in reward_fns:
                rewards[name] = float(np.mean(fn(prompts=[prompt], completions=[completion], answer=[answer])))
            batch_rewards.append(rewards)
            
        
        # Store results for this batch - modified to create separate entries
        for prompt, generations, answer in zip(prompts, completions, answers):
            for gen_idx, generation in enumerate(generations):
                result = {
                    "prompt": prompt,
                    "generation": generation,
                    "generation_idx": gen_idx,
                }
                # Add individual reward function scores
                for name, fn in reward_fns:
                    result[f"reward_{name}"] = fn(completions=[[generation]], prompts=[prompt], answer=[answer])
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
            
            batch_scores = [1] * len(correct_flags)
            all_reward_scores.append(batch_scores)

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
