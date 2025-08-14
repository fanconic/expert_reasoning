# evaluate.py
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.utils import set_seed
from src.data.dataset import get_dataset  # same as training
from src.models.model_module_trl import irl_load_model_and_tokenizer_trl
from src.rewards.reward_functions import (
    strict_format_reward_func,
    soft_format_reward_func,
    eval_correctness,
    int_reward_func,
    xmlcount_reward_func,
    correctness_reward_func,
)
import numpy as np
from src.eval.eval_module import compute_pass_at_k
from vllm import SamplingParams
import wandb
from trl.trainer.grpo_trainer import maybe_apply_chat_template


reward_fns = [
    ("xmlcount_reward_func", xmlcount_reward_func),
    ("soft_format_reward_func", soft_format_reward_func),
    ("strict_format_reward_func", strict_format_reward_func),
    ("int_reward_func", int_reward_func),
    ("correctness_reward_func", correctness_reward_func),
]


@hydra.main(config_path="configs", config_name="config_irl_eval", version_base="1.3")
def main(cfg: DictConfig):
    """
    Evaluate a trained model and compute pass@k on a dataset split.
    """

    print("Evaluation configuration:\n", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

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
    model, _, tokenizer, _ = irl_load_model_and_tokenizer_trl(cfg, pretrained=True, checkpoint=cfg.model.name)
    model.eval()

    # Generation loop
    all_correct_flags = []  # list[list[bool]]  (per-problem)
    n = cfg.sampling.n_samples
    sampling_params = {
        "max_new_tokens": int(cfg.model.max_seq_length),
        "temperature": float(cfg.sampling.temperature),
        "top_p": float(cfg.sampling.top_p),
        "do_sample": True,
        "num_return_sequences": n,
    }

    sums = {name: 0.0 for name, _ in reward_fns}
    sum_sqs = {name: 0.0 for name, _ in reward_fns}
    count = 0

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
                truncation=True,
                padding_side="left"
            )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        input_lengths = enc["attention_mask"].sum(dim=1).tolist()

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
            in_len = input_lengths[prompt_idx]
            seq = gen_out[row_idx]
            # slice off the prompt tokens so we only keep the completion
            gen_only = seq[in_len:]
            text = tokenizer.decode(gen_only, skip_special_tokens=True)
            decoded_per_prompt[prompt_idx].append(text)

        # Build completions structure: list over batch of list[{"content": str}] of length n
        completions = [[{"content": t} for t in texts] for texts in decoded_per_prompt]


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

    print("\n--- Final metrics ---")
    for k, v in pass_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/pass@{k}": v})
        print(f"pass@{k}: {v:.4f}")

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


if __name__ == "__main__":
    main()
