# evaluate.py
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.utils import set_seed
from src.data.dataset import get_dataset  # same as training
from src.models.model_module import load_model_and_tokenizer
from src.rewards.reward_functions import eval_correctness
from src.eval.eval_module import compute_pass_at_k
from vllm import SamplingParams
import wandb

@hydra.main(config_path="configs", config_name="config_eval", version_base="1.3")
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
            name=f"eval_{cfg.wandb.run_name}",
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

    # Generation loop
    all_correct_flags = []  # list[list[bool]]  (per-problem)
    sampling_params = SamplingParams(
        n=cfg.sampling.n_samples,
        seed=cfg.seed,
        max_tokens=cfg.model.max_seq_length - cfg.dataset.max_prompt_length,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
    )
    
    lora_req = model.load_lora(cfg.model.name)

    for batch in tqdm(loader):

        # each batch is a dict with “prompt” and “answer”
        prompts = [b["prompt"] for b in batch]  # list of lists of messages
        answers = [b["answer"] for b in batch]  # list of strings

        # turn each prompt into a single string ready for generation
        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]

        # generate with vllm
        
        outputs = model.fast_generate(
            texts,
            sampling_params=sampling_params,
            use_tqdm=False,
            #lora_request=lora_req,
        )
        
        gens = [[out.outputs[i].text for i in range(cfg.sampling.n_samples)]  for out in outputs]
        completions = [[{"content": g[i]} for i in range(cfg.sampling.n_samples)] for g in gens]
        
        for completion, answer in zip(completions, answers):
            correct_flags = eval_correctness(completions=completion, answer=answer)
            all_correct_flags.append(correct_flags)
            
        #import IPython; IPython.embed()

    pass_at_k = compute_pass_at_k(all_correct_flags, cfg.eval.ks)

    print("\n--- Final metrics ---")
    for k, v in pass_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/pass@{k}": v})
        print(f"pass@{k}: {v:.4f}")


if __name__ == "__main__":
    main()
