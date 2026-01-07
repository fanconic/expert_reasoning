# evaluate.py
from src.models.model_module import load_model_and_tokenizer, irl_load_model_and_tokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.utils import set_seed, save_results_to_jsonl
from src.data.dataset import get_dataset
from src.rewards.reward_functions import (
    strict_format_reward_func,
    soft_format_reward_func,
    int_reward_func,
    xmlcount_reward_func,
    gsm8k_correctness_reward_func,
    countdown_correctness_function,
    medical_correctness_reward_func,
    eval_correctness_gsm8k,
    eval_correctness_countdown,
    eval_correctness_medical
)
import torch
import numpy as np
from src.eval.eval_module import compute_pass_at_k, compute_success_at_k_from_scores, compute_oracle_at_1_from_N
from vllm import SamplingParams
import wandb
from trl.trainer.grpo_trainer import maybe_apply_chat_template, apply_chat_template
import os

# --- NEW IMPORTS FOR GUIDANCE ---
import copy

wandb.login()

    
class TopKRewardLogitsProcessor:
    def __init__(self, reward_model, reward_tokenizer, alpha=1.0, k=10, device="cuda"):
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.alpha = alpha
        self.k = k
        self.device = device
        
    def __call__(self, prompt_tokens_ids, generated_tokens_ids, logits):
        """
        Implements Algorithm 1: Reward-Augmented Decoding
        1. Get Top-K tokens from Policy (logits).
        2. Compute Rewards only for those K tokens.
        3. Reweight and return.
        """
        # 1. Identify Top-K candidates from the Policy Model
        # logits shape: [vocab_size]
        top_k_scores, top_k_indices = torch.topk(logits, self.k)
        
        # 2. Prepare inputs for the Reward Model
        # We need to construct K sequences: [Prompt + Generated + Candidate_i]
        base_seq = prompt_tokens_ids + generated_tokens_ids
        
        # Create a batch of K sequences
        # shape: [K, seq_len + 1]
        candidate_seqs = []
        for token_idx in top_k_indices:
            candidate_seqs.append(base_seq + [token_idx.item()])
            
        inputs = torch.tensor(candidate_seqs, device=self.device)
        
        # 3. Compute Rewards (Batched for Efficiency)
        with torch.no_grad():
            # Run the RM on the K candidates
            # Assuming RM outputs [Batch, Seq_Len, Vocab] (Dense) OR [Batch] (Scalar)
            output = self.reward_model(inputs)
            
            # Logic to extract the specific scalar reward for the last token
            if hasattr(output, 'logits'):

                rm_scores = output.logits[:, -1].mean(dim=-1) # Fallback heuristic
            else:
                # If Scalar RM
                rm_scores = output[:, -1] if output.ndim > 1 else output

        new_logits = torch.full_like(logits, float('-inf'))
        guided_scores = top_k_scores + (self.alpha * rm_scores)
        new_logits.scatter_(0, top_k_indices, guided_scores)
        
        return new_logits

# ==========================================
# 2. Chunk-Level Guidance (Step Search)
# ==========================================
def generate_with_chunk_guidance(
    model, 
    reward_model, 
    reward_tokenizer, 
    prompts_text, 
    sampling_params, 
    step_size=5, 
    n_candidates=4,
    max_tokens=256
):
    """
    Performs generation by stepping 'step_size' tokens at a time, 
    generating 'n_candidates', and selecting the best one via Reward Model.
    """
    # Initialize current generation with prompts
    current_gens = prompts_text
    
    # We iterate until we hit max length (simplified loop)
    # Note: vLLM is most efficient with batching, this manual loop 
    # splits the batch logic somewhat.
    
    for _ in range(0, max_tokens, step_size):
        # 1. Generate Candidates for the next step
        step_params = copy.deepcopy(sampling_params)
        step_params.max_tokens = step_size
        step_params.n = n_candidates
        
        # model.fast_generate typically returns a list of RequestOutputs
        outputs = model.fast_generate(current_gens, sampling_params=step_params, use_tqdm=False)
        
        new_current_gens = []
        
        # 2. Evaluate Candidates
        for i, out in enumerate(outputs):
            candidates_text = [o.text for o in out.outputs] # Just the NEW text
            parent_text = current_gens[i]
            
            # Construct full sequences for scoring
            full_candidates = [parent_text + c for c in candidates_text]
            
            # Tokenize for RM
            inputs = reward_tokenizer(full_candidates, return_tensors="pt", padding=True, truncation=True).to(reward_model.device)
            
            with torch.no_grad():
                rm_out = reward_model(**inputs)
                # Assuming dense rewards [B, Seq], take mean of the NEW chunk
                # or just the score of the last token.
                if hasattr(rm_out, 'logits'):
                    scores = rm_out.logits.mean(dim=1).squeeze().cpu().numpy()
                else:
                    scores = rm_out.mean(dim=1).squeeze().cpu().numpy()
                
                # Handle single candidate edge case
                if scores.ndim == 0: scores = [scores]
            
            # 3. Select Best Candidate
            best_idx = np.argmax(scores)
            best_extension = candidates_text[best_idx]
            new_current_gens.append(parent_text + best_extension)
            
        current_gens = new_current_gens
        
    final_outputs = []
    for gen in current_gens:
        final_outputs.append([{"content": gen[len(p):]} for p in prompts_text]) 
        
    return current_gens 


# ==========================================
# 3. Helper for Scoring (Existing)
# ==========================================
@torch.no_grad()
def score_with_reward_model(
    reward_model, reward_tokenizer, prompts_msgs, decoded_per_prompt, dense=False
):
    MICROBATCH_SIZE = 16 
    device = next(reward_model.parameters()).device
    texts = []
    idx_slices = [] 
    start = 0
    for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
        for c in completions:
            msgs = p_msgs + [{"role": "assistant", "content": c if isinstance(c, str) else c.get("content", "")}]
            texts.append(apply_chat_template({"messages": msgs}, reward_tokenizer)["text"])
        end = start + len(completions)
        idx_slices.append((start, end))
        start = end

    if len(texts) == 0: return [[] for _ in prompts_msgs]

    if dense:
        completion_texts = []
        for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
            for c in completions:
                content = c if isinstance(c, str) else c.get("content", "")
                completion_texts.append(content + (reward_tokenizer.eos_token or ""))
    pad_side = "left" if dense else "right"
    flat_logits = [] 
    for i in range(0, len(texts), MICROBATCH_SIZE):
        batch_texts = texts[i : i + MICROBATCH_SIZE]
        enc = reward_tokenizer(
            text=batch_texts, return_tensors="pt", padding=True, add_special_tokens=False,
            truncation=True, max_length=1124, padding_side=pad_side,
        ).to(device)
        out = reward_model(**enc).logits.squeeze(-1) 
        if dense:
            batch_comp = completion_texts[i : i + len(batch_texts)]
            response_mask = reward_tokenizer(
                text=batch_comp, return_tensors="pt", padding="max_length", add_special_tokens=False,
                truncation=True, padding_side="left", max_length=out.size(1),
            ).to(device)["attention_mask"]
            out = out.masked_fill(response_mask == 0, np.nan) 
            out_cpu = out.detach().float().cpu()
            for row in out_cpu: flat_logits.append(row.tolist())
        else:
            flat_logits.extend((out.detach().float().cpu().tolist()))
    logits = flat_logits
    all_scores = []
    for s, e in idx_slices: all_scores.append(logits[s:e])
    return all_scores


@hydra.main(config_path="configs", config_name="config_eval", version_base="1.3")
def main(cfg: DictConfig):
    print("Evaluation configuration:\n", OmegaConf.to_yaml(cfg))
    
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    config_save_path = os.path.join(cfg.training.output_dir, "evaluation_config.yaml")
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Configuration saved to: {config_save_path}")
    
    set_seed(cfg.seed)

    if cfg.dataset.name == "gsm8k" or cfg.dataset.name == "gsm8k_kd":
        reward_fns = [("xmlcount_reward_func", xmlcount_reward_func), ("correctness_reward_func", gsm8k_correctness_reward_func)]
        eval_correctness = eval_correctness_gsm8k
    elif cfg.dataset.name == "countdown" or cfg.dataset.name == "countdown_kd":
        reward_fns = [("correctness_reward_func", countdown_correctness_function)]
        eval_correctness = eval_correctness_countdown
    else:
        # Simplified for brevity - your original code had more
        reward_fns = [("correctness_reward_func", medical_correctness_reward_func)]
        eval_correctness = eval_correctness_medical

    if cfg.eval.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity, config=wandb_config,
            name=f"eval_{cfg.wandb.run_name}-cp{cfg.model.checkpoint}",
        )

    # Load data
    no_system = getattr(cfg.dataset, "no_system", False)
    dataset = get_dataset(cfg.dataset.name, split=cfg.dataset.split, ratio=0.1, no_system=no_system)
    loader = DataLoader(
        dataset, batch_size=cfg.eval.per_device_eval_batch_size, shuffle=False,
        collate_fn=lambda examples: examples,
    )

    # Load models
    if cfg.airl:
        model, reward_model, tokenizer, reward_tokenizer = irl_load_model_and_tokenizer(cfg, pretrained=True)
        reward_model.eval()
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        reward_model = None # Ensure variable exists
        
    model.eval()
    lora_req = model.load_lora(cfg.model.name, load_tensors=True)

    # Generation Parameters
    n = cfg.sampling.n_samples
    sampling_params = SamplingParams(
        n=n,
        seed=cfg.seed,
        max_tokens=cfg.model.max_completion_length,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
    )

    # ==========================================
    # CONFIGURABLE GUIDANCE LOGIC
    # ==========================================
    guidance_method = getattr(cfg, "guidance", {}).get("method", "none") # "top_k", "chunk", "none"

    if guidance_method == "topk" and cfg.airl:
        print(f"--- ACTIVATING REWARD-AUGMENTED DECODING (Top-K={cfg.guidance.k}) ---")
        
        # Instantiate the processor
        rw_processor = TopKRewardLogitsProcessor(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            alpha=getattr(cfg.guidance, "alpha", 1.0),
            k=getattr(cfg.guidance, "k", 5),  # Recommended K=5 or 10
            device=next(reward_model.parameters()).device
        )
        
        # Attach to sampling params
        sampling_params.logits_processors = [rw_processor]
        
    elif guidance_method == "chunk" and cfg.airl:
        print("--- ACTIVATING CHUNK-LEVEL GUIDANCE ---")
        # NOTE: Chunk guidance uses a custom loop, so we handle it inside the batch loop
    else:
        print("--- STANDARD GENERATION (No Active Guidance) ---")

    # Metrics storage
    all_correct_flags = []  
    all_reward_scores = []
    sums = {name: 0.0 for name, _ in reward_fns}
    sum_sqs = {name: 0.0 for name, _ in reward_fns}
    count = 0
    all_results = []

    for batch in tqdm(loader):
        prompts = [b["prompt"] for b in batch]
        answers = [b["answer"] for b in batch]
        prompts_text = [maybe_apply_chat_template({"prompt": p}, tokenizer)["prompt"] for p in prompts]

        # ==========================================
        # GENERATION SWITCH
        # ==========================================
        if guidance_method == "chunk" and cfg.airl:
            # Use custom chunk loop
            # This returns a list of pure strings (the full text)
            generated_texts = generate_with_chunk_guidance(
                model=model,
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                prompts_text=prompts_text,
                sampling_params=sampling_params,
                step_size=getattr(cfg.guidance, "step_size", 10),
                n_candidates=getattr(cfg.guidance, "n_candidates", 4)
            )
            # Reformat to match expected structure: [[{'content': ...} * n] * batch]
            # Chunk guidance typically produces 1 best path per prompt
            completions = [[{"content": t[len(p):]}] for p, t in zip(prompts_text, generated_texts)]
            # n is effectively 1 here for metrics
            
        else:
            # Standard vLLM Generate (includes LogitsProcessor if added above)
            outputs = model.fast_generate(
                prompts_text,
                sampling_params=sampling_params,
                use_tqdm=False,
                lora_request=lora_req,
            )
            gens = [[out.outputs[i].text for i in range(n)] for out in outputs]
            completions = [[{"content": g[i]} for i in range(n)] for g in gens]
        
        
        # ==========================================
        # EVALUATION (Unchanged)
        # ==========================================
        batch_rewards = []
        for prompt, completion, answer in zip(prompts, completions, answers):
            batch_rewards_list = []
            for c in completion:
                rewards = {}
                for name, fn in reward_fns:
                    rewards[name] = float(np.mean(fn(prompts=[prompt], completions=[[c]], answer=[answer])))
                batch_rewards_list.append(rewards)
            batch_rewards.append(batch_rewards_list)
        
        if cfg.airl:
            batch_scores = score_with_reward_model(
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                prompts_msgs=prompts,
                decoded_per_prompt=completions,
                dense=cfg.model.dense_rewards
            )  
        else:
            batch_scores = [[1] * len(c) for c in completions]
        
        # Store results
        for prompt, generations, scores, rewards in zip(prompts, completions, batch_scores, batch_rewards):
            for gen_idx, (generation, score, rews) in enumerate(zip(generations, scores, rewards)):
                result = {
                    "prompt": prompt,
                    "generation": generation,
                    "generation_idx": gen_idx,
                    "reward_model_score": score,
                }
                result = result | rews
                all_results.append(result)
                
                if cfg.model.dense_rewards:
                    all_reward_scores.append(np.nanmean(scores,axis=1).tolist())
                else:
                    all_reward_scores.append(scores)

        for completion, answer in zip(completions, answers):
            correct_flags = eval_correctness(completions=completion, answer=answer)
            all_correct_flags.append(correct_flags)

            for name, fn in reward_fns:
                batch_score = float(np.mean(fn(prompts=prompts, completions=completions, answer=answers)))
                sums[name] += batch_score   
                sum_sqs[name] += batch_score**2
            count += 1
            
    # --- METRICS COMPUTATION (Unchanged) ---
    pass_at_k = compute_pass_at_k(all_correct_flags, cfg.eval.ks)
    success_at_k = compute_success_at_k_from_scores(all_correct_flags, all_reward_scores, cfg.eval.ks)
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