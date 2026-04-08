# evaluate.py
import os
os.environ["UNSLOTH_COMPILE_OVERWRITE"] = "0"
#os.environ["TORCHDYNAMO_DISABLE"] = "1"

from src.models.model_module import load_model_and_tokenizer, irl_load_model_and_tokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.utils import set_seed, save_results_to_jsonl
from src.data.dataset import get_dataset
from src.rewards.reward_functions import (
    xmlcount_reward_func,
    gsm8k_correctness_reward_func,
    countdown_correctness_function,
    medical_correctness_reward_func,
    scienceqa_correctness_reward_func,
    mmlu_correctness_reward_func,
    eval_correctness_gsm8k,
    eval_correctness_countdown,
    eval_correctness_medical,
    eval_correctness_scienceqa,
    eval_correctness_mmlu
)
import torch
import numpy as np
from src.eval.eval_module import (
    compute_pass_at_k,
    compute_reward_weighted_pass_at_k_from_scores,
    compute_success_at_k_from_scores,
    compute_oracle_at_1_from_N,
)
from vllm import SamplingParams
import wandb
from trl.trainer.grpo_trainer import maybe_apply_chat_template, apply_chat_template

import pandas as pd
from unsloth import FastLanguageModel

# --- NEW IMPORTS FOR GUIDANCE ---
import copy
import re

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


# Module-level cache
_BOUNDARY_TOKEN_DECODE_CACHE = {}


def sentence_boundary_mask(reward_tokenizer, full_batch, base_completion_mask, device):
    """
    Robust step-boundary detector for process reward modelling.

    Args:
        full_batch: dict with key "input_ids" -> LongTensor [bs, L]
        base_completion_mask: Bool/0-1 tensor [bs, L], True only on assistant completion tokens
        reward_tokenizer: HuggingFace tokenizer used to decode token pieces

    Returns:
        boundary_mask: Bool tensor [bs, L]
    """
    global _BOUNDARY_TOKEN_DECODE_CACHE

    input_ids = full_batch["input_ids"]
    bs, L = input_ids.shape

    boundary_mask = torch.zeros((bs, L), dtype=torch.bool, device=device)

    explicit_boundaries = [
        "</think>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "####",
        "\r\n\r\n",
        "\n\n\n",
        "\n\n",
        ".\n",
        "!\n",
        "?\n",
        ";\n",
        ":\n",
        "\n- ",
        "\n* ",
        "\n• ",
        "\n1.",
        "\n2.",
        "\n3.",
        "\n4.",
        "\n5.",
        "\n6.",
        "\n7.",
        "\n8.",
        "\n9.",
        "\n10.",
    ]
    explicit_boundaries = sorted(explicit_boundaries, key=len, reverse=True)

    max_explicit_len = max(len(x) for x in explicit_boundaries)
    suffix_window = max(96, max_explicit_len + 48)

    _abbr = {
        "e.g.", "i.e.", "etc.", "vs.", "cf.",
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
        "no.", "fig.", "eq.", "sec.", "resp.",
    }

    _wrapper_tags = {
        "<think>", "</think>", "<answer>", "</answer>",
        "<reasoning>", "</reasoning>",
    }

    def decode_one(tok_id: int) -> str:
        if tok_id not in _BOUNDARY_TOKEN_DECODE_CACHE:
            _BOUNDARY_TOKEN_DECODE_CACHE[tok_id] = reward_tokenizer.decode(
                [tok_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        return _BOUNDARY_TOKEN_DECODE_CACHE[tok_id]

    def _last_nonspace_char(s: str):
        for ch in reversed(s):
            if not ch.isspace():
                return ch
        return None

    def _strip_trailing_space(s: str) -> str:
        return s.rstrip(" \t")

    def _looks_like_abbreviation(s: str) -> bool:
        s = _strip_trailing_space(s).lower()

        m = re.search(r'([a-z]{1,10}\.)$', s)
        if m and m.group(1) in _abbr:
            return True

        m2 = re.search(r'([a-z]\.[a-z]\.)$', s)
        if m2 and m2.group(1) in _abbr:
            return True

        return False

    def _piece_is_only_layout(piece: str) -> bool:
        return piece.strip() == ""

    def _normalise_visible_text(s: str) -> str:
        x = s
        for tag in _wrapper_tags:
            x = x.replace(tag, "")
        x = re.sub(r"<\|[^>]+?\|>", "", x)
        x = re.sub(r"\s+", "", x)
        return x

    def _starts_with_digit(piece: str) -> bool:
        if piece is None:
            return False
        m = re.match(r'^[ \t\r\n]*([0-9])', piece)
        return m is not None

    def _is_explicit_boundary(s: str) -> bool:
        return any(s.endswith(x) for x in explicit_boundaries)

    def _is_sentence_punct_boundary(s: str, just_added_piece: str, next_piece: str) -> bool:
        if just_added_piece != "" and just_added_piece.strip(" \t") == "":
            return False

        s = _strip_trailing_space(s)
        if not s:
            return False

        last = s[-1]
        if last not in ".!?;:":
            return False

        if last == "." and _looks_like_abbreviation(s):
            return False

        # Avoid splitting on decimal points like 90.2
        if last == "." and _starts_with_digit(next_piece):
            return False

        if last in "!?":
            return True

        if last in ";:":
            return True

        return True

    def _is_newline_boundary(s: str, just_added_piece: str) -> bool:
        if "\n" not in just_added_piece and "\r" not in just_added_piece:
            return False

        if not s.endswith("\n"):
            return False

        if s.endswith("\n\n"):
            return True

        prefix = s[:-1]
        ch = _last_nonspace_char(prefix)
        if ch is None:
            return False

        if ch in ".!?;:)":
            return True

        if prefix.endswith("</think>") or prefix.endswith("####"):
            return True

        return False

    def _ends_reasoning_step(s: str, just_added_piece: str, next_piece: str) -> bool:
        if _is_explicit_boundary(s):
            return True

        if _is_newline_boundary(s, just_added_piece):
            return True

        if _is_sentence_punct_boundary(s, just_added_piece, next_piece):
            return True

        return False

    for b in range(bs):
        completion_positions = torch.nonzero(
            base_completion_mask[b].bool(), as_tuple=False
        ).flatten()

        if completion_positions.numel() == 0:
            continue

        completion_positions_list = completion_positions.tolist()
        decoded_pieces = [
            decode_one(int(input_ids[b, pos].item()))
            for pos in completion_positions_list
        ]

        suffix = ""
        seen_meaningful_content = False
        prev_was_boundary = False

        for i, pos in enumerate(completion_positions_list):
            piece = decoded_pieces[i]
            next_piece = decoded_pieces[i + 1] if i + 1 < len(decoded_pieces) else None

            suffix += piece
            if len(suffix) > suffix_window:
                suffix = suffix[-suffix_window:]

            if not seen_meaningful_content and _normalise_visible_text(suffix) != "":
                seen_meaningful_content = True

            is_boundary = _ends_reasoning_step(suffix, piece, next_piece)

            if is_boundary and not seen_meaningful_content:
                is_boundary = False

            if is_boundary and prev_was_boundary and _piece_is_only_layout(piece):
                is_boundary = False

            if is_boundary:
                boundary_mask[b, pos] = True
                prev_was_boundary = True
            else:
                prev_was_boundary = False

        # Always include the final completion token so the last segment gets a reward
        boundary_mask[b, int(completion_positions[-1].item())] = True

    boundary_mask &= base_completion_mask.bool()
    return boundary_mask


def every_n_tokens_mask(full_batch, base_completion_mask, n: int):
    """
    Returns mask [bs, L] that is True every n tokens within the completion
    and always True at the final completion token.
    """
    input_ids = full_batch["input_ids"]  # [bs, L]
    bs, L = input_ids.shape
    device = input_ids.device

    # Count token positions within the completion (1-based)
    token_indices = base_completion_mask.long().cumsum(dim=1)  # [bs, L]

    # Mark every n-th token (ignore positions outside completion)
    every_n_mask = (token_indices % n == 0) & base_completion_mask

    # Find last completion token index per batch
    last_indices = token_indices.argmax(dim=1)  # [bs]

    # Ensure last completion token is always included
    every_n_mask[torch.arange(bs, device=device),last_indices] |= base_completion_mask.any(dim=1)
    return every_n_mask


def backfill_rewards(rewards, mask):
    B, T = rewards.shape
    indices = torch.arange(T, device=rewards.device).expand(B, T)
    masked_indices = torch.where(mask.bool(), indices, torch.tensor(T, device=rewards.device))
    next_valid_index = torch.cummin(masked_indices.flip(1), dim=1)[0].flip(1)
    next_valid_index = next_valid_index.clamp(max=T-1).long()
    result = torch.gather(rewards, 1, next_valid_index)
    
    return result



@torch.no_grad()
def score_with_policy_model(
    policy_model, policy_tokenizer, prompts_msgs, decoded_per_prompt, max_length=512, micro_batch=16
):
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(policy_model)
    
    device = next(policy_model.parameters()).device
    
    # 1. Flatten prompts and completions into a single list of strings
    texts = []
    completion_texts = []
    
    for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
        for c in completions:
            content = c if isinstance(c, str) else c.get("content", "")
            msgs = p_msgs + [{"role": "assistant", "content": content}]
            texts.append(apply_chat_template({"messages": msgs}, policy_tokenizer)["text"])
            
            comp_text = content + (policy_tokenizer.eos_token or "")
            completion_texts.append(comp_text)

    if not texts: return [[] for _ in prompts_msgs]

    # Calculate global max seq_len strictly for the final output array shape
    global_tokens = policy_tokenizer(
        completion_texts, return_attention_mask=True, add_special_tokens=False, padding=False
    )
    seq_len = min(max(len(t) for t in global_tokens['input_ids']), max_length)

    all_log_probs = []
    
    # 2. Dynamic Batching Loop
    for i in range(0, len(texts), micro_batch):
        batch_texts = texts[i : i + micro_batch]
        batch_completion_texts = completion_texts[i : i + micro_batch]

        batch_inputs = policy_tokenizer(
            text=batch_texts, return_tensors="pt", padding=True, add_special_tokens=False,
            truncation=True, max_length=max_length, padding_side="right"
        ).to(device)
        
        batch_completions = policy_tokenizer(
            text=batch_completion_texts, return_tensors="pt", padding=True, add_special_tokens=False,
            truncation=True, max_length=max_length,
        ).to(device)

        with torch.inference_mode():
            outputs = policy_model(**batch_inputs)
            
            # Find where the completions start
            completion_lens = batch_completions["attention_mask"].sum(dim=1).long()
            full_lens = batch_inputs["attention_mask"].sum(dim=1).long()
            start_indices = (full_lens - completion_lens).clamp(min=0)
            
            current_micro_batch_size = batch_inputs["input_ids"].size(0)
            batch_res = torch.full((current_micro_batch_size, seq_len), float('nan'), device=device)
            
            # --- MEMORY OPTIMIZATION: Process log probs sequence-by-sequence ---
            # We use CrossEntropyLoss which calculates -log(p) internally and is highly optimized,
            # avoiding the need to instantiate a massive log_softmax tensor.
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            
            for b in range(current_micro_batch_size):
                comp_len = completion_lens[b].item()
                # The logit at index i predicts the token at index i+1
                start_idx = max(start_indices[b].item() - 1, 0) 
                
                # Prevent out-of-bounds if the sequence hit max_length
                end_idx = min(start_idx + comp_len, batch_inputs["input_ids"].size(1) - 1)
                actual_len = end_idx - start_idx
                
                if actual_len > 0:
                    # Slice ONLY the logits we need for this specific completion
                    # Shape becomes: [actual_len, vocab_size] instead of [B, Seq, Vocab]
                    seq_logits = outputs.logits[b, start_idx : end_idx, :] 
                    seq_labels = batch_inputs["input_ids"][b, start_idx + 1 : end_idx + 1]
                    
                    # Cross entropy gives -log(p). We invert it to get log(p).
                    seq_log_probs = -loss_fct(seq_logits, seq_labels)
                    
                    copy_len = min(actual_len, seq_len)
                    batch_res[b, :copy_len] = seq_log_probs[:copy_len]
            
            all_log_probs.append(batch_res.cpu().numpy())
            
            # Aggressively free the massive logits tensor before the next batch loop
            del outputs
            torch.cuda.empty_cache()
            
    B = len(prompts_msgs)
    final_log_probs = np.concatenate(all_log_probs, axis=0).reshape(B, -1, seq_len)
    return final_log_probs


# ==========================================
# 3. Helper for Scoring (Existing)
# ==========================================
@torch.no_grad()
def score_with_reward_model(
    reward_model, reward_tokenizer, prompts_msgs, decoded_per_prompt, dense_reward=False, max_length=512, micro_batch=16,
    clip_reward_model=False, reward_lb=-5.0, reward_ub=5.0, dense_partial_fixed_n=10
):
    # --- Optimization 1: Enable Unsloth Inference Kernels ---
    FastLanguageModel.for_inference(reward_model)
    
    device = next(reward_model.parameters()).device
    
    # 1. Flatten prompts and completions into a single list of strings
    texts = []
    completion_texts = []
    
    for p_msgs, completions in zip(prompts_msgs, decoded_per_prompt):
        for c in completions:
            content = c if isinstance(c, str) else c.get("content", "")
            # Build full text
            msgs = p_msgs + [{"role": "assistant", "content": content}]
            texts.append(apply_chat_template({"messages": msgs}, reward_tokenizer)["text"])
            
            # Build completion text for length calculation
            comp_text = content + (reward_tokenizer.eos_token or "")
            completion_texts.append(comp_text)

    if not texts: return [[] for _ in prompts_msgs]

    # Calculate global max seq_len strictly for the final output array shape
    # We batch this strictly for CPU side length checking
    global_tokens = reward_tokenizer(
        completion_texts, return_attention_mask=True, add_special_tokens=False, padding=False
    )
    # The max length of any completion in the dataset (clamped to max_length limit)
    seq_len = min(max(len(t) for t in global_tokens['input_ids']), max_length)

    new_logits = []
    
    # --- Optimization 2: Dynamic Batching Loop ---
    for i in range(0, len(texts), micro_batch):
        batch_texts = texts[i : i + micro_batch]
        batch_completion_texts = completion_texts[i : i + micro_batch]

        # Tokenize ONLY this batch with padding=True (Dynamic Padding)
        # This makes the tensor width = length of longest sequence in THIS batch, not 512.
        batch_inputs = reward_tokenizer(
            text=batch_texts, return_tensors="pt", padding=True, add_special_tokens=False,
            truncation=True, max_length=max_length, padding_side="right"
        ).to(device)
        
        # Tokenize completions just for length calculations
        batch_completions = reward_tokenizer(
            text=batch_completion_texts, return_tensors="pt", padding=True, add_special_tokens=False,
            truncation=True, max_length=max_length,
        ).to(device)

        with torch.inference_mode():
            # Model Forward Pass
            # logits shape: [micro_batch, dynamic_seq_len] or [micro_batch]
            reward_outputs = reward_model(**batch_inputs)
            reward_logits = reward_outputs.logits.squeeze(-1)
            
            current_batch_max_len = batch_inputs["input_ids"].shape[1]

            # Handle Non-Dense (Scalar) Rewards
            if not dense_reward:
                # --- Optimization 3: Use expand instead of repeat (Memory View) ---
                reward_logits = reward_logits.unsqueeze(1).expand(-1, current_batch_max_len)
            
            if clip_reward_model:
                reward_logits = torch.clamp(reward_logits, min=reward_lb, max=reward_ub)

            # Calculate indices
            completion_lens = batch_completions["attention_mask"].sum(dim=1).long()
            full_lens = batch_inputs["attention_mask"].sum(dim=1).long()
            start_indices = (full_lens - completion_lens).clamp(min=0)
            
            # Generate gather indices
            # We must clamp to seq_len because the final output expects fixed width
            gather_indices = start_indices[:, None] + torch.arange(seq_len, device=device)[None, :]
            
            # Important: Clamp indices to the current batch's dynamic width to avoid out-of-bounds
            gather_indices_safe = gather_indices.clamp(max=current_batch_max_len - 1)
            
            # Gather
            reward_comp = reward_logits.gather(1, gather_indices_safe)

            # --- Handle Dense Partial Logic (Optional) ---
            if dense_reward in ["partial", "partial_fixed"]:
                # Note: These masks need to be regenerated for the dynamic batch shape
                if dense_reward == "partial":
                    end_of_thought_mask = sentence_boundary_mask(
                        reward_tokenizer, batch_inputs, batch_inputs["attention_mask"], device
                    )
                    
                    
                else:
                    end_of_thought_mask = every_n_tokens_mask(
                        batch_inputs, batch_inputs["attention_mask"], dense_partial_fixed_n
                    )
                
                # We need to act carefully here because gather_indices might be wider than the dynamic batch
                # But since we clamped gather_indices_safe, it is valid for gathering.
                end_of_thought_mask = end_of_thought_mask.gather(1, gather_indices_safe)
                reward_comp = backfill_rewards(reward_comp, end_of_thought_mask)
                
            # Apply NaN mask for padding/invalid 
            #import IPython; IPython.embed(); exit()
            output_mask = torch.arange(seq_len, device=device)[None, :] < completion_lens[:, None]
            reward_comp[~output_mask] = float('nan')

            out_cpu = reward_comp.detach().float().cpu()
            new_logits.append(out_cpu)
       
    B = len(prompts_msgs)
    all_scores = np.concatenate(new_logits, axis=0).reshape(B, -1, seq_len)
    return all_scores



@hydra.main(config_path="configs", config_name="config_eval", version_base="1.3")
def main(cfg: DictConfig):
    print("Evaluation configuration:\n", OmegaConf.to_yaml(cfg))
    
    os.makedirs(cfg.model.name, exist_ok=True)
    config_save_path = os.path.join(cfg.model.name, "evaluation_config.yaml")
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Configuration saved to: {config_save_path}")
    
    set_seed(cfg.seed)

    if cfg.dataset.name == "gsm8k" or cfg.dataset.name == "gsm8k_kd" or "aime" in cfg.dataset.name:
        reward_fns = [("xmlcount_reward_func", xmlcount_reward_func), ("correctness_reward_func", gsm8k_correctness_reward_func)]
        eval_correctness = eval_correctness_gsm8k
    elif cfg.dataset.name == "countdown" or cfg.dataset.name == "countdown_kd":
        reward_fns = [("correctness_reward_func", countdown_correctness_function)]
        eval_correctness = eval_correctness_countdown
    elif cfg.dataset.name == "medreason" or cfg.dataset.name == "medreason_kd":
        # Simplified for brevity - your original code had more
        reward_fns = [("correctness_reward_func", medical_correctness_reward_func)]
        eval_correctness = eval_correctness_medical
    elif cfg.dataset.name == "science" or cfg.dataset.name == "science_kd":
        # Simplified for brevity - your original code had more
        reward_fns = [("correctness_reward_func", scienceqa_correctness_reward_func)]
        eval_correctness = eval_correctness_scienceqa
    elif cfg.dataset.name == "mmlu" or cfg.dataset.name == "mmlu_kd":
        reward_fns = [("correctness_reward_func", mmlu_correctness_reward_func)]
        eval_correctness = eval_correctness_mmlu
    elif cfg.dataset.name == "medical" or cfg.dataset.name == "medical_kd":
        # Simplified for brevity - your original code had more
        reward_fns = [("correctness_reward_func", medical_correctness_reward_func)]
        eval_correctness = eval_correctness_medical

    if cfg.eval.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity, config=wandb_config,
            name=f"eval_{cfg.wandb.run_name}-cp{cfg.model.checkpoint}",
        )

    # Load dataset
    # Load data
    no_system = getattr(cfg.dataset, "no_system", False)
    dataset = get_dataset(cfg.dataset.name, split=cfg.dataset.split, ratio=1, no_system=no_system)
    loader = DataLoader(
        dataset, batch_size=cfg.eval.per_device_eval_batch_size, shuffle=False,
        collate_fn=lambda examples: examples,
    )
    jsonl_path = f"{cfg.model.name}/eval_results_new.jsonl"
    df = pd.read_json(jsonl_path, lines=True)
    
    # Load models
    if cfg.airl:
        model, reward_model, policy_tokenizer, reward_tokenizer = irl_load_model_and_tokenizer(cfg, pretrained=True)
        del reward_model
        torch.cuda.empty_cache()
        model.eval()


    # Metrics storage
    all_correct_flags = []  
    all_reward_scores = []
    sums = {name: 0.0 for name, _ in reward_fns}
    sum_sqs = {name: 0.0 for name, _ in reward_fns}
    count = 0
    all_results = []
    

    bs = cfg.eval.per_device_eval_batch_size
    n = cfg.sampling.n_samples
    batch_counter = 0
    for batch in tqdm(loader):
        sub_df = df.iloc[batch_counter*bs*n : (batch_counter+1)*bs*n].copy().reset_index(drop=True)
        assert sub_df.iloc[0]["prompt"][1]["content"] == batch[0]["prompt"][1]["content"], "DataLoader and JSONL are misaligned!"
        assert sub_df.iloc[-1]["prompt"][1]["content"] == batch[-1]["prompt"][1]["content"], "DataLoader and JSONL are misaligned!"
        batch_counter += 1
        prompts = [b["prompt"] for b in batch]
        answers = [b["answer"] for b in batch]

        gens = sub_df["generation"].tolist()
        completions = [gens[i : i + n] for i in range(0, len(gens), 16)]
        
        
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
            # batch_scores = score_with_reward_model(
            #     reward_model=reward_model,
            #     reward_tokenizer=reward_tokenizer,
            #     prompts_msgs=prompts,
            #     decoded_per_prompt=completions,
            #     dense_reward=cfg.model.dense_rewards,
            #     max_length=cfg.model.max_prompt_length + cfg.model.max_completion_length,
            #     micro_batch=cfg.eval.max_micro_batch,
            #     clip_reward_model=cfg.model.clip_reward_model,
            #     reward_lb=cfg.model.reward_lb,
            #     reward_ub=cfg.model.reward_ub,
            #     dense_partial_fixed_n=cfg.model.dense_partial_fixed_n
            # )
            
            batch_scores = [[[0]] * len(c) for c in completions]
            
            batch_log_probs = score_with_policy_model(
                policy_model=model,
                policy_tokenizer=policy_tokenizer, # Or reward_tokenizer if they share a vocab
                prompts_msgs=prompts,
                decoded_per_prompt=completions,
                max_length=cfg.model.max_prompt_length + cfg.model.max_completion_length,
                micro_batch=cfg.eval.max_micro_batch
            )
        else:
            batch_scores = [[[0]] * len(c) for c in completions]
            batch_log_probs = [[[0]] * len(c) for c in completions]
        
        # Store results
        for prompt, generations, scores, log_probs, rewards in zip(prompts, completions, batch_scores, batch_log_probs, batch_rewards):
            for gen_idx, (generation, score, log_prob, rews) in enumerate(zip(generations, scores, log_probs, rewards)):
                result = {
                    "prompt": prompt,
                    "generation": generation,
                    "generation_idx": gen_idx,
                    "reward_model_score": score[~np.isnan(score)].tolist() if isinstance(score, np.ndarray) and score.ndim > 0 else score,
                    "policy_log_probs": log_prob[~np.isnan(log_prob)].tolist() if isinstance(log_prob, np.ndarray) and log_prob.ndim > 0 else log_prob,
                }
                result = result | rews
                all_results.append(result)
                
                all_reward_scores.append(np.nanmean(scores,axis=1).tolist())
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
    reward_weighted_pass_at_k = compute_reward_weighted_pass_at_k_from_scores(
        all_correct_flags, all_reward_scores, cfg.eval.ks
    )
    success_at_k = compute_success_at_k_from_scores(all_correct_flags, all_reward_scores, cfg.eval.ks)
    oracle_at_1 = compute_oracle_at_1_from_N(all_correct_flags)

    print("\n--- Final metrics ---")
    for k, v in pass_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/pass@{k}": v})
        print(f"pass@{k}: {v:.4f}")

    for k, v in reward_weighted_pass_at_k.items():
        if cfg.eval.report_to == "wandb":
            wandb.log({f"test/reward_weighted_pass@{k}|N={n}": v})
        print(f"reward_weighted_pass@{k}|N={n}: {v:.4f}")
        
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
    output_file = f"{cfg.model.name}/eval_results_logprobs.jsonl"
    save_results_to_jsonl(output_file, all_results)
    print(f"\nSaved evaluation results to {output_file}")

if __name__ == "__main__":
    main()
