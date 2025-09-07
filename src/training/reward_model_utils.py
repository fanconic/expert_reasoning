"""Utility functions for reward model training and processing."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from transformers import PreTrainedTokenizerBase
from  trl.data_utils import apply_chat_template

def tokenize_examples(
    prompts: List[List[Dict[str, str]]],
    expert_completions: List[List[Dict]],
    policy_completions: List[List[Dict]],
    tokenizer: PreTrainedTokenizerBase,
    max_completion_length: int,
    device: torch.device,
    response_only: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Tokenize expert and policy examples for reward model training."""
    if response_only:
        expert_messages = [{"messages": [{"role": "system", "content": ""}] + c} for c in expert_completions]
        expert_texts = [apply_chat_template(x, tokenizer)["text"] for x in expert_messages]
        policy_messages = [{"messages": [{"role": "system", "content": ""}] + c} for c in policy_completions]
        policy_texts = [apply_chat_template(x, tokenizer)["text"] for x in policy_messages]
    else:
        # First tokenize just the prompts to get their lengths
        prompt_messages = [{"messages": p} for p in prompts]
        prompt_texts = [apply_chat_template(x, tokenizer)["text"] for x in prompt_messages]
        prompt_tokens = tokenizer(
            text=prompt_texts,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        prompt_lengths = [len(ids) for ids in prompt_tokens["input_ids"]]
        
        # Create texts with full sequences
        expert_messages = [{"messages": p + c} for p, c in zip(prompts, expert_completions)]
        expert_texts = [apply_chat_template(x, tokenizer)["text"] for x in expert_messages]
        policy_messages = [{"messages": p + c} for p, c in zip(prompts, policy_completions)]
        policy_texts = [apply_chat_template(x, tokenizer)["text"] for x in policy_messages]

    # Tokenize
    expert_tokens = tokenizer(
        text=expert_texts,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=False,
        padding_side="right",
        max_length=max_completion_length,
        truncation=True
    )
    policy_tokens = tokenizer(
        text=policy_texts,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=False,
        padding_side="right",
        max_length=max_completion_length,
        truncation=True
    )

    # Add response masks when not in response_only mode
    if not response_only:
        expert_response_mask = torch.zeros_like(expert_tokens["attention_mask"])
        policy_response_mask = torch.zeros_like(policy_tokens["attention_mask"])
        
        for i, prompt_len in enumerate(prompt_lengths):
            max_idx = min(max_completion_length, expert_tokens["attention_mask"].size(1))
            if prompt_len < max_idx:
                expert_response_mask[i, prompt_len:] = expert_tokens["attention_mask"][i, prompt_len:]
                policy_response_mask[i, prompt_len:] = policy_tokens["attention_mask"][i, prompt_len:]
        
        expert_tokens["response_mask"] = expert_response_mask
        policy_tokens["response_mask"] = policy_response_mask
    else:
        expert_tokens["response_mask"] = expert_tokens["attention_mask"].clone()
        policy_tokens["response_mask"] = policy_tokens["attention_mask"].clone()
    
    # Put on right device
    expert_tokens = {k: v.to(device) for k, v in expert_tokens.items()}
    policy_tokens = {k: v.to(device) for k, v in policy_tokens.items()}
    
    return expert_tokens, policy_tokens

def dedup_token_batch(
    tokens: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Deduplicate rows in a token batch by (input_ids, attention_mask).
    Returns:
    - unique_tokens: same dict with only unique rows
    - counts: float tensor [n_unique] with multiplicity per unique row
    - inverse: long tensor [n_rows] mapping each original row -> unique idx
    """
    ids = tokens["input_ids"]
    mask = tokens["attention_mask"]

    key_to_uidx = {}
    unique_idx = []
    counts_py = []
    inv = torch.empty(ids.size(0), dtype=torch.long)

    for i in range(ids.size(0)):
        key = (ids[i].detach().cpu().numpy().tobytes(),
               mask[i].detach().cpu().numpy().tobytes())
        j = key_to_uidx.get(key)
        if j is None:
            j = len(unique_idx)
            key_to_uidx[key] = j
            unique_idx.append(i)
            counts_py.append(1)
        else:
            counts_py[j] += 1
        inv[i] = j

    unique_tokens = {k: v[unique_idx] for k, v in tokens.items()}
    counts = torch.tensor(counts_py, dtype=torch.float32, device=ids.device)
    inverse = inv.to(ids.device)
    return unique_tokens, counts, inverse

def prepare_reward_batch(
    expert_tokens: Dict[str, torch.Tensor],
    policy_tokens: Dict[str, torch.Tensor],
    perturbed_tokens: Optional[Dict[str, torch.Tensor]],
    n_pos: int,
    n_pol: int,
    n_per: int,
    device: torch.device,
    eps: float,
    neg_label_smoothing: Optional[float],
    neg_sample_weight: float,
    num_neg_perturbations_per_expert: int,
    gradient_accumulation_steps: int
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Prepare batched data for reward model training."""
    # Build batch: [ unique_experts | policy | perturbed ]
    cat_keys = expert_tokens.keys()
    have_perturbed = perturbed_tokens is not None

    if have_perturbed:
        batch = {
            k: torch.cat([expert_tokens[k], policy_tokens[k], perturbed_tokens[k]], dim=0)
            for k in cat_keys
        }
    else:
        batch = {
            k: torch.cat([expert_tokens[k], policy_tokens[k]], dim=0)
            for k in cat_keys
        }

    # Labels (with smoothing)
    eps_pos = eps
    eps_neg = neg_label_smoothing if neg_label_smoothing is not None else eps

    pos_labels = torch.ones(n_pos, device=device) * (1 - eps_pos) + eps_pos * 0.5
    pol_labels = torch.zeros(n_pol, device=device) * (1 - eps_neg) + eps_neg * 0.5

    if have_perturbed:
        per_labels = torch.zeros(n_per, device=device) * (1 - eps_neg) + eps_neg * 0.5
        labels = torch.cat([pos_labels, pol_labels, per_labels], dim=0)
    else:
        labels = torch.cat([pos_labels, pol_labels], dim=0)

    # Sample weights
    weights = torch.ones_like(labels)
    multiplier = 1 if num_neg_perturbations_per_expert == 0 else 2
    weights[:n_pos] *= (multiplier * n_pol // gradient_accumulation_steps)  # account for deduplication
    
    if have_perturbed and (neg_sample_weight != 1.0):
        start_per = n_pos + n_pol
        weights[start_per : start_per + n_per] *= neg_sample_weight

    return batch, labels, weights
