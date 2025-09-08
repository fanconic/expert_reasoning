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
    dense_rewards: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Tokenize expert and policy examples for reward model training."""
    padding_side = "left" if dense_rewards else "right"
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
        padding_side=padding_side,
        max_length=max_completion_length,
        truncation=True
    )
    policy_tokens = tokenizer(
        text=policy_texts,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=False,
        padding_side=padding_side,
        max_length=max_completion_length,
        truncation=True
    )

    # Add response masks when not in response_only mode
    if not response_only:
        B, T = expert_tokens["attention_mask"].shape
        e_am = expert_tokens["attention_mask"]
        p_am = policy_tokens["attention_mask"]
        prompt_lens = torch.as_tensor(prompt_lengths)

        L_e = e_am.sum(dim=1) 
        L_p = p_am.sum(dim=1)
        e_left = T - L_e
        p_left = T - L_p

        e_start = e_left + torch.minimum(prompt_lens, L_e)
        p_start = p_left + torch.minimum(prompt_lens, L_p)

        e_real_end = e_left + L_e
        p_real_end = p_left + L_p

        if max_completion_length is None:
            e_end, p_end = e_real_end, p_real_end
        else:
            e_end = torch.minimum(e_start + max_completion_length, e_real_end)
            p_end = torch.minimum(p_start + max_completion_length, p_real_end)

        idx = torch.arange(T).unsqueeze(0)      # (1, T)

        e_mask = (idx >= e_start.unsqueeze(1)) & (idx < e_end.unsqueeze(1))
        p_mask = (idx >= p_start.unsqueeze(1)) & (idx < p_end.unsqueeze(1))

        expert_response_mask = (e_mask & e_am.bool()).to(e_am.dtype)
        policy_response_mask = (p_mask & p_am.bool()).to(p_am.dtype)
        
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


def get_last_token_indices(attention_mask):
    # Method 1: Using max with range
    arange = torch.arange(attention_mask.size(1), device=attention_mask.device)
    last_tok_idx = (attention_mask * arange.unsqueeze(0)).max(dim=1).indices
    
    # Handle rows with all zeros by setting their index to 0 or another default
    all_zeros = ~attention_mask.bool().any(dim=1)
    last_tok_idx = last_tok_idx.masked_fill(all_zeros, 0)
    
    return last_tok_idx


def left_to_right_pad(x, pad_value=0):
    """
    Convert a left-padded batch of sequences into right-padded.
    x: (batch, seq_len) tensor
    pad_value: value used for padding (e.g. 0)
    """
    batch_size, seq_len = x.shape
    out = torch.full_like(x, pad_value)

    for i in range(batch_size):
        seq = x[i]
        # find where padding ends
        non_pad = seq[seq != pad_value]
        # put tokens at the start (right padding at the end)
        out[i, :len(non_pad)] = non_pad

    return out