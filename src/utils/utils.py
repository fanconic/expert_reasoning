import numpy as np
import os
import random
import torch
import json
from trl.trainer.grpo_trainer import apply_chat_template

def set_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def save_results_to_jsonl(filename, results):
    """Save evaluation results to a JSONL file."""
    with open(filename, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
            
            
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