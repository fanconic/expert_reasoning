"""
Generation and Completion Scoring Module
-----------------------------------------
Handles generation of completions and computation of log probabilities.
Supports both vLLM and standard HF generation.
"""

from contextlib import nullcontext
from typing import Dict, List, Any, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizerBase
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from accelerate.utils import broadcast_object_list, gather_object


class CompletionGenerator:
    """Handles generation of completions using vLLM or standard HF generation."""

    def __init__(
        self,
        model,
        processing_class: PreTrainedTokenizerBase,
        accelerator,
        args,
        max_prompt_length: Optional[int] = None,
        max_completion_length: int = 512,
        use_vllm: bool = False,
        vllm_mode: str = "server",
    ):
        self.model = model
        self.model_wrapped = model
        self.processing_class = processing_class
        self.accelerator = accelerator
        self.args = args
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.use_vllm = use_vllm
        self.vllm_mode = vllm_mode
        self.is_fsdp_enabled = getattr(args, "fsdp", False)

    def generate_completions(
        self,
        inputs: List[Dict[str, Any]],
        generation_config,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate completions for the given inputs.
        
        Args:
            inputs: List of input dicts with 'prompt' field
            generation_config: HF generation config
            
        Returns:
            prompt_ids: [B, P] token IDs of prompts
            prompt_mask: [B, P] attention mask for prompts
            completion_ids: [B, C] token IDs of completions
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # Tokenize prompts
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, 
            padding_side="left", add_special_tokens=False
        )
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Trim prompts if needed
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            pad_token = self.processing_class.pad_token
            if pad_token is not None:
                prompts_text = [self._strip_leading_tokens(text, pad_token) for text in prompts_text]

        # Generate completions
        if self.use_vllm:
            completion_ids = self._generate_with_vllm(prompts_text)
        else:
            completion_ids = self._generate_with_hf(prompt_ids, prompt_mask, generation_config)

        return prompt_ids, prompt_mask, completion_ids

    def _strip_leading_tokens(self, text: str, pad_token: str) -> str:
        """Remove leading pad tokens from text."""
        while text.startswith(pad_token):
            text = text.removeprefix(pad_token)
        return text

    def _generate_with_hf(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        generation_config,
    ) -> torch.Tensor:
        """Generate completions using HuggingFace's generate method."""
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=generation_config
                )

        # Extract completion IDs (everything after prompt)
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        return completion_ids

    def _generate_with_vllm(self, prompts_text: List[str]) -> torch.Tensor:
        """Generate completions using vLLM (requires vllm_client to be set up)."""
        raise NotImplementedError("vLLM generation requires additional setup. Use standard HF generation.")


class LogProbabilityComputer:
    """Computes per-token log probabilities from model outputs."""

    def __init__(self, model, accelerator, max_completion_length: int = 512):
        self.model = model
        self.accelerator = accelerator
        self.max_completion_length = max_completion_length

    def get_per_token_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities.
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            logits_to_keep: Number of logits to compute (for efficiency)
            batch_size: Batch size for processing (for memory efficiency)
            
        Returns:
            per_token_logps: [B, logits_to_keep] log probabilities
        """
        if batch_size is None:
            return self._compute_logps_unbatched(input_ids, attention_mask, logits_to_keep)
        else:
            return self._compute_logps_batched(input_ids, attention_mask, logits_to_keep, batch_size)

    def _compute_logps_unbatched(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Compute logps for entire batch at once."""
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -logits_to_keep - 1 : -1, :]
            per_token_logps = torch.log_softmax(logits, dim=-1)

        return per_token_logps

    def _compute_logps_batched(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute logps in smaller batches to save memory."""
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]

            with torch.no_grad():
                outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits[:, -logits_to_keep - 1 : -1, :]
                batch_logps = torch.log_softmax(logits, dim=-1)

            all_logps.append(batch_logps)

        return torch.cat(all_logps, dim=0)


class CompletionMasker:
    """Handles masking and postprocessing of generated completions."""

    def __init__(self, processing_class: PreTrainedTokenizerBase, device: torch.device):
        self.processing_class = processing_class
        self.device = device

    def create_completion_mask(
        self,
        completion_ids: torch.Tensor,
        mask_truncated_completions: bool = False,
    ) -> torch.Tensor:
        """
        Create mask for completions, marking everything after the first EOS token.
        
        Args:
            completion_ids: [B, C] token IDs
            mask_truncated_completions: Whether to zero out sequences that don't end with EOS
            
        Returns:
            completion_mask: [B, C] binary mask (1 for valid tokens, 0 for padding/after-EOS)
        """
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        if mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        return completion_mask, is_eos

    def completion_ids_to_list(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> List[List[int]]:
        """Convert token tensors to list of token IDs, respecting mask."""
        return [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
