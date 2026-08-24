from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn.functional as F
from accelerate.utils import gather
from trl import GRPOTrainer


def opsd_reward_func(completions, **kwargs):
    return [0.0 for _ in completions]


class OPSDTrainer(GRPOTrainer):
    """GRPO trainer with OPSD implicit log-ratio rewards.

    The trainable student is the active LoRA policy. The frozen teacher and
    reference distributions are the same base model with adapters disabled:

      reward = mean_t log p_base(y_t | x, expert, y_<t)
             - mean_t log p_base(y_t | x, y_<t)
    """

    def __init__(
        self,
        *args,
        opsd_reward_temperature: float = 1.0,
        opsd_reward_lb: Optional[float] = -5.0,
        opsd_reward_ub: Optional[float] = 5.0,
        opsd_max_micro_batch: int = 2,
        opsd_log_first_batch: bool = True,
        **kwargs,
    ):
        self.opsd_reward_temperature = float(opsd_reward_temperature)
        self.opsd_reward_lb = opsd_reward_lb
        self.opsd_reward_ub = opsd_reward_ub
        self.opsd_max_micro_batch = max(1, int(opsd_max_micro_batch))
        self.opsd_log_first_batch = bool(opsd_log_first_batch)
        self._opsd_logged_example = False
        super().__init__(*args, **kwargs)

    def _completion_to_text(self, completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            parts = []
            for msg in completion:
                if isinstance(msg, dict):
                    parts.append(str(msg.get("content", "")))
                else:
                    parts.append(str(msg))
            return "".join(parts)
        return str(completion)

    def _normal_context_text(self, prompt: Any) -> str:
        if isinstance(prompt, list):
            return self.processing_class.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        return str(prompt)

    def _teacher_prompt(self, prompt: Any, expert_trace: str) -> Any:
        if not isinstance(prompt, list):
            return (
                f"Question:\n{prompt}\n\n"
                "Here is an expert reasoning trace for this problem:\n"
                f"{expert_trace}\n\n"
                "Now evaluate the student's reasoning trajectory token by token by "
                "predicting the continuation that best follows the question and expert reasoning.\n\n"
                "Student reasoning trajectory:\n"
            )

        messages = deepcopy(prompt)
        last_user_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                break

        teacher_suffix = (
            "\n\nHere is an expert reasoning trace for this problem:\n"
            f"{expert_trace}\n\n"
            "Now evaluate the student's reasoning trajectory token by token by "
            "predicting the continuation that best follows the question and expert reasoning.\n\n"
            "Student reasoning trajectory:\n"
        )

        if last_user_idx is None:
            messages.append({"role": "user", "content": teacher_suffix.lstrip()})
        else:
            messages[last_user_idx]["content"] = str(messages[last_user_idx].get("content", "")) + teacher_suffix
        return messages

    def _teacher_context_text(self, prompt: Any, expert_trace: str) -> str:
        teacher_prompt = self._teacher_prompt(prompt, expert_trace)
        if isinstance(teacher_prompt, list):
            return self.processing_class.apply_chat_template(
                teacher_prompt, tokenize=False, add_generation_prompt=True
            )
        return str(teacher_prompt)

    def _normalize_completion_ids(self, completion_ids_list, completions):
        normalized = []
        for ids, completion in zip(completion_ids_list or [], completions):
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().tolist()
            normalized.append([int(x) for x in ids])

        if len(normalized) == len(completions):
            return normalized

        normalized = []
        for completion in completions:
            text = self._completion_to_text(completion)
            ids = self.processing_class(text, add_special_tokens=False).input_ids
            normalized.append([int(x) for x in ids])
        return normalized

    def _context_token_ids(self, context_text: str) -> list[int]:
        return self.processing_class(context_text, add_special_tokens=False).input_ids

    def _score_completion_logps(
        self,
        model,
        context_texts: list[str],
        completion_ids: list[list[int]],
    ) -> torch.Tensor:
        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id

        max_length = int(self.args.max_prompt_length) + int(self.args.max_completion_length)
        rows = []
        metadata = []

        for context_text, comp_ids in zip(context_texts, completion_ids):
            comp_ids = list(comp_ids)
            if len(comp_ids) == 0:
                rows.append([pad_token_id])
                metadata.append((0, 0))
                continue

            if len(comp_ids) >= max_length:
                comp_ids = comp_ids[: max_length - 1]

            context_ids = self._context_token_ids(context_text)
            max_context = max(1, max_length - len(comp_ids))
            if len(context_ids) > max_context:
                context_ids = context_ids[-max_context:]

            input_ids = context_ids + comp_ids
            rows.append(input_ids)
            metadata.append((len(context_ids), len(comp_ids)))

        max_row_len = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), max_row_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row_idx, row in enumerate(rows):
            row_tensor = torch.tensor(row, dtype=torch.long, device=device)
            input_ids[row_idx, : row_tensor.numel()] = row_tensor
            attention_mask[row_idx, : row_tensor.numel()] = 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        scores = []
        for row_idx, (ctx_len, comp_len) in enumerate(metadata):
            if comp_len == 0 or ctx_len == 0:
                scores.append(torch.zeros((), dtype=torch.float32, device=device))
                continue

            start = ctx_len
            end = ctx_len + comp_len
            pred_logits = logits[row_idx, start - 1 : end - 1, :]
            labels = input_ids[row_idx, start:end]
            token_logps = F.log_softmax(pred_logits.float(), dim=-1).gather(
                -1, labels.unsqueeze(-1)
            ).squeeze(-1)
            scores.append(token_logps.mean())

        return torch.stack(scores, dim=0)

    def _score_contexts(
        self,
        model,
        context_texts: list[str],
        completion_ids: list[list[int]],
    ) -> torch.Tensor:
        chunks = []
        for i in range(0, len(context_texts), self.opsd_max_micro_batch):
            j = min(i + self.opsd_max_micro_batch, len(context_texts))
            chunks.append(
                self._score_completion_logps(
                    model,
                    context_texts[i:j],
                    completion_ids[i:j],
                )
            )
        return torch.cat(chunks, dim=0)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        targets = [example.get("target") for example in inputs]
        if any(target is None for target in targets):
            raise ValueError("OPSD requires a `target` expert trace in the training dataset. Use the *_kd datasets.")

        if len(targets) != len(prompts):
            if len(targets) * self.num_generations == len(prompts):
                targets = [targets[i // self.num_generations] for i in range(len(prompts))]
            else:
                raise ValueError(f"Cannot align OPSD targets with prompts: {len(targets)=}, {len(prompts)=}.")

        completion_ids = self._normalize_completion_ids(completion_ids_list, completions)
        normal_contexts = [self._normal_context_text(prompt) for prompt in prompts]
        teacher_contexts = [
            self._teacher_context_text(prompt, target) for prompt, target in zip(prompts, targets)
        ]

        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        adapter_context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()

        with torch.inference_mode(), adapter_context:
            teacher_logps = self._score_contexts(model, teacher_contexts, completion_ids)
            ref_logps = self._score_contexts(model, normal_contexts, completion_ids)

        if was_training:
            model.train()
            self.model.train()

        rewards = (teacher_logps - ref_logps) / max(self.opsd_reward_temperature, 1e-8)
        if self.opsd_reward_lb is not None or self.opsd_reward_ub is not None:
            lb = -float("inf") if self.opsd_reward_lb is None else float(self.opsd_reward_lb)
            ub = float("inf") if self.opsd_reward_ub is None else float(self.opsd_reward_ub)
            rewards = rewards.clamp(min=lb, max=ub)

        if (
            self.opsd_log_first_batch
            and not self._opsd_logged_example
            and self.accelerator.is_main_process
            and len(prompts) > 0
        ):
            self._opsd_logged_example = True
            print("\n[OPSD debug] student prompt:")
            print(normal_contexts[0][:1000])
            print("[OPSD debug] teacher prompt:")
            print(teacher_contexts[0][:1000])
            print("[OPSD debug] completion:")
            print(self._completion_to_text(completions[0])[:1000])
            print(
                "[OPSD debug] reward/logps: "
                f"reward={rewards[0].item():.4f}, "
                f"teacher={teacher_logps[0].item():.4f}, "
                f"ref={ref_logps[0].item():.4f}"
            )

        rewards_per_func = rewards.to(device=device, dtype=torch.float32).unsqueeze(1)
        return gather(rewards_per_func)
