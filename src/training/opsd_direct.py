from __future__ import annotations

import math
import os
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class DirectOPSDTrainer:
    """Non-RL OPSD-token trainer.

    This trainer samples trajectories from the current student policy and then
    applies a weighted token-level NLL update on those sampled tokens. The
    weights are a frozen-base log-ratio:

      w_t = log p_base(y_t | x, expert, y_<t) - log p_base(y_t | x, y_<t)
      loss = -mean_t stopgrad(w_t) log p_student(y_t | x, y_<t)
    """

    def __init__(self, model, tokenizer, train_dataset, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        opsd_cfg = getattr(cfg, "opsd", {})
        self.num_generations = int(cfg.training.num_generations)
        self.max_prompt_length = int(cfg.model.max_prompt_length)
        self.max_completion_length = int(cfg.model.max_completion_length)
        self.max_length = self.max_prompt_length + self.max_completion_length
        self.temperature = float(cfg.sampling.temperature)
        self.top_p = float(cfg.sampling.top_p)
        self.max_micro_batch = max(1, int(getattr(opsd_cfg, "max_micro_batch", 2)))
        self.reward_temperature = float(getattr(opsd_cfg, "reward_temperature", 1.0))
        self.weight_lb = getattr(opsd_cfg, "reward_lb", -5.0)
        self.weight_ub = getattr(opsd_cfg, "reward_ub", 5.0)
        self.normalize_weights = bool(getattr(opsd_cfg, "normalize_weights", False))
        self.log_first_batch = bool(getattr(opsd_cfg, "log_first_batch", True))
        self._logged_example = False
        self.context_token_cache_size = max(
            0, int(getattr(opsd_cfg, "context_token_cache_size", 4096))
        )
        self._context_token_cache: dict[str, tuple[int, ...]] = {}

        self.global_step = 0
        self.report_to_wandb = (
            str(getattr(cfg.training, "report_to", "")).lower() == "wandb"
            and wandb is not None
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self._build_optimizer(trainable_params)
        self.scheduler = get_scheduler(
            name=cfg.training.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=int(float(cfg.training.warmup_ratio) * int(cfg.training.max_steps)),
            num_training_steps=int(cfg.training.max_steps),
        )

    def _build_optimizer(self, params):
        betas = (float(self.cfg.training.adam_beta1), float(self.cfg.training.adam_beta2))
        optim_name = str(self.cfg.training.optim).lower()
        if "8bit" in optim_name:
            try:
                import bitsandbytes as bnb

                return bnb.optim.PagedAdamW8bit(
                    params,
                    lr=float(self.cfg.training.learning_rate),
                    betas=betas,
                    weight_decay=float(self.cfg.training.weight_decay),
                )
            except ImportError:
                pass

        return torch.optim.AdamW(
            params,
            lr=float(self.cfg.training.learning_rate),
            betas=betas,
            weight_decay=float(self.cfg.training.weight_decay),
        )

    def _set_train_mode(self):
        self.model.train()
        if hasattr(self.model, "for_training"):
            try:
                self.model.for_training()
            except Exception:
                pass

    def _set_eval_mode(self):
        self.model.eval()
        if hasattr(self.model, "for_inference"):
            try:
                self.model.for_inference()
            except Exception:
                pass

    def _normal_context_text(self, prompt: Any) -> str:
        if isinstance(prompt, list):
            return self.tokenizer.apply_chat_template(
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

        suffix = (
            "\n\nHere is an expert reasoning trace for this problem:\n"
            f"{expert_trace}\n\n"
            "Now evaluate the student's reasoning trajectory token by token by "
            "predicting the continuation that best follows the question and expert reasoning.\n\n"
            "Student reasoning trajectory:\n"
        )
        if last_user_idx is None:
            messages.append({"role": "user", "content": suffix.lstrip()})
        else:
            messages[last_user_idx]["content"] = str(messages[last_user_idx].get("content", "")) + suffix
        return messages

    def _teacher_context_text(self, prompt: Any, expert_trace: str) -> str:
        teacher_prompt = self._teacher_prompt(prompt, expert_trace)
        if isinstance(teacher_prompt, list):
            return self.tokenizer.apply_chat_template(
                teacher_prompt, tokenize=False, add_generation_prompt=True
            )
        return str(teacher_prompt)

    def _context_token_ids(self, text: str) -> list[int]:
        if self.context_token_cache_size > 0:
            cached = self._context_token_cache.get(text)
            if cached is not None:
                return list(cached)

        token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if self.context_token_cache_size > 0:
            if len(self._context_token_cache) >= self.context_token_cache_size:
                self._context_token_cache.clear()
            self._context_token_cache[text] = tuple(int(token_id) for token_id in token_ids)
        return list(token_ids)

    def _prepare_rows(self, context_texts: list[str], completion_ids: list[list[int]]):
        rows = []
        metadata = []
        for context_text, comp_ids in zip(context_texts, completion_ids):
            comp_ids = list(comp_ids)
            if len(comp_ids) >= self.max_length:
                comp_ids = comp_ids[: self.max_length - 1]

            context_ids = self._context_token_ids(context_text)
            max_context = max(1, self.max_length - len(comp_ids))
            if len(context_ids) > max_context:
                context_ids = context_ids[-max_context:]

            row = context_ids + comp_ids
            rows.append(row if row else [self.tokenizer.eos_token_id])
            metadata.append((len(context_ids), len(comp_ids)))
        return rows, metadata

    def _token_logps_list(
        self,
        context_texts: list[str],
        completion_ids: list[list[int]],
    ) -> list[torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        rows, metadata = self._prepare_rows(context_texts, completion_ids)
        max_row_len = max(len(row) for row in rows)

        input_ids = torch.full(
            (len(rows), max_row_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row_idx, row in enumerate(rows):
            row_tensor = torch.tensor(row, dtype=torch.long, device=self.device)
            input_ids[row_idx, : row_tensor.numel()] = row_tensor
            attention_mask[row_idx, : row_tensor.numel()] = 1

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        logps = []
        for row_idx, (ctx_len, comp_len) in enumerate(metadata):
            if comp_len == 0 or ctx_len == 0:
                logps.append(torch.zeros((0,), dtype=torch.float32, device=self.device))
                continue
            start = ctx_len
            end = ctx_len + comp_len
            pred_logits = logits[row_idx, start - 1 : end - 1, :]
            labels = input_ids[row_idx, start:end]
            token_logps = F.log_softmax(pred_logits.float(), dim=-1).gather(
                -1, labels.unsqueeze(-1)
            ).squeeze(-1)
            logps.append(token_logps)
        return logps

    def _score_no_grad(
        self,
        context_texts: list[str],
        completion_ids: list[list[int]],
    ) -> list[torch.Tensor]:
        scores = []
        for i in range(0, len(context_texts), self.max_micro_batch):
            j = min(i + self.max_micro_batch, len(context_texts))
            scores.extend(self._token_logps_list(context_texts[i:j], completion_ids[i:j]))
        return scores

    def _completion_text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _generate_rollouts(self, prompts: list[Any]):
        prompt_texts = [self._normal_context_text(prompt) for prompt in prompts]
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        ).to(self.device)
        self.tokenizer.padding_side = old_padding_side

        self._set_eval_mode()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_completion_length,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-8),
                top_p=self.top_p,
                num_return_sequences=self.num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[:, prompt_len:]
        completion_ids = []
        completion_texts = []
        decode_first_completion = self.log_first_batch and not self._logged_example
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        for row in generated.detach().cpu().tolist():
            ids = []
            for token_id in row:
                if token_id == pad_token_id:
                    continue
                ids.append(int(token_id))
                if eos_token_id is not None and token_id == eos_token_id:
                    break
            completion_ids.append(ids)
            if decode_first_completion and not completion_texts:
                completion_texts.append(self._completion_text(ids))

        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * self.num_generations)
        return expanded_prompts, completion_ids, completion_texts

    def _make_weights(
        self,
        prompts: list[Any],
        targets: list[str],
        completion_ids: list[list[int]],
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        normal_contexts = [self._normal_context_text(prompt) for prompt in prompts]
        teacher_contexts = [
            self._teacher_context_text(prompt, target) for prompt, target in zip(prompts, targets)
        ]

        adapter_context = self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else nullcontext()
        was_training = self.model.training
        self._set_eval_mode()
        with torch.no_grad(), adapter_context:
            teacher_logps = self._score_no_grad(teacher_contexts, completion_ids)
            ref_logps = self._score_no_grad(normal_contexts, completion_ids)
        if was_training:
            self._set_train_mode()

        weights = []
        teacher_means = []
        ref_means = []
        for t_logps, r_logps in zip(teacher_logps, ref_logps):
            seq_len = min(t_logps.numel(), r_logps.numel())
            if seq_len == 0:
                weights.append(torch.zeros((0,), dtype=torch.float32, device=self.device))
                teacher_means.append(0.0)
                ref_means.append(0.0)
                continue
            t_logps = t_logps[:seq_len]
            r_logps = r_logps[:seq_len]
            weight = (t_logps - r_logps) / max(self.reward_temperature, 1e-8)
            if self.weight_lb is not None or self.weight_ub is not None:
                lb = -float("inf") if self.weight_lb is None else float(self.weight_lb)
                ub = float("inf") if self.weight_ub is None else float(self.weight_ub)
                weight = weight.clamp(min=lb, max=ub)
            weights.append(weight.detach())
            teacher_means.append(float(t_logps.mean().detach().cpu()))
            ref_means.append(float(r_logps.mean().detach().cpu()))

        if self.normalize_weights:
            flat = torch.cat([w for w in weights if w.numel() > 0], dim=0)
            if flat.numel() > 1:
                mean = flat.mean()
                std = flat.std().clamp_min(1e-4)
                weights = [(w - mean) / std if w.numel() > 0 else w for w in weights]

        flat_weights = torch.cat([w for w in weights if w.numel() > 0], dim=0)
        metrics = {
            "opsd/teacher_logp": sum(teacher_means) / max(1, len(teacher_means)),
            "opsd/ref_logp": sum(ref_means) / max(1, len(ref_means)),
            "opsd/weight_mean": float(flat_weights.mean().detach().cpu()) if flat_weights.numel() else 0.0,
            "opsd/weight_std": float(flat_weights.std().detach().cpu()) if flat_weights.numel() > 1 else 0.0,
        }
        return weights, metrics

    def _student_backward(
        self,
        prompts: list[Any],
        weights: list[torch.Tensor],
        completion_ids: list[list[int]],
        loss_scale: float,
    ) -> dict[str, float]:
        normal_contexts = [self._normal_context_text(prompt) for prompt in prompts]
        total_tokens = sum(int(w.numel()) for w in weights)
        if total_tokens == 0:
            return {"loss": 0.0, "student_logp": 0.0, "tokens": 0.0}

        loss_total = 0.0
        logp_total = 0.0
        self._set_train_mode()
        for i in range(0, len(normal_contexts), self.max_micro_batch):
            j = min(i + self.max_micro_batch, len(normal_contexts))
            logps = self._token_logps_list(normal_contexts[i:j], completion_ids[i:j])
            loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
            for local_idx, token_logps in enumerate(logps):
                weight = weights[i + local_idx].to(self.device)
                seq_len = min(weight.numel(), token_logps.numel())
                if seq_len == 0:
                    continue
                weight = weight[:seq_len]
                token_logps = token_logps[:seq_len]
                loss_sum = loss_sum - (weight * token_logps).sum()
                logp_total += float(token_logps.detach().mean().cpu())

            if loss_sum.requires_grad:
                (loss_sum / float(total_tokens) * loss_scale).backward()
                loss_total += float(loss_sum.detach().cpu())

        return {
            "loss": loss_total / float(total_tokens),
            "student_logp": logp_total / max(1, math.ceil(len(normal_contexts) / self.max_micro_batch)),
            "tokens": float(total_tokens),
        }

    def _log_debug_example(self, prompt, target, completion_text, metrics):
        if self._logged_example or not self.log_first_batch:
            return
        self._logged_example = True
        print("\n[OPSD-direct debug] student prompt:")
        print(self._normal_context_text(prompt)[:1000])
        print("[OPSD-direct debug] teacher prompt:")
        print(self._teacher_context_text(prompt, target)[:1000])
        print("[OPSD-direct debug] sampled completion:")
        print(completion_text[:1000])
        print(f"[OPSD-direct debug] metrics: {metrics}")

    def train(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.training.per_device_train_batch_size),
            shuffle=True,
            collate_fn=lambda examples: examples,
        )
        iterator = iter(loader)
        grad_accum = int(self.cfg.training.gradient_accumulation_steps)
        max_steps = int(self.cfg.training.max_steps)
        logging_steps = int(self.cfg.training.logging_steps)

        progress = tqdm(range(max_steps), desc="Training OPSD-direct")
        for _ in progress:
            self.optimizer.zero_grad(set_to_none=True)
            step_metrics = {}

            for _accum_idx in range(grad_accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                prompts = [example["prompt"] for example in batch]
                targets = [example["target"] for example in batch]
                expanded_prompts, completion_ids, completion_texts = self._generate_rollouts(prompts)
                expanded_targets = [
                    target for target in targets for _ in range(self.num_generations)
                ]

                weights, weight_metrics = self._make_weights(
                    expanded_prompts, expanded_targets, completion_ids
                )
                loss_metrics = self._student_backward(
                    expanded_prompts,
                    weights,
                    completion_ids,
                    loss_scale=1.0 / float(grad_accum),
                )

                step_metrics = {**weight_metrics, **loss_metrics}
                if completion_texts:
                    self._log_debug_example(
                        expanded_prompts[0],
                        expanded_targets[0],
                        completion_texts[0],
                        step_metrics,
                    )

            max_grad_norm = float(self.cfg.training.max_grad_norm)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            step_metrics["train/lr"] = self.scheduler.get_last_lr()[0]
            progress.set_postfix(loss=f"{step_metrics.get('loss', 0.0):.4f}")
            if self.global_step % logging_steps == 0:
                print(f"[OPSD-direct] step={self.global_step} metrics={step_metrics}")
                if self.report_to_wandb:
                    wandb.log(step_metrics, step=self.global_step)

        self.save_model(os.path.join(self.cfg.training.output_dir, "best_model"))

    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Saved OPSD-direct model to {output_dir}")
