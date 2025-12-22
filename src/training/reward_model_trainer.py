"""
Reward Model Training Module
-----------------------------
Encapsulates all logic for training the discriminator/reward model.
Handles tokenization, batch preparation, and loss computation.
"""

from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn


class RewardModelTrainer:
    """Handles reward model training with support for BCE and WGAN losses."""

    def __init__(
        self,
        reward_model: nn.Module,
        reward_optimizer: torch.optim.Optimizer,
        reward_tokenizer,
        accelerator,
        args,
        eps: float = 0.1,
        disc_temperature: float = 1.0,
        clip_reward_model: bool = False,
        reward_lb: float = -1.0,
        reward_ub: float = 1.0,
        response_only: bool = False,
        dense_rewards: bool = False,
        classifier_loss: str = "bce",
        neg_label_smoothing: Optional[float] = None,
        neg_sample_weight: float = 1.0,
        max_completion_length: int = 512,
        max_micro_batch: int = 64,
        num_neg_perturbations_per_expert: int = 1,
    ):
        self.reward_model = reward_model
        self.reward_optimizer = reward_optimizer
        self.reward_tokenizer = reward_tokenizer
        self.accelerator = accelerator
        self.args = args
        self.eps = eps
        self.disc_temperature = disc_temperature
        self.clip_reward_model = clip_reward_model
        self.reward_lb = reward_lb
        self.reward_ub = reward_ub
        self.response_only = response_only
        self.dense_rewards = dense_rewards
        self.classifier_loss = classifier_loss
        self.neg_label_smoothing = neg_label_smoothing if neg_label_smoothing is not None else eps
        self.neg_sample_weight = neg_sample_weight
        self.max_completion_length = max_completion_length
        self.max_micro_batch = max_micro_batch
        self.num_neg_perturbations_per_expert = num_neg_perturbations_per_expert

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        weights: torch.Tensor,
        n_pos: int,
        n_pol: int,
        n_per: int,
        pos_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training loop for reward model with gradient accumulation and micro-batching.
        
        Args:
            batch: Dictionary with tokenized input_ids and attention_mask
            labels: Binary labels (1 for expert, 0 for policy)
            weights: Sample weights for weighted loss
            n_pos: Number of positive (expert) samples
            n_pol: Number of policy samples
            n_per: Number of perturbed samples
            pos_counts: Multiplicities for deduplicated samples
            
        Returns:
            Total loss accumulated across all gradient accumulation steps
        """
        total_loss = 0
        
        # Compute samples per accumulation step
        pos_per_step = n_pos // self.args.gradient_accumulation_steps
        pol_per_step = n_pol // self.args.gradient_accumulation_steps
        per_per_step = n_per // self.args.gradient_accumulation_steps if n_per > 0 else 0
        have_perturbed = n_per > 0

        for step in range(self.args.gradient_accumulation_steps):
            # Get slices for this step
            pos_slice = slice(
                step * pos_per_step,
                (step + 1) * pos_per_step if step < self.args.gradient_accumulation_steps - 1 else n_pos
            )
            pol_slice = slice(
                step * pol_per_step,
                (step + 1) * pol_per_step if step < self.args.gradient_accumulation_steps - 1 else n_pol
            )

            # Prepare step data
            if have_perturbed:
                per_slice = slice(
                    step * per_per_step,
                    (step + 1) * per_per_step if step < self.args.gradient_accumulation_steps - 1 else n_per
                )
                step_batch = {
                    k: torch.cat([
                        batch[k][:n_pos][pos_slice],
                        batch[k][n_pos:n_pos+n_pol][pol_slice],
                        batch[k][n_pos+n_pol:][per_slice]
                    ], dim=0)
                    for k in batch.keys()
                }
                step_labels = torch.cat([
                    labels[:n_pos][pos_slice],
                    labels[n_pos:n_pos+n_pol][pol_slice],
                    labels[n_pos+n_pol:][per_slice]
                ])
                step_weights = torch.cat([
                    weights[:n_pos][pos_slice],
                    weights[n_pos:n_pos+n_pol][pol_slice],
                    weights[n_pos+n_pol:][per_slice]
                ])
            else:
                step_batch = {
                    k: torch.cat([
                        batch[k][:n_pos][pos_slice],
                        batch[k][n_pos:][pol_slice]
                    ], dim=0)
                    for k in batch.keys()
                }
                step_labels = torch.cat([
                    labels[:n_pos][pos_slice],
                    labels[n_pos:][pol_slice]
                ])
                step_weights = torch.cat([
                    weights[:n_pos][pos_slice],
                    weights[n_pos:][pol_slice]
                ])

            # Process micro-batches
            step_loss = self._process_micro_batches(
                step_batch, step_labels, step_weights, num_micro_batches_total=self.args.gradient_accumulation_steps
            )
            total_loss += step_loss

        return total_loss

    def _process_micro_batches(
        self,
        step_batch: Dict[str, torch.Tensor],
        step_labels: torch.Tensor,
        step_weights: torch.Tensor,
        num_micro_batches_total: int,
    ) -> torch.Tensor:
        """Process a single gradient accumulation step into multiple micro-batches."""
        step_size = step_batch["input_ids"].size(0)
        num_micro_batches = (step_size + self.max_micro_batch - 1) // self.max_micro_batch
        micro_batch_size = (step_size + num_micro_batches - 1) // num_micro_batches

        step_loss = 0
        for micro_idx in range(num_micro_batches):
            start_idx = micro_idx * micro_batch_size
            end_idx = min((micro_idx + 1) * micro_batch_size, step_size)
            
            micro_batch = {k: v[start_idx:end_idx] for k, v in step_batch.items()}
            micro_labels = step_labels[start_idx:end_idx]
            micro_weights = step_weights[start_idx:end_idx]

            if self.dense_rewards:
                micro_labels = micro_labels.unsqueeze(1).repeat(1, micro_batch["input_ids"].size(1))

            micro_loss = self._compute_micro_batch_loss(
                micro_batch, micro_labels, micro_weights, step_weights, num_micro_batches, num_micro_batches_total
            )
            step_loss += micro_loss.detach()

        return step_loss

    def _compute_micro_batch_loss(
        self,
        micro_batch: Dict[str, torch.Tensor],
        micro_labels: torch.Tensor,
        micro_weights: torch.Tensor,
        step_weights: torch.Tensor,
        num_micro_batches: int,
        num_micro_batches_total: int,
    ) -> torch.Tensor:
        """Compute loss for a single micro-batch using configured loss function."""
        if self.classifier_loss == "bce":
            return self._compute_bce_loss(
                micro_batch, micro_labels, micro_weights, step_weights, num_micro_batches, num_micro_batches_total
            )
        elif self.classifier_loss == "wgan":
            return self._compute_wgan_loss(
                micro_batch, micro_labels, micro_weights, step_weights, num_micro_batches, num_micro_batches_total
            )
        else:
            raise NotImplementedError(f"Classifier loss function `{self.classifier_loss}` not implemented")

    def _compute_bce_loss(
        self,
        micro_batch: Dict[str, torch.Tensor],
        micro_labels: torch.Tensor,
        micro_weights: torch.Tensor,
        step_weights: torch.Tensor,
        num_micro_batches: int,
        num_micro_batches_total: int,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss."""
        with self.accelerator.autocast():
            micro_logits = self.reward_model(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"]
            ).logits.squeeze(-1)
            
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits, micro_labels, reduction="none"
            )

            if self.dense_rewards:
                # Mask out non-response tokens
                masked_micro_bce = micro_bce.masked_fill(micro_batch["response_mask"] == 0, 0.0)
                masked_micro_bce = masked_micro_bce / micro_batch["response_mask"].sum(1).unsqueeze(1)
                micro_loss = (masked_micro_bce * micro_weights.unsqueeze(1)).sum() / step_weights.sum()
            else:
                micro_loss = (micro_bce * micro_weights).sum() / step_weights.sum()

            scaled_loss = micro_loss / (num_micro_batches_total * num_micro_batches)

        scaled_loss.backward()
        return micro_loss

    def _compute_wgan_loss(
        self,
        micro_batch: Dict[str, torch.Tensor],
        micro_labels: torch.Tensor,
        micro_weights: torch.Tensor,
        step_weights: torch.Tensor,
        num_micro_batches: int,
        num_micro_batches_total: int,
    ) -> torch.Tensor:
        """Compute Wasserstein GAN critic loss."""
        with self.accelerator.autocast():
            # Critic scores (no sigmoid) — shape: [B, T] if dense, else [B]
            micro_scores = self.reward_model(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"]
            ).logits.squeeze(-1)

            micro_labels = micro_labels.float()  # 1 = expert, 0 = policy

            if self.dense_rewards:
                # token mask for the generated response region
                resp_mask = micro_batch["response_mask"].float()  # [B, T]
                # per-token masks for real/fake
                real_tok_mask = resp_mask * (micro_labels if micro_labels.dim() == 2 else micro_labels.unsqueeze(1))
                fake_tok_mask = resp_mask * (1.0 - (micro_labels if micro_labels.dim() == 2 else micro_labels.unsqueeze(1)))

                # mean score over tokens (avoid div-by-zero)
                real_count = real_tok_mask.sum().clamp_min(1.0)
                fake_count = fake_tok_mask.sum().clamp_min(1.0)
                real_mean = (micro_scores * real_tok_mask).sum() / real_count
                fake_mean = (micro_scores * fake_tok_mask).sum() / fake_count

                # WGAN critic loss: minimise (fake - real)
                micro_loss = (fake_mean - real_mean)

            else:
                # sequence-level case
                is_real = (micro_labels.view(-1) > 0.5)
                real_scores = micro_scores[is_real]
                fake_scores = micro_scores[~is_real]

                # guard against empty split in tiny micro-batches
                real_mean = real_scores.mean() if real_scores.numel() > 0 else micro_scores.new_zeros(())
                fake_mean = fake_scores.mean() if fake_scores.numel() > 0 else micro_scores.new_zeros(())

                micro_loss = (fake_mean - real_mean)

            # scale for gradient accumulation
            scaled_loss = micro_loss / (num_micro_batches_total * num_micro_batches)

        scaled_loss.backward()
        return micro_loss

    def optimizer_step(self, reward_updates_per_policy_step: int, global_step: int, standard_grpo: bool = False):
        """Apply optimizer step with gradient clipping if needed."""
        if global_step % reward_updates_per_policy_step == 0 and not standard_grpo:
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
            self.reward_optimizer.step()
            self.reward_optimizer.zero_grad()
