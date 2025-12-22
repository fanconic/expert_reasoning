"""
Advantage Calculation Module
-----------------------------
Handles reward normalization and advantage calculation for GRPO and PRIME.
Supports both sparse (scalar per completion) and dense (token-level) rewards.
"""

from typing import Tuple, Optional
import torch


class AdvantageCalculator:
    """Computes advantages from rewards using GRPO or PRIME methods."""

    def __init__(
        self,
        device: torch.device,
        advantage_calculation: str = "grpo",
        normalise_rewards: bool = True,
        scale_rewards: bool = True,
        dense_rewards: bool = False,
        dense_gamma: float = 0.99,
    ):
        self.device = device
        self.advantage_calculation = advantage_calculation
        self.normalise_rewards = normalise_rewards
        self.scale_rewards = scale_rewards
        self.dense_rewards = dense_rewards
        self.dense_gamma = dense_gamma

    def compute_advantages(
        self,
        rewards_per_func: torch.Tensor,
        reward_weights: torch.Tensor,
        completion_mask: torch.Tensor,
        num_generations: int,
        add_expert_to_policy: bool = False,
        num_experts_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute advantages from rewards.
        
        Args:
            rewards_per_func: [N, K] or [N, K, T] tensor of rewards per sample and reward function
            reward_weights: [K] tensor of weights for each reward function
            completion_mask: [N, T] tensor indicating valid completion tokens
            num_generations: Number of generations per prompt
            add_expert_to_policy: Whether expert examples are included
            num_experts_per_prompt: Number of expert examples per prompt
            
        Returns:
            advantages: [N, T] or [N] tensor of advantages
            metrics: Dictionary of metrics for logging
        """
        metrics = {}

        # Determine the effective group size (may include experts)
        if add_expert_to_policy:
            advantage_num_generation = num_generations + num_experts_per_prompt
        else:
            advantage_num_generation = num_generations

        # Combine rewards from different functions
        if not self.dense_rewards:
            advantages, metrics_reward = self._compute_advantages_sparse(
                rewards_per_func, reward_weights, advantage_num_generation
            )
        else:
            advantages, metrics_reward = self._compute_advantages_dense(
                rewards_per_func, reward_weights, completion_mask, advantage_num_generation
            )

        metrics.update(metrics_reward)
        return advantages, metrics

    def _compute_advantages_sparse(
        self,
        rewards_per_func: torch.Tensor,
        reward_weights: torch.Tensor,
        num_generations: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute advantages for scalar rewards (one per sequence)."""
        if self.advantage_calculation == "grpo":
            return self._grpo_sparse(rewards_per_func, reward_weights, num_generations)
        elif self.advantage_calculation == "prime":
            return self._prime_sparse(rewards_per_func, reward_weights, num_generations)
        else:
            raise NotImplementedError(f"Advantage calculation `{self.advantage_calculation}` not implemented")

    def _compute_advantages_dense(
        self,
        rewards_per_func: torch.Tensor,
        reward_weights: torch.Tensor,
        completion_mask: torch.Tensor,
        num_generations: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute advantages for token-level rewards."""
        if self.advantage_calculation == "grpo":
            return self._grpo_dense(rewards_per_func, reward_weights, completion_mask, num_generations)
        elif self.advantage_calculation == "prime":
            return self._prime_dense(rewards_per_func, reward_weights, completion_mask, num_generations)
        else:
            raise NotImplementedError(f"Advantage calculation `{self.advantage_calculation}` not implemented")

    def _grpo_sparse(
        self,
        rewards_per_func: torch.Tensor,
        reward_weights: torch.Tensor,
        num_generations: int,
    ) -> Tuple[torch.Tensor, dict]:
        """GRPO advantage: normalize by group mean and std."""
        # Combine rewards: [N]
        rewards = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)

        # Group-normalize: compute mean/std per prompt
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1, unbiased=False)

        # Expand back to full shape
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)

        # Compute advantages
        if self.normalise_rewards:
            advantages = rewards - mean_grouped_rewards
        else:
            advantages = rewards

        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        metrics = {
            "reward": mean_grouped_rewards.mean().item(),
            "reward_std": std_grouped_rewards.mean().item(),
            "frac_reward_zero_std": is_std_zero.float().mean().item(),
        }

        return advantages, metrics

    def _prime_sparse(
        self,
        rewards_per_func: torch.Tensor,
        reward_weights: torch.Tensor,
        num_generations: int,
    ) -> Tuple[torch.Tensor, dict]:
        """PRIME advantage: leave-one-out baseline."""
        raise NotImplementedError("PRIME advantage calculation for sparse rewards not implemented")

    def _grpo_dense(
        self,
        rewards_per_func: torch.Tensor,
        reward_weights: torch.Tensor,
        completion_mask: torch.Tensor,
        num_generations: int,
    ) -> Tuple[torch.Tensor, dict]:
        """GRPO advantage for token-level rewards."""
        # Combine rewards: [N, T]
        rewards = (
            rewards_per_func * reward_weights.unsqueeze(0).unsqueeze(-1)
        ).nansum(dim=1)

        # Align with token dimension
        reward_mask = torch.flip(completion_mask, dims=[-1])
        rewards = rewards[:, -completion_mask.size(1):]
        rewards = rewards.masked_fill(reward_mask == 0, torch.nan)

        # Compute trajectory-level statistics
        N, T = rewards.shape
        K = num_generations
        B = N // K

        # Mean reward per trajectory
        traj_mean = torch.nanmean(rewards, dim=1).view(B, K)  # [B, K]
        mean_last = torch.mean(traj_mean, dim=1)  # [B]
        std_last = torch.std(traj_mean, dim=1)  # [B]

        # Expand back to [N, 1] for broadcasting
        mean_grouped_rewards = mean_last.repeat_interleave(K, dim=0).unsqueeze(-1)
        std_grouped_rewards = std_last.repeat_interleave(K, dim=0).unsqueeze(-1)

        # Compute token-level advantages
        if self.normalise_rewards:
            a_tilde = rewards - mean_grouped_rewards
        else:
            a_tilde = rewards

        if self.scale_rewards:
            a_tilde = a_tilde / (std_grouped_rewards + 1e-4)

        # Apply discount factor
        advantages = self._apply_discount_factor(a_tilde, reward_mask)

        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        metrics = {
            "reward": (rewards * reward_mask).sum(1).mean().item() / reward_mask.sum(1).mean().item(),
            "reward_std": (((rewards**2 - (rewards * reward_mask).sum(1).unsqueeze(1)**2 / reward_mask.sum(1).unsqueeze(1)) * reward_mask).sum(1) / reward_mask.sum(1))**(1/2) .mean().item(),
            "frac_reward_zero_std": is_std_zero.float().mean().item(),
        }

        return advantages, metrics

    def _prime_dense(
        self,
        rewards_per_func: torch.Tensor,
        reward_weights: torch.Tensor,
        completion_mask: torch.Tensor,
        num_generations: int,
    ) -> Tuple[torch.Tensor, dict]:
        """PRIME advantage for token-level rewards (leave-one-out baseline)."""
        # Combine rewards: [N, T]
        rewards = (
            rewards_per_func * reward_weights.unsqueeze(0).unsqueeze(-1)
        ).nansum(dim=1)

        reward_mask = torch.flip(completion_mask, dims=[-1])
        rewards = rewards[:, -completion_mask.size(1):]
        rewards = rewards.masked_fill(reward_mask == 0, torch.nan)

        # Trajectory statistics
        N, T = rewards.shape
        K = num_generations
        B = N // K

        # Mean reward per trajectory
        traj_mean = torch.nanmean(rewards, dim=1).view(B, K)  # [B, K]

        # Leave-one-out baseline: (1/(K-1)) * Σ_{j≠i} r_φ(y^j)
        sum_mean = torch.nansum(traj_mean, dim=1, keepdim=True)  # [B, 1]
        count = (~torch.isnan(traj_mean)).sum(dim=1, keepdim=True).clamp(min=1)
        others = (count - 1).clamp(min=1)

        baseline_loo = (sum_mean - traj_mean) / others  # [B, K]
        baseline_loo = baseline_loo.view(N)  # [N]

        # Advantage: r(s,a) - baseline
        a_tilde = rewards - baseline_loo.unsqueeze(1)

        # Apply discount factor
        advantages = self._apply_discount_factor(a_tilde, reward_mask)

        metrics = {
            "reward": (rewards * reward_mask).sum(1).mean().item() / reward_mask.sum(1).mean().item(),
            "reward_std": (((rewards**2 - (rewards * reward_mask).sum(1).unsqueeze(1)**2 / reward_mask.sum(1).unsqueeze(1)) * reward_mask).sum(1) / reward_mask.sum(1))**(1/2) .mean().item(),
        }

        return advantages, metrics

    def _apply_discount_factor(
        self,
        a_tilde: torch.Tensor,
        reward_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exponential discount factor to token-level advantages."""
        if self.dense_gamma <= 0.001:
            # No discount: advantages are just per-token a_tilde
            advantages = a_tilde
        else:
            # Geometric discount: turn suffix-sum into prefix-sum using geometric weighting
            T = a_tilde.size(1)
            p = torch.arange(T, device=self.device, dtype=a_tilde.dtype)
            p_pow = (self.dense_gamma ** p).unsqueeze(0)  # [1, T]
            
            x_rev = torch.flip(a_tilde, dims=[1])  # [N, T]
            s = torch.cumsum(x_rev / p_pow, dim=1)  # [N, T]
            y_rev = s * p_pow  # [N, T]
            advantages = torch.flip(y_rev, dims=[1])  # [N, T]

        # Mask and convert to right-padding format
        advantages = advantages.masked_fill(reward_mask == 0, 0.0)
        from src.training.reward_model_utils import left_to_right_pad
        advantages = left_to_right_pad(advantages, 0.0)

        return advantages
