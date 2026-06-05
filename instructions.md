You are working in my R-AIRL / expert_reasoning training codebase. Please implement the next training experiment setup only, with minimal unrelated refactoring.

Goal: add a stabilized interval-GRPO baseline and an AIRL-structured two-head critic for verifier-free reasoning reward training. Do not introduce programmatic verifiers as training rewards.

First inspect the existing training code, configs, critic model, GRPO loss, replay buffer, reward computation, and experiment scripts. Preserve existing behavior behind config flags.

Tasks:

1. Stabilized Current Baseline
Add a config/experiment variant for the existing R-AIRL critic using:
- interval reward by default, not dense token reward;
- policy learning rate sweep: `5e-7`, `1e-6`;
- discriminator learning rate sweep: `1e-6`, `5e-6`;
- explicit KL-to-reference/init/SFT policy coefficient sweep: `0.001`, `0.003`, `0.01`;
- effective rollout batch target `>=256` sequences if feasible via grad accumulation;
- group size `G=8` initially, keep existing group-size config;
- discriminator warmup preserved;
- replay buffer preserved;
- 3 discriminator updates per policy update preserved initially.

Make sure metrics log:
- policy reward mean/std;
- KL to reference;
- response length;
- discriminator accuracy/AUROC if available;
- pass@1/eval accuracy;
- reward-correctness separation if existing evaluation supports it.

2. Add AIRL-Structured Critic
Implement a new critic mode, e.g. `critic_type: airl_segment`.

Segment each generated reasoning trace into intervals. Use existing interval boundaries if present; otherwise add configurable `segment_tokens`, default `15`.

For each segment k:
- `s_k` = prefix before the segment;
- `a_k` = segment tokens;
- `s_next` = prefix after the segment.

The critic should share the base transformer/backbone and have two scalar heads:
- `g_head`: local reward-like score for the segment, computed from the hidden state at the segment end or from a segment pooled representation.
- `h_head`: prefix-potential score, computed from hidden states at prefix boundaries.

Compute:
- `g_k = g_head(hidden_at_segment_end)`
- `h_prev = h_head(hidden_at_prefix_before_segment)`
- `h_next = h_head(hidden_at_prefix_after_segment)`
- `shape_k = gamma * h_next - h_prev`
- `f_k = g_k + shape_k`

Use `gamma=1.0` by default, configurable with `0.99` option.

AIRL discriminator:
- compute detached segment log probability `log_pi_seg = sum_t log pi_old(a_t | prefix_t)` for the segment;
- use `disc_logit_k = f_k - log_pi_seg`;
- train with BCE classification:
  - expert / positive segments label `1`;
  - policy-negative / corrupted / incorrect-answer-agreement segments label `0`, using the existing positive/negative construction.
- Important: `log_pi_seg` must be detached; discriminator loss should update only critic/backbone/head parameters, not the actor.

Add diagnostics:
- mean/std of `g_k`;
- mean/std of `shape_k`;
- mean/std of `f_k`;
- ratio or absolute magnitude of `shape_k` vs `g_k`;
- discriminator BCE loss and accuracy;
- AUROC if available;
- correlation/separation between `mean(g)`, `mean(f)`, and final correctness if existing eval labels exist.

Guardrail: prevent `h` from absorbing everything. Add config options:
- separate learning-rate multiplier for `h_head`, default `0.5`;
- optional L2 penalty on `h` and/or `shape`, default small, e.g. `1e-4`;
- optional clamp for `shape_k`, default disabled but configurable.

3. Policy RL Reward Choices
Add policy reward mode config for AIRL critic:
- `reward_mode: mean_g`
  - sequence reward is mean over segment `g_k`.
- `reward_mode: mean_f`
  - sequence reward is mean over segment `f_k`.
- `reward_mode: mean_g_plus_shape`
  - sequence reward is `mean(g_k) + lambda_shape * mean(shape_k)`.
  - sweep `lambda_shape in [0.1, 0.3, 1.0]`.

For all reward modes:
- normalize rewards with existing GRPO group normalization;
- keep PPO/GRPO clipping;
- apply KL-to-reference penalty from the stabilized baseline config;
- log all reward components separately before and after normalization.

4. Optional Segment-Local Shaping
Add a flag for segment-local auxiliary advantages, default off:
- `use_segment_local_advantage: false`

When enabled, compute token/segment advantages as:
- `A_seq_i = existing sequence-level GRPO advantage`
- for each segment k:
  - `A_{i,k} = A_seq_i + lambda_local * normalized_local_signal_{i,k}`

Support local signal choices:
- `local_signal: g`
  - use normalized `g_{i,k}`;
- `local_signal: clipped_delta_f`
  - compute `delta_f = f_k - f_{k-1}`, clipped to configurable range, default `[-2, 2]`;
- `local_signal: drop_f`
  - compute `drop = max(0, f_{k-1} - f_k)` and subtract it:
    `A_{i,k} = A_seq_i - lambda_local * norm(drop)`.

Default sweep:
- `lambda_local in [0.05, 0.1]`.

Assign the segment-local advantage only to tokens inside that segment.

5. Experiment Entrypoints
Create runnable configs/scripts for:
- stabilized baseline interval R-AIRL;
- AIRL critic with `mean_g`;
- AIRL critic with `mean_f`;
- AIRL critic with `mean_g_plus_shape`;
- AIRL critic with segment-local `drop_f` auxiliary.

Keep dataset/backbone defaults consistent with existing GSM8K or the smallest fast-debug setup. Include a short debug run config with tiny steps/batches to validate tensor shapes and logging.

6. Verification
Before finishing:
- run unit tests if present;
- run a tiny smoke training run for each new critic/reward mode if feasible;
- verify no NaNs in `g`, `h`, `shape`, `f`, reward, advantage, and KL;
- verify actor parameters do not receive gradients from discriminator BCE;
- verify `h_prev`, `h_next`, and segment boundaries align correctly;
- print or log one example trace with segment boundaries and `g/shape/f` values.

Deliverables:
- list changed files;
- explain the new config flags;
- provide exact commands for the smoke runs and main experiment runs;
- summarize any tests/smoke runs completed and any failures.