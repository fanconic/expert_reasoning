Use $phd-second-brain for this: I am currently going my own paper again [Learning Reasoning Rewards from Expert Demonstrations with Inverse Reinforcement Learning.md](wiki/sources/Learning Reasoning Rewards from Expert Demonstrations with Inverse Reinforcement Learning.md) . Can you go over the training parameters in the appendix, and compare it to [Black-Box On-Policy Distillation of Large Language Models.md](wiki/sources/Black-Box On-Policy Distillation of Large Language Models.md) and also [Unified Reinforcement and Imitation Learning for Vision-Language Models.md](raw/Unified Reinforcement and Imitation Learning for Vision-Language Models.md). I feel like my training is not ery stable yet. Maybe because of too small models, or also too small batch size etc. Can you investigate it and propose ways how it can be more stable? Also, Can you have a close look at how I do the GRPO in my paper. basically I do normal GRPO over the averaged reward of all the dense rewards, and not really a per token reward assigned by the token classifier. Can you compare this with [Process Reinforcement through Implicit Rewards.md](wiki/sources/Process Reinforcement through Implicit Rewards.md) and their method. I tried with their discounted future rewards but that did not lead to good results or stable treinaing. Can you try to figure out why this was the case? Potentially again in the traiing parameters. Or generally is there another way to do dense reward RL on reasoning traces?



Default app




Okay this is an interesting part. So based on this, can you remake the plan for the new experiments? Anything I should keep in consideration when having the two heads?
Thursday 10:51 PM

Yes. I’d remake the experiment plan around one clean question:

Can an AIRL-structured critic make demonstration-only reasoning rewards more stable by separating local reward from prefix shaping?

Core Setup
Segment the trace first. Use reasoning steps if reliably parseable; otherwise use interval chunks, e.g. every 10-20 tokens. For each segment:

s_k     = prompt + prefix before segment
a_k     = segment text
s_{k+1} = prompt + prefix after segment
Train:

f_k = g(s_k, a_k) + gamma h(s_{k+1}) - h(s_k)
D_k = exp(f_k) / (exp(f_k) + pi_old(a_k | s_k))
Then compare policy rewards:

g_k                 # reward-only
f_k                 # shaped reward
g_k + lambda shape  # explicit blend
where shape = gamma h_next - h_prev.

Experiment Plan

Stabilized current baseline
Keep your existing R-AIRL critic, but rerun with lower policy LR, larger effective batch if possible, explicit KL, and interval rewards. This gives a fair baseline.

AIRL critic without policy RL
Train the two-head critic only. Evaluate:

expert vs policy/corrupted AUROC;
reward separation correct vs incorrect traces;
Best-of-16 reranking with mean(g), mean(f), and final h;
error localisation using drops in f, g, and h.
Policy RL with three reward choices
Run interval-level GRPO:

mean(g_k) as sequence reward;
mean(f_k) as shaped sequence reward;
mean(g_k) + lambda * mean(shape_k) with lambda in {0.1, 0.3, 1.0}.
Optional local shaping
If sequence-level AIRL is stable, add a small segment-local advantage:

A_{i,k} = A_seq_i + lambda * norm_group(g_{i,k})
or more conservatively:

A_{i,k} = A_seq_i - lambda * norm_group(drop_{i,k})
Transfer test
This is important for the AIRL story. Train critic on one dataset/backbone, rerank or RL on another. If g transfers better than f, that supports the claim that g is less shaped/more reward-like.

Two-Head Considerations
The big risk is that h explains everything and g becomes useless. To prevent that:

keep g and h as separate scalar heads, but share the backbone;
regularize h, e.g. smaller head LR or L2 penalty;
track magnitudes of g, h_next - h_prev, and f;
optionally stop-gradient through h in early experiments when computing g diagnostics;
normalize g and shaping separately before combining for GRPO;
do not let pi(a|s) use the updated policy accidentally: use pi_old or reference logprobs for discriminator training;
choose gamma close to 1, e.g. 0.99 or 1.0 for reasoning traces, since segments are not physical time steps;
avoid token-level first. Segment-level is much cleaner.
Paper Framing
I’d frame this as:

The original dense critic entangles local reasoning quality with prefix-level shaping. Inspired by AIRL, we decompose the discriminator logit into a local reward head and a potential-difference head, then test whether the reward head gives more stable verifier-free reasoning RL.

That gives you a principled follow-up without abandoning the paper’s core “no programmatic verifier” premise.



Thursday 10:51 PM
How do you pretrain the two head critic, Is it still just classification?
Thursday 11:09 PM

I like it.

Can you based on these observations and plan (without the framing) write a prompt for a coding agent, such that it manupulates exaclty these things, and sets up the experiment

I mainly care about the training first, so hence step 1, 3, and 4.
Thursday 11:11 PM

Here’s a prompt you can hand to a coding agent:

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


Thursday 11:11 PM
what is the reward density now at this point? And how can it be fuest?
12:32 AM

At this point the clean setup is segment/interval-dense critic, but sequence-level GRPO by default.

Meaning:

Critic produces:      g_k, h_k, f_k for every segment k
Policy reward uses:   mean_k(g_k) or mean_k(f_k)
GRPO sees:            one scalar reward per rollout
So it is not token-dense policy RL yet. It is denser than sparse outcome reward because the critic scores every interval, but after averaging, GRPO still gets a trajectory-level reward.

If you enable the local auxiliary, then it becomes interval-dense policy shaping:

A_{i,k} = A_seq_i + lambda_local * local_signal_{i,k}
assigned only to tokens in segment k.

For fusion, I’d use this hierarchy:

Most conservative:
R_i = mean_k g_{i,k}
Use h only to train the AIRL discriminator, but optimize the policy with the reward-like head g.

Shaped sequence reward:
R_i = mean_k f_{i,k}
    = mean_k(g_{i,k} + gamma h_{i,k+1} - h_{i,k})
This may train faster, but f is more shaped/less portable.

Controlled fusion:
R_i = mean_k g_{i,k} + lambda_shape * mean_k(shape_{i,k})
with:

lambda_shape in {0.1, 0.3, 1.0}
Then optional interval-local shaping:

A_{i,k} = A_seq_i - lambda_local * norm(drop_f_{i,k})
drop_f = max(0, f_{i,k-1} - f_{i,k})
My recommendation: start with mean_g as the main reward, then compare mean_f, then add small drop_f shaping. That gives you a clear ablation from “most principled reward-like” to “more shaped but potentially more stable.”



12:33 AM
# Execution Plan for `instructions.md`

## Objective

Implement the next R-AIRL training experiment setup with minimal unrelated refactoring:

- Add a stabilized interval R-AIRL baseline.
- Add an AIRL-structured two-head segment critic.
- Add policy reward modes for the new critic.
- Add optional segment-local shaping.
- Add runnable configs and scripts.
- Verify tensor shapes, logging, gradients, and tiny smoke runs.

The implementation should preserve existing behavior behind config flags and should not add programmatic verifiers as training rewards.

## Guiding Constraints

- Keep changes narrow and local to the training, model-loading, and config paths.
- Reuse existing interval reward, replay buffer, discriminator warmup, GRPO loss, and evaluation code where possible.
- Avoid broad refactors of `AIRLTrainer`; add small helper functions/modules for segment-specific logic.
- Document new config flags where they are introduced.
- Keep default config values compatible with current behavior.

## Files Expected to Change

- `src/config/irl_config.py`
- `src/training/irl_module.py`
- `src/training/airl_trainer_new.py`
- `src/models/model_module.py`
- `src/models/model_module_trl.py`, if the non-Unsloth path needs parity
- New helper module, likely `src/training/airl_segment_utils.py`
- New or copied configs under `configs/math/qwen3b/` or a small dedicated experiment folder
- New runner script folder, likely `runner_scripts/airl_segment/`
- Focused tests, if a test directory is added
- `configs/README.md`, if config flags need user-facing documentation

## Phase 1: Baseline Audit

1. Inspect the existing AIRL flow:
   - reward/discriminator update in `AIRLTrainer._update_reward_model_step`;
   - reward computation in `AIRLTrainer._calculate_rewards`;
   - GRPO advantage computation in `AIRLTrainer._advantage_calculation`;
   - replay buffer usage in `_generate_and_score_completions`;
   - KL handling in `compute_loss`;
   - interval reward modes: `partial` and `partial_fixed`.

2. Confirm current baseline config values:
   - `model.dense_rewards`;
   - `model.dense_partial_fixed_n`;
   - `model.reward_updates_per_policy_step`;
   - `training.reward_warmup_steps`;
   - `training.buffer_size`;
   - `training.beta`;
   - `sampling.num_generations`.

3. Identify metrics already logged:
   - reward mean/std;
   - KL;
   - completion length;
   - discriminator BCE/accuracy;
   - evaluation correctness/pass@k.

## Phase 2: Stabilized Interval Baseline

Add a runnable config/script variant for the existing critic with:

- interval rewards by default:
  - prefer `model.dense_rewards: "partial_fixed"` for stable fixed intervals;
  - use `model.dense_partial_fixed_n: 15`.
- policy learning rate sweep:
  - `5e-7`;
  - `1e-6`.
- discriminator learning rate sweep:
  - `1e-6`;
  - `5e-6`.
- KL coefficient sweep:
  - `training.beta: 0.001`;
  - `training.beta: 0.003`;
  - `training.beta: 0.01`.
- rollout target:
  - `sampling.num_generations: 8`;
  - keep `training.per_device_train_batch_size * training.gradient_accumulation_steps * num_devices >= 256` where feasible.
- preserve:
  - discriminator warmup;
  - replay buffer;
  - `model.reward_updates_per_policy_step: 3`.

Make sure existing logging covers:

- policy reward mean/std;
- KL to reference;
- response length;
- discriminator accuracy;
- pass@1/eval accuracy.

Add AUROC and reward-correctness separation only if available without invasive evaluation changes.

## Phase 3: Add Config Flags

Extend `IRLConfig` with new fields, keeping defaults inactive:
yaml
model:
  critic_type: standard
  reward_mode: standard
  segment_tokens: 15
  airl_gamma: 1.0
  h_head_lr_mult: 0.5
  h_l2_penalty: 1e-4
  shape_l2_penalty: 1e-4
  shape_clamp:
  lambda_shape: 1.0
  use_segment_local_advantage: false
  lambda_local: 0.05
  local_signal: drop_f
  clipped_delta_f_min: -2.0
  clipped_delta_f_max: 2.0
Pass these values through `run_irl_training` into `AIRLTrainer`.

## Phase 4: AIRL Segment Helper Utilities

Create a small helper module for segment logic, likely `src/training/airl_segment_utils.py`.

Required helpers:

- build segment boundaries from:
  - existing interval masks;
  - fallback fixed `segment_tokens`.
- compute segment start/end token indices.
- compute prefix-before and prefix-after indices.
- build segment masks for token assignment.
- reduce segment-level rewards to sequence-level rewards.
- broadcast segment advantages back to token positions.
- validate no NaNs/infs in diagnostic tensors.

Keep these helpers tensor-oriented and unit-testable.

## Phase 5: AIRL-Structured Critic Model

When `model.critic_type == "airl_segment"`:

1. Modify reward model loading so the existing reward backbone has:
   - `g_head`: scalar local reward head;
   - `h_head`: scalar prefix-potential head.

2. Use the same base transformer/backbone for both heads.

3. Preserve current reward model behavior when `critic_type != "airl_segment"`.

4. Save/load both heads with the reward adapter:
   - include `g_head` and `h_head` in `modules_to_save` where PEFT is used.

5. Configure optimizer param groups:
   - backbone and `g_head` use normal reward learning rate;
   - `h_head` uses `reward_learning_rate * h_head_lr_mult`.

## Phase 6: Segment AIRL Discriminator

Add a segment-specific discriminator path in `_update_reward_model_step`.

For each segment `k`:

- `s_k`: prefix before segment;
- `a_k`: segment tokens;
- `s_next`: prefix after segment;
- `g_k = g_head(hidden_at_segment_end)`;
- `h_prev = h_head(hidden_at_prefix_before_segment)`;
- `h_next = h_head(hidden_at_prefix_after_segment)`;
- `shape_k = gamma * h_next - h_prev`;
- `f_k = g_k + shape_k`;
- `log_pi_seg = sum_t log pi_old(a_t | prefix_t)`;
- `disc_logit_k = f_k - log_pi_seg.detach()`.

Train with BCE:

- expert/positive segments: label `1`;
- policy/corrupted/incorrect-answer-agreement segments: label `0`;
- reuse existing positive/negative construction.

Guardrails:

- detach `log_pi_seg`;
- prevent actor gradients from discriminator BCE;
- add optional L2 penalty on `h` and/or `shape`;
- add optional `shape_k` clamp;
- keep clamp disabled by default.

## Phase 7: Policy Reward Modes

Add AIRL segment reward aggregation in `_calculate_rewards`.

Supported modes:

- `mean_g`:
  - sequence reward is mean over segment `g_k`.
- `mean_f`:
  - sequence reward is mean over segment `f_k`.
- `mean_g_plus_shape`:
  - sequence reward is `mean(g_k) + lambda_shape * mean(shape_k)`.

For all modes:

- keep existing GRPO group normalization;
- keep PPO/GRPO clipping;
- apply KL through existing `training.beta`;
- log reward components separately before and after normalization.

Sweep values:

- `lambda_shape: 0.1`;
- `lambda_shape: 0.3`;
- `lambda_shape: 1.0`.

## Phase 8: Optional Segment-Local Advantages

Add an inactive-by-default path in `_advantage_calculation`:
yaml
model:
  use_segment_local_advantage: false
When enabled:

- compute existing sequence-level GRPO advantage `A_seq_i`;
- compute segment-level local signal;
- assign local advantage only to tokens inside that segment.

Supported local signals:

- `g`:
  - `A_{i,k} = A_seq_i + lambda_local * normalized(g_{i,k})`.
- `clipped_delta_f`:
  - `delta_f = f_k - f_{k-1}`;
  - clip to `[clipped_delta_f_min, clipped_delta_f_max]`;
  - add normalized signal.
- `drop_f`:
  - `drop = max(0, f_{k-1} - f_k)`;
  - `A_{i,k} = A_seq_i - lambda_local * normalized(drop)`.

Default sweep:

- `lambda_local: 0.05`;
- `lambda_local: 0.1`.

## Phase 9: Diagnostics and Logging

Add training metrics for the segment critic:

- `airl_segment/g_mean`;
- `airl_segment/g_std`;
- `airl_segment/shape_mean`;
- `airl_segment/shape_std`;
- `airl_segment/f_mean`;
- `airl_segment/f_std`;
- `airl_segment/shape_to_g_abs_ratio`;
- `airl_segment/h_prev_mean`;
- `airl_segment/h_next_mean`;
- `airl_segment/disc_bce`;
- `airl_segment/disc_acc`;
- `airl_segment/disc_auroc`, if available;
- reward component mean/std before normalization;
- reward component mean/std after normalization.

Also log one example trace periodically or during smoke tests:

- decoded trace;
- segment boundaries;
- per-segment `g`, `shape`, and `f`.

## Phase 10: Configs and Runner Scripts

Create runnable entries for:

1. Stabilized interval R-AIRL baseline.
2. AIRL segment critic with `reward_mode: mean_g`.
3. AIRL segment critic with `reward_mode: mean_f`.
4. AIRL segment critic with `reward_mode: mean_g_plus_shape`.
5. AIRL segment critic with segment-local `drop_f`.
6. Tiny debug config.

Use GSM8K/Qwen3B as the default starting point unless another fast-debug setup is available.

Suggested layout:
text
configs/math/qwen3b/airl_segment/
  stabilized_interval.yaml
  mean_g.yaml
  mean_f.yaml
  mean_g_plus_shape.yaml
  drop_f_local.yaml
  debug.yaml

runner_scripts/airl_segment/
  smoke.sh
  main_sweep.sh
## Phase 11: Verification

Before finishing, run what is feasible in the current environment:

1. Unit tests or focused helper tests:
   - segment boundary construction;
   - previous/next prefix index alignment;
   - segment-to-token advantage broadcasting;
   - fixed `segment_tokens` fallback.

2. Tiny smoke runs:
   - stabilized interval baseline;
   - `airl_segment + mean_g`;
   - `airl_segment + mean_f`;
   - `airl_segment + mean_g_plus_shape`;
   - `airl_segment + drop_f` local advantage.

3. Runtime checks:
   - no NaNs/infs in `g`, `h`, `shape`, `f`;
   - no NaNs/infs in reward, advantage, and KL;
   - discriminator BCE does not produce actor gradients;
   - `h_prev`, `h_next`, and segment boundaries align correctly.

4. Evaluation checks:
   - pass@1/eval accuracy still logs;
   - reward model scores still serialize in eval jsonl;
   - existing non-segment configs still run or at least import cleanly.

## Smoke Command Templates

Stabilized baseline:
bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/stabilized_interval \
  training.output_dir=./outputs/smoke_stabilized_interval \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
AIRL segment `mean_g`:
bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/mean_g \
  training.output_dir=./outputs/smoke_airl_segment_mean_g \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
AIRL segment `mean_f`:
bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/mean_f \
  training.output_dir=./outputs/smoke_airl_segment_mean_f \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
AIRL segment `mean_g_plus_shape`:
bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/mean_g_plus_shape \
  training.output_dir=./outputs/smoke_airl_segment_mean_g_plus_shape \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
AIRL segment local `drop_f`:
bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/drop_f_local \
  training.output_dir=./outputs/smoke_airl_segment_drop_f \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
## Main Sweep Template
bash
bash runner_scripts/airl_segment/main_sweep.sh
Expected sweep dimensions:

- policy learning rate: `5e-7`, `1e-6`;
- discriminator learning rate: `1e-6`, `5e-6`;
- KL coefficient: `0.001`, `0.003`, `0.01`;
- reward mode: `mean_g`, `mean_f`, `mean_g_plus_shape`;
- `lambda_shape`: `0.1`, `0.3`, `1.0` for `mean_g_plus_shape`;
- optional local shaping: `drop_f`;
- `lambda_local`: `0.05`, `0.1`.

## Deliverables Checklist

- [ ] Changed files listed.
- [ ] New config flags explained.
- [ ] Smoke commands provided.
- [ ] Main experiment commands provided.
- [ ] Tests/smoke runs completed and summarized.
- [ ] Any failures or skipped checks documented.
- [ ] Existing non-segment behavior preserved by default.

This was my exectuion plan that has been done so far. Can you make a second exectuion plan, that also makes sure that I use dense/ interval rewards (based on the differne potentially, as discussed above?)

Show more
12:37 AM


Yes. I’d make the second plan explicitly separate critic density from policy reward density, because that is the easy place to accidentally think you are doing dense RL while GRPO still sees one scalar.

# Execution Plan 2: Dense / Interval AIRL Reward Usage

## Objective

Extend the implemented AIRL-segment setup so experiments can explicitly compare:

1. sequence-level rewards from interval/dense critic scores;
2. interval-level policy advantages from segment rewards;
3. delta/drop-based interval shaping;
4. optional token-broadcast of interval advantages.

The goal is to ensure we can test whether dense/interval rewards help training, rather than only using interval critic scores averaged into a scalar rollout reward.

Do not add programmatic verifiers as training rewards.

## Key Distinction

Add code/configs that make this distinction explicit:

- `critic_density`: where the critic scores are produced.
  - `sequence`
  - `interval`
  - `token` if already supported by the standard critic

- `policy_reward_density`: how GRPO consumes the reward.
  - `sequence`: one scalar reward per rollout, broadcast to all tokens.
  - `interval`: one advantage per interval/segment, broadcast only to tokens inside that interval.
  - `token`: one advantage per token, only if existing token-level critic path supports it safely.

Current AIRL segment setup likely has:

```text
critic_density = interval
policy_reward_density = sequence