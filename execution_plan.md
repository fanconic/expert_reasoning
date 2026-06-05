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

```yaml
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
```

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

```yaml
model:
  use_segment_local_advantage: false
```

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

```text
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
```

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

```bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/stabilized_interval \
  training.output_dir=./outputs/smoke_stabilized_interval \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
```

AIRL segment `mean_g`:

```bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/mean_g \
  training.output_dir=./outputs/smoke_airl_segment_mean_g \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
```

AIRL segment `mean_f`:

```bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/mean_f \
  training.output_dir=./outputs/smoke_airl_segment_mean_f \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
```

AIRL segment `mean_g_plus_shape`:

```bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/mean_g_plus_shape \
  training.output_dir=./outputs/smoke_airl_segment_mean_g_plus_shape \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
```

AIRL segment local `drop_f`:

```bash
python train_irl.py \
  --config-path=configs/math/qwen3b \
  --config-name=airl_segment/drop_f_local \
  training.output_dir=./outputs/smoke_airl_segment_drop_f \
  training.report_to=none \
  training.max_steps=2 \
  eval.do_eval=false
```

## Main Sweep Template

```bash
bash runner_scripts/airl_segment/main_sweep.sh
```

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
