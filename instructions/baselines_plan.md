# Coding Agent Task: Implement GAD and OPSD Baselines for R-AIRL/ReGAIL Rebuttal

## Goal

Implement exactly two rebuttal baselines for the R-AIRL/ReGAIL experiments:

1. **GAD-style black-box on-policy distillation**, on **Qwen2.5-7B**.
2. **OPSD-style on-policy self-distillation**, on the same Qwen2.5-7B GSM8K setting.

Do not implement source-label adversarial ablations, rePIRL, dense GAD, or other extra baselines.

## Baseline 1: GAD on Qwen2.5-7B

### Goal

Implement a faithful GAD-style sequence-level adversarial distillation baseline.

This tests whether black-box on-policy distillation from expert traces is enough to match R-AIRL/ReGAIL.

### Method

Use:

- **Student/policy:** Qwen2.5-7B.
- **Teacher data:** expert demonstrations/reference traces from the same training split.
- **Discriminator:** scalar sequence-level scorer.
- **Discriminator loss:** pairwise Bradley-Terry loss.
- **Policy update:** GRPO using the discriminator scalar score as reward.

For each prompt:

```text
expert trace: y_E
policy rollout: y_P ~ pi_theta(. | x)
```

Train discriminator with:

```text
loss_D = -log sigmoid(D_phi(x, y_E) - D_phi(x, y_P))
```

Train policy with:

```text
reward(y_P) = D_phi(x, y_P)
```

and the existing GRPO machinery.

### Important Constraints

- Use **Qwen2.5-7B**.
- Use sequence-level rewards only.
- Do not add dense/token rewards to GAD.
- Do not use answer-agreement relabelling.
- Do not use corrupted expert traces.
- Do not use external answer verifiers.
- Match R-AIRL/ReGAIL rollout count, generation settings, optimisation steps, and evaluation as closely as possible.

### GAD Outputs

Report:

- pass@1 on GSM8K;
- absolute reranked accuracy using the GAD discriminator score, if easy;
- training wall-clock / GPU-hours;
- training stability notes.

## Baseline 2: OPSD

### Goal

Implement an OPSD-style dense on-policy self-distillation baseline.

This tests whether dense on-policy supervision from expert demonstrations can match the learned reward approach, without learning an explicit reusable critic.

### Method

OPSD uses the same model in two roles:

- **Student:** sees only the task question/prompt.
- **Teacher:** sees the task question plus the expert demonstration/reference reasoning.

The student samples its own on-policy rollout. The teacher then scores the student's rollout token-by-token under the privileged demonstration-conditioned context. The student is trained to move towards the teacher distribution on its own sampled trajectories.

In notation:

```text
student distribution: pi_theta(. | x)
teacher distribution: pi_bar(. | x, c)
student rollout: y ~ pi_theta(. | x)
```

where:

```text
x = task question
c = expert CoT / reference solution
y = student-generated reasoning trace
```

The basic training signal is the on-policy reverse-KL / token distillation objective:

```text
sum_t KL[ pi_theta(. | x, y_<t) || pi_bar(. | x, c, y_<t) ]
```

For a simpler implementation, use the sampled-token form:

```text
loss = sum_t stopgrad(w_t) * log pi_theta(y_t | x, y_<t)

where w_t can be based on:
log pi_theta(y_t | x, y_<t) - log pi_bar(y_t | x, c, y_<t)
```

Equivalently, this can be implemented as maximising the implicit token reward:

```text
r_t = log pi_bar(y_t | x, c, y_<t) - log pi_theta_ref(y_t | x, y_<t)
```

If the existing RL/GRPO code is easier to reuse, aggregate `r_t` into a sequence reward or use token-level advantages. Prefer the cleanest implementation that is reliable in the current codebase.

## Required OPSD Variant

Implement the following first:

```text
OPSD-token
```

Configuration:

- Same backbone as GAD: **Qwen2.5-7B**.
- Same training prompts and expert demonstrations.
- Same train/validation/test splits.
- Same generation settings as the R-AIRL/ReGAIL policy rollouts where possible.
- Student prompt contains only the question.
- Teacher prompt contains the question and the corresponding expert demonstration.
- Teacher weights should be frozen initially.

Do not add OPSD-EMA for this rebuttal run. Use the frozen-teacher OPSD-token variant only.

## OPSD Prompt Templates

Use existing project prompt formatting if available. If not, use minimal templates.

Student:

```text
Question:
{question}

Answer with a complete reasoning process and final answer.
```

Teacher:

```text
Question:
{question}

Here is an expert reasoning trace for this problem:
{expert_cot}

Now evaluate the student's reasoning trajectory token by token by predicting the continuation that best follows the question and expert reasoning.

Student reasoning prefix:
{student_prefix}
```

Do not let the teacher simply regenerate the expert trace. The teacher must be evaluated on the **student trajectory prefixes**.

## Shared Implementation Steps

1. Locate the existing R-AIRL/ReGAIL data loader, rollout generation, and evaluation code.
2. Reuse the same task formatting, answer extraction, and accuracy evaluation.
3. Add a GAD trainer with:
   - Qwen2.5-7B student rollout generation;
   - scalar sequence discriminator;
   - pairwise expert-vs-policy discriminator loss;
   - GRPO policy update using discriminator reward.
4. Add an OPSD trainer with:
   - student rollout generation;
   - teacher-forced log-probability computation under the demonstration-conditioned teacher;
   - student log-probability computation under the normal prompt;
   - token-level distillation or implicit-reward optimisation.
5. Cache teacher log-probs for OPSD if practical, but only when the student rollout is fixed.
6. Add config entries for:
   - number of optimisation steps;
   - rollout batch size;
   - max generation length;
   - baseline type: `gad` or `opsd`;
   - OPSD teacher type: frozen or EMA;
   - KL/reward temperature if used;
   - clipping/normalisation of token rewards if used.

## Minimal Experiment

```text
Dataset: GSM8K
Backbone: Qwen2.5-7B
Baselines: GAD and OPSD-token
Compare against: SFT and best sparse R-AIRL/ReGAIL
```

Use the same evaluation protocol as Table 2.

Report:

- pass@1;
- training wall-clock / GPU-hours;
- whether training was stable;
- optional validation accuracy curve;
- optional mean token reward/log-ratio statistics.

## Optional OPSD Reranking Analysis

OPSD does not learn a reusable critic, so do **not** present it as a direct D2/D3 replacement.

If easy, add exploratory reranking using the implicit log-ratio:

```text
score(y) = sum_t [log pi_bar(y_t | x, c, y_<t) - log pi_student_ref(y_t | x, y_<t)]
```

Report this only as:

```text
OPSD implicit-score reranking
```

Do not compare it too strongly against the learned R-AIRL/ReGAIL critic, because it requires the expert demonstration at inference time and is not a standalone reward model.

## Non-goals

Do not implement:

- corrupted expert traces;
- answer-agreement relabelling;
- source-label adversarial baseline;
- D3 localisation as a main result;
- dense GAD;
- rePIRL.

Those belong to other baselines or ablations.

## Expected Rebuttal Claim

If GAD underperforms R-AIRL/ReGAIL:

```text
Black-box on-policy adversarial distillation is not sufficient to explain the gains from the proposed reward supervision.
```

If OPSD underperforms R-AIRL/ReGAIL:

```text
On-policy dense self-distillation from demonstrations is not sufficient to recover the benefits of a reusable learned reasoning reward.
```

If OPSD matches or beats R-AIRL/ReGAIL on pass@1:

```text
OPSD is a strong training baseline, but it still does not provide the same reusable reward interface for reranking and process-level localisation.
```

Either outcome is useful for the rebuttal.
