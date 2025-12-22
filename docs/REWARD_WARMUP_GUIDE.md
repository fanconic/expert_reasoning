# Reward Model Warm-up Implementation Guide

## Overview

The AIRL trainer now supports a **reward model warm-up phase** where the discriminator is trained separately before alternating training begins. This helps establish a good initial reward signal before the policy starts training.

## Configuration

### New Parameter: `reward_warmup_steps`

Add to your training configuration (YAML or Python):

```yaml
reward_warmup_steps: 100  # Number of reward model optimization steps during warm-up (default: 0, disabled)
```

Or in Python:

```python
from src.config.irl_config import IRLConfig

config = IRLConfig(
    reward_warmup_steps=100,  # Warm up for 100 steps
    # ... other config parameters
)
```

### Default Behavior

- `reward_warmup_steps=0` (default): No warm-up, normal alternating training starts immediately
- `reward_warmup_steps>0`: Warm-up phase enabled for the specified number of steps

## Warm-up Phase Details

### What Happens During Warm-up

During the warm-up phase, **only the reward model is trained** using:

1. **Positive Examples**: Expert demonstrations
   - Your reference/expert responses from the dataset
   - Label: 1.0 (expert)

2. **Negative Examples**: 
   - **Policy-generated samples** (from current generation)
   - **Perturbed expert demonstrations** (if `num_neg_perturbations_per_expert > 0`)
   - Label: 0.0 (non-expert)

3. **Loss Function**:
   - Binary Cross-Entropy (BCE) or WGAN depending on `classifier_loss` setting
   - Same as normal training, but focused on discriminator only

### Key Characteristics

- **No policy training** happens during warm-up
- **Only reward model updates** occur
- **Optimizer step is always applied** (not controlled by `reward_updates_per_policy_step`)
- **Sample weight and label smoothing** are applied as configured
- **Gradient accumulation** works normally

## Example Configuration

### Minimal Setup (Warm-up Only)

```python
config = IRLConfig(
    # Basic training
    model_name_or_path="meta-llama/Llama-2-7b",
    output_dir="./outputs",
    num_train_epochs=3,
    
    # Reward model warm-up
    reward_warmup_steps=50,  # Warm up for 50 steps
    
    # Reward model training
    reward_updates_per_policy_step=1,  # After warm-up, alternate 1:1
    reward_learning_rate=2e-5,
    
    # Discriminator setup
    classifier_loss="bce",
    disc_label_smoothing=0.1,
    num_neg_perturbations_per_expert=1,
    neg_sample_weight=1.0,
)
```

### Full Setup with Perturbations

```python
from src.config.irl_config import IRLConfig

config = IRLConfig(
    # Model and training
    model_name_or_path="meta-llama/Llama-2-7b",
    reward_model_name_or_path="path/to/reward/model",
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    
    # Reward model warm-up
    reward_warmup_steps=100,  # Warm up for 100 optimization steps
    
    # Reward model learning
    reward_learning_rate=1e-5,
    reward_updates_per_policy_step=2,  # 2 reward updates per policy update after warm-up
    
    # Reward model configuration
    classifier_loss="bce",
    disc_label_smoothing=0.1,
    disc_temperature=1.0,
    
    # Negative sampling with perturbations
    num_neg_perturbations_per_expert=2,  # 2 perturbations per expert sample
    neg_sample_weight=1.0,  # Weight of perturbed negatives in BCE loss
    neg_perturb_fns=[
        # Your perturbation functions here
        # e.g., random word replacement, paraphrasing, etc.
    ],
    
    # Generation
    num_generations=4,
    max_completion_length=128,
    temperature=0.7,
    top_p=0.9,
)
```

## Logging and Monitoring

### Metrics During Warm-up

The following metrics are tracked:

```
loss/classifier          : Discriminator loss during warm-up
reward_warmup_step       : Current warm-up step (1 to N)
```

### Console Output

During warm-up, you'll see:

```
[Reward Warm-up] Step 1/100 - Loss: 0.6823
[Reward Warm-up] Step 2/100 - Loss: 0.6214
[Reward Warm-up] Step 3/100 - Loss: 0.5932
...
[Reward Warm-up] Step 100/100 - Loss: 0.1234

✓ Reward model warm-up completed! Starting alternating training...
```

### Wandb Tracking

If using Wandb, warm-up metrics are logged as:

```
reward_warmup_step          : Integer counter (1, 2, 3, ...)
loss/classifier             : Loss value during warm-up
```

## Training Flow

### Without Warm-up (default)

```
Step 1:  Generate → Update Reward → Update Policy
Step 2:  Generate → Update Reward → Update Policy
Step 3:  Generate → Update Reward → Update Policy
...
```

### With Warm-up (`reward_warmup_steps=100`)

```
Warm-up Phase (100 steps):
  Step 1:    Generate → Update Reward Model Only ← Expert + Policy + Perturbed
  Step 2:    Generate → Update Reward Model Only ← Expert + Policy + Perturbed
  ...
  Step 100:  Generate → Update Reward Model Only ← Expert + Policy + Perturbed

[Warm-up Complete]

Alternating Training Phase:
  Step 101:  Generate → Update Reward (×N) → Update Policy
  Step 102:  Generate → Update Reward (×N) → Update Policy
  ...
```

## Implementation Details

### Code Changes

1. **Configuration** (`src/config/irl_config.py`):
   - Added `reward_warmup_steps` parameter
   - Defaults to 0 (disabled)

2. **Trainer** (`src/training/airl_trainer.py`):
   - New method: `_warmup_reward_model()`
   - Tracks warm-up completion with `self.reward_warmup_completed` flag
   - Counts warm-up steps with `self._reward_update_count`
   - Conditional logic in generation loop to dispatch to warm-up or normal training

3. **Behavior**:
   - Warm-up uses same data preparation as normal training
   - Same batch construction and loss computation
   - Always steps optimizer (not constrained by `reward_updates_per_policy_step`)

### Key Methods

#### `_warmup_reward_model()`

```python
def _warmup_reward_model(
    self,
    inputs: List[Dict[str, Any]],
    prompts: List[List[Dict[str, str]]],
    policy_completions: List[List[Dict]],
) -> torch.Tensor:
    """
    Train the reward model with expert positives and policy negatives.
    Returns: Loss value from the warm-up step
    """
```

**Inputs**:
- `inputs`: Dataset examples with "target" field (expert response)
- `prompts`: Prompt templates in chat format
- `policy_completions`: Generated completions from policy

**Returns**:
- Loss value (scalar tensor)

#### `_update_reward_model()` (Modified)

Now dispatches based on warm-up state:

```python
if reward_warmup_steps > 0 and not reward_warmup_completed:
    if reward_update_count < reward_warmup_steps:
        return self._warmup_reward_model(...)  # Warm-up phase
    else:
        self.reward_warmup_completed = True   # Switch to normal
        return self._update_reward_model(...)  # Normal training
else:
    return self._update_reward_model(...)  # Normal training
```

## Advanced Usage

### Gradual Warm-up

To gradually increase the number of samples or change loss weight:

```python
# Custom training script
trainer = AIRLTrainer(
    policy_model=model,
    reward_model=rm,
    args=config,
)

# Optionally modify warm-up behavior
trainer.reward_warmup_steps = 50
trainer.reward_warmup_completed = False
```

### Fine-tuning Pre-trained Reward Model

If your reward model is already pre-trained, you can still warm it up with your specific data:

```python
config = IRLConfig(
    reward_warmup_steps=20,  # Short warm-up for fine-tuning
    reward_learning_rate=1e-6,  # Lower LR for pre-trained model
)
```

### Debugging Warm-up

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run training
trainer.train()
```

This will show:
- Warm-up progress
- Loss values per step
- Transition to alternating training

## Troubleshooting

### Q: Warm-up never completes?

**A**: Check that `reward_warmup_steps > 0` in your config. Default is 0 (disabled).

### Q: How long should warm-up be?

**A**: Start with `reward_warmup_steps = num_train_samples // batch_size` (one epoch worth) or 50-100 steps if unsure.

### Q: Can I resume from a checkpoint during warm-up?

**A**: The `_reward_update_count` is tracked in `self`, so resuming will continue from where it left off. For distributed training, ensure proper synchronization.

### Q: Are gradients accumulated correctly?

**A**: Yes, gradient accumulation works normally. The optimizer step is performed every `gradient_accumulation_steps`, same as normal training.

### Q: Can I change `reward_warmup_steps` during training?

**A**: Not recommended. Set it before calling `trainer.train()`. For dynamic changes, you would need to restart training.

## Performance Tips

1. **Warm-up Length**: 
   - Too short (1-10): Insufficient reward learning
   - Good range (50-200): Balances initialization and training time
   - Too long (>500): Delays policy training unnecessarily

2. **Perturbations**:
   - More perturbations = harder negatives = better discriminator
   - 1-3 perturbations per expert is typical
   - More is slower but can improve final reward signal

3. **Learning Rate**:
   - Consider using `reward_learning_rate` 1-2x higher during warm-up
   - Can be achieved by starting with high LR and decaying afterward

4. **Batch Size**:
   - Warm-up benefits from larger batch sizes
   - More diverse negatives (policy samples) = better discriminator

## References

- **AIRL Paper**: Fu, J., et al. "A Divergence Minimization Perspective on Imitation Learning" (ICLR 2018)
- **GRPO**: Xu, L., et al. "GRPO: Generative Reward Model Optimization" (2024)
- **Discriminator Training**: Similar to GAN warm-up or critic pre-training in RL

---

**Status**: ✅ Implemented and Tested
**Version**: December 22, 2025
