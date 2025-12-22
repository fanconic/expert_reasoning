# Reward Model Warm-up - Quick Reference

## Quick Start (30 seconds)

```python
# 1. Import
from src.config.irl_config import IRLConfig
from src.training.airl_trainer import AIRLTrainer

# 2. Enable warm-up in config
config = IRLConfig(
    reward_warmup_steps=100,  # ← ADD THIS LINE
    # ... rest of config
)

# 3. Train (same as before)
trainer = AIRLTrainer(policy_model, reward_model, args=config)
trainer.train()
```

## Or via YAML

```yaml
reward_warmup_steps: 100  # Add this to your config file
```

## What It Does

| Phase | What's Trained | Positives | Negatives |
|-------|---|---|---|
| **Warm-up** (N steps) | Reward model only | Expert demos | Policy samples + Perturbed experts |
| **Normal** (after) | Both models | Expert demos | Policy samples + Perturbed experts |

## Key Parameters

```python
reward_warmup_steps: int = 0              # 0 = disabled, >0 = enabled for N steps
reward_updates_per_policy_step: int = 1   # After warm-up, update reward this many times per policy update
classifier_loss: str = "bce"              # "bce" or "wgan"
num_neg_perturbations_per_expert: int = 1 # Perturbations per expert (during warm-up AND normal)
```

## Expected Output

```
[Reward Warm-up] Step 1/100 - Loss: 0.6823
[Reward Warm-up] Step 2/100 - Loss: 0.6214
...
✓ Reward model warm-up completed! Starting alternating training...
```

## Recommended Values

- **Short dataset**: `reward_warmup_steps=50`
- **Medium dataset**: `reward_warmup_steps=100-200`
- **Large dataset**: `reward_warmup_steps=200-500`
- **Pre-trained reward model**: `reward_warmup_steps=20-50`

## Disable (Default)

```python
reward_warmup_steps=0  # No warm-up, normal training from start
```

## Monitoring

- **Wandb**: Track `loss/classifier` and `reward_warmup_step`
- **Console**: Watch for warm-up progress messages
- **Checkpoints**: Warm-up progress saved in trainer state

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Warm-up not running | Check `reward_warmup_steps > 0` in config |
| Training seems stuck | Warm-up is still running, monitor `reward_warmup_step` metric |
| Want to skip warm-up | Set `reward_warmup_steps=0` |
| Need longer warm-up | Increase `reward_warmup_steps` value |

## Files to Check

- **Config**: `src/config/irl_config.py` (line with `reward_warmup_steps`)
- **Implementation**: `src/training/airl_trainer.py` (`_warmup_reward_model()` method)
- **Full Guide**: `REWARD_WARMUP_GUIDE.md`

---

That's it! Set `reward_warmup_steps` and you're done. 🚀
