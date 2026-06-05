"""Small compatibility helpers for fast-moving Transformers/TRL releases."""

from __future__ import annotations

import os


def configure_pytorch_transformers_runtime() -> None:
    """Keep Transformers on the PyTorch path in mixed framework envs."""
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")


def ensure_transformers_cache_alias() -> None:
    """Restore the Transformers 4 cache constant expected by some TRL deps.

    Transformers 5 removed ``transformers.utils.hub.TRANSFORMERS_CACHE``.
    TRL 0.24 can still import optional callback dependencies that reference
    that name, so we provide the old alias before importing TRL GRPO modules.
    """
    configure_pytorch_transformers_runtime()

    try:
        import transformers.utils.hub as transformers_hub
    except Exception:
        return

    if hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
        return

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_path = str(HF_HUB_CACHE)
    except Exception:
        cache_path = os.path.expanduser("~/.cache/huggingface/hub")

    transformers_hub.TRANSFORMERS_CACHE = os.environ.get(
        "TRANSFORMERS_CACHE",
        cache_path,
    )


def ensure_trl_grpo_optional_symbols() -> None:
    """Populate optional TRL GRPO symbols absent when optional deps are missing."""
    ensure_transformers_cache_alias()

    try:
        import trl.trainer.grpo_trainer as grpo_trainer
    except Exception:
        return

    for name in ("GuidedDecodingParams", "LLM", "SamplingParams"):
        if not hasattr(grpo_trainer, name):
            setattr(grpo_trainer, name, None)

    if not hasattr(grpo_trainer, "wandb"):
        grpo_trainer.wandb = None
