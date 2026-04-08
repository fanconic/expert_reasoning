"""Helpers for unified evaluation modes and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

MODE_GENERATE = "generate"
MODE_AIME = "aime"
MODE_PREGENERATED_POLICY = "pregenerated_policy"
MODE_PREGENERATED_POLICY_AND_REWARD = "pregenerated_policy_and_reward"

VALID_MODES = {
    MODE_GENERATE,
    MODE_AIME,
    MODE_PREGENERATED_POLICY,
    MODE_PREGENERATED_POLICY_AND_REWARD,
}

MODE_ALIASES = {
    "default": MODE_GENERATE,
    "evaluate": MODE_GENERATE,
    "generate_and_eval": MODE_GENERATE,
    "generate": MODE_GENERATE,
    "aime": MODE_AIME,
    "aime_generate": MODE_AIME,
    "generate_aime": MODE_AIME,
    "pregenerated": MODE_PREGENERATED_POLICY,
    "pregenerated_policy_only": MODE_PREGENERATED_POLICY,
    "logprobs": MODE_PREGENERATED_POLICY,
    "pregenerated_policy": MODE_PREGENERATED_POLICY,
    "pregenerated_sft": MODE_PREGENERATED_POLICY_AND_REWARD,
    "pregenerated_policy_and_reward": MODE_PREGENERATED_POLICY_AND_REWARD,
    "logprobs_sft": MODE_PREGENERATED_POLICY_AND_REWARD,
}


def canonical_eval_mode(raw_mode: Optional[str]) -> str:
    """Normalize mode aliases to canonical mode names."""
    if raw_mode is None:
        return MODE_GENERATE
    key = str(raw_mode).strip().lower()
    mode = MODE_ALIASES.get(key, key)
    if mode not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Unknown eval mode '{raw_mode}'. Valid modes: {valid}")
    return mode


def eval_mode_uses_pregenerated(mode: str) -> bool:
    """Whether this mode reads generations from an existing jsonl."""
    canonical_mode = canonical_eval_mode(mode)
    return canonical_mode in {
        MODE_PREGENERATED_POLICY,
        MODE_PREGENERATED_POLICY_AND_REWARD,
    }


def default_output_filename(mode: str, dataset_name: str) -> str:
    """Default output filename for each evaluation mode."""
    canonical_mode = canonical_eval_mode(mode)
    if canonical_mode == MODE_GENERATE:
        return f"eval_results_{dataset_name}.jsonl"
    if canonical_mode == MODE_AIME:
        return "eval_results_aime.jsonl"
    if canonical_mode == MODE_PREGENERATED_POLICY:
        return "eval_results_logprobs.jsonl"
    if canonical_mode == MODE_PREGENERATED_POLICY_AND_REWARD:
        return "eval_results_logprobs_sft.jsonl"
    raise ValueError(f"Unhandled eval mode: {canonical_mode}")


def _default_source_dir(mode: str, model_name: str, policy_name: Optional[str]) -> str:
    canonical_mode = canonical_eval_mode(mode)
    if canonical_mode == MODE_PREGENERATED_POLICY_AND_REWARD:
        return policy_name or model_name
    return model_name


def _default_candidates(mode: str) -> list[str]:
    canonical_mode = canonical_eval_mode(mode)
    if canonical_mode == MODE_PREGENERATED_POLICY:
        return ["eval_results_new.jsonl"]
    if canonical_mode == MODE_PREGENERATED_POLICY_AND_REWARD:
        return ["eval_results_new.jsonl", "debug.jsonl", "eval_results.jsonl"]
    return []


def resolve_pregenerated_jsonl_path(
    mode: str,
    model_name: str,
    policy_name: Optional[str] = None,
    explicit_path: Optional[str] = None,
    source_dir_override: Optional[str] = None,
    candidate_filenames: Optional[Iterable[str]] = None,
) -> str:
    """Resolve the input jsonl path for pregenerated modes."""
    canonical_mode = canonical_eval_mode(mode)
    if canonical_mode not in {
        MODE_PREGENERATED_POLICY,
        MODE_PREGENERATED_POLICY_AND_REWARD,
    }:
        raise ValueError(
            "resolve_pregenerated_jsonl_path only applies to pregenerated evaluation modes."
        )

    if explicit_path:
        explicit = Path(explicit_path)
        if explicit.exists():
            return str(explicit)
        raise FileNotFoundError(f"Pregenerated jsonl not found: {explicit}")

    source_dir = source_dir_override or _default_source_dir(
        canonical_mode, model_name, policy_name
    )
    candidates = (
        list(candidate_filenames)
        if candidate_filenames is not None
        else _default_candidates(canonical_mode)
    )

    attempted_paths = []
    for filename in candidates:
        path = Path(source_dir) / filename
        attempted_paths.append(str(path))
        if path.exists():
            return str(path)

    attempted = ", ".join(attempted_paths) if attempted_paths else "<none>"
    raise FileNotFoundError(
        f"Could not resolve pregenerated jsonl for mode='{canonical_mode}'. "
        f"Tried: {attempted}."
    )
