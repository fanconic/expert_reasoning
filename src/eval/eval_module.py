import math
import numpy as np


def _pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    """
    Exact pass@k from Chen et al. (2021):
        pass@k = 1 − C(n − m, k) / C(n, k)
    where n = num_samples, m = num_correct.
    Returns 0 if k > n or no correct samples.
    """
    if num_correct == 0 or k > num_samples:
        return 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def compute_pass_at_k(all_correct_flags, ks):
    """
    Args:
        all_correct_flags: list[list[bool]]
            Outer list → each problem. Inner list → correctness of each sample.
        ks: iterable[int]
    Returns:
        dict{k: pass@k}
    """
    totals = {k: 0.0 for k in ks}
    for flags in all_correct_flags:
        n = len(flags)
        m = sum(flags)
        for k in ks:
            totals[k] += _pass_at_k(m, n, k)
    num_problems = len(all_correct_flags)
    return {k: totals[k] / num_problems for k in ks}


def _softmax_scores(scores, temperature: float = 1.0):
    """
    Convert arbitrary scores into a probability distribution with softmax.
    Non-finite scores are ignored; if all scores are non-finite, fallback is uniform.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.asarray([], dtype=float)

    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.ones(n, dtype=float) / n

    temp = max(float(temperature), 1e-8)
    shifted = scores[finite_mask] / temp
    shifted -= np.max(shifted)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)

    probs = np.zeros(n, dtype=float)
    if not np.isfinite(denom) or denom <= 0:
        probs[finite_mask] = 1.0 / finite_mask.sum()
        return probs

    probs[finite_mask] = exp_vals / denom
    return probs


def _pass_at_k_from_effective_mass(
    effective_num_correct: float, num_samples: int, k: int
) -> float:
    """
    Generalized pass@k using an effective (possibly fractional) number of correct samples.
    Reduces to exact Chen et al. pass@k when effective_num_correct is an integer.
    """
    if effective_num_correct <= 0.0 or k <= 0 or k > num_samples:
        return 0.0
    if effective_num_correct >= num_samples:
        return 1.0

    remaining = num_samples - effective_num_correct
    if remaining <= 0.0:
        return 1.0

    # Product form of C(remaining, k) / C(num_samples, k), valid for integer k.
    fail_prob = 1.0
    for j in range(k):
        fail_prob *= (remaining - j) / (num_samples - j)
    fail_prob = float(np.clip(fail_prob, 0.0, 1.0))
    return 1.0 - fail_prob


def compute_reward_weighted_pass_at_k_from_scores(
    all_correct_flags,
    all_scores,
    ks,
    temperature: float = 1.0,
):
    """
    Reward-weighted pass@k with a "uniform-reward consistency" property:
      if all rewards are equal, this equals classical pass@k.

    For each problem:
      1) Softmax reward scores to probabilities.
      2) Compute p_correct = probability mass on correct candidates.
      3) Convert to effective correct count: m_eff = n * p_correct.
      4) Use generalized pass@k with m_eff.
    """
    ks = [int(k) for k in ks]
    num_problems = len(all_correct_flags)
    if num_problems == 0:
        return {k: 0.0 for k in ks}

    totals = {k: 0.0 for k in ks}

    for flags, scores in zip(all_correct_flags, all_scores):
        flags = np.asarray(flags, dtype=bool)
        scores = np.asarray(scores, dtype=float)

        if len(flags) != len(scores):
            n = min(len(flags), len(scores))
            flags = flags[:n]
            scores = scores[:n]

        if len(flags) == 0:
            continue

        probs = _softmax_scores(scores, temperature=temperature)
        p_correct = float(np.sum(probs[flags])) if flags.any() else 0.0
        p_correct = float(np.clip(p_correct, 0.0, 1.0))
        n = len(flags)
        m_eff = n * p_correct

        for k in ks:
            if k <= 0:
                continue
            totals[k] += _pass_at_k_from_effective_mass(m_eff, n, k)

    return {k: totals[k] / num_problems for k in ks}


def compute_success_at_k_from_scores(all_correct_flags, all_scores, ks):
    """
    success@k|N with a reranker:
      For each problem i, take the top-k by 'scores_i' among its N candidates,
      and mark success if any of those k is correct according to flags_i.
    Args:
        all_correct_flags: list[list[bool]]   # per problem, length N
        all_scores:        list[list[float]]  # per problem, length N (aligned to flags)
        ks: iterable[int]
    Returns:
        dict{k: success@k|N}
    """
    num_problems = len(all_correct_flags)
    totals = {k: 0.0 for k in ks}
    for flags, scores in zip(all_correct_flags, all_scores):
        scores = np.asarray(scores, dtype=float)
        flags  = np.asarray(flags,  dtype=bool)
        N = len(flags)
        order = np.argsort(scores)[::-1]  # descending
        for k in ks:
            if k > N:
                continue
            topk = order[:k]
            totals[k] += float(flags[topk].any())
    
    return {k: totals[k] / num_problems for k in ks}


def compute_oracle_at_1_from_N(all_correct_flags):
    """
    'oracle@1|N' upper bound for top-1 selection:
    1 if any of the N is correct for a problem, else 0; averaged over problems.
    """
    vals = [1.0 if any(flags) else 0.0 for flags in all_correct_flags]
    return float(np.mean(vals))
