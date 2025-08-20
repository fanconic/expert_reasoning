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