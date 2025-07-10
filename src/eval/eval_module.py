import math


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
