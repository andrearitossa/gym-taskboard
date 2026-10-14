"""NIST score sheets and reliability statistics (RAL-SI-2020-B / -P)."""

import numpy as np
from scipy.stats import binom

# points per item, as in the NIST benchmark score sheets
ASSEMBLY = {"inserted": 3, "seated": 2, "placed": 1}   # max 6 per part
DISASSEMBLY = {"removed": 2, "in_kit": 1}              # max 3 per part


def part_score(status, mode):
    sheet = ASSEMBLY if mode == "assembly" else DISASSEMBLY
    return sum(pts for key, pts in sheet.items() if status[key])


def max_score(n_parts, mode):
    return n_parts * sum((ASSEMBLY if mode == "assembly" else DISASSEMBLY).values())


def success_probability(m, n, confidence=0.95, tol=1e-6):
    """Success-probability bound P_S after m successes in n trials (NIST test method).

    P_S solves F(m - 1; n, P_S) = CL, with F the binomial CDF: the probability
    of success that can be claimed with confidence CL (30/30 -> 0.905).
    """
    if m == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        p = (lo + hi) / 2
        if binom.cdf(m - 1, n, p) >= confidence:
            lo = p
        else:
            hi = p
    return lo


def summarize(times, completions):
    """Mean, std and 95% CI of completion times / completion fractions over trials."""
    out = {}
    for key, x in (("time", np.asarray(times, float)), ("completion", np.asarray(completions, float))):
        mean, std = x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0
        half = 1.96 * std / np.sqrt(len(x))
        out[key] = {"mean": mean, "std": std, "ci95": (mean - half, mean + half)}
    return out
