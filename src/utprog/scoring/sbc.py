"""Simulation-based calibration (Talts et al. 2018).

With the truth drawn from the estimator's own prior and data from its own
likelihood, the rank of the true parameter among L independent posterior draws
is uniform on {0, ..., L}. A chi-square test on binned ranks detects an
estimator that is too narrow (U shape), too wide (hump) or biased (slope).
"""
import numpy as np
from scipy import stats


def rank(draws, truth):
    return int(np.sum(np.asarray(draws) < truth))


def uniformity(ranks, n_draws, bins=20):
    """Chi-square statistic and p-value of binned ranks; (n_draws + 1) must be divisible by bins."""
    ranks = np.asarray(ranks)
    if (n_draws + 1) % bins:
        raise ValueError("choose bins dividing n_draws + 1")
    counts = np.histogram(ranks, bins=bins, range=(-0.5, n_draws + 0.5))[0]
    stat, p = stats.chisquare(counts)
    return float(stat), float(p), counts
