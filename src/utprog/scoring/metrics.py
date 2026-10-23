"""Forecast scores for one forecast against the true remaining life.

    u90          width in decades of the central 90 % forecast interval (as in the
                 sensitivity study), log10(q95) - log10(q05)
    log_bias     log10(forecast median) - log10(true RUL)
    covered      true RUL inside the central 90 % interval
    crps_log     continuous ranked probability score of log10 RUL, a proper score
                 for the whole forecast distribution (Gneiting & Raftery 2007)
    pit          probability integral transform, P(forecast < truth); uniform if calibrated
    p_alpha_lambda  forecast mass within +-alpha_rel of the true RUL (alpha-lambda accuracy)
    p_ph         forecast mass within +-alpha_abs x life of the true RUL (prognostic-horizon band)
                 (both after Saxena et al. 2010)
Forecasts may contain inf (arrest predicted); quantiles use the inverted CDF,
so no inf is interpolated, and RUL is capped at CAP cycles for the CRPS.
"""
import numpy as np
from scipy import stats

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
CAP = 1e12


def crps_sample(x, y):
    """Sample CRPS: E|X - y| - E|X - X'| / 2, with the sorted-sample identity for the second term."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / n ** 2)


def forecast_record(samples, true_rul, life, alpha_rel=0.2, alpha_abs=0.05):
    x = np.asarray(samples, float)
    q05, q25, q50, q75, q95 = np.quantile(x, QUANTILES, method="inverted_cdf")
    t = max(float(true_rul), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u90 = float(np.log10(q95) - np.log10(max(q05, 1.0)))
        log_bias = float(np.log10(max(q50, 1.0)) - np.log10(t))
        err = np.abs(x - true_rul)
    return dict(q05=q05, q25=q25, q50=q50, q75=q75, q95=q95, u90=u90, log_bias=log_bias,
                covered=bool(q05 <= true_rul <= q95),
                crps_log=crps_sample(np.log10(np.clip(x, 1.0, CAP)), np.log10(min(t, CAP))),
                pit=float(np.mean(x < true_rul) + 0.5 * np.mean(x == true_rul)),
                p_alpha_lambda=float(np.mean(err <= alpha_rel * true_rul)),
                p_ph=float(np.mean(err <= alpha_abs * life)))


def wilson(k, n, level=0.95):
    """Wilson score interval for a binomial proportion (Wilson 1927)."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(0.5 + level / 2.0)
    p = k / n
    d = 1.0 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return float(c - h), float(c + h)
