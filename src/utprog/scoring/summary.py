"""Summary tables from per-forecast records.

RUL scores (U90, bias, coverage, CRPS) use specimens whose crack fails, as the
sensitivity study's U90 does; the arrest forecast is scored separately with the
Brier score of p_arrest against whether the true crack arrests.
"""
import numpy as np
import pandas as pd

from .metrics import wilson


def _row(g):
    fin = g[~g["arrested"].astype(bool)] if "arrested" in g else g
    n = len(fin)
    k = int(fin["covered"].astype(bool).sum())
    lo, hi = wilson(k, n)
    u = fin["u90"].to_numpy(float)
    ok = np.isfinite(u)
    row = dict(n=n, n_arrested=len(g) - n,
               u90_mean=float(u[ok].mean()) if ok.any() else np.nan,
               u90_median=float(np.median(u[ok])) if ok.any() else np.nan,
               bias_median=float(fin["log_bias"].median()), abs_bias_median=float(fin["log_bias"].abs().median()),
               coverage=k / n if n else np.nan, coverage_lo=lo, coverage_hi=hi,
               crps_mean=float(fin["crps_log"].mean()),
               last_obs_age_median=float(fin["last_obs_age"].median()))
    if "p_arrest" in g:
        row["brier_arrest"] = float(np.mean((g["p_arrest"] - g["arrested"].astype(float)) ** 2))
    return row


def summarise(records, by=("arm",)):
    rows = []
    for key, g in records.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key), **_row(g)))
    return pd.DataFrame(rows)


def summarise_depth(records, by=("arm", "sensor", "policy", "estimator", "oracle")):
    """Per arm at each scoring depth: the numbers compared with the sensitivity study."""
    d = records[records["kind"] == "depth"]
    return summarise(d, tuple(by) + ("depth_mm",))
