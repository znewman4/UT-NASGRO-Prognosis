"""Summary tables from per-forecast records."""
import numpy as np
import pandas as pd

from .metrics import wilson


def summarise(records, by=("arm",)):
    rows = []
    for key, g in records.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g)
        k = int(g["covered"].sum())
        lo, hi = wilson(k, n)
        u = g["u90"].to_numpy(float)
        fin = np.isfinite(u)
        rows.append(dict(zip(by, key), n=n,
                         u90_median=float(np.median(u)), u90_mean=float(u[fin].mean()) if fin.any() else np.nan,
                         frac_u90_inf=float(1.0 - fin.mean()),
                         bias_median=float(g["log_bias"].median()),
                         abs_bias_median=float(g["log_bias"].abs().median()),
                         coverage=k / n, coverage_lo=lo, coverage_hi=hi,
                         crps_mean=float(g["crps_log"].mean()),
                         last_obs_age_median=float(g["last_obs_age"].median())))
    return pd.DataFrame(rows)
