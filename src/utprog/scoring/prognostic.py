"""Prognostic metrics over a specimen's sequence of forecasts (Saxena et al. 2010).

Both use the forecasts at fixed life fractions (records with kind == "fixed").

alpha_lambda       at each fraction t_lambda of life, a forecast passes if at least
                   beta of its mass lies within +-alpha of the true RUL
                   (p_alpha_lambda >= beta); reported as the pass rate per arm and fraction.
prognostic_horizon the time from the first forecast after which every forecast keeps at
                   least beta of its mass inside the +-alpha x life band around the true
                   RUL, to failure; reported as a fraction of life (0 if never).
"""
import numpy as np
import pandas as pd


def alpha_lambda(records, beta=0.5, by=("arm",)):
    d = records[records["kind"] == "fixed"].copy()
    d["frac_fixed"] = d["frac"].round(3)
    d["pass"] = d["p_alpha_lambda"] >= beta
    return d.groupby(list(by) + ["frac_fixed"])["pass"].mean().rename("alpha_lambda_pass").reset_index()


def prognostic_horizon(records, beta=0.5, by=("arm",)):
    d = records[records["kind"] == "fixed"]
    rows = []
    for key, g in d.groupby(list(by) + ["specimen"], sort=False):
        g = g.sort_values("N")
        ok = (g["p_ph"] >= beta).to_numpy()
        if not ok[-1]:
            ph = 0.0
        else:
            bad = np.flatnonzero(~ok)
            first = 0 if bad.size == 0 else bad[-1] + 1
            ph = float(1.0 - g["frac"].iloc[first])
        rows.append(dict(zip(list(by) + ["specimen"], key), ph_frac=ph))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.groupby(list(by))["ph_frac"].agg(["mean", "median"]).add_prefix("ph_frac_").reset_index()
