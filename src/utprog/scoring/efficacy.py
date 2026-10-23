"""Realised efficacy: what a real sensor and estimator achieve, against what perfect knowledge gives.

Inputs are summary_depth tables (per arm and scoring depth) from runs that include a
study_baseline arm and oracle arms. At each depth:

    U90_base   mean forecast U90 of the study baseline (the sensitivity study's population)
    predicted  1 - U90_oracle / U90_base       (the oracle that knows what the approach measures)
    realised   1 - U90_arm / U90_base
    fraction   realised / predicted

Coverage and bias are carried alongside: a narrow forecast that misses the truth is not
efficacy. oracle_check compares the oracle's reduction with the frozen prediction
(docs/predictions.yaml); a mismatch beyond tolerance means a definition mismatch or a bug.
"""
import numpy as np
import pandas as pd
import yaml

from .._external import ROOT

PREDICTIONS = ROOT / "docs" / "predictions.yaml"


def predictions():
    return yaml.safe_load(PREDICTIONS.read_text())["predictions"]


def _by_depth(sd, arm):
    d = sd[sd["arm"] == arm]
    return dict(zip(d["depth_mm"], d["u90_mean"]))


def efficacy_table(sd, oracle="oracle_depth", baseline="study_baseline"):
    base, orc = _by_depth(sd, baseline), _by_depth(sd, oracle)
    rows = []
    for _, r in sd.iterrows():
        b = base.get(r["depth_mm"], np.nan)
        o = orc.get(r["depth_mm"], np.nan)
        pred = 1.0 - o / b
        real = 1.0 - r["u90_mean"] / b
        rows.append(dict(arm=r["arm"], depth_mm=r["depth_mm"], oracle=r.get("oracle", False), n=r["n"],
                         u90=r["u90_mean"], u90_baseline=b, predicted=pred, realised=real,
                         fraction=real / pred if pred > 0 else np.nan, coverage=r["coverage"],
                         coverage_lo=r["coverage_lo"], coverage_hi=r["coverage_hi"],
                         bias_median=r["bias_median"], reference_oracle=oracle))
    return pd.DataFrame(rows)


def oracle_check(sd, case, key="P1.1", oracle="oracle_depth", baseline="study_baseline", tolerance=0.03):
    """Oracle reduction at each scored depth against the frozen prediction."""
    want = predictions()[key]["values"][case]
    base, orc = _by_depth(sd, baseline), _by_depth(sd, oracle)
    rows = []
    for depth, p in want.items():
        depth = float(depth)
        if depth not in base or depth not in orc:
            continue
        got = 1.0 - orc[depth] / base[depth]
        rows.append(dict(prediction=key, case=case, depth_mm=depth, predicted=p, oracle=got,
                         u90_baseline=base[depth], difference=got - p, passes=bool(abs(got - p) <= tolerance)))
    return pd.DataFrame(rows)


def check_b_table(records):
    """Check-B replica: forecast U90 after the sizing window, 1 / 2 / 10 sizings, per sensor and estimator."""
    d = records[(records["kind"] == "time") & ~records["arrested"].astype(bool)]
    t = d.groupby(["sensor", "estimator", "policy"])["u90"].mean().unstack("policy")
    out = t.reset_index()
    if {"w1", "w10"} <= set(t.columns):
        out["reduction_10_vs_1"] = (1.0 - t["w10"] / t["w1"]).to_numpy()
    return out
