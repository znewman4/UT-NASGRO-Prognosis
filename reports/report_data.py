"""Every number the supervisor report quotes, computed from the run directories.

Each function returns plain data (dicts, DataFrames) and the run it came from, so the
report can cite its source. Calibration is judged on forecasts made at times chosen
independently of the truth (scoring kinds 'random' and 'fixed'); forecasts made when the
true crack reaches a given depth ('depth') match the study's definition of U90 but are a
truth-dependent stopping time, so they are used for sharpness, never for coverage.
"""
import numpy as np
import pandas as pd

import results as R
from utprog.scoring import efficacy

POLICY_RATE = {"periodic_0.4": 2.5, "periodic_0.2": 5, "periodic_0.1": 10, "periodic_0.05": 20, "continuous": 200}


def _wilson_rows(rec, by, n_boot=1000, seed=0):
    """Coverage per group with a 95 % interval from a bootstrap over specimens (a specimen's forecasts
    are correlated, so resampling forecasts would overstate the precision)."""
    rng = np.random.default_rng(seed)
    out = []
    for key, d in rec.groupby(by, observed=True):
        g = d.groupby("specimen")["covered"].agg(["sum", "size"])
        k, m = g["sum"].to_numpy(float), g["size"].to_numpy(float)
        idx = rng.integers(0, len(k), size=(n_boot, len(k)))
        boot = k[idx].sum(1) / m[idx].sum(1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), coverage=k.sum() / m.sum(),
                        lo=lo, hi=hi, n=int(m.sum()), n_specimens=len(k)))
    return pd.DataFrame(out)


def calibration(run, kinds=("random", "fixed")):
    """Coverage of nominal 90 % intervals per arm, forecasts at truth-independent times, Wilson 95 % CI."""
    rec = R.records(run)
    rec = rec[rec["kind"].isin(kinds) & ~rec["arrested"].astype(bool) & rec["covered"].notna()]
    return _wilson_rows(rec, ["arm", "sensor", "policy", "estimator"])


def coverage_by_readings(run, estimators=("grid", "is"), bins=(-1, 0, 5, 10, 30, 10 ** 6)):
    rec = R.records(run)
    rec = rec[rec["kind"].isin(["random", "fixed"]) & ~rec["arrested"].astype(bool) & rec["estimator"].isin(estimators)]
    rec = rec.assign(nb=pd.cut(rec["n_sized"], list(bins)))
    out = _wilson_rows(rec, ["estimator", "sensor", "nb"])
    return out


def toy(run):
    rec = R.records(run)
    rec = rec[rec["kind"].isin(["random", "fixed"]) & rec["covered"].notna()]
    return _wilson_rows(rec, ["arm", "sensor", "policy", "estimator"])


def oracle_check(run, case):
    sd = R.summary_depth(run)
    oc = efficacy.oracle_check(sd, case)
    rng = efficacy.predictions()["P1.1"]["seed_range"][case]
    oc["seed_lo"] = oc["depth_mm"].map(lambda d: rng[d][0] if d in rng else rng[float(d)][0])
    oc["seed_hi"] = oc["depth_mm"].map(lambda d: rng[d][1] if d in rng else rng[float(d)][1])
    oc["in_seed_range"] = (oc["oracle"] >= oc["seed_lo"] - 0.005) & (oc["oracle"] <= oc["seed_hi"] + 0.005)
    return oc


def efficacy_at(run, depth, oracle="oracle_depth"):
    sd = R.summary_depth(run)
    e = efficacy.efficacy_table(sd, oracle=oracle)
    return e[np.isclose(e["depth_mm"], depth)].set_index("arm")


def u90_depth(run):
    """Mean U90 per arm and scoring depth (study's definition), with arm parts."""
    sd = R.summary_depth(run)
    return sd


def sizing_curve(run, depth=1.0):
    """P1.2: current practice (one fresh sizing, NASGRO MC around it) against sizing sd at a study depth."""
    e = efficacy_at(run, depth)
    rows = []
    for arm, r in e.iterrows():
        parts = arm.split("|")
        if len(parts) == 3 and parts[1] == "sized_now":
            rows.append(dict(sensor=parts[0], estimator=parts[2], sd_mm=float(parts[0][2:]), u90=r["u90"],
                             realised=r["realised"], fraction=r["fraction"]))
    out = pd.DataFrame(rows).sort_values(["estimator", "sd_mm"])
    out.loc[out["sensor"] == "sd0.17", "sd_mm"] = 0.167
    return out, float(e.loc["study_baseline", "u90"]), float(e.loc["oracle_depth", "u90"])


def check_b(run):
    """U90 at the end of a sizing window (20 % of life): no reading, 1 sizing at the start, 1 fresh sizing
    at the forecast moment, 2 (start and end) and 10 sizings; grid and is."""
    rec = R.records(run)
    d = rec[(rec["kind"] == "time") & ~rec["arrested"].astype(bool)]
    t = d.groupby(["sensor", "estimator", "policy"])["u90"].agg(["mean", "count"]).reset_index()
    cov = _wilson_rows(d, ["sensor", "estimator", "policy"])
    return t.merge(cov, on=["sensor", "estimator", "policy"])


def learning_curve(run, depth, sensors=("sd0.1", "sd0.2", "sd0.5")):
    """U90 at a study depth against readings per nominal life, grid filter, plus references."""
    sd = R.summary_depth(run)
    sd = sd[np.isclose(sd["depth_mm"], depth)]
    rows = []
    for _, r in sd.iterrows():
        if r["estimator"] == "grid" and r["policy"] in POLICY_RATE and r["sensor"] in sensors:
            rows.append(dict(sensor=r["sensor"], policy=r["policy"], rate=POLICY_RATE[r["policy"]], u90=r["u90_mean"]))
    refs = {a: float(sd.loc[sd["arm"] == a, "u90_mean"].iloc[0]) for a in
            ("study_baseline", "oracle_depth", "oracle_depth_lambda", "oracle_state_lambda") if (sd["arm"] == a).any()}
    return pd.DataFrame(rows).sort_values(["sensor", "rate"]), refs
