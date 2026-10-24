"""Collect the latest run of each experiment variant for the supervisor report.

Each experiment script writes its own summary files, but a later invocation with a
different --variant overwrites them; this module reads the run directories instead,
so every number in the report traces to one run directory and its manifest.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utprog._external import ROOT
from utprog.scoring import efficacy
from utprog.scoring.metrics import wilson

RESULTS = ROOT / "results"


def latest(exp, variant):
    """Newest run directory with a manifest for (experiment, variant), or None."""
    d = RESULTS / exp / variant
    runs = sorted(p for p in d.glob("*") if (p / "manifest.json").is_file()) if d.is_dir() else []
    return runs[-1] if runs else None


def manifest(run):
    return json.loads((run / "manifest.json").read_text())


def summary(run, kind=None):
    s = pd.read_csv(run / "summary.csv")
    return s if kind is None else s[s["kind"] == kind]


def summary_depth(run):
    p = run / "summary_depth.csv"
    return pd.read_csv(p) if p.is_file() else None


def records(run):
    return pd.read_csv(run / "records.csv.gz", low_memory=False)


def decisions(run):
    p = run / "decisions.csv.gz"
    return pd.read_csv(p) if p.is_file() else None


def runs_table():
    """One row per run used: experiment, variant, n, arrested, commit, dirty, pipeline commit, date."""
    rows = []
    for exp_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        for var in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            run = latest(exp_dir.name, var.name)
            if run is None:
                continue
            m = manifest(run)
            rows.append(dict(experiment=exp_dir.name, variant=var.name, run=run.name,
                             n=m.get("n_specimens"), arrested=m.get("n_arrested"),
                             runtime_min=round(m.get("runtime_s", np.nan) / 60, 1),
                             commit=str(m.get("git", {}).get("commit", ""))[:7],
                             dirty=m.get("git", {}).get("dirty"),
                             config_hash=str(m.get("config_hash", ""))[:8]))
    return pd.DataFrame(rows)


def oracle_check(run, case, tolerance=0.03):
    return efficacy.oracle_check(summary_depth(run), case, tolerance=tolerance)


def bootstrap_reduction(rec, arm, ref, depth=None, kind="depth", n_boot=2000, seed=0):
    """1 - mean U90(arm) / mean U90(ref) with a paired bootstrap 90 % interval over specimens."""
    d = rec[(rec["kind"] == kind) & ~rec["arrested"].astype(bool)]
    if depth is not None:
        d = d[np.isclose(d["depth_mm"], depth)]
    a = d[d["arm"] == arm].set_index(["specimen", "N"])["u90"]
    b = d[d["arm"] == ref].set_index(["specimen", "N"])["u90"]
    j = a.index.intersection(b.index)
    a, b = a.loc[j].to_numpy(), b.loc[j].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5:
        return np.nan, np.nan, np.nan, len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boot = 1.0 - a[idx].mean(1) / b[idx].mean(1)
    return 1.0 - a.mean() / b.mean(), *np.quantile(boot, [0.05, 0.95]), len(a)


def coverage_ci(k, n):
    """Observed coverage with its 95 % Wilson interval."""
    lo, hi = wilson(k, n)
    return k / n if n else np.nan, lo, hi
