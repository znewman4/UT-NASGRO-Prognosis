"""Test 2: nominal 90 % intervals cover about 90 %, and SBC ranks are uniform (Talts et al. 2018).

Runs the toy end to end through the runner (truth drawn from the estimator's own
prior, so the estimator is well-specified). One random-time forecast per
specimen keeps the trials independent.
"""
import numpy as np
import pytest

from utprog import runner
from utprog.scoring import metrics, sbc


def toy_cfg(n):
    return dict(
        experiment="test_coverage", seed=12345, n_specimens=n, workers=1, time_grid_frac=0.001,
        truth=dict(kind="toy", toy=dict(C=1.0e-11, m=3.0, dS_MPa=100.0, Y=1.12, a_c_mm=10.0),
                   a_start=dict(median_mm=1.0, cov=0.3, trunc=4.0), log_lambda=dict(mu=0.0, sd=0.4)),
        prior=dict(log_lambda=dict(mu=0.0, sd=0.4), a0="same_as_truth"),
        sensors=dict(sd02=dict(sd_abs_mm=0.2)),
        policies=dict(periodic=dict(kind="periodic", T_frac=0.15)),
        estimators=dict(grid=dict(kind="grid", n_lambda=81, n_a0=41)),
        arms=dict(product=dict(sensor=["sd02"], policy=["periodic"], estimator=["grid"])),
        scoring=dict(n_random=1, frac_max=0.9, fixed_fractions=[], n_samples=2000, sbc_draws=99))


@pytest.mark.slow
def test_coverage_and_sbc_toy(tmp_path):
    n = 400
    cfg = toy_cfg(n)
    runner._init(cfg)
    rec = [r for i in range(n) for r in runner.run_specimen(i)[1]]
    k = sum(r["covered"] for r in rec)
    lo, hi = metrics.wilson(k, len(rec), level=0.99)
    assert lo <= 0.90 <= hi, (k / len(rec), lo, hi)
    _, p, _ = sbc.uniformity(np.array([r["sbc_rank"] for r in rec]), 99)
    assert p > 1e-3
