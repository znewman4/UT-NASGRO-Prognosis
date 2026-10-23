"""Scores behave as defined on distributions with known answers."""
import numpy as np
import pandas as pd
import pytest

from utprog.scoring import cost, metrics, prognostic, sbc


def test_crps_point_mass_and_normal():
    assert metrics.crps_sample(np.full(100, 2.0), 3.5) == pytest.approx(1.5)
    x = np.random.default_rng(0).standard_normal(200000)
    # closed form for N(0, 1) at y = 0: 2 phi(0) - 1/sqrt(pi)
    assert metrics.crps_sample(x, 0.0) == pytest.approx(2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi), abs=2e-3)


def test_forecast_record_u90_and_coverage():
    rng = np.random.default_rng(1)
    s = 0.2
    x = 1e5 * np.exp(s * np.log(10) * rng.standard_normal(100000))
    r = metrics.forecast_record(x, 1e5, 3e5)
    assert r["u90"] == pytest.approx(2 * 1.6449 * s, rel=0.02)
    assert r["covered"] and abs(r["log_bias"]) < 0.01 and r["pit"] == pytest.approx(0.5, abs=0.01)
    assert not metrics.forecast_record(x, 1e7, 3e7)["covered"]


def test_forecast_record_handles_arrest_samples():
    x = np.concatenate([np.full(900, 1e5), np.full(100, np.inf)])
    r = metrics.forecast_record(x, 1e5, 2e5)
    assert np.isinf(r["q95"]) and np.isinf(r["u90"]) and np.isfinite(r["crps_log"])


def test_wilson_contains_proportion():
    lo, hi = metrics.wilson(90, 100)
    assert lo < 0.9 < hi and 0.8 < lo and hi < 0.96


def test_sbc_uniform_ranks_pass():
    ranks = np.random.default_rng(2).integers(0, 100, 5000)
    assert sbc.uniformity(ranks, 99)[1] > 1e-3
    assert sbc.uniformity(np.full(500, 50), 99)[1] < 1e-6


def test_prognostic_horizon():
    rows = [dict(arm="a", specimen=0, kind="fixed", N=f * 100, frac=f, p_ph=p, p_alpha_lambda=p)
            for f, p in [(0.1, 0.9), (0.3, 0.2), (0.5, 0.8), (0.7, 0.9)]]
    ph = prognostic.prognostic_horizon(pd.DataFrame(rows), beta=0.5)
    assert ph["ph_frac_mean"].iloc[0] == pytest.approx(0.5)
    al = prognostic.alpha_lambda(pd.DataFrame(rows), beta=0.5)
    assert al["alpha_lambda_pass"].tolist() == [True, False, True, True]


def test_break_even():
    d = lambda **k: pd.DataFrame([dict(n_readings=5, service_frac=1.0, acted=1, failed=0, unused_frac=0.1) | k])  # noqa: E731
    per, cont = d(), d(n_readings=200, unused_frac=0.02)
    c_per = cost.Costs(per_reading=0.1, repair=1.0, failure=50.0, unused_per_life=2.0)
    c_cont = cost.Costs(per_reading=0.0, repair=1.0, failure=50.0, unused_per_life=2.0)
    cstar = cost.break_even_per_life(cont, per, c_cont, c_per)
    assert cstar == pytest.approx(0.5 + 0.2 - 0.04)
    assert cost.break_even_sd([0.1, 0.2, 0.3], [1.0, 0.5, 0.1], 0.3) == pytest.approx(0.25)
