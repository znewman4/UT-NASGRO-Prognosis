"""Test 1: the grid estimator recovers a known lambda in the closed-form case."""
import numpy as np
import pytest

from conftest import toy_prior, toy_record
from utprog.estimators.grid_bayes import GridBayes
from utprog.estimators.likelihood import NoiseModel


def _fit(toy, a0_cov, log_lam, sd=5e-5, n_obs=40, seed=1):
    rng = np.random.default_rng(seed)
    a0 = 1.1e-3 if a0_cov else 1e-3
    life = toy.rul(np.array([a0]), {"log_lambda": np.array([log_lam])})[0]
    obs = toy_record(toy, log_lam, a0, np.linspace(0, 0.8 * life, n_obs), sd, rng)
    est = GridBayes(toy, NoiseModel(sd_abs=sd), n_lambda=121, n_a0=61)
    est.reset(toy_prior(a0_cov))
    for o in obs:
        est.update(o)
    return est, life


def test_recovers_lambda_with_known_start_depth(toy):
    est, _ = _fit(toy, 0.0, 0.3)
    s = est.summary()
    assert abs(s["post_log_lambda_mean"] - 0.3) < 3 * s["post_log_lambda_sd"] + 1e-3
    assert s["post_log_lambda_sd"] < 0.05


def test_recovers_lambda_and_start_depth(toy):
    est, life = _fit(toy, 0.3, -0.25)
    s = est.summary()
    assert abs(s["post_log_lambda_mean"] + 0.25) < 3 * s["post_log_lambda_sd"] + 1e-3
    assert abs(s["post_a0_mm"] - 1.1) < 0.05
    assert est.n_regrid >= 1                       # the posterior outgrew the prior grid
    rul = est.forecast(0.8 * life, 4000, np.random.default_rng(2))
    assert np.median(rul) == pytest.approx(0.2 * life, rel=0.1)


def test_forecast_conditions_on_survival(toy):
    """With no readings, a forecast long after the nominal life keeps only growth rates slow enough
    for the component to be still in service: no forecast says it has already failed."""
    est = GridBayes(toy, NoiseModel(sd_abs=1e-4), n_lambda=121, n_a0=31)
    est.reset(toy_prior(0.3))
    nominal = toy.rul(np.array([1e-3]), {"log_lambda": np.array([0.0])})[0]
    x = est.forecast(2.0 * nominal, 4000, np.random.default_rng(3))
    assert np.mean(x > 0) > 0.99
    assert est.summary()["post_log_lambda_mean"] < 0.1     # the unconditioned posterior is still the prior
