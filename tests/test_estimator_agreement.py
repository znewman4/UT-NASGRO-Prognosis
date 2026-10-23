"""Test 4: the particle filter (IBIS) matches the grid estimator on the one-parameter problem."""
import numpy as np
import pytest

from conftest import toy_prior, toy_record
from utprog.estimators.grid_bayes import GridBayes
from utprog.estimators.likelihood import NoiseModel
from utprog.estimators.particle_filter import ParticleFilter


@pytest.mark.parametrize("sd", [1e-4, 3e-4])
def test_particle_filter_agrees_with_grid(toy, sd):
    rng = np.random.default_rng(4)
    life = toy.rul(np.array([1e-3]), {"log_lambda": np.array([0.2])})[0]
    obs = toy_record(toy, 0.2, 1e-3, np.linspace(0, 0.6 * life, 15), sd, rng)
    nz = NoiseModel(sd_abs=sd)
    grid = GridBayes(toy, nz, n_lambda=401, n_a0=3)
    pf = ParticleFilter(toy, nz, n_particles=4000, n_moves=4)
    grid.reset(toy_prior(0.0))
    pf.reset(toy_prior(0.0), rng=np.random.default_rng(5))
    for o in obs:
        grid.update(o)
        pf.update(o)
    g, p = grid.summary(), pf.summary()
    sd_post = g["post_log_lambda_sd"]
    assert abs(g["post_log_lambda_mean"] - p["post_log_lambda_mean"]) < 0.15 * sd_post
    assert p["post_log_lambda_sd"] == pytest.approx(sd_post, rel=0.15)
    N = 0.6 * life
    qg = np.quantile(grid.forecast(N, 20000, np.random.default_rng(6)), [0.05, 0.5, 0.95])
    qp = np.quantile(pf.forecast(N, 20000, np.random.default_rng(6)), [0.05, 0.5, 0.95])
    np.testing.assert_allclose(qp, qg, rtol=0.05)
