"""The importance-sampling estimator over full NASGRO inputs, and the stress channel."""
import numpy as np
import pytest
from scipy import stats

from utprog.estimators.likelihood import NoiseModel, StressNoise, stress_loglik
from utprog.estimators.nasgro_is import NasgroIS, Population
from utprog.forward import nasgro_adapter as ad
from utprog.sensors.base import Observation, StressObservation
from utprog.sensors.stress import layer_means
from utprog.truth import scenarios
from utprog.truth.priors import InitialDepth, StudyDepth


@pytest.fixture(scope="module")
def pop():
    return Population("plate", 600, 3, InitialDepth(1e-3, 0.3))


def test_population_depth_matches_trajectory(pop):
    i = int(np.flatnonzero(np.isfinite(pop.life))[0])
    N = 0.37 * pop.life[i]
    cN = pop.cum[i, :pop.stop[i] + 1]
    want = np.exp(np.interp(N, cN, pop.log_edges[i, :pop.stop[i] + 1]))
    assert pop.depth_at(N)[i] == pytest.approx(want, rel=1e-9)
    assert pop.depth_at(2.0 * pop.life[i])[i] == pytest.approx(pop.a_fail[i])


def test_no_readings_gives_the_prior_population(pop):
    est = NasgroIS(pop, NoiseModel(sd_abs=1e-4))
    est.reset()
    x = est.forecast(0.0, 20000, np.random.default_rng(0))
    assert np.mean(np.isinf(x)) == pytest.approx(np.mean(np.isinf(pop.life)), abs=0.02)


def test_readings_from_a_member_concentrate_on_it(pop):
    fin = np.flatnonzero(np.isfinite(pop.life))
    i = int(fin[np.argsort(pop.life[fin])[len(fin) // 2]])
    est = NasgroIS(pop, NoiseModel(sd_abs=2e-5))
    est.reset()
    for N in np.linspace(0, 0.5 * pop.life[i], 8):
        est.update(Observation(float(N), "sized", float(pop.depth_at(N)[i])))
    x = est.forecast(0.5 * pop.life[i], 4000, np.random.default_rng(1))
    assert np.median(x) == pytest.approx(0.5 * pop.life[i], rel=0.15)


def test_stress_loglik_equals_dense():
    nz = StressNoise(sd=30.0, drift_sd=20.0)
    means = np.array([[100.0, 50.0, 10.0], [0.0, 0.0, 0.0]])
    obs = StressObservation(0.0, (90.0, 60.0, -5.0), (1e-3, 2e-3, 4e-3))
    cov = 30.0 ** 2 * np.eye(3) + 20.0 ** 2
    want = [stats.multivariate_normal(m, cov).logpdf(obs.values) for m in means]
    np.testing.assert_allclose(stress_loglik(means, obs, nz), want, rtol=1e-10)


def test_layer_mean_closed_form():
    sig, L, d = 200.0, 2e-3, 3e-3
    z = np.linspace(0, d, 200001)
    num = np.trapezoid(sig * (1 - z / L) * np.exp(-z / L), z) / d
    assert layer_means(sig, L, [d])[0] == pytest.approx(num, rel=1e-6)
    assert layer_means(150.0, np.inf, [d])[0] == 150.0


def test_study_depth_prior_matches_the_study_sampler(ns):
    p = StudyDepth(1e-3, 0.5e-3)
    a = p.sample(np.random.default_rng(0), 200000)
    s = ns.cases.sample("plate", 1e-3, 200000, seed=9)["a_i"]
    assert np.mean(a) == pytest.approx(np.mean(s), rel=0.01) and np.std(a) == pytest.approx(np.std(s), rel=0.01)
    grid = np.linspace(2e-5, 3e-3, 30001)
    assert np.trapezoid(np.exp(p.logpdf(grid)), grid) == pytest.approx(1.0, abs=2e-3)
