"""The sequential likelihood equals the dense multivariate normal, censoring and POD terms are right."""
import numpy as np
import pytest
from scipy import stats

from utprog.estimators.likelihood import GaussianAccumulator, NoiseModel
from utprog.sensors.base import Observation


def dense_loglik(nz, a, y, N):
    sd = nz.sd(a)
    R = np.exp(-np.abs(N[:, None] - N[None, :]) / nz.corr_cycles) if nz.corr_cycles > 0 else np.eye(N.size)
    cov = sd[:, None] * R * sd[None, :] + nz.bias_sd ** 2
    return stats.multivariate_normal(mean=a + nz.bias_mean, cov=cov).logpdf(y)


@pytest.mark.parametrize("corr", [0.0, 3000.0])
@pytest.mark.parametrize("bias_sd", [0.0, 1e-4])
@pytest.mark.parametrize("sd_rel", [0.0, 0.1])
def test_sequential_equals_dense(corr, bias_sd, sd_rel):
    rng = np.random.default_rng(0)
    nz = NoiseModel(sd_abs=1e-4, sd_rel=sd_rel, bias_mean=-2e-5, bias_sd=bias_sd, corr_cycles=corr)
    N = np.cumsum(rng.uniform(500, 4000, 12))
    G = 5
    a_pred = 1e-3 * (1 + N[None, :] / 3e4) * rng.uniform(0.9, 1.1, (G, 1))
    y = a_pred[0] + 1e-4 * rng.standard_normal(N.size)
    acc = GaussianAccumulator(nz, G)
    for k in range(N.size):
        acc.update(a_pred[:, k], Observation(float(N[k]), "sized", float(y[k])))
    want = [dense_loglik(nz, a_pred[g], y, N) for g in range(G)]
    np.testing.assert_allclose(acc.loglik(), want, rtol=1e-10)


def test_miss_censored_and_pod_terms():
    nz = NoiseModel(sd_abs=1e-4, pod_a50=2e-4, pod_a90=4e-4, a_floor=5e-5, a_size_min=1.5e-4)
    a = np.array([1e-4, 3e-4, 1e-3])
    acc = GaussianAccumulator(nz, 3)
    acc.update(a, Observation(0.0, "miss"))
    acc.update(a, Observation(1.0, "censored"))
    pod = nz.pod(a)
    want = np.log(1 - pod) + np.log(pod) + stats.norm.logcdf((1.5e-4 - a) / 1e-4)
    np.testing.assert_allclose(acc.loglik(), want, rtol=1e-10)


def test_pod_matches_a50_a90_and_floor():
    nz = NoiseModel(sd_abs=1e-4, pod_a50=2e-4, pod_a90=4e-4, a_floor=1e-4)
    np.testing.assert_allclose(nz.pod(np.array([2e-4, 4e-4])), [0.5, 0.9], rtol=1e-10)
    assert nz.pod(np.array([0.9e-4]))[0] == 0.0


def test_take_and_where_keep_states():
    nz = NoiseModel(sd_abs=1e-4, corr_cycles=1000.0)
    acc = GaussianAccumulator(nz, 4)
    for N, y in [(0.0, 1e-3), (500.0, 1.1e-3)]:
        acc.update(np.array([1e-3, 1.05e-3, 0.95e-3, 1.1e-3]), Observation(N, "sized", y))
    t = acc.take(np.array([3, 0]))
    np.testing.assert_allclose(t.loglik(), acc.loglik()[[3, 0]])
    w = acc.where(np.array([True, False, True, False]), acc.take(np.array([1, 1, 1, 1])))
    np.testing.assert_allclose(w.loglik(), acc.loglik()[[1, 1, 1, 3]])
