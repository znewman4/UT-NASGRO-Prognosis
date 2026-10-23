"""Growth curves a(N) from the validated pipeline, which is used unchanged.

nasgro_scenarios.life.ca_life returns only the total life. `trajectory` repeats
its arithmetic step for step (the same depth grid, stress intensity, growth
law and integrate call) but also keeps the cumulative cycles at every grid
depth, which the test bed needs for a(N). tests/test_adapter_regression.py
holds it to ca_life and to the stored results of the sensitivity study.

Units as in the pipeline: depth in m, stress in MPa, K in MPa sqrt(m).
"""
from dataclasses import dataclass

import numpy as np

from .._external import nasgro


def rate_on_grid(case, s, a, k_res_fn=None):
    """Growth per cycle and peak K at depths a (n, m) for the samples s (as in ca_life)."""
    ns = nasgro()
    smax = ns.cases.max_stress(case, s)[:, None]
    k1 = ns.cases.k_unit(case, s, a)
    kmax = k1 * smax
    kmin = kmax * s["R"][:, None]
    if k_res_fn is not None:
        kr = k_res_fn(s, a)
        kmax, kmin = kmax + kr, kmin + kr
    return ns.nasgro.growth_rate(kmax, kmin, a, ns.cases.material(s)), kmax


@dataclass
class Trajectory:
    edges: np.ndarray       # (n, m+1) depth grid per sample
    rate: np.ndarray        # (n, m) growth per cycle at bin midpoints
    kmax: np.ndarray        # (n, m)
    cum: np.ndarray         # (n, m+1) cycles from a_i to each edge (inf after an arrest)
    stop: np.ndarray        # (n,) edge index of failure
    life: np.ndarray        # (n,) cycles to failure, identical to ca_life (inf if arrested)

    @property
    def a_fail(self):
        return self.edges[np.arange(self.stop.size), self.stop]

    def finite_curve(self, i):
        """(cycles, log depth) of sample i up to failure or arrest, for interpolation."""
        cum = self.cum[i, :self.stop[i] + 1]
        fin = np.isfinite(cum)
        return cum[fin], np.log(self.edges[i, :self.stop[i] + 1][fin])


def trajectory(case, s, k_res_fn=None, n_grid=400, rate_multiplier=None):
    """Growth from s['a_i'] to failure for every sample in s.

    rate_multiplier(a) -> array broadcastable to (n, m): an optional factor on the
    growth rate along the path (truth-side variability only; None in validation).
    """
    ns = nasgro()
    edges, a, da = ns.life.depth_grid(s["a_i"], ns.cases.a_limit(case, s), n_grid)
    rate, kmax = rate_on_grid(case, s, a, k_res_fn)
    if rate_multiplier is not None:
        rate = rate * rate_multiplier(a)
    life = ns.life.integrate(rate, kmax, s["Kc"], da)
    failed = kmax >= s["Kc"][:, None]
    stop = np.where(failed.any(axis=1), failed.argmax(axis=1), rate.shape[1])
    with np.errstate(divide="ignore"):
        dn = np.where(rate > 0.0, da / np.where(rate > 0.0, rate, 1.0), np.inf)
    cum = np.concatenate([np.zeros((rate.shape[0], 1)), np.cumsum(dn, axis=1)], axis=1)
    return Trajectory(edges, rate, kmax, cum, stop, life)


def cycles_to_depth(case, s, a_to, k_res_fn=None, n_grid=400):
    """Cycles from s['a_i'] to depth a_to; the life if it fails first, inf if it arrests first."""
    tr = trajectory(case, s, k_res_fn, n_grid)
    n, m = tr.rate.shape
    la = np.log(tr.edges)
    x = np.log(np.broadcast_to(np.asarray(a_to, float), (n,)))
    j = np.clip((la <= x[:, None]).sum(axis=1) - 1, 0, m - 1)
    r = np.arange(n)
    lo, hi = tr.cum[r, j], tr.cum[r, j + 1]
    f = (x - la[r, j]) / (la[r, j + 1] - la[r, j])
    with np.errstate(invalid="ignore"):
        val = np.where(f > 0.0, lo + f * (hi - lo), lo)
    val = np.maximum(val, 0.0)
    return np.where(np.exp(x) >= tr.a_fail, tr.life, val)


def chunked(fn, s, size=1000):
    """fn over sample chunks (bounds memory), as nasgro_scenarios.life.chunked."""
    return nasgro().life.chunked(fn, s, size)
