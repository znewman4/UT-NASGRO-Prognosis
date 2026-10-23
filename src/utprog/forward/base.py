"""Forward-model interface: crack depth after a number of cycles, and cycles to failure.

theta is a dict of equal-length arrays:
    log_lambda   log of the lumped growth-rate multiplier (an exact factor on NASGRO's C)
    a0           true depth (m) at N = 0, the start of monitoring
    <axis>       any table axis of the model (a NASGRO variable such as S, or rs_amp)
Every method is vectorised over theta.

CurveModel represents growth by curves of cumulative cycles against depth at
lambda = 1 on a fixed geometric depth grid (one curve per distinct table-axis
value). A Curve knows where growth arrests (zero rate, e.g. below threshold or
in a compressive residual-stress zone) and where it fails (Kmax >= Kc or the
geometric limit), with the same conventions as nasgro_scenarios.life.integrate.
"""
from abc import ABC, abstractmethod

import numpy as np

# Rates below this (m/cycle) count as arrest. No crack of interest grows this slowly,
# and it keeps interpolated near-zero rates from producing astronomically large cycle counts.
DEAD_RATE = 1e-40


class ForwardModel(ABC):
    axes = ()
    C_nominal = np.nan          # for reporting lambda as an effective C

    @abstractmethod
    def advance(self, a, dN, theta):
        """Depth after dN cycles from depth a."""

    @abstractmethod
    def rul(self, a, theta):
        """Cycles to failure from depth a (inf if the crack arrests first)."""

    def depth_at(self, N, theta):
        """Depth at cycle N given theta (growth from theta['a0'] at N = 0)."""
        return self.advance(theta["a0"], N, theta)


class Curve:
    """Cycles at lambda = 1 against depth on a fixed grid of bin edges.

    rate, kmax: growth per cycle and peak K at the bin midpoints (geometric means
    of the edges). Bins with rate < DEAD_RATE are dead (arrest). Failure is at the
    left edge of the first bin with kmax >= Kc, else at the last edge.
    Within a bin, cycles are linear in log depth, so advance and rul are exact
    inverses of each other.
    """

    def __init__(self, edges, rate, kmax=None, Kc=np.inf):
        edges = np.asarray(edges, float)
        rate = np.asarray(rate, float)
        m = rate.size
        self.log_a = np.log(edges)
        self.dead = ~(rate >= DEAD_RATE)
        da = np.diff(edges)
        dn = np.where(self.dead, 0.0, da / np.where(self.dead, 1.0, rate))
        self.cum = np.concatenate([[0.0], np.cumsum(dn)])
        idx = np.where(self.dead, np.arange(m), m)
        self.next_dead = np.minimum.accumulate(idx[::-1])[::-1]      # first dead bin at or after each bin
        failed = np.zeros(m, bool) if kmax is None else np.asarray(kmax, float) >= Kc
        self.stop = int(failed.argmax()) if failed.any() else m
        self.a_fail = float(edges[self.stop])

    def _bin(self, a):
        return np.clip(np.searchsorted(self.log_a, np.log(a), side="right") - 1, 0, self.dead.size - 1)

    def cycles_at(self, a):
        return np.interp(np.log(a), self.log_a, self.cum)

    def advance(self, a, dN, lam):
        a = np.maximum(a, np.exp(self.log_a[0]))
        j = self._bin(a)
        s = np.minimum(self.stop, self.next_dead[j])          # edge where growth stops (arrest or failure)
        target = self.cycles_at(a) + lam * dN
        a_new = np.minimum(np.exp(np.interp(target, self.cum, self.log_a)), np.exp(self.log_a[s]))
        return np.where(a >= self.a_fail, a, np.maximum(a, a_new))

    def rul(self, a, lam):
        a = np.maximum(a, np.exp(self.log_a[0]))
        j = self._bin(a)
        blocked = self.next_dead[j] < self.stop
        r = np.maximum((self.cum[self.stop] - self.cycles_at(a)) / lam, 0.0)
        return np.where(a >= self.a_fail, 0.0, np.where(blocked, np.inf, r))


class CurveModel(ForwardModel):
    """A forward model built from Curves, one per distinct value of the table axes."""

    def curve(self, psi):
        raise NotImplementedError

    def _groups(self, theta, n):
        if not self.axes:
            return [(self.curve(()), slice(None))]
        if "_groups" in theta:                    # precomputed by the caller (GridBayes knows its grid nodes)
            return [(self.curve(psi), idx) for psi, idx in theta["_groups"]]
        if len(self.axes) == 1:
            x = np.broadcast_to(np.asarray(theta[self.axes[0]], float), (n,))
            uniq, inv = np.unique(x, return_inverse=True)
            uniq = uniq[:, None]
        else:
            psi = np.column_stack([np.broadcast_to(np.asarray(theta[k], float), (n,)) for k in self.axes])
            uniq, inv = np.unique(psi, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
        return [(self.curve(tuple(u)), order[bounds[j]:bounds[j + 1]]) for j, u in enumerate(uniq)]

    def _apply(self, method, a, other, theta):
        a, other, lam = np.broadcast_arrays(np.asarray(a, float), np.asarray(other, float),
                                            np.exp(np.asarray(theta["log_lambda"], float)))
        a, other, lam = a.ravel(), other.ravel(), lam.ravel()
        out = np.empty(a.size)
        for c, idx in self._groups(theta, a.size):
            out[idx] = method(c, a[idx], other[idx], lam[idx])
        return out

    def advance(self, a, dN, theta):
        return self._apply(lambda c, a_, n_, l_: c.advance(a_, n_, l_), a, dN, theta)

    def rul(self, a, theta):
        return self._apply(lambda c, a_, _, l_: c.rul(a_, l_), a, 0.0, theta)
