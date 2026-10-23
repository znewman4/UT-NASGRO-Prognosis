"""Grid posterior over (log lambda, a0[, table axes]) with adaptive refinement.

The continuous dimensions sit on a regular grid in (log lambda, log a0); table
axes (e.g. a residual-stress amplitude) sit on their table nodes, so the
forward model is evaluated exactly where it was tabulated. Prior weights are
density x cell volume and are not renormalised, so sum(prior x likelihood)
approximates the evidence p(record | model) used for Bayes factors.

Many readings make the posterior far narrower than the prior. When it becomes
narrower than `min_cells` spacings along a dimension, or puts more than
`edge_mass` on an open edge, the grid is rebuilt around it and the stored
record replayed. This is exact for this model: the likelihood is a function of
the whole record, not of a filtered state.

Forecasts at N condition on the component still being in service at N: grid points
whose crack would already have failed are dropped (the observation "not failed yet").
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

from .base import Estimator, weighted_summary
from .likelihood import replay, stress_loglik


class GridBayes(Estimator):
    def __init__(self, forward, noise, n_lambda=121, n_a0=61, width=5.0, min_cells=1.5, edge_mass=1e-3,
                 max_lambda_sd=8.0, stress_noise=None):
        if n_lambda < 3 or n_a0 < 3:
            raise ValueError("need at least 3 nodes per continuous dimension")
        self.forward, self.noise = forward, noise
        self.n_lambda, self.n_a0 = int(n_lambda), int(n_a0)
        self.width, self.min_cells, self.edge_mass = float(width), float(min_cells), float(edge_mass)
        self.max_lambda_sd = float(max_lambda_sd)
        self.stress_noise = stress_noise

    def reset(self, prior, rng=None):
        self.prior, self.history, self.n_regrid = prior, [], 0
        mu, sd = prior.log_lambda_mu, prior.log_lambda_sd
        self._bounds = {"log_lambda": (mu - self.max_lambda_sd * sd, mu + self.max_lambda_sd * sd)}
        ranges = {"log_lambda": (mu - self.width * sd, mu + self.width * sd)}
        if not prior.a0.known:
            self._bounds["a0"] = (float(np.log(prior.a0.lo)), float(np.log(prior.a0.hi)))
            ranges["a0"] = (float(np.log(prior.a0.ppf(1e-4))), float(np.log(prior.a0.ppf(1.0 - 1e-4))))
        self._build(ranges)

    def _build(self, ranges):
        p = self.prior
        dims = []                                  # (name, values, log weight, spacing in the grid variable)
        lo, hi = ranges["log_lambda"]
        x = np.linspace(lo, hi, self.n_lambda)
        h = x[1] - x[0]
        dims.append(("log_lambda", x, stats.norm.logpdf(x, p.log_lambda_mu, p.log_lambda_sd) + np.log(h), h))
        if "a0" in ranges:
            lo, hi = ranges["a0"]
            la = np.linspace(lo, hi, self.n_a0)
            h = la[1] - la[0]
            a = np.exp(la)
            dims.append(("a0", a, p.a0.logpdf(a) + la + np.log(h), h))
        else:
            dims.append(("a0", np.array([p.a0.median]), np.zeros(1), 0.0))
        for name, (nodes, w) in p.axes.items():
            w = np.asarray(w, float)
            dims.append((name, np.asarray(nodes, float), np.log(w / w.sum()), 0.0))
        idx = [i.ravel() for i in np.meshgrid(*[np.arange(d[1].size) for d in dims], indexing="ij")]
        self.theta = {d[0]: d[1][i] for d, i in zip(dims, idx)}
        self.log_prior = sum(d[2][i] for d, i in zip(dims, idx))
        self._index = {d[0]: i for d, i in zip(dims, idx)}
        self._grid = {d[0]: (d[1], d[3]) for d in dims}
        self._ranges = dict(ranges)
        self._theta_fw = dict(self.theta)
        axes = tuple(getattr(self.forward, "axes", ()))
        if axes:                                  # group grid points by table node once per grid
            key = np.ravel_multi_index([self._index[a] for a in axes], [self._grid[a][0].size for a in axes])
            order = np.argsort(key, kind="stable")
            uniq, start = np.unique(key[order], return_index=True)
            stops = np.append(start[1:], key.size)
            self._theta_fw["_groups"] = [(tuple(float(self.theta[a][order[s]]) for a in axes), order[s:e])
                                         for s, e in zip(start, stops)]
        self.acc = replay(self.noise, self.forward, self._theta_fw, self.history)
        self._life = None
        self.other_ll = np.zeros(self.log_prior.size)
        for obs in self.history:
            if obs.channel != "depth":
                self.other_ll += self._channel_ll(obs)

    def _channel_ll(self, obs):
        if obs.channel == "stress":
            if self.stress_noise is None:
                raise ValueError("a stress reading needs the estimator's stress_noise")
            return stress_loglik(self.forward.layer_means(self.theta, obs.depths), obs, self.stress_noise)
        raise ValueError(f"no likelihood for channel {obs.channel}")

    def update(self, obs):
        self.history.append(obs)
        if obs.channel == "depth":
            self.acc.update(self.forward.depth_at(obs.N, self._theta_fw), obs)
        else:
            self.other_ll = self.other_ll + self._channel_ll(obs)
        self._maybe_regrid()

    def log_joint(self):
        return self.log_prior + self.acc.loglik() + self.other_ll

    def log_evidence(self):
        return float(logsumexp(self.log_joint()))

    def weights(self):
        lp = self.log_joint()
        w = np.exp(lp - lp.max())
        return w / w.sum()

    def _maybe_regrid(self):
        for _ in range(6):
            w = self.weights()
            new, need = {}, False
            for name in self._ranges:
                vals, h = self._grid[name]
                xs = vals if name == "log_lambda" else np.log(vals)
                x = self.theta[name] if name == "log_lambda" else np.log(self.theta[name])
                m = float(w @ x)
                sd = float(np.sqrt(max(w @ (x - m) ** 2, 0.0)))
                i = self._index[name]
                blo, bhi = self._bounds[name]
                edge = 0.0
                if xs[0] > blo + 1e-9:
                    edge += float(w[i == 0].sum())
                if xs[-1] < bhi - 1e-9:
                    edge += float(w[i == xs.size - 1].sum())
                half = self.width * max(sd, h / 4.0)
                if edge > self.edge_mass:
                    need = True
                    half = max(half, 0.75 * (xs[-1] - xs[0]))
                elif sd < self.min_cells * h:
                    need = True
                new[name] = (max(m - half, blo), min(m + half, bhi))
            if not need or all(np.allclose(new[k], self._ranges[k]) for k in new):
                return
            self._build(new)
            self.n_regrid += 1

    def life(self):
        """Cycles from N = 0 to failure at every grid point (cached per grid)."""
        if self._life is None:
            self._life = self.forward.rul(self.theta["a0"], self._theta_fw)
        return self._life

    def survival_mass(self, N):
        """Posterior probability that the component is still in service at N."""
        return float(self.weights() @ (self.life() > N))

    def posterior_draws(self, n, rng, N=None):
        """Posterior draws; given N, conditioned on the component not having failed by N."""
        w = self.weights()
        if N is not None:
            alive = w * (self.life() > N)
            if alive.sum() > 0:                   # else the record contradicts survival: keep the posterior
                w = alive / alive.sum()
        idx = rng.choice(w.size, size=n, p=w)
        node = {k: v[idx] for k, v in self.theta.items()}
        th = dict(node)
        th["log_lambda"] = node["log_lambda"] + (rng.random(n) - 0.5) * self._grid["log_lambda"][1]
        th["a0"] = node["a0"] * np.exp((rng.random(n) - 0.5) * self._grid["a0"][1])
        if N is not None:                          # a jitter must not move a surviving draw past failure
            dead = self.forward.rul(th["a0"], th) <= N
            for k in ("log_lambda", "a0"):
                th[k] = np.where(dead, node[k], th[k])
        return th

    def summary(self):
        w = self.weights()
        out = weighted_summary(self.theta, w, self.forward.C_nominal)
        for name in self.prior.axes:
            out[f"post_{name}"] = float(w @ self.theta[name])
        out.update(log_evidence=self.log_evidence(), n_regrid=self.n_regrid)
        return out
