"""Importance sampling over full NASGRO input vectors: the well-specified estimator.

A population of n specimens is drawn once from the same priors as the truth
(every NASGRO input from the validated sampler, the start depth from the depth
prior, and the residual-stress class mixture if the experiment has one) and
grown to failure with the validated pipeline. Each particle therefore carries
its own complete a(N) curve, with no lumping into lambda: threshold, Kc, closure
and residual stress all act as they do in the truth.

Readings reweight the particles through the same likelihood as the grid
estimator (depth: estimators/likelihood.py; stress: layer averages). A forecast
at N keeps only particles that have not failed by N (the component is still in
service) and returns their remaining lives, inf where the particle arrests.

With a NASGRO-only truth this estimator is correct by construction, like check B
("the true crack was drawn from the same model"), so it measures the information
in the readings without model-form error; the grid estimator adds the error of
lumping into lambda. Many precise readings make the weights degenerate: the
effective sample size is recorded with every forecast, and arms where it falls
below about 100 should be read with care (use GridBayes for continuous arms).
"""
import numpy as np
from scipy.special import logsumexp

from ..forward import nasgro_adapter as adapter
from ..sensors.stress import layer_means
from ..truth import priors, scenarios
from .likelihood import GaussianAccumulator, stress_loglik


class Population:
    def __init__(self, case, n, seed, a0_prior, nominal=None, spread=None, rs_classes=None, n_grid=400):
        rng = np.random.default_rng([int(seed), 7])
        s = priors.sample_inputs(case, n, seed, nominal, spread)
        s["a_i"] = np.asarray(a0_prior.sample(rng, n), float)
        names = list(rs_classes or {})
        self.labels = np.array([""] * n, dtype=object)
        self.rs_sig, self.rs_len = np.zeros(n), np.full(n, np.inf)
        m = int(n_grid)
        self.edges, self.cum = np.empty((n, m + 1)), np.empty((n, m + 1))
        self.stop, self.life = np.empty(n, int), np.empty(n)
        if names:
            p = np.array([rs_classes[k][0] for k in names], float)
            idx = np.minimum(np.searchsorted(np.cumsum(p / p.sum()), rng.random(n), side="right"), len(names) - 1)
        else:
            idx = np.zeros(n, int)
        for k in np.unique(idx):
            sel = np.flatnonzero(idx == k)
            sub = {key: v[sel] for key, v in s.items()}
            kfn = None
            if names:
                self.labels[sel] = names[k]
                sc = rs_classes[names[k]][1]
                if sc is not None:
                    sub = scenarios.add_rs_inputs(sub, sc, seed)
                    kfn = scenarios.rs_k_fn(case, sc)
                    self.rs_sig[sel], self.rs_len[sel] = sub["rs_sig"], sub["rs_len"]
            tr = adapter.trajectory(case, sub, kfn, m)
            self.edges[sel], self.cum[sel], self.stop[sel], self.life[sel] = tr.edges, tr.cum, tr.stop, tr.life
        self.log_edges = np.log(self.edges)
        self.a_fail = self.edges[np.arange(n), self.stop]
        self.class_names = names
        self.class_prior = {k: rs_classes[k][0] / sum(v[0] for v in rs_classes.values()) for k in names}

    @property
    def n(self):
        return self.life.size

    def depth_at(self, N):
        """True depth of every particle at cycle N (failure depth once failed; arrest depth if stopped)."""
        n, m1 = self.cum.shape
        r = np.arange(n)
        j = np.clip((self.cum <= N).sum(axis=1) - 1, 0, m1 - 2)
        lo, hi = self.cum[r, j], self.cum[r, j + 1]
        with np.errstate(invalid="ignore"):
            f = np.where(np.isfinite(hi), np.clip((N - lo) / (hi - lo), 0.0, 1.0), 0.0)
        la = self.log_edges[r, j] + f * (self.log_edges[r, j + 1] - self.log_edges[r, j])
        return np.where(N >= self.life, self.a_fail, np.exp(la))


class NasgroIS:
    def __init__(self, population, noise, stress_noise=None):
        self.pop, self.noise, self.stress_noise = population, noise, stress_noise

    def reset(self, prior=None, rng=None):
        self.acc = GaussianAccumulator(self.noise, self.pop.n)
        self.other = np.zeros(self.pop.n)
        self.N_last = 0.0

    def update(self, obs):
        self.N_last = max(self.N_last, obs.N)
        if obs.channel == "depth":
            self.acc.update(self.pop.depth_at(obs.N), obs)
        elif obs.channel == "stress":
            means = layer_means(self.pop.rs_sig, self.pop.rs_len, obs.depths)
            self.other = self.other + stress_loglik(means, obs, self.stress_noise)
        else:
            raise ValueError(f"no likelihood for channel {obs.channel}")

    def log_weights(self, N=None):
        lw = self.acc.loglik() + self.other
        if N is not None:
            lw = np.where(self.pop.life > N, lw, -np.inf)          # still in service at N
        return lw

    def weights(self, N=None):
        lw = self.log_weights(N)
        if not np.isfinite(lw).any():
            lw = np.zeros_like(lw)
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def forecast(self, N_now, n, rng):
        self.N_last = max(self.N_last, N_now)
        idx = rng.choice(self.pop.n, size=n, p=self.weights(N_now))
        return self.pop.life[idx] - N_now

    def summary(self, N=None):
        w = self.weights(N if N is not None else self.N_last)
        out = dict(ess=float(1.0 / np.sum(w ** 2)))
        if self.pop.class_names:
            p = {k: float(w[self.pop.labels == k].sum()) for k in self.pop.class_names}
            order = sorted(p, key=lambda k: -p[k])
            out.update({f"p_{k}": v for k, v in p.items()})
            out["pred_class"] = order[0]
            lw = self.log_weights()
            ev = {k: logsumexp(lw[self.pop.labels == k]) - np.log(max((self.pop.labels == k).sum(), 1))
                  for k in self.pop.class_names}
            out["two_ln_bf"] = float(2.0 * (ev[order[0]] - ev[order[1]])) if len(order) > 1 else np.inf
        return out
