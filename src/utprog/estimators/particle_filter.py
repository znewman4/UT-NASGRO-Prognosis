"""Particle filter for the static parameters: iterated batch importance sampling
(IBIS; Chopin 2002).

Particles carry theta = (log lambda[, a0]). Each reading reweights them by its
incremental likelihood; when the effective sample size falls below
ess_frac x n, they are resampled (systematic) and moved by random-walk
Metropolis-Hastings steps that target the posterior given the whole record,
with the proposal scaled to the particle covariance (2.38^2 / d; Roberts &
Rosenthal 2001). Moves are in (log lambda, log a0) with the Jacobian of the log
transform in the target. Table axes are not supported (use GridBayes for those).

Its job here is test 4: agreement with GridBayes on the one-parameter problem.
"""
import numpy as np

from .base import Estimator, weighted_summary
from .likelihood import GaussianAccumulator, replay


def systematic(w, rng):
    n = w.size
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), u), n - 1)


class ParticleFilter(Estimator):
    def __init__(self, forward, noise, n_particles=2000, ess_frac=0.5, n_moves=3):
        if forward.axes:
            raise NotImplementedError("ParticleFilter does not move on table axes; use GridBayes")
        self.forward, self.noise = forward, noise
        self.n, self.ess_frac, self.n_moves = int(n_particles), float(ess_frac), int(n_moves)

    def reset(self, prior, rng=None):
        self.prior = prior
        self.rng = rng if rng is not None else np.random.default_rng()
        self.theta = prior.sample(self.rng, self.n)
        self.logw = np.zeros(self.n)
        self.history = []
        self.acc = GaussianAccumulator(self.noise, self.n)
        self.n_resample = 0

    def _to_x(self, th):
        cols = [th["log_lambda"]] + ([np.log(th["a0"])] if not self.prior.a0.known else [])
        return np.column_stack(cols)

    def _from_x(self, x):
        th = {"log_lambda": x[:, 0].copy()}
        th["a0"] = np.exp(x[:, 1]) if not self.prior.a0.known else np.full(x.shape[0], self.prior.a0.median)
        return th

    def _log_target_prior(self, th):
        lp = self.prior.logpdf(th)
        return lp if self.prior.a0.known else lp + np.log(th["a0"])

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def ess(self):
        w = self.weights()
        return 1.0 / np.sum(w ** 2)

    def update(self, obs):
        if obs.channel != "depth":
            raise NotImplementedError("ParticleFilter uses depth readings only")
        self.history.append(obs)
        old = self.acc.loglik()
        self.acc.update(self.forward.depth_at(obs.N, self.theta), obs)
        self.logw = self.logw + (self.acc.loglik() - old)
        if self.ess() < self.ess_frac * self.n:
            self._resample_move()

    def _resample_move(self):
        idx = systematic(self.weights(), self.rng)
        self.theta = {k: v[idx] for k, v in self.theta.items()}
        self.acc = self.acc.take(idx)
        self.logw = np.zeros(self.n)
        self.n_resample += 1
        x = self._to_x(self.theta)
        d = x.shape[1]
        cov = np.atleast_2d(np.cov(x.T)) * (2.38 ** 2 / d) + 1e-12 * np.eye(d)
        L = np.linalg.cholesky(cov)
        cur = self._log_target_prior(self.theta) + self.acc.loglik()
        for _ in range(self.n_moves):
            xp = x + self.rng.standard_normal(x.shape) @ L.T
            thp = self._from_x(xp)
            accp = replay(self.noise, self.forward, thp, self.history)
            prop = self._log_target_prior(thp) + accp.loglik()
            with np.errstate(invalid="ignore"):
                ok = np.log(self.rng.random(self.n)) < prop - cur
            x = np.where(ok[:, None], xp, x)
            self.acc = self.acc.where(ok, accp)
            cur = np.where(ok, prop, cur)
        self.theta = self._from_x(x)

    def posterior_draws(self, n, rng, N=None):
        w = self.weights()
        if N is not None:
            alive = w * (self.forward.rul(self.theta["a0"], self.theta) > N)
            if alive.sum() > 0:
                w = alive / alive.sum()
        idx = rng.choice(self.n, size=n, p=w)
        return {k: v[idx] for k, v in self.theta.items()}

    def summary(self):
        out = weighted_summary(self.theta, self.weights(), self.forward.C_nominal)
        out.update(ess=float(self.ess()), n_resample=self.n_resample)
        return out
