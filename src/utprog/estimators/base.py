"""Estimator interface and the parameter prior.

    reset(prior, rng)        start from the prior
    update(observation)      condition on one reading
    forecast(N_now, n, rng)  remaining-life samples (cycles; inf where the crack arrests)

An estimator never sees the Specimen: only Observations and its own forward
model and noise model, both built from nominal design information.

Parameters theta: log_lambda (normal prior, from derive_lambda_prior or the
toy), a0 (InitialDepth, known if its CoV is 0) and any discrete table axes
(e.g. the residual-stress amplitude of a hypothesis), with node weights.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..truth.priors import InitialDepth


@dataclass
class Prior:
    log_lambda_mu: float
    log_lambda_sd: float
    a0: InitialDepth
    axes: dict = field(default_factory=dict)          # name -> (nodes, weights)

    def free(self):
        return ("log_lambda",) if self.a0.known else ("log_lambda", "a0")

    def sample(self, rng, n):
        th = {"log_lambda": self.log_lambda_mu + self.log_lambda_sd * rng.standard_normal(n),
              "a0": np.asarray(self.a0.sample(rng, n), float)}
        for name, (nodes, w) in self.axes.items():
            w = np.asarray(w, float)
            th[name] = rng.choice(np.asarray(nodes, float), size=n, p=w / w.sum())
        return th

    def logpdf(self, theta):
        lp = stats.norm.logpdf(theta["log_lambda"], self.log_lambda_mu, self.log_lambda_sd)
        if not self.a0.known:
            lp = lp + self.a0.logpdf(theta["a0"])
        for name, (nodes, w) in self.axes.items():
            nodes, w = np.asarray(nodes, float), np.asarray(w, float) / np.sum(w)
            j = np.searchsorted(nodes, theta[name])
            ok = (j < nodes.size) & (nodes[np.minimum(j, nodes.size - 1)] == theta[name])
            lp = lp + np.where(ok, np.log(w[np.minimum(j, nodes.size - 1)]), -np.inf)
        return lp


def weighted_summary(theta, w, C_nominal):
    """Posterior summary in NASGRO terms: lambda reported as an effective C."""
    ll = theta["log_lambda"]
    m = float(w @ ll)
    sd = float(np.sqrt(max(w @ (ll - m) ** 2, 0.0)))
    out = dict(post_log_lambda_mean=m, post_log_lambda_sd=sd, post_C_eff=float(C_nominal * np.exp(m)))
    a0 = np.asarray(theta["a0"], float)
    out["post_a0_mm"] = float(w @ a0) * 1e3
    return out


class Estimator(ABC):
    forward = None

    @abstractmethod
    def reset(self, prior, rng=None):
        ...

    @abstractmethod
    def update(self, obs):
        ...

    @abstractmethod
    def posterior_draws(self, n, rng):
        """Independent draws of theta from the current posterior."""

    def forecast(self, N_now, n, rng):
        th = self.posterior_draws(n, rng)
        return self.forward.rul(self.forward.depth_at(N_now, th), th)

    def summary(self):
        return {}
