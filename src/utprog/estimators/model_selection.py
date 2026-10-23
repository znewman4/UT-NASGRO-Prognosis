"""Bayesian model comparison between hypotheses (Approach 2: residual stress).

Each hypothesis is a GridBayes with its own forward model (e.g. a residual-
stress scenario tabulated over its amplitude) and prior. All are updated on the
same record. The evidence of each gives posterior model probabilities and
Bayes factors (Kass & Raftery 1995; 2 ln BF > 6 is "strong"). The forecast is
the Bayesian model average. At a forecast time N the evidence includes the probability
that the component is still in service, which is itself informative (a peened surface
that stops a crack makes a long survival likely).
"""
import numpy as np
from scipy.special import logsumexp

from .base import Estimator


class ModelSelection(Estimator):
    def __init__(self, members, priors, prior_probs=None):
        self.members, self.priors = dict(members), dict(priors)
        names = list(self.members)
        p = np.array([(prior_probs or {}).get(k, 1.0) for k in names], float)
        self.names, self.log_prior_probs = names, np.log(p / p.sum())

    def reset(self, prior=None, rng=None):
        for k in self.names:
            self.members[k].reset(self.priors[k], rng)

    def update(self, obs):
        for k in self.names:
            self.members[k].update(obs)

    def log_evidence(self, N=None):
        """log p(record | hypothesis), and with N also log P(still in service at N | record, hypothesis)."""
        le = np.array([self.members[k].log_evidence() for k in self.names])
        if N is not None:
            with np.errstate(divide="ignore"):
                le = le + np.log([self.members[k].survival_mass(N) for k in self.names])
        return le

    def probs(self, N=None):
        lp = self.log_prior_probs + self.log_evidence(N)
        if not np.isfinite(lp).any():
            lp = self.log_prior_probs + self.log_evidence()
        return np.exp(lp - logsumexp(lp))

    def posterior_draws(self, n, rng, N=None):
        raise NotImplementedError("theta differs between hypotheses; use the members' draws")

    def forecast(self, N_now, n, rng):
        self._N = N_now
        counts = rng.multinomial(n, self.probs(N_now))
        parts = [self.members[k].forecast(N_now, c, rng) for k, c in zip(self.names, counts) if c > 0]
        return np.concatenate(parts)

    def summary(self):
        N = getattr(self, "_N", None)
        le = self.log_evidence(N)
        if not np.isfinite(le).any():
            le = self.log_evidence()
        p = self.probs(N)
        order = np.argsort(-p)
        out = {f"p_{k}": float(pk) for k, pk in zip(self.names, p)}
        out.update({f"log_evidence_{k}": float(v) for k, v in zip(self.names, le)})
        out["pred_class"] = self.names[order[0]]
        out["two_ln_bf"] = float(2.0 * (le[order[0]] - le[order[1]])) if len(self.names) > 1 else np.inf
        return out
