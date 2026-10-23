"""Bayesian model comparison between hypotheses (Approach 2: residual stress).

Each hypothesis is a GridBayes with its own forward model (e.g. a residual-
stress scenario tabulated over its amplitude) and prior. All are updated on the
same record. The evidence of each gives posterior model probabilities and
Bayes factors (Kass & Raftery 1995; 2 ln BF > 6 is "strong"). The forecast is
the Bayesian model average.
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

    def log_evidence(self):
        return np.array([self.members[k].log_evidence() for k in self.names])

    def probs(self):
        lp = self.log_prior_probs + self.log_evidence()
        return np.exp(lp - logsumexp(lp))

    def posterior_draws(self, n, rng):
        raise NotImplementedError("theta differs between hypotheses; use the members' draws")

    def forecast(self, N_now, n, rng):
        counts = rng.multinomial(n, self.probs())
        parts = [self.members[k].forecast(N_now, c, rng) for k, c in zip(self.names, counts) if c > 0]
        return np.concatenate(parts)

    def summary(self):
        le = self.log_evidence()
        p = self.probs()
        order = np.argsort(-p)
        out = {f"p_{k}": float(pk) for k, pk in zip(self.names, p)}
        out.update({f"log_evidence_{k}": float(v) for k, v in zip(self.names, le)})
        out["pred_class"] = self.names[order[0]]
        out["two_ln_bf"] = float(2.0 * (le[order[0]] - le[order[1]])) if len(self.names) > 1 else np.inf
        return out
