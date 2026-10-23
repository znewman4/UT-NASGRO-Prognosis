"""Likelihood of a depth record, shared by every estimator.

For the sized readings the errors are Gaussian with covariance
    Sigma = S R S + bias_sd^2 1 1^T,   S = diag(sd(a_k)),  R_kl = exp(-|N_k - N_l| / corr_cycles),
the model the sensor simulates. R is the correlation of an Ornstein-Uhlenbeck
process, whose precision is tridiagonal, so whitening is sequential:
    w_k = (x_k - rho_k x_{k-1}) / c_k,   x_k = r_k / sd_k,   c_k = sqrt(1 - rho_k^2),
and likewise u_k for the vector of ones. The per-specimen bias is integrated
out exactly with the Woodbury identity and the matrix determinant lemma:
    r^T Sigma^-1 r = sum w^2 - b^2 (sum u w)^2 / (1 + b^2 sum u^2)
    log det Sigma  = 2 sum log sd_k + 2 sum log c_k + log(1 + b^2 sum u^2).
GaussianAccumulator keeps these sums for every grid point or particle, so each
reading costs O(points) whatever the record length. tests/test_likelihood.py
checks it against the dense multivariate normal.

Misses contribute log(1 - POD(a)) and detections log POD(a). A censored reading
contributes log POD(a) + log Phi((a_size_min - a - b_mean) / sqrt(sd^2 + b^2)).
Misses and censored readings are treated as independent of the Gaussian errors
(an approximation, stated in docs/assumptions.md).
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

LOG2PI = float(np.log(2.0 * np.pi))
TINY = 1e-300


@dataclass(frozen=True)
class NoiseModel:
    sd_abs: float                 # m
    sd_rel: float = 0.0           # fraction of depth
    bias_mean: float = 0.0        # m (negative = undersizing)
    bias_sd: float = 0.0          # m, per specimen
    corr_cycles: float = 0.0      # OU correlation length; 0 = independent
    pod_a50: float = 0.0          # m; 0 = perfect detection
    pod_a90: float = 0.0          # m
    a_floor: float = 0.0          # m, never detected below
    a_size_min: float = 0.0       # m, detected but reported as censored below

    def __post_init__(self):
        if not (self.sd_abs > 0.0 or self.sd_rel > 0.0):
            raise ValueError("sizing sd must be positive")
        if self.pod_a50 > 0.0 and not self.pod_a90 > self.pod_a50:
            raise ValueError("POD needs a90 > a50")
        if not np.isfinite(self.corr_cycles) or self.corr_cycles < 0.0:
            raise ValueError("corr_cycles must be finite and >= 0 (a fully shared error is bias_sd)")

    @classmethod
    def from_config(cls, d, n_nominal=None):
        """From config units: *_mm in mm, corr_cycles in cycles or corr_frac of the nominal life."""
        mm = lambda k: float(d.get(k, 0.0)) * 1e-3  # noqa: E731
        corr = float(d.get("corr_cycles", 0.0))
        if d.get("corr_frac"):
            corr = float(d["corr_frac"]) * float(n_nominal)
        return cls(sd_abs=mm("sd_abs_mm"), sd_rel=float(d.get("sd_rel", 0.0)), bias_mean=mm("bias_mean_mm"),
                   bias_sd=mm("bias_sd_mm"), corr_cycles=corr, pod_a50=mm("pod_a50_mm"),
                   pod_a90=mm("pod_a90_mm"), a_floor=mm("a_floor_mm"), a_size_min=mm("a_size_min_mm"))

    def sd(self, a):
        return np.sqrt(self.sd_abs ** 2 + (self.sd_rel * np.asarray(a, float)) ** 2)

    def pod(self, a):
        a = np.asarray(a, float)
        if self.pod_a50 > 0.0:
            beta = np.log(self.pod_a90 / self.pod_a50) / ndtri(0.9)
            p = ndtr(np.log(np.maximum(a, TINY) / self.pod_a50) / beta)
        else:
            p = np.ones_like(a)
        return np.where(a >= self.a_floor, p, 0.0)

    def rho(self, gap):
        return 0.0 if self.corr_cycles <= 0.0 else float(np.exp(-abs(gap) / self.corr_cycles))


class GaussianAccumulator:
    FIELDS = ("sw2", "suw", "su2", "slogsd", "slogc", "other", "xprev", "vprev")

    def __init__(self, noise, n):
        self.noise = noise
        self.k = 0
        self.last_N = None
        for f in self.FIELDS:
            setattr(self, f, np.zeros(n))

    def update(self, a_pred, obs):
        nz = self.noise
        a = np.asarray(a_pred, float)
        pod = nz.pod(a)
        if obs.kind == "miss":
            self.other = self.other + np.log(np.maximum(1.0 - pod, TINY))
            return
        self.other = self.other + np.log(np.maximum(pod, TINY))
        sd = nz.sd(a)
        if obs.kind == "censored":
            tot = np.sqrt(sd ** 2 + nz.bias_sd ** 2)
            self.other = self.other + log_ndtr((nz.a_size_min - a - nz.bias_mean) / tot)
            return
        x = (obs.value - a - nz.bias_mean) / sd
        v = 1.0 / sd
        rho = 0.0 if self.last_N is None else nz.rho(obs.N - self.last_N)
        c = np.sqrt(1.0 - rho ** 2)
        w = (x - rho * self.xprev) / c
        u = (v - rho * self.vprev) / c
        self.sw2 = self.sw2 + w * w
        self.suw = self.suw + u * w
        self.su2 = self.su2 + u * u
        self.slogsd = self.slogsd + np.log(sd)
        self.slogc = self.slogc + np.log(c)
        self.xprev, self.vprev = x, v
        self.k += 1
        self.last_N = obs.N

    def loglik(self):
        b2 = self.noise.bias_sd ** 2
        quad = self.sw2 - b2 * self.suw ** 2 / (1.0 + b2 * self.su2)
        return self.other - 0.5 * (self.k * LOG2PI + 2.0 * (self.slogsd + self.slogc)
                                   + np.log1p(b2 * self.su2) + quad)

    def take(self, idx):
        out = GaussianAccumulator(self.noise, 0)
        out.k, out.last_N = self.k, self.last_N
        for f in self.FIELDS:
            setattr(out, f, np.broadcast_to(getattr(self, f), self.other.shape)[idx].copy())
        return out

    def where(self, mask, other):
        """Per point: other's state where mask, else this one's (same record, so k and last_N agree)."""
        out = self.take(slice(None))
        for f in self.FIELDS:
            setattr(out, f, np.where(mask, np.broadcast_to(getattr(other, f), mask.shape),
                                     np.broadcast_to(getattr(self, f), mask.shape)))
        return out


def replay(noise, forward, theta, history):
    """Accumulator for parameter points theta over a whole record."""
    acc = GaussianAccumulator(noise, np.size(theta["log_lambda"]))
    for obs in history:
        acc.update(forward.depth_at(obs.N, theta), obs)
    return acc
