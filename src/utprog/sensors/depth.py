"""Ultrasonic depth sizing: the measurement model.

    y(N) = a(N) + b_mean + b + sd(a) e(N)
        b      per-specimen (per-installation) bias ~ N(0, bias_sd^2)
        e(N)   standardised Ornstein-Uhlenbeck error, correlation exp(-|dN| / corr_cycles)
               (corr_cycles = 0: independent readings)
        sd(a)  = sqrt(sd_abs^2 + (sd_rel a)^2)
    detection  hit with probability POD(a) = Phi(ln(a / a50) / beta), the log-normal
               hit/miss model of MIL-HDBK-1823A, with beta from a90; never below a_floor
    sizing     a detected crack whose reading is below a_size_min is reported as
               "censored" (detected, too small to size)

Parameter values and their sources are in docs/assumptions.md. The same
NoiseModel is the estimators' likelihood (estimators/likelihood.py), so an
estimator is well-specified unless its config overrides the noise.

Bond failure (optional, `dropout`): a permanently bonded array stops reporting
at a random time T_b, lognormal with median `median_frac` x nominal life and
CoV `cov` (An et al. lost a bonded array after about 10,000 cycles). The loss is
silent: later readings simply never arrive, and the estimator is not told.

Common random numbers: the bias draw, the detection uniforms and the error
field are generated on a fixed time grid (spacing `grid` cycles, up to the
specimen's life) from three child streams of the specimen's sensor seed. Arms
reading at the same cycle therefore see the same error, and two sensors that
differ only in sd see errors that differ only in scale.
"""
import numpy as np
from scipy.signal import lfilter

from ..estimators.likelihood import NoiseModel
from ..rng import child
from .base import Observation, Sensor


def ou_field(z, rho):
    """Stationary AR(1) with unit variance from standard normals z (rho = lag-one correlation)."""
    if rho <= 0.0:
        return z
    zz = z * np.sqrt(1.0 - rho ** 2)
    zz[0] = z[0]
    return lfilter([1.0], [1.0, -rho], zz)


class DepthSensor(Sensor):
    channel = "depth"

    def __init__(self, noise: NoiseModel, grid, horizon, dropout=None, n_nominal=None):
        self.noise = noise
        self.grid = float(grid)
        self.horizon = float(horizon)       # field length for specimens that never fail
        self.dropout = None
        if dropout:
            cov = float(dropout.get("cov", 0.0))
            self.dropout = (float(dropout["median_frac"]) * float(n_nominal), np.sqrt(np.log1p(cov ** 2)))

    def failure_time(self, seed):
        if self.dropout is None:
            return np.inf
        med, s = self.dropout
        return med * float(np.exp(s * np.random.default_rng(child(seed, 3)).standard_normal()))

    def observe(self, specimen, times, seed):
        times = np.asarray(times, float)
        times = times[times < self.failure_time(seed)]
        if times.size == 0:
            return []
        nz = self.noise
        h = specimen.life if np.isfinite(specimen.life) else self.horizon
        n = int(np.ceil(max(h, times.max()) / self.grid)) + 2
        zb = np.random.default_rng(child(seed, 0)).standard_normal()
        u = np.random.default_rng(child(seed, 1)).random(n)
        z = np.random.default_rng(child(seed, 2)).standard_normal(n)
        e = ou_field(z, nz.rho(self.grid))
        j = np.clip(np.rint(times / self.grid).astype(int), 0, n - 1)
        a = specimen.depth(times)
        y = a + nz.bias_mean + nz.bias_sd * zb + nz.sd(a) * e[j]
        hit = u[j] < nz.pod(a)
        out = []
        for N, yy, h_ in zip(times, y, hit):
            if not h_:
                out.append(Observation(float(N), "miss"))
            elif yy < nz.a_size_min:
                out.append(Observation(float(N), "censored"))
            else:
                out.append(Observation(float(N), "sized", float(yy)))
        return out
