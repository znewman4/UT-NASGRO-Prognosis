"""Residual-stress channel (Approach 2): LCR layer averages of the residual stress.

Longitudinal critically refracted (LCR) waves give the stress averaged over a
near-surface layer whose depth falls with frequency (about 1 to 6 mm from 5 to
1 MHz in Javadi et al.'s weld measurements), against a stress-free reference.
For the profile sigma(z) = sigma_s (1 - z/L) e^(-z/L) of
nasgro_scenarios.residual_stress the layer average has the closed form

    (1/d) int_0^d sigma(z) dz = sigma_s e^(-d/L)         (uniform profile: sigma_s)

A survey records every layer at once: value_k = mean_k + drift + e_k, with
e_k ~ N(0, sd^2) per layer and drift ~ N(0, drift_sd^2) shared by the survey
(uncompensated temperature or reference error). Surveys happen at the listed
fractions of the nominal life (default: once, at installation), whatever the
arm's depth policy. sd and drift_sd are placeholders until the supervisors
agree them (Javadi et al. report agreement with FE within about +-40 MPa).
"""
import numpy as np

from ..rng import child
from .base import Sensor, StressObservation


def layer_means(sig, L, depths):
    """Layer averages (..., K) for surface stress sig and decay length L (inf = uniform)."""
    sig = np.asarray(sig, float)[..., None]
    L = np.asarray(L, float)[..., None]
    d = np.asarray(depths, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(np.isinf(L), 1.0, np.exp(-d / L))
    return sig * f


class LCRSensor(Sensor):
    channel = "stress"

    def __init__(self, noise, depths, times):
        self.noise, self.depths, self.times_ = noise, tuple(float(d) for d in depths), np.asarray(times, float)

    @classmethod
    def from_config(cls, d, noise, n_nominal):
        return cls(noise, [x * 1e-3 for x in d.get("layer_depths_mm", [1.0, 2.0, 4.0, 6.0])],
                   np.asarray(d.get("at_frac", [0.0]), float) * n_nominal)

    def observe(self, specimen, times, seed):
        h = specimen.life if np.isfinite(specimen.life) else np.inf
        t = self.times_[self.times_ < h]
        rng_d = np.random.default_rng(child(seed, 10))
        rng_e = np.random.default_rng(child(seed, 11))
        mean = layer_means(specimen.inputs.get("rs_sig", 0.0), specimen.inputs.get("rs_len", np.inf), self.depths)
        out = []
        for N in t:
            v = mean + self.noise.drift_sd * rng_d.standard_normal() + self.noise.sd * rng_e.standard_normal(mean.size)
            out.append(StressObservation(float(N), tuple(float(x) for x in v), self.depths))
        return out
