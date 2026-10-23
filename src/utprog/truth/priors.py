"""Prior distributions of the virtual specimens and of the estimators.

NASGRO inputs come from the validated sampler (nasgro_scenarios.cases.sample)
with the study's typical spreads, in NASGRO-only mode: the crack-opening factor
kappa and the SIF-solution error are not terms of the NASGRO equation and sit
at 1 (nasgro_scenarios.runner.NON_NASGRO). Crack tilt is held at 0 until the
mixed-mode work (M3). The initial depth has its own prior (InitialDepth) so the
starting size can be swept on common random numbers.

The lumped multiplier lambda is an internal reparametrisation for the filters:
lambda = N_nominal / N_specimen over a reference depth interval, i.e. the factor
on C that makes nominal NASGRO match the specimen there. Its prior is derived
from the NASGRO input distributions (derive_lambda_prior) and results are
reported back as an effective C (C_eff = lambda C_nominal), which absorbs the
stress, closure and threshold differences that depth data cannot separate.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri

from .._external import nasgro

NON_NASGRO = ("kappa", "sif")
DEPTH_ONLY_FIXED = NON_NASGRO + ("tilt",)


def all_inputs():
    c = nasgro().cases
    return tuple(sorted(set(c.BAR_INPUTS.values()) | set(c.PLATE_INPUTS.values()) | {"material", "Cn"}))


def sample_inputs(case, n, seed, nominal=None, spread=None, fixed=()):
    """n draws of every NASGRO input (depth is set separately by the caller)."""
    fx = DEPTH_ONLY_FIXED + ("depth",) + tuple(fixed)
    return nasgro().cases.sample(case, 1e-3, n, seed=int(seed), fixed=fx, spread=spread, nominal=nominal)


def nominal_inputs(case, nominal=None):
    """One sample with every input at its nominal value."""
    return sample_inputs(case, 1, seed=0, nominal=nominal, fixed=all_inputs())


@dataclass(frozen=True)
class InitialDepth:
    """Lognormal true depth at N = 0, truncated to [median / trunc, median * trunc]; cov = 0 means known."""
    median: float          # m
    cov: float = 0.0
    trunc: float = 4.0

    @classmethod
    def from_config(cls, d):
        return cls(float(d["median_mm"]) * 1e-3, float(d.get("cov", 0.0)), float(d.get("trunc", 4.0)))

    @property
    def known(self):
        return self.cov <= 0.0

    @property
    def s(self):
        return float(np.sqrt(np.log1p(self.cov ** 2)))

    @property
    def lo(self):
        return self.median if self.known else self.median / self.trunc

    @property
    def hi(self):
        return self.median if self.known else self.median * self.trunc

    def _mass(self):
        z = np.log(self.trunc) / self.s
        return ndtr(-z), ndtr(z)

    def ppf(self, q):
        q = np.asarray(q, float)
        if self.known:
            return np.full(q.shape, self.median)
        pl, ph = self._mass()
        return self.median * np.exp(self.s * ndtri(pl + q * (ph - pl)))

    def sample(self, rng, n=None):
        """Inverse-CDF draw: one uniform per sample, so a changed median keeps the same z (CRN)."""
        return self.ppf(rng.random(n))

    def logpdf(self, a):
        a = np.asarray(a, float)
        if self.known:
            return np.where(a == self.median, 0.0, -np.inf)
        pl, ph = self._mass()
        with np.errstate(divide="ignore"):
            z = np.log(a / self.median) / self.s
            out = stats.norm.logpdf(z) - np.log(self.s * a) - np.log(ph - pl)
        return np.where((a >= self.lo) & (a <= self.hi), out, -np.inf)


def _cache_path(cache_dir, spec):
    key = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"lambda_prior_{key}.json"


def derive_lambda_prior(case, interval, n=4000, seed=1, nominal=None, spread=None, n_grid=400, cache_dir=None):
    """Prior of log lambda implied by the NASGRO input distributions over depths [a1, a2] (m).

    lambda_j = N_nominal(a1 -> a2) / N_j(a1 -> a2). Samples that arrest inside the
    interval (lambda = 0) are counted and left out of the fit. mu is the median of
    log lambda and sd half the width of its central 68 %, which is robust to the
    long tails that threshold effects give.
    """
    from ..forward import nasgro_adapter as ad
    a1, a2 = map(float, interval)
    spec = dict(case=case, interval=[a1, a2], n=n, seed=seed, nominal=nominal or {}, spread=spread or {},
                n_grid=n_grid, commit=nasgro().lock["commit"])
    path = _cache_path(cache_dir, spec) if cache_dir else None
    if path is not None and path.is_file():
        return json.loads(path.read_text())
    nom = nominal_inputs(case, nominal)
    nom["a_i"] = np.array([a1])
    n_nom = float(ad.cycles_to_depth(case, nom, a2, n_grid=n_grid)[0])
    if not np.isfinite(n_nom):
        raise ValueError(f"the nominal {case} crack arrests between {a1 * 1e3:.2f} and {a2 * 1e3:.2f} mm")
    s = sample_inputs(case, n, seed, nominal, spread)
    s["a_i"] = np.full(n, a1)
    n_s = ad.chunked(lambda x: ad.cycles_to_depth(case, x, a2, n_grid=n_grid), s)
    ok = np.isfinite(n_s) & (n_s > 0)
    ll = np.log(n_nom / n_s[ok])
    q16, q50, q84 = np.percentile(ll, [16, 50, 84])
    out = dict(mu=float(q50), sd=float(0.5 * (q84 - q16)), mean=float(ll.mean()), std=float(ll.std()),
               frac_arrested=float(1.0 - ok.mean()), n_nominal=n_nom, **spec)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
    return out
