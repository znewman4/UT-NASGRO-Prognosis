"""Virtual specimens: draw(rng) -> Specimen with its true a(N) and failure life.

A Specimen holds the truth; only sensors and policies may look at it. The
estimators see observations only.

NasgroTruth draws every NASGRO input from the validated sampler, the starting
depth from InitialDepth, and optionally a residual-stress class and path
variability, then integrates validated NASGRO growth to failure. ToyTruth draws
(lambda, a0) from the toy's own prior, so the toy estimators are exactly
well-specified (needed for the coverage and SBC tests).

The random draws are made in a fixed order (seed, depth, class, path) whatever
the configuration, so variants of one experiment share their random numbers.
"""
from dataclasses import dataclass, field

import numpy as np

from ..forward import nasgro_adapter as adapter
from . import priors, scenarios


@dataclass
class Specimen:
    index: int
    case: str
    a0: float                   # true depth at N = 0 (m)
    life: float                 # cycles from N = 0 to failure (inf if the crack arrests)
    a_fail: float
    truth_theta: dict           # true log_lambda (lumped, diagnostic for NASGRO truth) and a0
    curve_N: np.ndarray = field(default_factory=lambda: np.empty(0))
    curve_log_a: np.ndarray = field(default_factory=lambda: np.empty(0))
    inputs: dict = field(default_factory=dict)
    label: str = ""             # e.g. the residual-stress class
    events: np.ndarray = field(default_factory=lambda: np.empty(0))
    depth_fn: object = None     # exact a(N) where available (toy)

    @property
    def arrested(self):
        return not np.isfinite(self.life)

    def depth(self, N):
        N = np.asarray(N, float)
        if self.depth_fn is not None:
            return self.depth_fn(N)
        return np.exp(np.interp(N, self.curve_N, self.curve_log_a))


class NasgroTruth:
    def __init__(self, case, a_start, nominal=None, spread=None, rs_classes=None, variability=None,
                 n_grid=1000, lambda_interval=None):
        """rs_classes: {name: (probability, scenario dict or None)}; variability: {sd, corr_log_a}."""
        self.case, self.a_start = case, a_start
        self.nominal, self.spread = dict(nominal or {}), spread or None
        self.n_grid = int(n_grid)
        rs_classes = rs_classes or {}
        self.class_names = list(rs_classes)
        self.class_sc = [sc for _, sc in rs_classes.values()]
        p = np.array([pr for pr, _ in rs_classes.values()], float)
        self.class_cum = np.cumsum(p / p.sum()) if p.size else p
        v = variability or {}
        self.var_sd, self.var_corr = float(v.get("sd", 0.0)), float(v.get("corr_log_a", 0.5))
        self.lambda_interval = lambda_interval
        self._n_ref = None
        if lambda_interval is not None:
            nom = priors.nominal_inputs(case, self.nominal)
            nom["a_i"] = np.array([lambda_interval[0]])
            self._n_ref = float(adapter.cycles_to_depth(case, nom, lambda_interval[1])[0])

    def draw(self, rng, index=0):
        seed = int(rng.integers(2 ** 31 - 1))
        a0 = float(self.a_start.sample(rng))
        u_class = rng.random()
        z_path = rng.standard_normal(self.n_grid)
        s = priors.sample_inputs(self.case, 1, seed, self.nominal, self.spread)
        s["a_i"] = np.array([a0])
        label, kfn = "", None
        if self.class_names:
            k = int(min(np.searchsorted(self.class_cum, u_class, side="right"), len(self.class_names) - 1))
            label, sc = self.class_names[k], self.class_sc[k]
            if sc is not None:
                s = scenarios.add_rs_inputs(s, sc, seed)
                kfn = scenarios.rs_k_fn(self.case, sc)
        mult = scenarios.path_variability(self.var_sd, self.var_corr, z_path) if self.var_sd > 0 else None
        tr = adapter.trajectory(self.case, s, kfn, self.n_grid, mult)
        log_lam = np.nan
        if self._n_ref is not None:
            a1, a2 = self.lambda_interval
            s_ref = dict(s, a_i=np.array([a1]))
            n_ref = float(adapter.cycles_to_depth(self.case, s_ref, a2, k_res_fn=kfn)[0])
            log_lam = float(np.log(self._n_ref / n_ref)) if np.isfinite(n_ref) else -np.inf
        cN, cla = tr.finite_curve(0)
        return Specimen(index=index, case=self.case, a0=a0, life=float(tr.life[0]), a_fail=float(tr.a_fail[0]),
                        truth_theta=dict(log_lambda=log_lam, a0=a0), curve_N=cN, curve_log_a=cla,
                        inputs={k: float(v[0]) for k, v in s.items()}, label=label)


class ToyTruth:
    def __init__(self, model, a_start, log_lambda_mu, log_lambda_sd):
        self.model, self.a_start = model, a_start
        self.mu, self.sd = float(log_lambda_mu), float(log_lambda_sd)

    def draw(self, rng, index=0):
        rng.integers(2 ** 31 - 1)                  # keeps the stream layout of NasgroTruth
        a0 = float(self.a_start.sample(rng))
        log_lam = self.mu + self.sd * float(rng.standard_normal())
        th = {"log_lambda": np.array([log_lam])}
        life = float(self.model.rul(np.array([a0]), th)[0])
        model = self.model

        def depth(N):
            return model.advance(a0, N, {"log_lambda": log_lam})
        return Specimen(index=index, case="toy", a0=a0, life=life, a_fail=model.a_c,
                        truth_theta=dict(log_lambda=log_lam, a0=a0), depth_fn=depth)
