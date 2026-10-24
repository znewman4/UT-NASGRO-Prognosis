"""The sensitivity study's Monte Carlo used as a forecaster: the oracle and baseline arms.

At forecast time N, the true crack depth a(N) is read from the specimen and the
validated sampler and life integrator are run exactly as in the study:

    s = nasgro_scenarios.cases.sample(case, a(N), n, seed, fixed=NON_NASGRO + known)
    RUL samples = nasgro_scenarios.life.ca_life(case, s)

so the depth spread around a(N) is the study's (sized: sd 0.5 mm; below 1 mm:
lognormal CoV 0.6), and every other input keeps its industrial spread. Inputs in
`known` are set to the specimen's TRUE values (the study set them to nominal;
at the nominal specimen the two coincide). This arm sees the truth by design:
it is the "perfect information" reference, never a realisable estimator.

With known = () it is the study's baseline; with known = ("depth",) it is the
study's "crack depth known exactly". Scored at a fixed true depth equal to one of
the study's depths, with the same seed and n, it reproduces the stored U90 and
U90 reduction to rounding (tests/test_oracle.py). That is the definition check
the efficacy comparison rests on.

Groups (cases.sample keys): depth, F, S, D, t, W, ac, alpha, smax_sigma0, q, p,
dKth0, Kc, a0, R, cth_pos, cth_neg, tilt, material (bar: C, n, p, dKth0), Cn
(plate: C, n), and lambda, the check-A growth-rate group (plate S, Cn, alpha;
bar F, material). Residual stress (Approach 2): rs_class (the specimen's class,
with that class's amplitude and decay spread) and rs_params (its true amplitude
and decay length as well). Without rs_class the population is the prior mixture
of classes.
"""
import numpy as np
from scipy.special import ndtr

from .._external import nasgro
from ..truth import scenarios
from ..truth.priors import NON_NASGRO

MEMBERS = {"material": ("C", "n", "p", "dKth0"), "Cn": ("C", "n")}
LAMBDA = {"plate": ("S", "Cn", "alpha"), "bar": ("F", "material")}
RS_GROUPS = ("rs_class", "rs_params")


class StudyMC:
    sees_truth = True

    def __init__(self, case, known=(), n=2000, seed=1, nominal=None, spread=None, rs_classes=None,
                 fix_tilt=False):
        self.case, self.n, self.seed = case, int(n), int(seed)
        self.nominal, self.spread = nominal or None, spread or None
        groups = []
        for g in known:
            groups += list(LAMBDA[case]) if g == "lambda" else [g]
        self.groups = tuple(dict.fromkeys(groups))
        self.rs_classes = rs_classes or {}
        self.fixed = NON_NASGRO + (("tilt",) if fix_tilt else ()) + tuple(
            g for g in self.groups if g not in RS_GROUPS)
        # Without truth-valued groups the forecast depends only on the depth (and class), so it is
        # cached: the oracle at a scoring depth is computed once for all specimens.
        self._cacheable = not (set(self.groups) - {"depth", "rs_class"})
        self._cache = {}

    def reset(self, prior=None, rng=None):
        pass

    def update(self, obs):
        pass                                     # it does not use readings

    def summary(self):
        return {"oracle_known": "+".join(self.groups) or "none"}

    def population(self, a_now, inputs=None, label=""):
        """Input samples at depth a_now, with known groups set to `inputs` (the truth)."""
        ns = nasgro()
        s = ns.cases.sample(self.case, a_now, self.n, seed=self.seed, fixed=self.fixed,
                            spread=self.spread, nominal=self.nominal)
        for g in self.groups:
            if g in ("depth",) + RS_GROUPS:
                continue
            for k in MEMBERS.get(g, (g,)):
                s[k] = np.full(self.n, float(inputs[k]))
        return s

    def lives(self, s, inputs=None, label=""):
        ns = nasgro()
        if not self.rs_classes:
            return ns.life.chunked(lambda x: ns.life.ca_life(self.case, x), s)
        names = list(self.rs_classes)
        if "rs_class" in self.groups:
            idx = np.full(self.n, names.index(label))
        else:
            p = np.array([self.rs_classes[k][0] for k in names], float)
            u = ndtr(ns.cases._z("rs_class", self.n, self.seed))
            idx = np.minimum(np.searchsorted(np.cumsum(p / p.sum()), u, side="right"), len(names) - 1)
        out = np.empty(self.n)
        for k, name in enumerate(names):
            sel = np.flatnonzero(idx == k)
            if sel.size == 0:
                continue
            sub = {key: v[sel] for key, v in s.items()}
            sc = self.rs_classes[name][1]
            kfn = None
            if sc is not None:
                sub = scenarios.add_rs_inputs(sub, sc, self.seed)
                if "rs_params" in self.groups and name == label:
                    sub["rs_sig"] = np.full(sel.size, float(inputs["rs_sig"]))
                    sub["rs_len"] = np.full(sel.size, float(inputs["rs_len"]))
                kfn = scenarios.rs_k_fn(self.case, sc)
            out[sel] = ns.life.chunked(lambda x, f=kfn: ns.life.ca_life(self.case, x, f), sub, size=500)
        return out

    def forecast_truth(self, specimen, N):
        a_now = float(specimen.depth(N))
        key = (round(a_now, 12), specimen.label if "rs_class" in self.groups else None)
        if self._cacheable and key in self._cache:
            return self._cache[key]
        s = self.population(a_now, specimen.inputs, specimen.label)
        x = self.lives(s, specimen.inputs, specimen.label)
        if self._cacheable and len(self._cache) < 256:
            self._cache[key] = x
        return x


class StaticMC(StudyMC):
    """Current practice: size the crack, then run the NASGRO Monte Carlo around the reading.

    The realistic form of the study's assumption and the L0 baseline of Approaches 1 and 3.
    It uses only the latest sized reading y (taken as the current depth) and its sizing sd:
    depth samples y + sd z, reflected at 20 um as in the study, every other input at its
    industrial spread; no growth history, no survival. Without a sized reading yet, the depth
    is the study's undetected model (lognormal, CoV 0.6) around the sizing limit. It does not
    see the truth.
    """
    sees_truth = False

    def __init__(self, case, noise, n=2000, seed=1, nominal=None, spread=None, rs_classes=None, fix_tilt=False):
        super().__init__(case, (), n, seed, nominal, spread, rs_classes, fix_tilt)
        self.noise = noise
        self._cacheable = False

    def reset(self, prior=None, rng=None):
        self.last = None

    def update(self, obs):
        if obs.channel == "depth" and obs.kind == "sized":
            self.last = float(obs.value)

    def summary(self):
        return {"static_depth_mm": self.last * 1e3 if self.last is not None else np.nan}

    def forecast(self, N_now, n, rng):
        ns = nasgro()
        if self.last is None:
            a_ref = max(self.noise.a_size_min, 1e-4)
            s = ns.cases.sample(self.case, a_ref, self.n, seed=self.seed, fixed=self.fixed, undetected=True,
                                spread=self.spread, nominal=self.nominal)
        else:
            # a sized crack is detected: without undetected=False the study's case switches to its
            # undetected model (tilt sd 15 deg instead of 5) for any reading below 1 mm
            s = ns.cases.sample(self.case, self.last, self.n, seed=self.seed, fixed=self.fixed + ("depth",),
                                undetected=False, spread=self.spread, nominal=self.nominal)
            a = self.last + float(self.noise.sd(self.last)) * ns.cases._z("depth", self.n, self.seed)
            s["a_i"] = np.where(a > 2e-5, a, 2e-5 + np.abs(a - 2e-5))
        return self.lives(s)
