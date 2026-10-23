"""Precomputed NASGRO growth curves for the estimators.

A filter with thousands of grid points or particles over thousands of specimens
cannot call full NASGRO at every step. NasgroTable evaluates the validated growth
law once, at nominal inputs, on a fixed geometric depth grid for every node of
the table axes (NASGRO variables such as the stress S, or the residual-stress
amplitude rs_amp), and stores log growth rate and Kmax per depth.

    lambda   exact: the rate is proportional to C, so cycles scale as 1/lambda
             (C itself is therefore never a table axis)
    axes     multilinear interpolation of log rate and Kmax between nodes;
             exact at the nodes, which is where the grid estimator evaluates it

Interpolation error between nodes is what test 5 (tests/test_surrogate.py)
guards. Tables are cached on disk keyed by their full specification and the
pinned pipeline commit.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .._external import nasgro
from ..truth import priors, scenarios
from . import nasgro_adapter as adapter
from .base import Curve, CurveModel


class NasgroTable(CurveModel):
    def __init__(self, case, a_min, n_grid=600, nominal=None, axes=None, rs=None, cache_dir=None):
        axes = dict(axes or {})
        if "C" in axes:
            raise ValueError("C is carried by the lumped multiplier lambda, not by a table axis")
        for k, v in axes.items():
            if len(v) < 2:
                raise ValueError(f"table axis {k} needs at least two nodes")
        self.case, self.nominal, self.rs = case, dict(nominal or {}), rs
        self.axes = tuple(axes)
        self.nodes = [np.asarray(sorted(float(x) for x in v)) for v in axes.values()]
        nom = priors.nominal_inputs(case, self.nominal)
        self._nom = nom
        self.Kc = float(nom["Kc"][0])
        self.C_nominal = float(nom["C"][0])
        a_lim = float(nasgro().cases.a_limit(case, nom)[0])
        if not a_min < a_lim:
            raise ValueError("a_min must be below the geometric limit")
        self.edges = a_min * (a_lim / a_min) ** np.linspace(0.0, 1.0, n_grid + 1)
        spec = dict(case=case, a_min=a_min, n_grid=n_grid, nominal=self.nominal,
                    axes={k: v.tolist() for k, v in zip(self.axes, self.nodes)}, rs=rs,
                    commit=nasgro().lock["commit"])
        self.key = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]
        self._values = self._load_or_build(cache_dir, nom)
        self._rgi = RegularGridInterpolator(tuple(self.nodes), self._values) if self.axes else None
        self._curves = {}

    def _load_or_build(self, cache_dir, nom):
        path = Path(cache_dir) / f"table_{self.key}.npz" if cache_dir else None
        if path is not None and path.is_file():
            z = np.load(path)
            if np.array_equal(z["edges"], self.edges):
                return z["values"]
        values = self._build(nom)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, edges=self.edges, values=values)
        return values

    def _build(self, nom):
        combos = list(itertools.product(*self.nodes)) or [()]
        n = len(combos)
        cols = np.array(combos, float).reshape(n, len(self.axes))
        s = {k: np.repeat(v, n) for k, v in nom.items()}
        s["a_i"] = np.full(n, self.edges[0])
        amp = None
        for j, name in enumerate(self.axes):
            if name == "rs_amp":
                amp = cols[:, j]
            elif name in s:
                s[name] = cols[:, j].copy()
            else:
                raise ValueError(f"unknown table axis {name}")
        kfn = None
        if self.rs is not None:
            if amp is None:
                amp = np.full(n, 0.5 * sum(self.rs["A"]))
            s = scenarios.rs_inputs_at(s, self.rs, amp)
            kfn = scenarios.rs_k_fn(self.case, self.rs)
        elif amp is not None:
            raise ValueError("an rs_amp axis needs a residual-stress scenario")
        a_mid = np.sqrt(self.edges[1:] * self.edges[:-1])
        rate, kmax = adapter.rate_on_grid(self.case, s, np.tile(a_mid, (n, 1)), kfn)
        values = np.concatenate([np.log(np.maximum(rate, 1e-300)), kmax], axis=1)
        return values.reshape(tuple(len(v) for v in self.nodes) + (2 * a_mid.size,))

    def layer_means(self, theta, depths):
        """LCR layer averages (G, K) of this hypothesis's residual stress (MPa); 0 without one.

        Uses the same surface stress as the table (amplitude x sigma_y, first-cycle shakedown
        at nominal load if the scenario has it) and the median decay length."""
        from ..sensors.stress import layer_means
        n = np.size(theta["log_lambda"])
        if self.rs is None:
            return np.zeros((n, len(depths)))
        lo, hi = self.rs["A"]
        amp = np.broadcast_to(np.asarray(theta.get("rs_amp", 0.5 * (lo + hi)), float), (n,))
        sy = float(self._nom["sigma_y"][0])
        sig = amp * sy
        if self.rs["shakedown"]:
            smax = float(nasgro().cases.max_stress(self.case, self._nom)[0])
            smin = smax * float(self._nom["R"][0])
            sig = np.clip(sig, -sy - smin, sy - smax)
        L = np.inf if self.rs["L"] is None else float(self.rs["L"][0])
        return layer_means(sig, np.full(n, L), depths)

    def curve(self, psi=()):
        key = tuple(float(x) for x in psi)
        c = self._curves.get(key)
        if c is None:
            if self.axes:
                x = np.array([np.clip(v, nd[0], nd[-1]) for v, nd in zip(key, self.nodes)])
                v = self._rgi(x[None, :])[0]
            else:
                v = self._values
            m = self.edges.size - 1
            c = Curve(self.edges, np.exp(v[:m]), v[m:], self.Kc)
            if len(self._curves) < 4096:
                self._curves[key] = c
        return c
