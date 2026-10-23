"""The scratch toy: closed-form Paris growth with a lumped multiplier.

    da/dN = lambda C (dS Y sqrt(pi a))^m,    failure at a = a_c.

With k = C (dS Y sqrt(pi))^m and p = 1 - m/2, N(a1 -> a2) = (g(a2) - g(a1)) / (lambda k)
with g(a) = a^p / p (log a when m = 2), so a(N) and N(a) are exact. The toy has
no surrogate error, so it isolates the estimators: tests 1, 2 and 4 run on it.
"""
import numpy as np

from .base import ForwardModel


class ToyParis(ForwardModel):
    axes = ()

    def __init__(self, C, m, dS, Y, a_c):
        self.C_nominal = float(C)
        self.m = float(m)
        self.a_c = float(a_c)
        self.k = self.C_nominal * (dS * Y * np.sqrt(np.pi)) ** self.m
        self.p = 1.0 - self.m / 2.0

    @classmethod
    def from_config(cls, d):
        return cls(float(d["C"]), float(d["m"]), float(d["dS_MPa"]), float(d["Y"]), float(d["a_c_mm"]) * 1e-3)

    def _g(self, a):
        return np.log(a) if self.p == 0.0 else a ** self.p / self.p

    def _ginv(self, g):
        return np.exp(g) if self.p == 0.0 else (self.p * g) ** (1.0 / self.p)

    def advance(self, a, dN, theta):
        a, dN, lam = np.broadcast_arrays(np.asarray(a, float), np.asarray(dN, float),
                                         np.exp(np.asarray(theta["log_lambda"], float)))
        g = self._g(np.minimum(a, self.a_c)) + lam * self.k * dN
        gc = self._g(self.a_c)
        return np.where(g >= gc, self.a_c, self._ginv(np.minimum(g, gc)))

    def rul(self, a, theta):
        a, lam = np.broadcast_arrays(np.asarray(a, float), np.exp(np.asarray(theta["log_lambda"], float)))
        r = (self._g(self.a_c) - self._g(np.minimum(a, self.a_c))) / (lam * self.k)
        return np.maximum(r, 0.0)
