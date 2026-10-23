"""Forward models: the closed-form toy and the Curve machinery are self-consistent."""
import numpy as np
import pytest

from utprog.forward.base import Curve


def test_toy_advance_and_rul_are_inverse(toy):
    th = {"log_lambda": np.array([-0.3, 0.0, 0.4])}
    a = np.array([0.5e-3, 1e-3, 3e-3])
    r = toy.rul(a, th)
    np.testing.assert_allclose(toy.advance(a, r, th), toy.a_c, rtol=1e-9)
    a1 = toy.advance(a, 0.3 * r, th)
    np.testing.assert_allclose(toy.advance(a1, 0.4 * r, th), toy.advance(a, 0.7 * r, th), rtol=1e-9)
    np.testing.assert_allclose(toy.rul(a, {"log_lambda": th["log_lambda"] + np.log(2)}), r / 2, rtol=1e-12)


def test_toy_matches_numerical_paris(toy):
    a = np.geomspace(1e-3, toy.a_c, 200001)
    rate = toy.k * a ** (toy.m / 2)
    N = np.trapezoid(1 / rate, a) if hasattr(np, "trapezoid") else np.trapz(1 / rate, a)
    assert toy.rul(np.array([1e-3]), {"log_lambda": np.array([0.0])})[0] == pytest.approx(N, rel=1e-6)


def test_curve_constant_rate_and_inverse():
    edges = np.geomspace(1e-4, 1e-2, 101)
    c = Curve(edges, np.full(100, 1e-8))
    a = np.array([2e-4, 1e-3])
    np.testing.assert_allclose(c.rul(a, 1.0), (1e-2 - a) / 1e-8, rtol=1e-3)
    np.testing.assert_allclose(c.advance(a, c.rul(a, 2.0), 2.0), 1e-2, rtol=1e-9)


def test_curve_arrest_and_failure():
    edges = np.geomspace(1e-4, 1e-2, 101)
    rate = np.full(100, 1e-8)
    rate[40:45] = 0.0                                   # dead zone
    kmax = np.linspace(1, 50, 100)
    c = Curve(edges, rate, kmax, Kc=40.0)
    below, above = edges[20], edges[60]
    assert np.isinf(c.rul(np.array([below]), 1.0))[0]
    assert c.advance(np.array([below]), np.array([1e12]), 1.0)[0] == pytest.approx(edges[40])
    assert np.isfinite(c.rul(np.array([above]), 1.0))[0]
    assert c.a_fail == pytest.approx(edges[int(np.argmax(kmax >= 40))])
    assert c.advance(np.array([above]), np.array([1e12]), 1.0)[0] == pytest.approx(c.a_fail)
    inside = np.sqrt(edges[42] * edges[43])
    assert c.advance(np.array([inside]), np.array([1e6]), 1.0)[0] == inside
