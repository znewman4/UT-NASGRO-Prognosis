"""Test 5: the cached tables match full NASGRO where they are used.

  * at a node, a table curve is the full NASGRO curve (same grid, same arithmetic);
  * lambda is an exact multiplier: cycles scale as 1/lambda when lambda acts on C;
  * between nodes, interpolation stays within a stated tolerance;
  * N proportional to 1/lambda is only approximate when the rate change comes from
    stress, because the threshold and Kc terms do not scale (this is why lambda is
    defined on C and stress is a table axis when it matters).
"""
import numpy as np
import pytest

from utprog.forward import nasgro_adapter as ad
from utprog.forward.base import DEAD_RATE
from utprog.forward.table_cache import NasgroTable
from utprog.truth import priors, scenarios

A_MIN = 0.05e-3
TOL_DECADES = 0.02


def full_rul(case, a_from, nominal=None, kfn=None, sc=None, amp=None):
    s = priors.nominal_inputs(case, nominal)
    s["a_i"] = np.array([a_from])
    if sc is not None:
        s = scenarios.rs_inputs_at(s, sc, amp)
    return float(ad.trajectory(case, s, kfn, n_grid=2000).life[0])


def th(**kw):
    return {k: np.atleast_1d(np.asarray(v, float)) for k, v in kw.items()}


@pytest.mark.parametrize("case", ["plate", "bar"])
def test_node_curve_equals_full_nasgro(case):
    tab = NasgroTable(case, A_MIN, n_grid=600)
    s = priors.nominal_inputs(case)
    s["a_i"] = np.array([A_MIN])
    tr = ad.trajectory(case, s, n_grid=600)
    c = tab.curve(())
    np.testing.assert_allclose(c.log_a, np.log(tr.edges[0]), rtol=1e-12)
    assert c.stop == tr.stop[0]
    rate = tr.rate[0]
    np.testing.assert_array_equal(c.dead, rate < DEAD_RATE)
    live = ~c.dead
    np.testing.assert_allclose(np.diff(c.cum)[live], (np.diff(tr.edges[0]) / np.where(live, rate, 1.0))[live],
                               rtol=1e-9)


@pytest.mark.parametrize("case", ["plate", "bar"])
def test_rul_from_table_matches_full_nasgro(case):
    tab = NasgroTable(case, A_MIN, n_grid=600)
    for a in (0.5e-3, 1e-3, 3e-3):
        r_tab = tab.rul(np.array([a]), th(log_lambda=0.0))[0]
        r_full = full_rul(case, a)
        if np.isinf(r_full):
            assert np.isinf(r_tab)
        else:
            assert np.log10(r_tab) == pytest.approx(np.log10(r_full), abs=0.005)


def test_lambda_is_exact_on_C():
    case = "plate"
    tab = NasgroTable(case, A_MIN, n_grid=600)
    C = priors.nominal_inputs(case)["C"][0]
    r_full = full_rul(case, 1e-3, nominal={"C": 2.0 * C})
    r_tab = tab.rul(np.array([1e-3]), th(log_lambda=np.log(2.0)))[0]
    assert np.log10(r_tab) == pytest.approx(np.log10(r_full), abs=0.005)
    np.testing.assert_allclose(r_tab, tab.rul(np.array([1e-3]), th(log_lambda=0.0))[0] / 2.0, rtol=1e-12)


def test_interpolation_between_stress_nodes():
    tab = NasgroTable("plate", A_MIN, n_grid=600, axes={"S": [160.0, 200.0, 240.0]})
    for S in (180.0, 220.0):
        r_tab = tab.rul(np.array([1e-3]), th(log_lambda=0.0, S=S))[0]
        r_full = full_rul("plate", 1e-3, nominal={"S": S})
        assert np.log10(r_tab) == pytest.approx(np.log10(r_full), abs=TOL_DECADES)


@pytest.mark.parametrize("name", ["plate_weld", "plate_uniform"])
def test_interpolation_between_residual_stress_nodes(name):
    sc = scenarios.rs_scenario({"from": name}, "plate")
    kfn = scenarios.rs_k_fn("plate", sc)
    nodes = np.linspace(0.5, 1.0, 6)
    tab = NasgroTable("plate", A_MIN, n_grid=600, axes={"rs_amp": nodes}, rs=sc)
    for amp in (nodes[2], 0.5 * (nodes[2] + nodes[3])):
        r_tab = tab.rul(np.array([1e-3]), th(log_lambda=0.0, rs_amp=amp))[0]
        r_full = full_rul("plate", 1e-3, kfn=kfn, sc=sc, amp=np.array([amp]))
        assert np.log10(r_tab) == pytest.approx(np.log10(r_full), abs=TOL_DECADES)


def test_stress_scaling_is_only_approximately_inverse_lambda():
    """Characterises the approximation: report, and bound it for +-10 % stress at 1 mm."""
    tab = NasgroTable("plate", A_MIN, n_grid=600)
    n = priors.nominal_inputs("plate")["n"][0]
    base = tab.rul(np.array([1e-3]), th(log_lambda=0.0))[0]
    for f in (0.9, 1.1):
        exact = full_rul("plate", 1e-3, nominal={"S": 200.0 * f})
        scaled = base / f ** n                           # N proportional to 1/lambda, lambda = f^n
        dev = abs(np.log10(exact) - np.log10(scaled))
        assert 1e-4 < dev < 0.1, dev
