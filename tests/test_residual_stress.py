"""M2: residual-stress scenarios and the Bayes-factor machinery."""
import numpy as np
import pytest

from utprog.estimators.base import Prior
from utprog.estimators.grid_bayes import GridBayes
from utprog.estimators.likelihood import NoiseModel
from utprog.estimators.model_selection import ModelSelection
from utprog.forward import nasgro_adapter as ad
from utprog.forward.table_cache import NasgroTable
from utprog.sensors.base import Observation
from utprog.truth import priors, scenarios
from utprog.truth.priors import InitialDepth

A_MIN = 0.05e-3


def nominal(a_i=1e-3):
    s = priors.nominal_inputs("plate")
    s["a_i"] = np.array([a_i])
    return s


def test_redistribution_with_full_retention_is_superposition():
    sup = scenarios.rs_scenario({"from": "plate_weld"}, "plate")
    red = scenarios.rs_scenario({"from": "plate_weld", "form": "redistribution", "r_inf": 1.0,
                                 "a_ref_mm": 1.0}, "plate")
    s = scenarios.rs_inputs_at(nominal(), sup, np.array([0.8]))
    l1 = ad.trajectory("plate", s, scenarios.rs_k_fn("plate", sup)).life
    l2 = ad.trajectory("plate", s, scenarios.rs_k_fn("plate", red)).life
    np.testing.assert_allclose(l2, l1, rtol=1e-12)


def test_relaxation_shortens_the_residual_effect():
    sup = scenarios.rs_scenario({"from": "plate_weld"}, "plate")
    red = scenarios.rs_scenario({"from": "plate_weld", "form": "redistribution", "r_inf": 0.2,
                                 "ell_mm": 0.5, "a_ref_mm": 1.0}, "plate")
    s = scenarios.rs_inputs_at(nominal(), sup, np.array([0.8]))
    base = ad.trajectory("plate", nominal()).life[0]
    l_sup = ad.trajectory("plate", s, scenarios.rs_k_fn("plate", sup)).life[0]
    l_red = ad.trajectory("plate", s, scenarios.rs_k_fn("plate", red)).life[0]
    assert l_sup < l_red < base


def test_zero_amplitude_node_equals_no_residual_stress():
    sc = scenarios.rs_scenario({"A": [-0.5, 0.5], "L_mm": [2.5, 0.3]}, "plate")
    tab = NasgroTable("plate", A_MIN, n_grid=400, axes={"rs_amp": [-0.5, 0.0, 0.5]}, rs=sc)
    none = NasgroTable("plate", A_MIN, n_grid=400)
    th = {"log_lambda": np.zeros(1), "rs_amp": np.zeros(1)}
    np.testing.assert_allclose(tab.rul(np.array([1e-3]), th), none.rul(np.array([1e-3]), {"log_lambda": np.zeros(1)}),
                               rtol=1e-9)


def test_compressive_and_tensile_change_life_in_opposite_directions():
    ten = scenarios.rs_scenario({"from": "plate_weld"}, "plate")
    com = scenarios.rs_scenario({"A": [-0.8, -0.4], "L_mm": [0.8, 0.3], "shakedown": True}, "plate")
    base = ad.trajectory("plate", nominal()).life[0]
    s_t = scenarios.rs_inputs_at(nominal(), ten, np.array([0.75]))
    s_c = scenarios.rs_inputs_at(nominal(), com, np.array([-0.6]))
    assert ad.trajectory("plate", s_t, scenarios.rs_k_fn("plate", ten)).life[0] < base
    assert ad.trajectory("plate", s_c, scenarios.rs_k_fn("plate", com)).life[0] > base


def test_identical_hypotheses_give_zero_bayes_factor():
    tab = NasgroTable("plate", A_MIN, n_grid=400)
    nz = NoiseModel(sd_abs=1e-4)
    pr = Prior(0.0, 0.3, InitialDepth(1e-3, 0.2))
    ms = ModelSelection({"a": GridBayes(tab, nz, 41, 21), "b": GridBayes(tab, nz, 41, 21)}, {"a": pr, "b": pr})
    ms.reset()
    for N, y in [(0.0, 1.0e-3), (5000.0, 1.2e-3), (10000.0, 1.5e-3)]:
        ms.update(Observation(N, "sized", y))
    assert ms.summary()["two_ln_bf"] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(ms.probs(), [0.5, 0.5])


def test_bayes_factor_picks_the_generating_hypothesis_when_lambda_is_tight():
    """Machinery check, not a finding: with lambda nearly known the classes separate."""
    ten = scenarios.rs_scenario({"from": "plate_weld"}, "plate")
    com = scenarios.rs_scenario({"A": [-0.8, -0.4], "L_mm": [0.8, 0.3], "shakedown": True}, "plate")
    t_ten = NasgroTable("plate", A_MIN, n_grid=400, axes={"rs_amp": [0.5, 0.75, 1.0]}, rs=ten)
    t_com = NasgroTable("plate", A_MIN, n_grid=400, axes={"rs_amp": [-0.8, -0.6, -0.4]}, rs=com)
    th = {"log_lambda": np.zeros(1), "a0": np.array([1e-3]), "rs_amp": np.array([0.75])}
    life = t_ten.rul(np.array([1e-3]), th)[0]
    rng = np.random.default_rng(0)
    nz = NoiseModel(sd_abs=5e-5)
    base = Prior(0.0, 0.05, InitialDepth(1e-3, 0.0))
    p_ten = Prior(0.0, 0.05, base.a0, {"rs_amp": ([0.5, 0.75, 1.0], [1, 1, 1])})
    p_com = Prior(0.0, 0.05, base.a0, {"rs_amp": ([-0.8, -0.6, -0.4], [1, 1, 1])})
    ms = ModelSelection({"tensile": GridBayes(t_ten, nz, 41, 3), "compressive": GridBayes(t_com, nz, 41, 3)},
                        {"tensile": p_ten, "compressive": p_com})
    ms.reset()
    for N in np.linspace(0, 0.7 * life, 20):
        a = t_ten.depth_at(N, th)[0]
        ms.update(Observation(float(N), "sized", float(a + 5e-5 * rng.standard_normal())))
    s = ms.summary()
    assert s["pred_class"] == "tensile" and s["two_ln_bf"] > 6.0
