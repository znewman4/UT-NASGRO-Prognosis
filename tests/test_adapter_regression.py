"""Test 3 (M0 exit): the adapter reproduces the validated pipeline.

If this fails, nothing downstream is trusted.
  * trajectory() lives equal nasgro_scenarios.life.ca_life, with and without residual stress;
  * its cumulative curve ends at that life and gives a monotone a(N);
  * it reproduces the stored statistics of the sensitivity study (results/baseline.json,
    extended inputs; results/2026-10-23_nasgro_only/rankings.json, NASGRO-only inputs) on the
    same random numbers. The stored files are hash-locked with the pipeline.
"""
import json

import numpy as np
import pytest

from utprog.forward import nasgro_adapter as ad

STATS = ("mu", "sigma", "u90", "pf", "median")


def _lives(ns, case, s):
    return ad.chunked(lambda x: ad.trajectory(case, x).life, s)


@pytest.mark.parametrize("case", ["bar", "plate"])
@pytest.mark.parametrize("a_mm", [0.5, 1.0, 4.0])
def test_trajectory_life_equals_ca_life(ns, case, a_mm):
    s = ns.cases.sample(case, a_mm * 1e-3, 400, seed=7)
    tr = ad.trajectory(case, s)
    ref = ns.life.ca_life(case, s)
    np.testing.assert_array_equal(np.isinf(tr.life), np.isinf(ref))
    fin = np.isfinite(ref)
    np.testing.assert_allclose(tr.life[fin], ref[fin], rtol=1e-12)
    end = tr.cum[np.arange(tr.stop.size), tr.stop]
    np.testing.assert_allclose(np.maximum(end[fin], 1.0), ref[fin], rtol=1e-9)


@pytest.mark.parametrize("name", ["plate_weld", "bar_tensile", "bar_compressive", "plate_uniform"])
def test_trajectory_with_residual_stress_equals_ca_life(ns, name):
    sc = ns.residual_stress.SCENARIOS[name]
    case = sc["case"]
    s = ns.cases.sample(case, 1e-3, 300, seed=3)
    s = ns.residual_stress.add_inputs(s, sc, 3)
    kfn = ns.residual_stress.k_res(case, sc["shape"])
    tr = ad.trajectory(case, s, kfn)
    ref = ns.life.ca_life(case, s, kfn)
    np.testing.assert_array_equal(np.isinf(tr.life), np.isinf(ref))
    fin = np.isfinite(ref)
    np.testing.assert_allclose(tr.life[fin], ref[fin], rtol=1e-12)


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_depth_curve_is_monotone_and_reaches_failure(ns, case):
    s = ns.cases.sample(case, 1e-3, 50, seed=11, fixed=("kappa", "sif"))
    tr = ad.trajectory(case, s)
    for i in np.flatnonzero(np.isfinite(tr.life)):
        N, la = tr.finite_curve(i)
        assert np.all(np.diff(N) > 0) and np.all(np.diff(la) > 0)
        assert np.exp(la[-1]) == pytest.approx(tr.a_fail[i])


def test_cycles_to_depth_matches_curve_and_life(ns):
    s = ns.cases.sample("plate", 0.5e-3, 100, seed=5, fixed=("kappa", "sif"))
    tr = ad.trajectory("plate", s)
    to_fail = ad.cycles_to_depth("plate", s, 0.99 * ns.cases.a_limit("plate", s) + 1.0)
    np.testing.assert_allclose(to_fail, tr.life, rtol=1e-12)
    n2 = ad.cycles_to_depth("plate", s, 2e-3)
    ok = np.isfinite(tr.life) & (tr.a_fail > 2e-3)
    assert np.all(n2[ok] < tr.life[ok]) and np.all(n2[ok] > 0)


def _stored(ns, rel):
    return json.loads((ns.path / rel).read_text())["results"]


def _check(ns, case, depth_mm, n, fixed, stored):
    s = ns.cases.sample(case, depth_mm * 1e-3, n, seed=1, fixed=fixed)
    st = ns.life.stats(_lives(ns, case, s))
    for k in STATS:
        assert st[k] == pytest.approx(stored[k], abs=1e-9), (case, depth_mm, k)


def test_reproduces_stored_baseline_extended(ns):
    for r in _stored(ns, "results/baseline.json"):
        _check(ns, r["case"], r["depth_mm"], r["n"], (), r["stats"])


def test_reproduces_stored_rankings_nasgro_only(ns):
    rows = [r for r in _stored(ns, "results/2026-10-23_nasgro_only/rankings.json")
            if r["config"] == "standard" and r["seed"] == 1]
    assert rows
    for r in rows:
        _check(ns, r["case"], r["depth_mm"], r["n"], ("kappa", "sif"), r["stats"])
