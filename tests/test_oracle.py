"""Step 2 of the plan: oracle arms reproduce the sensitivity study exactly.

The study_mc forecaster run at a fixed true depth, with the study's seed and
sample size, must give the stored U90 (baseline) and the stored U90 reduction
when an input is known. If this fails there is a definition mismatch or a bug,
and no efficacy number is trusted.
"""
import json

import numpy as np
import pytest

from utprog.estimators.study_mc import StudyMC
from utprog.scoring.metrics import u90_finite
from utprog.truth import priors
from utprog.truth.specimen import Specimen

GROUP = {"crack depth": "depth", "crack shape a/c": "ac", "stress amplitude": "S", "load amplitude": "F",
         "material fit (C, m, p, dKth0)": "material", "growth curve C-n": "Cn", "constraint factor alpha": "alpha"}


def stored(ns, case, depth_mm):
    rows = json.loads((ns.path / "results/2026-10-23_nasgro_only/rankings.json").read_text())["results"]
    return next(r for r in rows if r["config"] == "standard" and r["seed"] == 1
                and r["case"] == case and r["depth_mm"] == depth_mm)


def at_depth(case, a):
    nom = priors.nominal_inputs(case)
    return Specimen(index=0, case=case, a0=0.5 * a, life=1e9, a_fail=1.0, truth_theta={},
                    inputs={k: float(v[0]) for k, v in nom.items()}, depth_fn=lambda N: np.full(np.shape(N), a))


@pytest.mark.parametrize("case,depth_mm", [("plate", 1.0), ("bar", 1.0), ("plate", 0.5), ("bar", 2.0)])
def test_oracle_reproduces_stored_reductions(ns, case, depth_mm):
    r = stored(ns, case, depth_mm)
    spec = at_depth(case, depth_mm * 1e-3)
    base = StudyMC(case, (), n=r["n"], seed=1).forecast_truth(spec, 0.0)
    assert u90_finite(base) == pytest.approx(r["stats"]["u90"], abs=1e-9)
    assert np.isfinite(base).mean() == pytest.approx(r["stats"]["pf"], abs=1e-12)
    for name, group in GROUP.items():
        if name not in r["reduction"]:
            continue
        x = StudyMC(case, (group,), n=r["n"], seed=1).forecast_truth(spec, 0.0)
        assert 1.0 - u90_finite(x) / r["stats"]["u90"] == pytest.approx(r["reduction"][name], abs=1e-9), name


def test_lambda_group_expands_per_case():
    assert StudyMC("plate", ("lambda",)).groups == ("S", "Cn", "alpha")
    assert StudyMC("bar", ("depth", "lambda")).groups == ("depth", "F", "material")


def test_known_groups_take_the_true_values():
    spec = at_depth("plate", 1e-3)
    spec.inputs["S"] = 230.0
    mc = StudyMC("plate", ("S",), n=50)
    s = mc.population(1e-3, spec.inputs)
    assert np.all(s["S"] == 230.0)


def test_frozen_predictions_match_stored_results(ns):
    """P1.1 in docs/predictions.yaml is the five-seed mean of the stored crack-depth reductions."""
    from utprog.scoring.efficacy import predictions
    p = predictions()
    assert {"P1.1", "P1.2", "P1.3", "P1.4", "P2.1", "P2.2", "P3.1"} <= set(p)
    rows = json.loads((ns.path / "results/2026-10-23_nasgro_only/rankings.json").read_text())["results"]
    for case, vals in p["P1.1"]["values"].items():
        for depth, want in vals.items():
            got = np.mean([r["reduction"]["crack depth"] for r in rows if r["config"] == "standard"
                           and r["case"] == case and r["depth_mm"] == float(depth)])
            assert got == pytest.approx(want, abs=6e-4), (case, depth)


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_current_practice_reproduces_the_study_sizing_sweep(ns, case):
    """static_mc given a reading exactly at 1 mm reproduces the study: sd 0.5 mm is its baseline,
    sd 0.5/3 mm its Appendix A 'x1/3' level (P1.2), on the same random numbers."""
    from utprog.estimators.likelihood import NoiseModel
    from utprog.estimators.study_mc import StaticMC
    from utprog.sensors.base import Observation
    sweeps = json.loads((ns.path / "results/2026-10-23_nasgro_only/sweeps.json").read_text())["results"]
    levels = {lv["level"]: lv for x in sweeps if x["case"] == case and x["key"] == "depth" for lv in x["levels"]}
    for level, sd in (("typical", 0.5e-3), ("x1/3", 0.5e-3 / 3)):
        est = StaticMC(case, NoiseModel(sd_abs=sd), n=10000, seed=1)
        est.reset()
        est.update(Observation(0.0, "sized", 1e-3))
        x = est.forecast(0.0, 0, None)
        assert u90_finite(x) == pytest.approx(levels[level]["u90"], abs=1e-9), level
        assert np.isfinite(x).mean() == pytest.approx(levels[level]["pf"], abs=1e-12), level
