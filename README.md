# UT–NASGRO prognosis test bed

**Question.** How accurate must ultrasonic crack sizing be before repeated sizing improves a
NASGRO remaining-life forecast, and when does continuous monitoring beat periodic inspection?
Then: can the sized-depth record alone tell tensile, none or compressive residual stress apart?

Virtual specimens are grown with the validated NASGRO pipeline (read-only, pinned, hash-checked),
observed by a modelled UT sensor under a measurement policy, and forecast by Bayesian
estimators. Every arm sees the same specimens and noise, so differences are the arms'.

## Status

| Milestone | Scope | Exit criterion | State |
|---|---|---|---|
| M0 | Skeleton, adapter, regression tests | Test 3 passes | built, **not yet run** |
| M1 | Depth-only test bed (Approaches 1 and 4) | Tests 1, 2, 4 pass; first table agreed | built, **not yet run** |
| M2 | Residual-stress hypotheses (Approach 3) | Classification rate vs sd and life fraction | built, **not yet run** |
| M3–M5 | Angle switch, opening channel, joint model | | stubs only |

## Rerun

```bash
git clone --recurse-submodules <this repo>        # or: git submodule update --init
pip install -e ".[test]"
python -m pytest                                   # fast tests, incl. test 3 (M0 exit)
python -m pytest -m slow                           # test 2: coverage and SBC on the toy
python experiments/00_toy.py
python experiments/01_depth_only.py                # all starting sizes and sensor variants
python experiments/01_depth_only.py configs/01b_depth_only_bar.yaml
python experiments/02_intervals.py
python experiments/03_residual_stress.py
python experiments/03_residual_stress.py configs/03b_residual_forms.yaml
python reports/build.py
```

A run is reproducible from its config, the git commit and the seed; each writes a
`manifest.json`, and `reports/build.py` ignores any result without one.

## Layout

```
external/nasgro_sa/      validated pipeline (git submodule, pinned; never edited)
external/nasgro_sa.lock.json   SHA-256 of every module and stored result used for regression
src/utprog/
  truth/                 priors.py, specimen.py, scenarios.py (residual stress, path variability)
  forward/               toy_paris.py, nasgro_adapter.py, table_cache.py, base.py (Curve)
  sensors/               depth.py (opening, load, angle: later milestones)
  policies/              none, one_off, periodic, continuous, event_triggered
  estimators/            likelihood.py, grid_bayes.py, particle_filter.py, model_selection.py
  scoring/               metrics, prognostic (α-λ, horizon), sbc, cost, summary
  runner.py              truth × sensor × policy × estimator, from a config
configs/ experiments/ tests/ results/ reports/ docs/
```

## The five interfaces

| Part | Contract |
|---|---|
| Truth | `draw(rng) -> Specimen` (true inputs, true a(N), failure life) |
| Sensor | `observe(specimen, times, seed) -> [Observation]` (noise, bias, correlation, POD, floors) |
| Policy | `times(specimen, ctx) -> schedule` |
| Estimator | `reset(prior)`, `update(observation)`, `forecast(N) -> remaining-life samples`; never sees the truth |
| Scorer | forecasts + truths → U90, bias, coverage, CRPS, α-λ, prognostic horizon, SBC, cost |

## Tests

| # | Test | File |
|---|---|---|
| 1 | Toy recovery of a known λ | `tests/test_toy_recovery.py` |
| 2 | Coverage of nominal 90 % intervals, SBC | `tests/test_coverage.py` (slow) |
| 3 | Adapter reproduces the pipeline's lives and stored results | `tests/test_adapter_regression.py` |
| 4 | Particle filter agrees with the grid on one parameter | `tests/test_estimator_agreement.py` |
| 5 | Tables match full NASGRO where used | `tests/test_surrogate.py` |

Plus: pipeline lock, likelihood vs dense normal, forward models, sensors and policies (CRN),
scoring, residual-stress scenarios and Bayes factors, runner reproducibility.

Assumptions and their sources: `docs/assumptions.md`. Decisions and results: `docs/decisions.md`.
