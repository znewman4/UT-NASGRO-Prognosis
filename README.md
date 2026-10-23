# UT–NASGRO prognosis test bed

**Question.** The sensitivity study says how much forecast spread disappears if an input is
*known exactly*. This test bed measures how much disappears when a *real ultrasonic sensor and
estimator learn it*. The ratio is each approach's **realised efficacy**.

Virtual specimens are grown with the validated NASGRO pipeline (read-only, pinned, hash-checked),
observed by modelled sensors under a measurement policy, and forecast by three kinds of estimator,
all on the same specimens and noise:

| Kind | What it is | Role |
|---|---|---|
| `study_mc` | the sensitivity engine at the true current depth, chosen inputs at their true values | oracle and study baseline; reproduces the study to 1e-9 (D-005) |
| `nasgro_is` | importance sampling over full NASGRO input vectors | well-specified filter (like check B) |
| `grid` / `model_selection` | lumped growth multiplier λ on C; Bayes factors between residual-stress hypotheses | practical filter |

Approaches follow the scoping page: 1 growth rate, 2 residual stress, 3 intervals, 4 opening,
5 scattering geometry, 6 joint model. Frozen predictions: `docs/predictions.md`.

## Status

| Step | State |
|---|---|
| 1 · Predictions frozen with exact definitions | done (`docs/predictions.md`, `.yaml`); P1.3 and P1.4 come from an uncommitted rerun of the original pipeline (D-001) |
| 2 · Oracle arms, fixed-depth scoring, same cases and spreads | done; oracle reproduces stored U90 and every tested reduction to 1e-9 |
| 3 · Code: renumbering, LCR sensor + E-03c, bond failure, biased sensor, perfect inspection, mismatched truth, sweeps | done for Approaches 1–3; E-04, E-05 are the second round |
| 4 · Inputs agreed with supervisors (sensor values, gates, costs) | **open**: every such value is marked PLACEHOLDER |
| 5 · Runs | tests pass (82); every config smoke-tested on 2–3 specimens; **no experiment run yet** |

## Rerun

```bash
git clone --recurse-submodules <this repo>        # or: git submodule update --init
pip install -e ".[test]"
python -m pytest                                   # 82 tests, incl. test 3 and the oracle check (~40 s)

# after the inputs are agreed, in this order:
python experiments/00_toy.py
python experiments/01_depth_only.py                                  # Approach 1, plate
python experiments/01_depth_only.py configs/01b_depth_only_bar.yaml  # Approach 1, bar
python experiments/01_depth_only.py configs/01c_sensor_sweeps.yaml   # Approach 1, placeholder sweeps
python experiments/02_intervals.py                                   # Approach 3
python experiments/03_residual_stress.py                             # Approach 2, depth alone
python experiments/03_residual_stress.py configs/03b_residual_forms.yaml
python experiments/03_residual_stress.py configs/03c_residual_lcr.yaml
python reports/efficacy.py                                           # reports/efficacy.md
python reports/build.py                                              # per-run tables and figures
```

Each run writes a `manifest.json` (config hash, commit, pipeline commit, versions, seed); reports
ignore any result without one. Add `--n 2 --out <scratch dir>` to any script for a smoke test that
never touches `results/`.

## Layout

```
external/nasgro_sa/          validated pipeline (submodule at 28f217c; never edited) + lock file
src/utprog/
  truth/      priors.py (incl. the study's depth model), specimen.py, scenarios.py (residual stress, path variability)
  forward/    nasgro_adapter.py, table_cache.py, toy_paris.py, base.py
  sensors/    depth.py (noise, bias, correlation, POD, bond failure), stress.py (LCR layer averages)
  policies/   none, one_off, periodic, continuous, window, event_triggered
  estimators/ study_mc.py (oracle), nasgro_is.py, grid_bayes.py, model_selection.py, particle_filter.py, likelihood.py
  scoring/    metrics, efficacy, prognostic (α-λ, horizon), sbc, cost, summary
  runner.py   truth × sensor × policy × estimator, from a config
configs/      00 toy · 01/01b/01c Approach 1 · 02 Approach 3 · 03/03b/03c Approach 2
docs/         predictions, assumptions (with sources and status), decisions, glossary
```

## Tests

| # | Test | File |
|---|---|---|
| 1 | Toy recovery of a known λ | `tests/test_toy_recovery.py` |
| 2 | Coverage of nominal 90 % intervals, SBC | `tests/test_coverage.py` |
| 3 | Adapter reproduces the pipeline's lives and stored results | `tests/test_adapter_regression.py` |
| 4 | Particle filter agrees with the grid on one parameter | `tests/test_estimator_agreement.py` |
| 5 | Tables match full NASGRO where used | `tests/test_surrogate.py` |
| — | Oracle reproduces the study; frozen predictions match stored results | `tests/test_oracle.py` |
| — | IS estimator, stress likelihood, study depth model | `tests/test_nasgro_is.py` |

Plus: pipeline lock, likelihood vs dense normal, forward models, sensors and policies (common
random numbers), scoring, residual-stress scenarios and Bayes factors, runner reproducibility.
