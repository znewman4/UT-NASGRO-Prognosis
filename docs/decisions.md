# Decisions

One entry per design decision or experiment: the question, the result, and what it changes.
Experiment entries are written after the run, from its `results/summary/` tables and manifest.
Approach numbers follow the scoping page: 1 growth rate, 2 residual stress, 3 intervals,
4 opening, 5 scattering geometry, 6 joint model.

## D-001 · Which code the predictions come from (M0; revised 23 Oct)

- **Question.** The plan names `nasgro_sa` as a read-only dependency. Which code is it, and is
  it the run the 23 October report quotes?
- **Result.** Two pipelines exist in `znewman4/NASGRO-Sensitivity-Analysis`:
  - the rebuild, `nasgro_scenarios`, on the branch the report links to (head `28f217c`; code
    identical to `7bf9697`, plus the NASGRO-only variable-amplitude results). Every number in the
    report's sections 2–8 comes from it. Test 3 reproduces its stored statistics to 1e-9.
  - the original `nasgro_sa`, on `main` (`f9a6555`, 22 Oct). The scoping page's checks A and B
    were rerun on it with κ and the SIF error at 1, but that rerun is not in any commit.
- **Changes.** The rebuild is pinned (`external/nasgro_sa`, hash-locked). Predictions P1.3 and
  P1.4 (checks A, B) are marked not reproducible and compared as reductions and ratios; the
  simulation recomputes them on the rebuild. If the rerun scripts are found, commit and pin them.

## D-002 · λ is an exact factor on C; other effects are table axes (M0)

- **Result.** NASGRO's rate is proportional to C, so defining λ on C makes N ∝ 1/λ exact;
  through stress it would not be (test 5 measures the gap).
- **Changes.** Tables are computed at λ = 1 and scaled; stress or residual-stress amplitude enter
  as table axes. Results are reported as C_eff = λ C_nominal.

## D-003 · Three kinds of estimator (M1; revised 23 Oct)

- **Question.** How to separate the value of a measurement from the error of the filter?
- **Changes.**
  - `study_mc`: the sensitivity engine itself, run at the true current depth with chosen inputs at
    their true values. The perfect-information oracle and the study baseline. Sees the truth.
  - `nasgro_is`: importance sampling over full NASGRO input vectors drawn from the truth's own
    priors. Well-specified, like check B; degenerates with many precise readings (ESS recorded).
  - `grid`: lumped λ on a nominal curve, with adaptive refinement. The practical filter; its gap
    to `nasgro_is` is the cost of lumping.
  `particle_filter` (IBIS) remains only to cross-check `grid` (test 4).

## D-004 · Fair scoring, and scoring where the study scored (M1; revised 23 Oct)

- **Changes.** Forecasts are scored at random times (fair to periodic inspection), at fixed life
  fractions (Chen-style cutoffs), at the moment the true crack reaches each study depth (to line
  up with the sensitivity study), and at fixed times (check B's end of window).

## D-005 · The oracle must reproduce the study before anything else is read (M1, 23 Oct)

- **Result.** `tests/test_oracle.py`: at 1 mm (plate, bar), 0.5 mm (plate) and 2 mm (bar), with
  the study's seed and n, the oracle reproduces the stored baseline U90, Pf and every tested input's
  U90 reduction (depth, a/c, stress, load, material fit, C–n, α) to 1e-9.
- **Changes.** Every experiment with depth scoring writes `<exp>_oracle_check.csv`; if the oracle
  misses the frozen prediction by more than the tolerance, the efficacy tables are not used.
- **Note.** The study keeps tilt uncertain (sd 5° sized, 15° undetected); the oracle does too, so it
  matches the study, while the depth-only truth has tilt 0. `fix_tilt: true` aligns the oracle with
  the truth instead.

## D-006 · Cracks that never grow are kept (M1, 23 Oct)

- **Result.** Dropping them biased Approach 2 (a peened surface that stops a crack is the easiest
  class to call) and hid the value of seeing whether a small crack grows.
- **Changes.** They are scored over one nominal life: arrest forecast (Brier score of p_arrest)
  and classification; RUL scores use failing cracks only, as the study's U90 does.

## D-007 · Gate thresholds and sensor values are placeholders (M1, M2)

- **Changes.** Every `gates:` block carries `placeholder: true`. Thresholds, sensor values (sizing
  error, correlation, POD, LCR error, bond life) and costs are agreed with the supervisors before
  the runs, and edited here with the date agreed. Sweeps (`01c`, `03c`, `failure_levels`) turn the
  results into conditions rather than single numbers.

## D-008a · Forecasts condition on survival to the forecast time (23 Oct)

- **Result.** Smoke runs showed a filter forecasting "already failed" for a slow specimen: its
  prior said every growth rate fails before the forecast time. A component in service has not
  failed, and that is information.
- **Changes.** `grid`, `particle_filter`, `nasgro_is` and the Bayes-factor comparison all condition
  on survival to N (`model_selection` adds the survival probability to each hypothesis's evidence).

## D-008b · Current practice as its own arm, and what "realised fraction > 1" means (23 Oct)

- **Result.** The study's populations have no time axis: depth is known to ±0.5 mm and everything
  else is at its industrial spread. A filter in service also knows the elapsed time and the
  start-depth prior, so even one fresh sizing partly informs the growth rate, and its realised
  reduction can exceed the depth oracle's.
- **Changes.** `static_mc` is current practice: size once, then run the NASGRO Monte Carlo around
  the reading, with no history and no survival. With a reading at the true depth it reproduces the
  study's baseline (sd 0.5 mm) and its one-third sizing level to 1e-9 (tests/test_oracle.py), so it
  tests P1.2 like-for-like. Arms `sized_now` (one fresh sizing at each forecast, estimator reset
  each time) compare current practice, the well-specified filter and the practical filter on the
  same reading. Realised fractions of the filters are reported against both oracles; a fraction
  above 1 against `oracle_depth` means the record (time, survival, repeated sizing) carries growth-rate
  information, which `oracle_state_lambda` bounds.

## D-008 · Second round (23 Oct)

- **Changes.** E-04 (R change with and without an opening sensor) and E-05 (angle sensor with a
  mixed-mode truth and model averaging) are not built; the first report covers Approaches 1–3.

## D-009 · First runs, before the inputs were agreed (24 Oct)

- **Decision.** The student asked for every Approach 1–3 experiment to be run for a supervisor
  report before the PLACEHOLDER inputs were agreed (step 4). The runs use the configured values,
  every one of which is labelled PLACEHOLDER in the report, and reduced specimen counts (100–400
  per variant) because the container has four cores. Rerun at the configured counts once the
  inputs are agreed.
- **Correction.** The first consistency run showed current practice losing 5–15 points of the
  exact-depth benefit at small sizing errors. Cause: `StaticMC` passed the reading to
  `cases.sample` without `undetected=False`, so any reading below 1 mm switched tilt to the
  study's undetected sd (15° instead of 5°). Fixed in 57a031c with a regression test; every run
  holding a `static_mc` arm was repeated after the fix.
- **Scoring note.** Forecasts made when the true crack reaches a study depth are a truth-dependent
  stopping time: they match the study's definition of U90 (sharpness), but coverage there is biased
  low even for a correct forecaster (a crack that reaches 1 mm early is a fast one). Coverage is
  read at random and fixed fractions of each specimen's life, which select far less (the
  well-specified toy covers 89–92 % there), and at fixed cycle counts (check B), which do not
  select at all.

## Experiments

| ID | Config | Question | Result | What it changes |
|---|---|---|---|---|
| E-00 | `00_toy.yaml` | Do coverage and SBC hold when the estimator is well-specified? | 400 specimens: coverage 89–92 % on every arm | machinery trusted |
| E-01 | `01_depth_only.yaml` | Approach 1: realised fraction of P1.1–P1.4 against sizing error, spacing, start depth | oracle within 1.6 points of P1.1 at all 5 depths; current practice 89 % of the depth benefit at sd 0.17 mm (P1.2: 86 %); check B 31 % / 14 % (P1.4: 30–40 % / ~10 %); continuous λ filter removes 83 % at 1 mm but covers 64–85 % (70 % after >30 readings) | Approach 1 needs a model-discrepancy term or the full-NASGRO filter |
| E-01b | `01b_depth_only_bar.yaml` | Same, round bar | oracle within 0.6 points at 4 depths; check B 21 % / 17 % | as E-01 |
| E-01c | `01c_sensor_sweeps.yaml` | Approach 1 under relative error, correlation, shared bias and POD | not yet run | |
| E-02 | `02_intervals.yaml` | Approach 3: when continuous beats periodic per unit cost; robustness to bias and de-bonding | 120 specimens: continuous gives up 0.02 nominal lives vs 0.15–1.06 periodic; λ filter continuous fails 5 % from 1 mm (periodic 0 %), perfect inspection 15 %, perfect information 0 %; silent de-bond 84 % | continuous pays only with an honest filter and a sensor-health check |
| E-03 | `03_residual_stress.yaml` | Approach 2: can depth alone classify tensile / none / compressive? | `a1mm_report`, 100 specimens: 64 % correct by half of life (continuous, sd 0.2 mm), 81 % by 90 %; amplitude and decay length ≤ 6 % (P2.2 holds) | depth alone is not enough |
| E-03b | `03b_residual_forms.yaml` | Approach 2: superposition / BS 7910 uniform / redistribution | not yet run | |
| E-03c | `03c_residual_lcr.yaml` | Approach 2: LCR layer averages, alone and with depth | one survey: 97 / 90 / 83 / 70 % correct at 10 / 20 / 40 / 80 MPa per layer | gate met up to ~40 MPa |
