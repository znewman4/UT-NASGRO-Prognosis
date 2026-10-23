# Decisions

One entry per design decision or experiment: the question, the result, and what it changes.
Experiment entries are written after the run, from its `results/summary/` tables and manifest.

## D-001 · Which package is the validated `nasgro_sa` pipeline (M0)

- **Question.** The plan names `nasgro_sa` as a read-only dependency. Which code is it?
- **Result.** The original `nasgro_sa` is not in any repository available to this project.
  `znewman4/NASGRO-Sensitivity-Analysis` contains `nasgro_scenarios`, which rebuilds it and is
  checked against the study's published 1 mm statistics (its README, `results/baseline.json`).
- **Changes.** `external/nasgro_sa` is that repository as a git submodule pinned at
  `7bf96977`; `external/nasgro_sa.lock.json` holds SHA-256 hashes of every module and of the
  stored results used for regression. `utprog._external` verifies them on every import and
  refuses an edited copy. If the original `nasgro_sa` turns up, it replaces the submodule and
  test 3 is rerun against it.

## D-002 · λ is an exact factor on C; other effects are table axes (M0)

- **Question.** How should the filter's lumped multiplier relate to NASGRO's variables?
- **Result.** NASGRO's rate is proportional to C, so defining λ on C makes N ∝ 1/λ exact.
  Defining it through stress would not be (threshold and Kc terms do not scale); test 5
  characterises that gap.
- **Changes.** Tables are computed at λ = 1 and scaled; stress, residual-stress amplitude or
  other NASGRO variables enter as table axes when a hypothesis needs them. Results are reported
  as C_eff = λ C_nominal.

## D-003 · Grid estimator with adaptive refinement, particle filter as the check (M1)

- **Question.** Continuous monitoring gives hundreds of readings; the posterior becomes far
  narrower than any fixed grid spacing.
- **Result.** The likelihood (OU errors plus a per-specimen bias) has exact sequential
  sufficient statistics, so the grid can be rebuilt around the posterior and the record
  replayed at O(points × readings) cost.
- **Changes.** `GridBayes` is the main estimator (and gives the evidence for Bayes factors);
  `ParticleFilter` (IBIS, Chopin 2002) is kept to cross-check it (test 4).

## D-004 · Fair scoring of periodic inspection (M1)

- **Question.** The toy scored "every 20k" at an inspection time, flattering periodic.
- **Changes.** Forecasts are scored at random times U(0, 0.95 × life), drawn per specimen
  from its own stream and shared by every arm; the last reading's age is recorded.

## D-005 · Gate thresholds are placeholders (M1, M2)

- **Changes.** Every `gates:` block carries `placeholder: true`. Thresholds are to be agreed
  with the supervisors before the runs and edited here with the date agreed.

## Experiments

| ID | Config | Question | Result | What it changes |
|---|---|---|---|---|
| E-00 | `00_toy.yaml` | Do coverage and SBC hold when the estimator is well-specified? | not yet run | |
| E-01 | `01_depth_only.yaml` | How accurate must sizing be before repeated sizing improves the forecast? | not yet run | |
| E-01b | `01b_depth_only_bar.yaml` | Same, round bar | not yet run | |
| E-02 | `02_intervals.yaml` | When does continuous beat periodic per unit cost? | not yet run | |
| E-03 | `03_residual_stress.yaml` | Can depth alone classify tensile / none / compressive? | not yet run | |
| E-03b | `03b_residual_forms.yaml` | Can depth distinguish superposition / BS 7910 uniform / redistribution? | not yet run | |
