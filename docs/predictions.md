# Frozen predictions

Written before any simulation run. The machine-readable copy, with seed ranges and
baselines, is `docs/predictions.yaml`; the efficacy report reads it.

**The comparison.** The sensitivity study says how much forecast spread disappears
if an input is *known exactly*. The simulation measures how much disappears when a
*real sensor and estimator learn it*. Per approach:

    predicted reduction  = 1 − U90(oracle) / U90(study baseline)     (simulation's oracle; must match the study)
    realised reduction   = 1 − U90(arm)    / U90(study baseline)     (same specimens, same depth)
    realised fraction    = realised reduction / predicted reduction

**Definitions** (all predictions): U90 = log10(N95/N05) over sampled cracks that
fail; reduction = 1 − U90(input fixed)/U90(all uncertain) on the same random
numbers; NASGRO-equation inputs only (κ and SIF error at 1), tilt uncertain;
input spreads as `nasgro_scenarios.cases` (sized depth sd 0.5 mm from 1 mm,
lognormal CoV 0.6 below); n = 10,000; mean of seeds 1–5.

## The table

| Approach | Prediction (exact definition) | Engine | Simulation that tests it |
|---|---|---|---|
| 1 · growth rate | **P1.1** Depth known removes U90 at 1 mm: plate 30.3 % [29.2, 31.4], bar 40.0 % [38.9, 40.8]. Other depths in the YAML | rebuild | E-01 / E-01b `consistency`: `oracle_depth` against `study_baseline`, scored when the true crack reaches each depth |
| 1 | **P1.2** Sizing sd 0.167 mm gives 82 % (bar) / 86 % (plate) of the exact-depth reduction at 1 mm | rebuild (seed 1) | E-01 `sized_now` arms: current practice (`static_mc`) at sd 0.167 vs 0.5 mm, against `oracle_depth`; reproduces the study's sweep exactly when the reading is exact |
| 1 | **P1.3** With crack state exact, the λ group (S or P, C–n, α) holds 62–84 % of U90 (plate 62–81, bar 79–84) | original, not committed | E-01 `oracle_state_lambda` (recomputed on the rebuild) |
| 1 | **P1.4** 10 sizings at 0.1 mm cut forecast U90 30–40 % against one sizing; ~10 % at 0.5 mm | original, not committed | E-01 `checkB` (window of 1, 2, 10 sizings over 20 % of life; `is` is check B's same-model setting) |
| 2 · residual stress | **P2.1** Tensile profile shortens 0.2–1 mm life by 0.26–0.40 decades; tensile makes almost all small cracks grow (plate 0.2 mm Pf 80 → 99 %), compressive stops more (bar 0.5 mm Pf 73 → 36 %) | rebuild | E-03 (depth record alone: `bf`, `is`; class-blind vs class-known oracles); E-03c (LCR layer averages + depth) |
| 2 | **P2.2** Given the class, amplitude and decay length ≤ 7 % of U90 each | rebuild | E-03 `oracle_depth_rs` against `oracle_depth_class` |
| 3 · intervals | **P3.1** Undetected crack state leads U90 below 1 mm (plate 0.2 mm 11 %, 0.5 mm 19 %; bar 0.5 mm depth 25 %, level with the material fit 26 %). No interval prediction exists | rebuild | E-02 `a0.2mm`, `a0.5mm`, with perfect-inspection and perfect-information arms as the reference |
| 4 · opening | **P4.1** α ≤ 4 % (8–12 % plate, 5–9 % bar if its spread doubles); a rate learned at R = 0.1 leaves U90 1.9 at 4 mm after R rises to 0.5 | rebuild / original | E-04, second round (not built) |
| 5 · scattering | **P5.1** Tilt ≤ 1.5 % once sized to ±5°; 16 % (bar) / 8 % (plate) at 4 mm if inclined (30°) and unsized | rebuild | E-05, second round (not built) |
| 6 · joint | No prediction | | left out of this report |

## Provenance

- **Rebuild** (`nasgro_scenarios`, commit `28f217c`, pinned and hash-locked in
  `external/`). Test 3 reproduces its stored statistics to 1e-9, and
  `tests/test_oracle.py` reproduces its stored U90 reductions from the oracle
  arm to 1e-9. These predictions are fully reproducible.
- **Original** (`nasgro_sa`, branch `main`, commit `f9a6555`). Checks A and B on
  the scoping page were rerun on it with κ and the SIF error fixed at 1, but no
  commit holds that rerun, and its scatter is narrower than the rebuild's
  (σ log10 N 0.27 vs 0.36 bar). P1.3 and P1.4 are therefore compared as
  reductions and ratios, and the simulation recomputes them on the rebuild.
  If the rerun scripts are found, they should be committed and pinned.
- The scoping HTML in the uploads is an older (κ-era) version; the numbers here
  follow the 23 October NASGRO-only report and the live scoping page.
