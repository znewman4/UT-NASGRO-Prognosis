# Glossary

| Term | Meaning here |
|---|---|
| a, a0 | Crack depth (m in code, mm in configs and tables); a0 is the true depth at N = 0, the start of monitoring |
| λ (lambda) | Lumped growth-rate multiplier, an exact factor on NASGRO's C. Internal to the filters; reported as C_eff |
| C_eff | λ × C_nominal: the effective Paris coefficient that absorbs stress, closure and threshold differences depth cannot separate |
| Arm | One combination of sensor × policy × estimator, run on every specimen |
| CRN | Common random numbers: every arm sees the same specimens, sensor error field and forecast draws |
| U90 | Width in decades of the central 90 % forecast interval of remaining life (as in the sensitivity study) |
| Coverage | Share of forecasts whose central 90 % interval contains the true remaining life |
| Bias | log10(forecast median) − log10(true remaining life) |
| CRPS | Continuous ranked probability score of log10 remaining life; lower is better; proper |
| PIT | P(forecast < truth); uniform across forecasts if calibrated |
| α-λ accuracy | Pass if ≥ β of the forecast lies within ±α of the true RUL at a given life fraction (Saxena et al. 2010) |
| Prognostic horizon | Time before failure from which every forecast keeps ≥ β of its mass within ±α × life of the truth |
| SBC | Simulation-based calibration: rank of the true parameter among posterior draws is uniform if inference is correct |
| POD, a50, a90 | Probability of detection; depths detected with probability 0.5 and 0.9 (log-normal hit/miss model) |
| Censored reading | Detected, but below the sizing limit, so only "smaller than a_size_min" is known |
| OU error | Ornstein–Uhlenbeck (continuous-time AR(1)) sizing error, correlated over `corr_cycles` |
| IBIS | Iterated batch importance sampling: particle filter for static parameters with resample–move steps |
| Evidence, Bayes factor | p(record | hypothesis); ratio between two hypotheses. 2 ln BF > 6 is "strong" |
| Gate | A pre-agreed threshold that decides whether an approach continues |
| Manifest | `manifest.json` beside each run: config hash, commit, pipeline commit, versions, seed, date, runtime |
| Oracle arm | `study_mc` with some inputs set to the specimen's true values: the sensitivity engine with perfect information. A reference, not an estimator |
| Study baseline | `study_mc` with nothing known: the sensitivity study's own population at the true current depth (depth spread sd 0.5 mm, or CoV 0.6 below 1 mm) |
| Predicted reduction | 1 − U90(oracle) / U90(study baseline), at the same depth; must match the frozen prediction |
| Realised reduction, realised fraction | 1 − U90(arm) / U90(study baseline); its ratio to the predicted reduction. The approach's efficacy |
| Check A, check B | Scoping-page checks on the original pipeline: the λ group's share of U90 with crack state known (A); U90 after 1, 2, 10 sizings over 20 % of life (B) |
| ESS | Effective sample size of importance weights, 1 / Σw²; recorded with every `nasgro_is` forecast |
| Brier score (arrest) | Mean of (p_arrest − arrested)²: how well the forecast says whether the crack will ever fail |
| LCR | Longitudinal critically refracted waves: travel time gives the residual stress averaged over a near-surface layer |
