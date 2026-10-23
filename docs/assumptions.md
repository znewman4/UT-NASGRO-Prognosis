# Assumptions

Every modelling choice the results depend on, with its source and status.
**Status:** *cited* (taken from the source), *pipeline* (inherited unchanged from the validated
NASGRO pipeline), *placeholder* (a working value to agree with the supervisors **before** the runs),
or *approximation* (a known simplification, with its consequence).

## 1. Truth: virtual specimens

| # | Assumption | Status | Source |
|---|---|---|---|
| T1 | Growth law: NASGRO equation with Newman closure and the NASGRO threshold with El Haddad length; integrated in depth with the crack shape held fixed | pipeline | `nasgro_scenarios` README; Forman & Mettu (1992); Newman (1984) |
| T2 | Two cases and their input spreads: plate (Newman–Raju SIF, t 10 mm, R 0.1, 200 MPa) and round bar (Forman–Shivakumar SIF, D 18 mm, R −1, 250 MPa), with the study's typical spreads | pipeline | `nasgro_scenarios.cases` |
| T3 | NASGRO-only inputs: κ and the SIF-solution error fixed at 1 (not terms of the NASGRO equation) | pipeline rule | `nasgro_scenarios.runner.NON_NASGRO` |
| T4 | Crack tilt fixed at 0 in M1–M2 (mixed mode is M3) | placeholder | plan, Approach 5 |
| T5 | Start depth: lognormal, CoV 0.3, truncated to ×/÷ 4 of the median; median swept 0.2–2 mm | placeholder | plan (starting-size sweep) |
| T6 | Truth is deterministic NASGRO once inputs are drawn; optional path variability: rate × exp(sd·Z − sd²/2), Z an AR(1) in log depth | approximation (variant `a1mm_path_variability`) | Yang & Manning (1996); scatter of replicate tests: Virkler et al. (1979) |
| T7 | Residual stress: K_res(a) superposed on Kmax and Kmin (ΔK unchanged, R_eff moves) by the Shen & Glinka weight function; profile A σy (1 − z/L) e^(−z/L) or uniform A σy | pipeline | `nasgro_scenarios.residual_stress`; Parker (1982); LaRue & Daniewicz (2007); Shen & Glinka (1991); BS 7910:2019 |
| T8 | Residual-stress classes and ranges: tensile = `plate_weld` (A ~ U(0.5, 1.0), L ~ 2.5 mm); compressive A ~ U(−0.8, −0.4), L ~ 0.8 mm, first-cycle shakedown | placeholder (tensile from the pipeline) | engineering judgement anchored on BS 7910 practice |
| T9 | Redistribution: K_res,eff = K_res [r∞ + (1 − r∞) e^(−(a − a_ref)/ℓ)], r∞ = 0.4, ℓ = 1 mm | placeholder, phenomenological | motivated by Edwards (2006), who attributes early/late life misprediction to redistribution |
| T10 | Specimens whose crack never grows are kept: scored for the arrest forecast and classification over one nominal life; RUL scores use failing cracks, as the study's U90 does | design choice (D-006) | |
| T11 | Mismatched truth (`extended: true`): κ (CoV 0.1) and the SIF error (CoV 0.05) uncertain in the truth only, plus rate wander along the path | design choice | the study's previous ("extended") inputs; Yang & Manning (1996) |
| T12 | Check-B replica start depth: the study's sized-depth model, normal(1 mm, 0.5 mm) reflected at 20 µm (`dist: study`) | pipeline | `nasgro_scenarios.cases._depth` |

## 2. Sensor: ultrasonic depth sizing (`sensors/depth.py`)

y(N) = a(N) + b_mean + b + sd(a)·e(N); see the module docstring for the full model.

| # | Parameter | Value in configs | Status | Source / to extract from |
|---|---|---|---|---|
| S1 | Sizing sd, `sd_abs_mm` | swept 0.05–0.5 mm; 0.2 mm study target, 0.5 mm typical | placeholder range | study document (±0.5 mm, as in `nasgro_scenarios.cases._depth`); **extract small-crack sizing error from** Peng et al. (2018) and Felice et al. (2014) |
| S2 | Depth-proportional sd, `sd_rel` | 0 | placeholder | Peng et al. (2018): image-based (TFM) sizing works for cracks larger than about two wavelengths, so error grows as cracks shrink; to be parameterised |
| S3 | Mean bias, `bias_mean_mm` | 0 | placeholder | Felice et al. (2014); Peng et al. (2018) |
| S4 | Per-installation bias, `bias_sd_mm` | 0; 0.1 mm in `a1mm_correlated` | placeholder | to agree (transducer placement, coupling) |
| S5 | Error correlation, `corr_frac` | 0; 0.05 of nominal life in `a1mm_correlated` | placeholder | to agree; matters most for continuous monitoring, where independent errors would average away unrealistically |
| S6 | POD, log-normal hit/miss, `pod_a50_mm`, `pod_a90_mm` | 0.15, 0.30 mm | placeholder form cited | MIL-HDBK-1823A (2009) model form; values to come from the chosen technique |
| S7 | Detection floor `a_floor_mm`, sizing limit `a_size_min_mm` | 0.05, 0.10 mm | placeholder | to agree |
| S8 | Relative sizing error `sd_rel` | 0 in E-01; swept 0, 0.11, 0.19 in E-01c | placeholder anchored on a source | Peng et al. (2018), tested errors on real fatigue cracks: image sizing < 11 %, scattering matrix < 19 %. These are maximum errors on tested cases, not standard deviations |
| S9 | Bond failure of a permanently bonded array: silent loss at lognormal time, median 0.4 × nominal life, CoV 0.5 | E-02 `bond0.2` | placeholder anchored on a source | An et al. lost a bonded array after about 10,000 cycles in a fatigue test (scoping page ref. [26]) |
| S10 | Undersizing bias unknown to the filter: −0.1 mm | E-02 `biased0.2` with `grid_unaware` | placeholder | scoping page: biased monitoring data can delay an inspection unsafely |
| S11 | LCR residual-stress survey: layer averages at 1, 2, 4, 6 mm, error sd 10–80 MPa per layer plus 20 MPa shared per survey, once at installation | E-03c | placeholder anchored on a source | Javadi et al.: 5 to 1 MHz sampled about 1 to 6 mm, agreement with FE within ±40 MPa (scoping page ref. [32]) |

## 3. Estimators

| # | Assumption | Status | Consequence |
|---|---|---|---|
| E1 | Only the lumped multiplier λ on C (and the start depth a0) is inferred from depth; λ is reported as C_eff = λ C_nominal | design choice | Growth cannot be separated from stress and closure with depth alone (plan, Approach 1 "what could kill it") |
| E2 | λ prior: lognormal, from the NASGRO input distributions over [1, 3] × start depth (median and central-68 % half-width of log λ) | design choice | well-specified on average; a specimen's curve shape still differs from the nominal shape (model error, measured by coverage) |
| E3 | Estimator noise model = the sensor's, unless `noise_override` | design choice | M1 tables are for a correctly specified sensor model |
| E4 | Misses and censored readings independent of the Gaussian errors | approximation | slight over-confidence when errors are correlated and many readings are censored |
| E5 | Residual-stress hypotheses fix the decay length L at its median; amplitude on table nodes (midpoint rule for the uniform prior) | approximation | truth draws L lognormal (CoV 0.3): model error inside each hypothesis |
| E6 | Rates below 1e-40 m/cycle count as arrest | approximation | none in practice |
| E7 | Tables interpolate log rate and Kmax linearly between axis nodes | approximation | bounded by test 5 (0.02 decades) |
| E8 | Oracle and study-baseline arms (`study_mc`) are the sensitivity engine at the true current depth, with known inputs at their true values; tilt uncertain as in the study | design choice (D-005) | they reproduce the stored numbers to 1e-9 (tests/test_oracle.py); they see the truth and are references, not estimators |
| E9 | `nasgro_is` draws its 4,000 particles from the truth's own priors (well-specified for a NASGRO-only truth) | design choice | weights degenerate with many precise readings; ESS is recorded with every forecast |
| E11 | Filters condition every forecast on the component being in service at the forecast time | design choice (D-008a) | survival is information the static study does not use |
| E12 | `static_mc` (current practice) takes the latest sized reading as the current depth, with its sizing sd, reflected at 20 µm; before any sized reading, the study's undetected model around the sizing limit | design choice (D-008b) | reproduces the study's sizing sweep to 1e-9 |
| E10 | Grid refinement triggers at 1.5 nodes per posterior sd | approximation | Riemann sums of a Gaussian at this spacing are accurate to better than 0.1 %; the jitter within a cell adds at most ~2 % to the posterior sd |

## 4. Scoring

| # | Assumption | Source |
|---|---|---|
| R1 | Forecasts scored at random times U(0, 0.95 × life) common to all arms, so the last periodic reading is 0 to T old | plan (fair periodic scoring) |
| R2 | U90 in decades, as the sensitivity study; coverage of central 90 % with Wilson intervals | study; Wilson (1927) |
| R3 | CRPS on log10 RUL as the proper overall score | Gneiting & Raftery (2007) |
| R4 | α-λ accuracy (α = 0.2) and prognostic horizon (±0.05 × life band), β = 0.5 | Saxena et al. (2010) |
| R5 | SBC on log λ with 99 draws, toy only (where the estimator is well-specified) | Talts et al. (2018) |
| R6 | Decision rule: act when P(failure before next planned reading + lead) ≥ p_act; costs in units of one repair and one nominal life | after Straub & Faber (2006); cost values placeholder |
| R7 | Model comparison by evidence; 2 ln BF > 6 "strong" | Kass & Raftery (1995) |

## References

- BSI (2019) *BS 7910:2019 Guide to methods for assessing the acceptability of flaws in metallic structures*. British Standards Institution.
- Chopin, N. (2002) A sequential particle filter method for static models. *Biometrika* 89(3), 539–551. [doi:10.1093/biomet/89.3.539](https://doi.org/10.1093/biomet/89.3.539)
- Edwards, L. (2006) Influence of residual stress redistribution on fatigue crack growth and damage tolerant design. *Materials Science Forum* 524–525, 363–372. [doi:10.4028/www.scientific.net/MSF.524-525.363](https://doi.org/10.4028/www.scientific.net/MSF.524-525.363) ([open copy](https://oro.open.ac.uk/16012))
- Felice, M.V., Velichko, A. & Wilcox, P.D. (2014) Accurate depth measurement of small surface-breaking cracks using an ultrasonic array post-processing technique. *NDT & E International* 68, 105–112. [doi:10.1016/j.ndteint.2014.08.004](https://doi.org/10.1016/j.ndteint.2014.08.004)
- Forman, R.G. & Mettu, S.R. (1992) Behavior of surface and corner cracks subjected to tensile and bending loads in Ti-6Al-4V alloy. In *Fracture Mechanics: 22nd Symposium*, ASTM STP 1131, 519–546.
- Gneiting, T. & Raftery, A.E. (2007) Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477), 359–378. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
- Kass, R.E. & Raftery, A.E. (1995) Bayes factors. *JASA* 90(430), 773–795. [doi:10.1080/01621459.1995.10476572](https://doi.org/10.1080/01621459.1995.10476572)
- LaRue, J.E. & Daniewicz, S.R. (2007) Predicting the effect of residual stress on fatigue crack growth. *International Journal of Fatigue* 29(3), 508–515. [doi:10.1016/j.ijfatigue.2006.05.008](https://doi.org/10.1016/j.ijfatigue.2006.05.008)
- Newman, J.C. Jr. (1984) A crack opening stress equation for fatigue crack growth. *International Journal of Fracture* 24, R131–R135. [doi:10.1007/BF00020751](https://doi.org/10.1007/BF00020751)
- Parker, A.P. (1982) Stress intensity factors, crack profiles, and fatigue crack growth rates in residual stress fields. In *Residual Stress Effects in Fatigue*, ASTM STP 776, 13–31.
- Peng, C., Bai, L., Zhang, J. & Drinkwater, B.W. (2018) The sizing of small surface-breaking fatigue cracks using ultrasonic arrays. *NDT & E International* 99, 64–71. [doi:10.1016/j.ndteint.2018.06.005](https://doi.org/10.1016/j.ndteint.2018.06.005)
- Roberts, G.O. & Rosenthal, J.S. (2001) Optimal scaling for various Metropolis–Hastings algorithms. *Statistical Science* 16(4), 351–367. [doi:10.1214/ss/1015346320](https://doi.org/10.1214/ss/1015346320)
- Saxena, A., Celaya, J., Saha, B., Saha, S. & Goebel, K. (2010) Metrics for offline evaluation of prognostic performance. *International Journal of Prognostics and Health Management* 1(1), 001. [NASA NTRS 20100039169](https://ntrs.nasa.gov/citations/20100039169)
- Shen, G. & Glinka, G. (1991) Weight functions for a surface semi-elliptical crack in a finite thickness plate. *Theoretical and Applied Fracture Mechanics* 15(3), 247–255. [doi:10.1016/0167-8442(91)90023-D](https://doi.org/10.1016/0167-8442(91)90023-D) (valid for 0.2 ≤ a/c ≤ 1, the range `nasgro_scenarios.cases` clips a/c to)
- Straub, D. & Faber, M.H. (2006) Computational aspects of risk-based inspection planning. *Computer-Aided Civil and Infrastructure Engineering* 21(3), 179–192. [doi:10.1111/j.1467-8667.2006.00426.x](https://doi.org/10.1111/j.1467-8667.2006.00426.x)
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. (2018) Validating Bayesian inference algorithms with simulation-based calibration. [arXiv:1804.06788](https://arxiv.org/abs/1804.06788)
- U.S. Department of Defense (2009) *MIL-HDBK-1823A: Nondestructive Evaluation System Reliability Assessment*.
- Virkler, D.A., Hillberry, B.M. & Goel, P.K. (1979) The statistical nature of fatigue crack propagation. *Journal of Engineering Materials and Technology* 101(2), 148–153. [doi:10.1115/1.3443666](https://doi.org/10.1115/1.3443666)
- Wilson, E.B. (1927) Probable inference, the law of succession, and statistical inference. *JASA* 22(158), 209–212. [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
- Yang, J.N. & Manning, S.D. (1996) A simple second order approximation for stochastic crack growth analysis. *Engineering Fracture Mechanics* 53(5), 677–686.

**Reference check (23 Oct 2026).** Newman (1984) is confirmed against Springer's record. LaRue &
Daniewicz (2007) and Shen & Glinka (1991) were corrected (DOI suffix .008, not .006; the semi-elliptical
weight-function paper, pp. 247–255, not the reference-SIF paper) and match their ScienceDirect /
Semantic Scholar listings. Straub & Faber (2006) and Chopin (2002) are confirmed bibliographically
(journal, volume, issue, pages), but their DOI strings could not be resolved from this environment
(doi.org and Crossref are blocked by its network policy): confirm those two at doi.org before the report.
The others were checked against the publisher or repository record.
