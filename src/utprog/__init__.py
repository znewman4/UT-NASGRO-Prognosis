"""utprog: does repeated ultrasonic crack sizing improve a NASGRO remaining-life forecast?

Five replaceable parts, joined by the runner:
    truth       draw(rng) -> Specimen            (virtual specimens on validated NASGRO)
    sensors     observe(specimen, times, seed)   (depth sizing with noise, bias, correlation, POD)
    policies    times(specimen, ctx)             (none, one-off, periodic, continuous, event-triggered)
    estimators  reset / update / forecast        (grid Bayes, particle filter, model selection)
    scoring     U90, bias, coverage, CRPS, prognostic horizon, alpha-lambda, SBC, cost
"""
__version__ = "0.1.0"
