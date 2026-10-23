"""Approach 3: turn forecast accuracy into inspection and repair decisions.

Decision rule (risk-based, after Straub & Faber 2006): after each reading, act
(repair or retire) if the forecast probability of failing before the next
planned reading plus a lead time is at least p_act. Simulated per specimen in
runner.decide_arm; each record has n_readings, service (cycles until action or
failure), acted, failed and unused (true RUL left at action), all also as
fractions of the nominal life.

Expected cost per specimen, in units where the nominal life is 1:
    per_reading x n_readings + per_life x service_frac + repair x acted
    + failure x failed + unused_per_life x unused_frac
per_reading is the periodic inspection cost; per_life the cost of keeping a
continuous system installed for one nominal life. The break-even monitoring
cost is the per_life at which continuous and periodic cost the same.
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Costs:
    per_reading: float = 0.0
    per_life: float = 0.0
    repair: float = 1.0
    failure: float = 100.0
    unused_per_life: float = 0.0


def total(dec, c):
    return (c.per_reading * dec["n_readings"] + c.per_life * dec["service_frac"] + c.repair * dec["acted"]
            + c.failure * dec["failed"] + c.unused_per_life * dec["unused_frac"])


def expected(dec, c):
    x = total(dec, c).to_numpy(float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan


def break_even_per_life(cont, periodic, c_cont, c_periodic):
    """Monitoring cost per nominal life at which continuous costs the same as periodic."""
    e_per, _ = expected(periodic, c_periodic)
    e_cont, _ = expected(cont, replace(c_cont, per_life=0.0))
    return (e_per - e_cont) / float(cont["service_frac"].mean())


def break_even_sd(sds, cstar, c_monitor):
    """Sizing sd below which continuous pays at monitoring cost c_monitor (cstar decreasing in sd).

    Returns nan if continuous never pays, inf if it pays at every sd in the range.
    """
    sds, cstar = np.asarray(sds, float), np.asarray(cstar, float)
    order = np.argsort(sds)
    sds, cstar = sds[order], cstar[order]
    pays = cstar >= c_monitor
    if not pays.any():
        return np.nan
    if pays.all():
        return np.inf
    k = np.flatnonzero(pays[:-1] & ~pays[1:])
    if k.size == 0:
        return np.nan
    k = k[0]
    return float(np.interp(c_monitor, [cstar[k + 1], cstar[k]], [sds[k + 1], sds[k]]))
