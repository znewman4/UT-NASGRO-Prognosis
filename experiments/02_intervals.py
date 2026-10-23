"""E-02 · Approach 3 (intervals): break-even monitoring cost and sizing error, continuous vs periodic.

For every sensor, estimator, decision threshold p_act, periodic spacing and failure cost:
the expected cost of periodic inspection, and the monitoring cost per nominal life at which
continuous monitoring with the same sensor and estimator costs the same (break-even). Then,
for each monitoring cost level, the sizing sd below which continuous pays. A cost table for
every arm (including perfect inspection, perfect information, the biased sensor and the
de-bonding array) gives the references and the robustness cases.

    python experiments/02_intervals.py [configs/02_intervals.yaml]
"""
import json
import sys

import numpy as np
import pandas as pd

from _bootstrap import ROOT, SUMMARY

from utprog import runner
from utprog.scoring import cost


def arm_costs(dec, cfg):
    c = cfg["costs"]
    rows = []
    for failure in c.get("failure_levels", [c["failure"]]):
        common = dict(repair=c["repair"], failure=failure, unused_per_life=c["unused_per_life"])
        for (arm, policy, p_act), d in dec.groupby(["arm", "policy", "p_act"]):
            kind = "continuous" if policy == "continuous" else "periodic"
            e, se = cost.expected(d, cost.Costs(per_reading=c[kind]["per_reading"], **common))
            rows.append(dict(arm=arm, p_act=p_act, failure_cost=failure, cost_excl_monitoring=e, se=se,
                             fail_rate=d["failed"].mean(), unused=d["unused_frac"].mean(),
                             readings=d["n_readings"].mean()))
    return pd.DataFrame(rows)


def tables(dec, cfg):
    c = cfg["costs"]
    sd = {k: v.get("sd_abs_mm") for k, v in cfg["sensors"].items()}
    rows = []
    for failure in c.get("failure_levels", [c["failure"]]):
        common = dict(repair=c["repair"], failure=failure, unused_per_life=c["unused_per_life"])
        c_per = cost.Costs(per_reading=c["periodic"]["per_reading"], **common)
        c_cont = cost.Costs(per_reading=c["continuous"]["per_reading"], **common)
        rows += _break_even(dec, c_per, c_cont, sd, failure)
    be = pd.DataFrame(rows)
    out = []
    for (est, pol, p_act, failure), d in be.groupby(["estimator", "periodic", "p_act", "failure_cost"]):
        for level in c["monitoring_per_life_levels"]:
            out.append(dict(estimator=est, periodic=pol, p_act=p_act, failure_cost=failure,
                            monitoring_per_life=level,
                            break_even_sd_mm=cost.break_even_sd(d["sd_mm"], d["break_even_per_life"], level)))
    return be, pd.DataFrame(out)


def _break_even(dec, c_per, c_cont, sd, failure):
    rows = []
    for (sensor, est, p_act), d in dec.groupby(["sensor", "estimator", "p_act"]):
        cont = d[d["policy"] == "continuous"]
        if cont.empty or sd.get(sensor) is None:
            continue
        for pol, per in d[d["policy"].str.startswith("periodic")].groupby("policy"):
            e, se = cost.expected(per, c_per)
            rows.append(dict(sensor=sensor, estimator=est, sd_mm=sd[sensor], p_act=p_act, failure_cost=failure,
                             periodic=pol, cost_periodic=e,
                             cost_periodic_se=se, fail_periodic=per["failed"].mean(),
                             fail_continuous=cont["failed"].mean(),
                             unused_periodic=per["unused_frac"].mean(), unused_continuous=cont["unused_frac"].mean(),
                             break_even_per_life=cost.break_even_per_life(cont, per, c_cont, c_per)))
    return rows


def main():
    done = runner.main(sys.argv[1:], default_config=str(ROOT / "configs" / "02_intervals.yaml"))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for name, cfg, out in done:
        dec = pd.read_csv(out / "decisions.csv.gz")
        arm_costs(dec, cfg).assign(run_dir=str(out)).to_csv(SUMMARY / f"02_intervals_{name}_arm_costs.csv", index=False)
        be, sd_table = tables(dec, cfg)
        be, sd_table = be.assign(run_dir=str(out)), sd_table.assign(run_dir=str(out))
        be.to_csv(out / "break_even.csv", index=False)
        sd_table.to_csv(out / "break_even_sd.csv", index=False)
        be.to_csv(SUMMARY / f"02_intervals_{name}_break_even.csv", index=False)
        sd_table.to_csv(SUMMARY / f"02_intervals_{name}_break_even_sd.csv", index=False)
        levels = cfg["costs"]["monitoring_per_life_levels"]
        exists = bool(((be["break_even_per_life"] > 0) & (be["break_even_per_life"] <= max(levels))).any())
        (SUMMARY / f"02_intervals_{name}_gates.json").write_text(json.dumps(
            dict(placeholder=True, break_even_inside_range=exists, run_dir=str(out)), indent=1))
        print(be.to_string(index=False))
        print(sd_table.replace(np.inf, "pays at every sd").to_string(index=False))


if __name__ == "__main__":
    main()
