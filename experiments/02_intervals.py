"""02 · M1, Approach 4: break-even monitoring cost and sizing error, continuous vs periodic.

For every sizing sd, decision threshold p_act and periodic spacing: the expected
cost of periodic inspection, and the monitoring cost per nominal life at which
continuous monitoring costs the same (break-even). Then, for each monitoring
cost level, the sizing sd below which continuous pays.

    python experiments/02_intervals.py [configs/02_intervals.yaml]
"""
import json
import sys

import numpy as np
import pandas as pd

from _bootstrap import ROOT, SUMMARY

from utprog import runner
from utprog.scoring import cost


def tables(dec, cfg):
    c = cfg["costs"]
    common = dict(repair=c["repair"], failure=c["failure"], unused_per_life=c["unused_per_life"])
    c_per = cost.Costs(per_reading=c["periodic"]["per_reading"], **common)
    c_cont = cost.Costs(per_reading=c["continuous"]["per_reading"], **common)
    sd = {k: v["sd_abs_mm"] for k, v in cfg["sensors"].items()}
    rows = []
    for (sensor, p_act), d in dec.groupby(["sensor", "p_act"]):
        cont = d[d["policy"] == "continuous"]
        for pol, per in d[d["policy"].str.startswith("periodic")].groupby("policy"):
            e, se = cost.expected(per, c_per)
            rows.append(dict(sensor=sensor, sd_mm=sd[sensor], p_act=p_act, periodic=pol, cost_periodic=e,
                             cost_periodic_se=se, fail_periodic=per["failed"].mean(),
                             fail_continuous=cont["failed"].mean(),
                             unused_periodic=per["unused_frac"].mean(), unused_continuous=cont["unused_frac"].mean(),
                             break_even_per_life=cost.break_even_per_life(cont, per, c_cont, c_per)))
    be = pd.DataFrame(rows)
    out = []
    for (pol, p_act), d in be.groupby(["periodic", "p_act"]):
        for level in c["monitoring_per_life_levels"]:
            out.append(dict(periodic=pol, p_act=p_act, monitoring_per_life=level,
                            break_even_sd_mm=cost.break_even_sd(d["sd_mm"], d["break_even_per_life"], level)))
    return be, pd.DataFrame(out)


def main():
    done = runner.main(sys.argv[1:], default_config=str(ROOT / "configs" / "02_intervals.yaml"))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for name, cfg, out in done:
        be, sd_table = tables(pd.read_csv(out / "decisions.csv.gz"), cfg)
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
