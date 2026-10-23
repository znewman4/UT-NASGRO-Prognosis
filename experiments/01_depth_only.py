"""E-01 · Approach 1 (growth rate): depth-only test bed on validated NASGRO.

Runs every variant, then writes:
  <exp>_table.csv          U90, bias and coverage against sizing sd, spacing and starting size
  <exp>_oracle_check.csv   oracle U90 reduction against the frozen prediction P1.1 (must pass first)
  <exp>_efficacy.csv       predicted (oracle), realised and realised fraction per arm and depth
  <exp>_checkB.csv         check-B replica (variant checkB): U90 after 1, 2, 10 sizings
  <exp>_gates.csv/.json    continuous vs periodic at the same sd

    python experiments/01_depth_only.py [configs/01_depth_only.yaml] [--variant a1mm] [--n 200]
    python experiments/01_depth_only.py configs/01b_depth_only_bar.yaml
"""
import json
import sys

import pandas as pd

from _bootstrap import ROOT, SUMMARY

from utprog import runner
from utprog.scoring import efficacy


def gate_table(summary, g):
    rows = []
    s = summary[summary["kind"] == "random"]
    for (variant, sensor), d in s.groupby(["variant", "sensor"], sort=False):
        cont = d[(d["policy"] == g["continuous_policy"]) & (d["estimator"] == "grid")]
        if cont.empty:
            continue
        c = cont.iloc[0]
        cov_ok = g["coverage_band"][0] <= c["coverage"] <= g["coverage_band"][1]
        for _, r in d[d["policy"].str.startswith("periodic") & (d["estimator"] == "grid")].iterrows():
            red = 1.0 - c["u90_mean"] / r["u90_mean"]
            rows.append(dict(variant=variant, sensor=sensor, periodic=r["policy"], u90_periodic=r["u90_mean"],
                             u90_continuous=c["u90_mean"], u90_reduction=red,
                             coverage_periodic=r["coverage"], coverage_continuous=c["coverage"],
                             passes=bool(red >= g["u90_reduction_min"] and cov_ok)))
    return pd.DataFrame(rows)


def main():
    done = runner.main(sys.argv[1:], default_config=str(ROOT / "configs" / "01_depth_only.yaml"))
    if not done:
        return
    cfg0 = done[0][1]
    exp, case, g = cfg0["experiment"], cfg0["truth"]["case"], cfg0["gates"]
    tables, effs, checks, checkb = [], [], [], []
    for name, cfg, out in done:
        sd = {k: v.get("sd_abs_mm") for k, v in cfg["sensors"].items()}
        s = pd.read_csv(out / "summary.csv")
        tables.append(s.assign(variant=name, a_start_mm=cfg["truth"]["a_start"]["median_mm"],
                               sd_mm=s["sensor"].map(sd), run_dir=str(out)))
        if (out / "summary_depth.csv").is_file():
            d = pd.read_csv(out / "summary_depth.csv")
            for ref in ("oracle_depth", "oracle_state_lambda"):
                effs.append(efficacy.efficacy_table(d, oracle=ref).assign(variant=name, run_dir=str(out)))
            checks.append(efficacy.oracle_check(d, case, tolerance=g["oracle_tolerance"]).assign(variant=name))
        if cfg["scoring"].get("at_time_frac"):
            checkb.append(efficacy.check_b_table(pd.read_csv(out / "records.csv.gz")).assign(variant=name))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(SUMMARY / f"{exp}_table.csv", index=False)
    for part, rows in (("efficacy", effs), ("oracle_check", checks), ("checkB", checkb)):
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(SUMMARY / f"{exp}_{part}.csv", index=False)
    gates = gate_table(table, g)
    gates.to_csv(SUMMARY / f"{exp}_gates.csv", index=False)
    oracle_ok = bool(pd.concat(checks)["passes"].all()) if checks else None
    (SUMMARY / f"{exp}_gates.json").write_text(json.dumps(
        dict(thresholds=g, oracle_matches_prediction=oracle_ok,
             any_pass=bool(gates["passes"].any()) if len(gates) else None,
             runs={n: str(o) for n, _, o in done}), indent=1))
    if checks:
        print(pd.concat(checks).to_string(index=False))
    if oracle_ok is False:
        print("ORACLE DOES NOT REPRODUCE THE PREDICTION: fix the definition mismatch before reading any efficacy.")


if __name__ == "__main__":
    main()
