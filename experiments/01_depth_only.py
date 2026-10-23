"""01 · M1, Approach 1: depth-only test bed on validated NASGRO.

Runs every variant (starting size, correlated errors, path variability), then
writes one table of U90, bias and coverage against sizing sd, spacing and
starting size, and the decision gate (continuous vs periodic at the same sd).

    python experiments/01_depth_only.py [configs/01_depth_only.yaml] [--variant a1mm] [--n 200]
    python experiments/01_depth_only.py configs/01b_depth_only_bar.yaml
"""
import json
import sys

import pandas as pd

from _bootstrap import ROOT, SUMMARY

from utprog import runner


def gate_table(summary, g):
    rows = []
    for (variant, sensor), d in summary.groupby(["variant", "sensor"], sort=False):
        cont = d[d["policy"] == g["continuous_policy"]]
        if cont.empty:
            continue
        c = cont.iloc[0]
        cov_ok = g["coverage_band"][0] <= c["coverage"] <= g["coverage_band"][1]
        for _, r in d[d["policy"].str.startswith("periodic")].iterrows():
            red = 1.0 - c["u90_median"] / r["u90_median"]
            rows.append(dict(variant=variant, sensor=sensor, periodic=r["policy"], u90_periodic=r["u90_median"],
                             u90_continuous=c["u90_median"], u90_reduction=red,
                             coverage_periodic=r["coverage"], coverage_continuous=c["coverage"],
                             passes=bool(red >= g["u90_reduction_min"] and cov_ok)))
    return pd.DataFrame(rows)


def main():
    done = runner.main(sys.argv[1:], default_config=str(ROOT / "configs" / "01_depth_only.yaml"))
    if not done:
        return
    exp = done[0][1]["experiment"]
    parts = []
    for name, cfg, out in done:
        s = pd.read_csv(out / "summary.csv")
        sd = {k: v.get("sd_abs_mm") for k, v in cfg["sensors"].items()}
        parts.append(s.assign(variant=name, a_start_mm=cfg["truth"]["a_start"]["median_mm"],
                              sd_mm=s["sensor"].map(sd), run_dir=str(out)))
    table = pd.concat(parts, ignore_index=True)
    gates = gate_table(table, done[0][1]["gates"])
    SUMMARY.mkdir(parents=True, exist_ok=True)
    table.to_csv(SUMMARY / f"{exp}_table.csv", index=False)
    gates.to_csv(SUMMARY / f"{exp}_gates.csv", index=False)
    (SUMMARY / f"{exp}_gates.json").write_text(json.dumps(
        dict(thresholds=done[0][1]["gates"], any_pass=bool(gates["passes"].any()) if len(gates) else None,
             runs={n: str(o) for n, _, o in done}), indent=1))
    cols = ["variant", "sensor", "policy", "u90_median", "bias_median", "coverage", "crps_mean"]
    print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
