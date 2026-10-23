"""E-03 · Approach 2 (residual stress): classification from the sized-depth record (and, in E-03c,
LCR layer averages).

Per estimator (bf: Bayes factors between grid posteriors; is: class mixture in the well-specified
importance sampler): correct-classification rate against life fraction and sizing sd, the share of
correct calls that are "strong" (2 ln BF above the gate), the confusion matrix at the gate's life
fraction, and the gate: the earliest life fraction at which the rate reaches min_correct at the
gate's sd. Specimens whose crack never grows are included (a peened surface that stops a crack is
the easiest case to classify). Also the forecast bias of a class-blind forecast (P2.1) and the U90
worth of amplitude and decay length once the class is known (P2.2), from the oracle arms.

    python experiments/03_residual_stress.py [configs/03_residual_stress.yaml]
    python experiments/03_residual_stress.py configs/03b_residual_forms.yaml
"""
import json
import sys

import numpy as np
import pandas as pd

from _bootstrap import ROOT, SUMMARY

from utprog import runner


def oracle_tables(rec, exp, name, out):
    """P2.1: median-life bias of the class-blind oracle against the class-known one, by class and depth.
    P2.2: U90 worth of amplitude and decay length once the class is known."""
    d = rec[(rec["kind"] == "depth") & rec["oracle"].astype(bool) & ~rec["arrested"].astype(bool)]
    if d.empty:
        return
    piv = d.pivot_table(index=["true_class", "depth_mm"], columns="estimator", values=["q50", "u90"], aggfunc="mean")
    rows = []
    for (cls, depth), r in piv.iterrows():
        row = dict(true_class=cls, depth_mm=depth)
        if ("q50", "oracle_depth") in r and ("q50", "oracle_depth_class") in r:
            row["blind_bias_decades"] = float(np.log10(r[("q50", "oracle_depth")] / r[("q50", "oracle_depth_class")]))
        if ("u90", "oracle_depth_rs") in r and ("u90", "oracle_depth_class") in r:
            row["rs_params_u90_reduction"] = float(1.0 - r[("u90", "oracle_depth_rs")] / r[("u90", "oracle_depth_class")])
        rows.append(row)
    pd.DataFrame(rows).assign(run_dir=str(out)).to_csv(SUMMARY / f"{exp}_{name}_oracle.csv", index=False)


def main():
    done = runner.main(sys.argv[1:], default_config=str(ROOT / "configs" / "03_residual_stress.yaml"))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for name, cfg, out in done:
        exp, g = cfg["experiment"], cfg["gates"]
        sd = {k: v.get("sd_abs_mm") for k, v in cfg["sensors"].items()}
        all_rec = pd.read_csv(out / "records.csv.gz")
        oracle_tables(all_rec, exp, name, out)
        rec = all_rec[(all_rec["kind"] == "fixed") & all_rec["pred_class"].notna()].copy()
        # life fraction for growing cracks; for cracks that never grow, the fraction of the scoring window
        span = rec["life"].where(~rec["arrested"].astype(bool), rec.groupby("specimen")["N"].transform("max") / 0.9)
        rec["frac_fixed"] = (rec["N"] / span).round(1)
        rec["sd_mm"] = rec["sensor"].map(lambda s: sd.get(s.split("+")[0]))
        rec["correct"] = rec["pred_class"] == rec["true_class"]
        rec["strong"] = rec["correct"] & (rec["two_ln_bf"] >= g["strong_two_ln_bf"])
        rate = rec.groupby(["estimator", "sensor", "sd_mm", "policy", "frac_fixed"])[["correct", "strong"]].mean().reset_index()
        at = rec[(rec["sd_mm"] == g["sd_mm"]) & (rec["frac_fixed"] == round(g["by_life_fraction"], 1))]
        confusion = pd.crosstab([at["estimator"], at["sensor"], at["policy"], at["true_class"]], at["pred_class"],
                                normalize="index")
        gate = {}
        for pol, d in rate[rate["sd_mm"] == g["sd_mm"]].groupby(["estimator", "sensor", "policy"]):
            pol = "|".join(pol)
            ok = d[d["correct"] >= g["min_correct"]]
            first = float(ok["frac_fixed"].min()) if len(ok) else None
            gate[pol] = dict(first_life_fraction=first,
                             passes=first is not None and first <= g["by_life_fraction"])
        rate.assign(run_dir=str(out)).to_csv(SUMMARY / f"{exp}_{name}_classification.csv", index=False)
        confusion.to_csv(SUMMARY / f"{exp}_{name}_confusion.csv")
        (SUMMARY / f"{exp}_{name}_gates.json").write_text(json.dumps(
            dict(thresholds=g, by_policy=gate, run_dir=str(out)), indent=1))
        print(rate.to_string(index=False))
        print(confusion.to_string())


if __name__ == "__main__":
    main()
