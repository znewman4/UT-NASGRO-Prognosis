"""03 · M2, Approach 3: residual-stress classification from the sized-depth record.

Correct-classification rate against life fraction and sizing sd, the share of
correct calls that are "strong" (2 ln BF above the gate), the confusion matrix
at the gate's life fraction, and the gate itself: the earliest life fraction at
which the rate reaches min_correct at the gate's sd.

    python experiments/03_residual_stress.py [configs/03_residual_stress.yaml]
    python experiments/03_residual_stress.py configs/03b_residual_forms.yaml
"""
import json
import sys

import pandas as pd

from _bootstrap import ROOT, SUMMARY

from utprog import runner


def main():
    done = runner.main(sys.argv[1:], default_config=str(ROOT / "configs" / "03_residual_stress.yaml"))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for name, cfg, out in done:
        exp, g = cfg["experiment"], cfg["gates"]
        sd = {k: v["sd_abs_mm"] for k, v in cfg["sensors"].items()}
        rec = pd.read_csv(out / "records.csv.gz")
        rec = rec[rec["kind"] == "fixed"].assign(frac_fixed=lambda d: d["frac"].round(3), sd_mm=lambda d: d["sensor"].map(sd))
        rec["correct"] = rec["pred_class"] == rec["true_class"]
        rec["strong"] = rec["correct"] & (rec["two_ln_bf"] >= g["strong_two_ln_bf"])
        rate = rec.groupby(["sd_mm", "policy", "frac_fixed"])[["correct", "strong"]].mean().reset_index()
        at = rec[(rec["sd_mm"] == g["sd_mm"]) & (rec["frac_fixed"] == round(g["by_life_fraction"], 3))]
        confusion = pd.crosstab([at["policy"], at["true_class"]], at["pred_class"], normalize="index")
        gate = {}
        for pol, d in rate[rate["sd_mm"] == g["sd_mm"]].groupby("policy"):
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
