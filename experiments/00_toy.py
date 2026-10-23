"""00 · Toy: closed-form Paris growth, depth-only sensor.

The estimator is well-specified, so nominal 90 % intervals must cover about 90 %
and simulation-based calibration ranks must be uniform (Talts et al. 2018).
SBC uses one random-time forecast per specimen so the ranks are independent.

    python experiments/00_toy.py [configs/00_toy.yaml] [--n 400] [--workers 4]
"""
import sys

import pandas as pd

from _bootstrap import ROOT, SUMMARY

from utprog import runner
from utprog.scoring import sbc


def main():
    for name, cfg, out in runner.main(sys.argv[1:], default_config=str(ROOT / "configs" / "00_toy.yaml")):
        rec = pd.read_csv(out / "records.csv.gz")
        one = rec[rec["kind"] == "random"].groupby(["arm", "specimen"]).head(1)
        L = int(cfg["scoring"]["sbc_draws"])
        rows = []
        for arm, g in one.groupby("arm"):
            stat, p, _ = sbc.uniformity(g["sbc_rank"].to_numpy(int), L)
            rows.append(dict(arm=arm, n=len(g), coverage=g["covered"].mean(), sbc_chi2=stat, sbc_p=p))
        table = pd.DataFrame(rows).assign(run_dir=str(out))
        table.to_csv(out / "sbc.csv", index=False)
        SUMMARY.mkdir(parents=True, exist_ok=True)
        table.to_csv(SUMMARY / f"00_toy_{name}_sbc.csv", index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
