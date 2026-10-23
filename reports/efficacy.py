"""Build the efficacy report (reports/efficacy.md) from results/summary/ and docs/predictions.yaml.

For each approach: the frozen prediction, the oracle simulation (consistency check), the
realistic simulation (efficacy), the realised fraction, coverage and fixed-cutoff life error,
how the result degrades with sensor error and under the mismatched truth, and what it would
take to pay. Summary tables carry the run directory they came from; a table whose run has no
manifest is not used.

    python reports/efficacy.py [--summary results/summary] [--out reports/efficacy.md]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utprog import manifest  # noqa: E402

PRED = yaml.safe_load((ROOT / "docs" / "predictions.yaml").read_text())["predictions"]


def load(summary, name):
    """Concatenate summary CSVs matching name, keeping only rows whose run has a manifest."""
    parts = []
    for f in sorted(summary.glob(name)):
        d = pd.read_csv(f)
        if "run_dir" in d:
            ok = d["run_dir"].map(lambda r: (Path(r) / "manifest.json").is_file())
            d = d[ok]
        parts.append(d.assign(source=f.name))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def md(df, cols=None, digits=3):
    if df.empty:
        return "_no results yet_\n"
    d = df[cols] if cols else df
    return d.to_markdown(index=False, floatfmt=f".{digits}f") + "\n"


def approach1(summary):
    out = ["## Approach 1 · growth rate\n",
           f"**Predictions.** P1.1 {PRED['P1.1']['statement']}. P1.2 {PRED['P1.2']['statement']}. "
           f"P1.3 {PRED['P1.3']['statement']} (original pipeline). P1.4 {PRED['P1.4']['statement']} "
           "(original pipeline).\n",
           "### Oracle against prediction (must pass before efficacy is read)\n",
           md(load(summary, "01*_oracle_check.csv"),
              ["case", "variant", "depth_mm", "predicted", "oracle", "difference", "passes"])]
    eff = load(summary, "01*_efficacy.csv")
    if not eff.empty:
        real = eff[~eff["oracle"].astype(bool)]
        for ref in ("oracle_depth", "oracle_state_lambda"):
            r = real[real["reference_oracle"] == ref]
            if r.empty:
                continue
            r = r.assign(estimator=r["arm"].str.split("|").str[-1], policy=r["arm"].str.split("|").str[1],
                         sensor=r["arm"].str.split("|").str[0])
            piv = r.pivot_table(index=["source", "variant", "depth_mm", "estimator", "policy"], columns="sensor",
                                values="fraction", aggfunc="mean").reset_index()
            out += [f"### Realised fraction against `{ref}` (1 = all the predicted reduction)\n", md(piv, digits=2)]
        cov = real[real["reference_oracle"] == "oracle_depth"]
        out += ["### Coverage of the 90 % intervals at the scoring depths\n",
                md(cov.pivot_table(index=["source", "variant", "arm"], columns="depth_mm",
                                   values="coverage").reset_index(), digits=2)]
    cb = load(summary, "01*_checkB.csv")
    out += ["### Check-B replica (U90 after 1, 2, 10 sizings over 20 % of life)\n",
            "Predicted 10-vs-1 reduction: plate 0.40 at 0.1 mm, 0.10 at 0.5 mm; bar 0.31 and 0.10 "
            "(original pipeline).\n", md(cb)]
    return out


def approach2(summary):
    out = ["## Approach 2 · residual stress\n",
           f"**Predictions.** P2.1 {PRED['P2.1']['statement']}. P2.2 {PRED['P2.2']['statement']}.\n",
           "### Oracle: bias of a class-blind forecast (P2.1) and worth of amplitude and decay (P2.2)\n",
           md(load(summary, "03*_oracle.csv"))]
    cls = load(summary, "03*_classification.csv")
    if not cls.empty:
        at = cls[cls["frac_fixed"].isin([0.2, 0.5, 0.8])]
        out += ["### Correct classification by life fraction\n",
                md(at.pivot_table(index=["source", "estimator", "sensor", "policy"], columns="frac_fixed",
                                  values="correct").reset_index(), digits=2)]
    for f in sorted(summary.glob("03*_gates.json")):
        g = json.loads(f.read_text())
        if (Path(g.get("run_dir", "")) / "manifest.json").is_file():
            out += [f"- `{f.name}`: " + "; ".join(f"{k}: first at {v['first_life_fraction']}"
                                                  for k, v in g["by_policy"].items()) + "\n"]
    return out


def approach3(summary):
    out = ["## Approach 3 · intervals\n",
           f"**Prediction.** P3.1 {PRED['P3.1']['statement']}. The references are perfect inspection "
           "(`perfect` sensor) and perfect information (`oracle_*` arms).\n",
           "### Break-even monitoring cost per nominal life (continuous vs periodic)\n",
           md(load(summary, "02_intervals_*_break_even.csv"),
              ["source", "sensor", "estimator", "p_act", "failure_cost", "periodic", "break_even_per_life",
               "fail_periodic", "fail_continuous"]),
           "### Sizing sd below which continuous pays\n",
           md(load(summary, "02_intervals_*_break_even_sd.csv")),
           "### Cost of every arm, including the references and robustness cases\n",
           md(load(summary, "02_intervals_*_arm_costs.csv"))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "summary"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "efficacy.md"))
    args = ap.parse_args()
    summary = Path(args.summary)
    text = ["# Realised efficacy of Approaches 1–3\n",
            "Prediction → oracle simulation (consistency) → realistic simulation (efficacy). "
            "Definitions and provenance: `docs/predictions.md`. Only runs with a manifest are used.\n"]
    for part in (approach1, approach2, approach3):
        text += part(summary)
    Path(args.out).write_text("\n".join(text))
    print(args.out)


if __name__ == "__main__":
    main()
