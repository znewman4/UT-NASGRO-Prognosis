"""Build report tables and figures from results/ only.

Every run directory is checked for its manifest first; a result without one is
skipped and listed, never reported. Uses the newest run of each experiment and
variant. Output goes to reports/<experiment>/.

    python reports/build.py
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utprog import manifest  # noqa: E402

RESULTS, REPORTS = ROOT / "results", ROOT / "reports"
# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def latest_runs():
    """{(experiment, variant): newest run dir with a manifest}; plus the dirs skipped for lacking one."""
    runs, skipped = {}, []
    for summary in RESULTS.glob("*/*/*/summary.csv"):
        d = summary.parent
        try:
            manifest.require(d)
        except FileNotFoundError:
            skipped.append(d)
            continue
        key = (d.parents[1].name, d.parent.name)
        if key not in runs or d.name > runs[key].name:
            runs[key] = d
    return runs, skipped


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def u90_figure(summary, sensors_sd, path, title):
    s = summary.assign(sd_mm=summary["sensor"].map(sensors_sd)).sort_values("sd_mm")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for colour, (pol, d) in zip(SERIES, s.groupby("policy", sort=False)):
        ax.plot(d["sd_mm"], d["u90_median"], color=colour, linewidth=2, marker="o", markersize=6, label=pol)
    style(ax, title, "sizing sd (mm)", "median U90 of remaining life (decades)")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def classification_figure(rate, path, title):
    pols = list(rate["policy"].unique())
    fig, axes = plt.subplots(1, len(pols), figsize=(4.2 * len(pols), 3.8), sharey=True, squeeze=False)
    for ax, pol in zip(axes[0], pols):
        d = rate[rate["policy"] == pol]
        for colour, (sd, g) in zip(SERIES, d.groupby("sd_mm")):
            ax.plot(g["frac_fixed"], g["correct"], color=colour, linewidth=2, marker="o", markersize=6,
                    label=f"sd {sd:g} mm")
        style(ax, f"{title}: {pol}", "life fraction", "correctly classified")
        ax.set_ylim(0, 1)
    axes[0][-1].legend(frameon=False, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    runs, skipped = latest_runs()
    for (exp, variant), d in sorted(runs.items()):
        m = manifest.require(d)
        out = REPORTS / exp
        out.mkdir(parents=True, exist_ok=True)
        summary = pd.read_csv(d / "summary.csv")
        cols = ["sensor", "policy", "n", "u90_median", "bias_median", "coverage", "coverage_lo", "coverage_hi",
                "crps_mean", "last_obs_age_median"]
        header = (f"# {exp} / {variant}\n\nRun `{d.relative_to(ROOT)}`, config hash `{m['config_hash'][:12]}`, "
                  f"commit `{m['git']['commit'][:10]}`{' (dirty tree)' if m['git']['dirty'] else ''}, "
                  f"NASGRO pipeline `{m['nasgro_sa']['commit'][:10]}`, seed {m['seed']}, "
                  f"{m['n_specimens']} specimens ({m['n_arrested']} arrested).\n\n")
        (out / f"{variant}.md").write_text(header + summary[[c for c in cols if c in summary]]
                                           .to_markdown(index=False, floatfmt=".3f") + "\n")
        sd = {k: v.get("sd_abs_mm") for k, v in m["config"]["sensors"].items()}
        u90_figure(summary, sd, out / f"{variant}_u90.png", f"{exp} / {variant}")
        cls = ROOT / "results" / "summary" / f"{exp}_{variant}_classification.csv"
        if cls.is_file():
            classification_figure(pd.read_csv(cls), out / f"{variant}_classification.png", exp)
    for d in skipped:
        print(f"skipped (no manifest): {d}")


if __name__ == "__main__":
    main()
