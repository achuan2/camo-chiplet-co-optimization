#!/usr/bin/env python3
"""Plot both Figure 2 panels from the regenerated numerical source table."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/reproduced-results/figure2_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figure2")
    args = parser.parse_args()
    with args.data.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure = plt.figure(figsize=(5.48, 2.5))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.55)
    left = figure.add_subplot(grid[0, 0])
    scenarios = ["Normal", "Temperature-tight", "Power-tight", "SLO-tight"]
    for j, (metric, title, color) in enumerate([
        ("hypervolume_5d", "HV improvement", "#56B4E9"),
        ("igd_plus_5d", "IGD+ reduction", "#E69F00"),
    ]):
        source = [next(x for x in rows if x["panel"] == "a" and x["scenario"] == s and x["metric"] == metric) for s in scenarios]
        positions = [i + (j - 0.5) * 0.36 for i in range(4)]
        values = [float(x["value"]) for x in source]
        left.bar(positions, values, 0.36, label=title, color=color, edgecolor="#333333", linewidth=0.5)
        for x, y, row in zip(positions, values, source):
            label = "%.1f" % y + ("\n(n.s.)" if row["not_significant"] == "True" else "")
            left.text(x, y + 0.7, label, ha="center", va="bottom", fontsize=7.5)
    left.set_xticks(range(4), ["Normal", "Temperature-\ntight", "Power-\ntight", "SLO-tight"], fontsize=7.5)
    left.set_ylim(0, 45)
    left.set_ylabel("Relative change vs. best evaluated SOTA (%)", fontsize=7.5)
    left.legend(frameon=False, ncol=2, fontsize=7.5, loc="lower left", bbox_to_anchor=(-0.05, 1.01), handlelength=0.8, columnspacing=0.9)
    right_grid = grid[0, 1].subgridspec(1, 3, width_ratios=[1, 1, 0.52], wspace=0.25)
    hv, igd, counts = [figure.add_subplot(right_grid[0, i]) for i in range(3)]
    methods = ["MOHaM", "C-NSGA-II", "C-NSGA-III", "C-TAEA", "CAMO"]
    for i, method in enumerate(methods):
        for axis, metric in [(hv, "hypervolume_5d"), (igd, "igd_plus_5d")]:
            value = next(float(r["value"]) for r in rows if r["panel"] == "b" and r["method"] == method and r["metric"] == metric)
            axis.scatter(value, i, marker="D" if method == "CAMO" else "o", color="#56B4E9" if method == "CAMO" else "#999999", edgecolors="#333333", linewidths=0.5, zorder=3)
        evaluations = next(float(r["value"]) for r in rows if r["panel"] == "b" and r["method"] == method and r["metric"] == "evaluations")
        counts.text(0.5, i, "{:,.0f}".format(evaluations), ha="center", va="center", fontsize=7.5)
    for axis in (hv, igd, counts):
        axis.set_ylim(4.5, -0.5)
        axis.set_yticks(range(5))
    hv.set_yticklabels(methods, fontsize=7.5)
    igd.set_yticklabels([])
    counts.set_yticklabels([])
    for axis in (hv, igd):
        axis.grid(axis="y", color="#D9DEE4", linewidth=0.5)
        axis.tick_params(axis="x", labelsize=7.5)
        axis.tick_params(axis="y", length=0)
    hv.set_xlim(0.22, 0.48)
    hv.set_xticks([0.25, 0.35, 0.45])
    igd.set_xlim(0.04, 0.27)
    igd.set_xticks([0.05, 0.15, 0.25])
    hv.set_title("HV ↑", fontsize=8)
    igd.set_title("IGD+ ↓", fontsize=8)
    counts.set_title("Evaluations", fontsize=7.5)
    counts.axis("off")
    figure.subplots_adjust(left=0.1, right=0.99, bottom=0.22, top=0.82)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(args.output) + ".pdf", bbox_inches="tight")
    figure.savefig(str(args.output) + ".png", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
