"""Plot measured wrong-link and recovery tradeoffs for a closed-loop pilot."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


FAMILIES = {
    "assignment": ("Native association cost", "#545b66", "o"),
    "margin": ("Competitor margin", "#9a602d", "^"),
    "linear": ("Linear score", "#706bb3", "s"),
    "structured": ("Interaction score", "#007861", "D"),
    "structured_rank": ("Interaction + ranking loss", "#3372a0", "v"),
}


def plot(pilot, out):
    setup = json.loads((pilot / "setup.json").read_bytes())
    summary = json.loads((pilot / "summary.json").read_bytes())
    points = [{**row, "name": name, "family": name.rsplit("_", 1)[0]}
              for name, row in summary.items()]
    points.append({**setup["reference"], "name": "all", "family": "unrestricted"})
    out.mkdir(parents=True, exist_ok=True)
    with (out / "recovery-tradeoff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(ncols=2, figsize=(7.2, 3.15), layout="constrained")
    for family, (label, color, marker) in FAMILIES.items():
        rows = [row for row in points if row["family"] == family and row["accepted"]]
        for ax, values in zip(axes, ([row["correct_per_gap"] for row in rows],
                                     [row["accepted"] - row["wrong"] for row in rows]), strict=True):
            ax.plot([row["wrong_per_acceptance"] for row in rows], values, marker=marker,
                    color=color, label=label, alpha=.9, markersize=4, lw=1.1)
    ref = setup["reference"]
    for ax, label, value, letter in zip(
        axes, ("Natural gaps retaining original ID", "Correct accepted reactivations"),
        (ref["correct_per_gap"], ref["accepted"] - ref["wrong"]), ("a", "b"), strict=True
    ):
        ax.scatter([ref["wrong_per_acceptance"]], [value], color="#252632", marker="x",
                   s=25, zorder=6, label="Unrestricted ByteTrack")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set(xlabel="Wrong reactivations / accepted", ylabel=label)
        ax.grid(axis="y", lw=.4, color="#c9cbd4", alpha=.65)
        ax.text(-.14, 1.03, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncols=3, fontsize=7)
    fig.savefig(out / "recovery-tradeoff.svg")
    fig.savefig(out / "recovery-tradeoff.pdf")
    fig.savefig(out / "recovery-tradeoff.png", dpi=600)
    plt.close(fig)


def official_plot(personpath, out):
    """Use locked official-test replays and retain all evaluated work points."""
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    axes_data = ((("byte", "ByteTrack"), ("bot", "BoT-SORT (motion only)")))
    fig, axes = plt.subplots(ncols=2, figsize=(7.2, 3.15), layout="constrained")
    exported = []
    for label, ax, (tracker, title) in zip(("a", "b"), axes, axes_data, strict=True):
        payload = json.loads((personpath / f"test-{tracker}.json").read_bytes())["aggregate"]
        points = [{"tracker": tracker, "policy": name, **entry} for name, entry in payload.items()]
        exported.extend(points)
        for family, (name, color, marker) in FAMILIES.items():
            data = [r for r in points if r["policy"].rsplit("_", 1)[0] == family]
            ax.scatter([r["wrong_per_acceptance"] for r in data], [r["correct_per_gap"] for r in data],
                       color=color, marker=marker, label=name, alpha=.9, s=27)
        for key, name, mark in (("unconstrained_tracker", "Unrestricted", "x"),
                                ("cap_only", "Cap only", "+")):
            row = payload[key]
            ax.scatter(row["wrong_per_acceptance"], row["correct_per_gap"], color="#242832",
                       label=name, marker=mark, zorder=5, s=34)
        ax.set(title=title, xlabel="Evaluated failures / accepted reactivations",
               ylabel="Natural gaps retaining original ID", xlim=(.10, .66), ylim=(.097, .18))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", lw=.4, color="#c9cbd4", alpha=.65)
        ax.text(-.14, 1.03, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncols=4, fontsize=7)
    fig.savefig(out / "official-tradeoff.svg")
    fig.savefig(out / "official-tradeoff.pdf")
    fig.savefig(out / "official-tradeoff.png", dpi=600)
    plt.close(fig)
    with (out / "official-tradeoff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args()
    if args.official:
        official_plot(args.pilot, args.output)
    else:
        plot(args.pilot, args.output)
