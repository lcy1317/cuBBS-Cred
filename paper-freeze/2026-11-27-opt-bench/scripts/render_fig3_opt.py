#!/usr/bin/env python3
"""Render updated Figure 3 (canonical 21 message counts M=16..65536) with new opt regimes:
Lim-Lee fixed-base (K <= 2100, M <= 2098) and 10-bit GLV Pippenger (K > 2100, M >= 2099)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

REPO = Path("/home/lcy/work/cuBBS-Cred")
FIG3 = REPO / "paper-figures/fig3"
sys.path.insert(0, str(FIG3.parent))
from eps_export import save_eps

CANONICAL_21 = [
    16, 32, 64, 128, 256, 512, 1024, 1536, 2048,
    3072, 4095, 4096, 4097, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536,
]
OPERATIONS = ("Sign", "Verify", "ProofGen", "ProofVerify")
TEXT = "#202630"
MUTED = "#626A74"
GRID = "#D9DEE7"
FONT = "Liberation Serif"
LIMLEE_FILL = "#F6EEE1"
PIPP_FILL = "#E7F1ED"
COLORS = ("#5068A9", "#3F887B", "#B67635", "#785B96")
EDGES = ("#34497E", "#28685D", "#80501F", "#553D73")
MARKERS = ("o", "s", "^", "D")
PANELS = ("(a)", "(b)", "(c)", "(d)")

def _compact(v: float) -> str:
    if v >= 1000:
        return f"{v/1000.0:.2f}k" if v < 10000 else f"{v/1000.0:.1f}k"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"

def main():
    rows = list(csv.DictReader((FIG3 / "fig3_message_length_summary.csv").open()))
    by = {(r["operation"], int(r["message_count"])): r for r in rows}
    # Rewrite canonical 21-point summary & mat (preserving full 23-point raw in raw/)
    kept_rows = [by[(op, m)] for op in OPERATIONS for m in CANONICAL_21]
    with (FIG3 / "fig3_message_length_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept_rows[0].keys()))
        w.writeheader()
        w.writerows(kept_rows)

    med = np.zeros((4, len(CANONICAL_21)), dtype=float)
    mn = np.zeros((4, len(CANONICAL_21)), dtype=float)
    mx = np.zeros((4, len(CANONICAL_21)), dtype=float)
    for oi, op in enumerate(OPERATIONS):
        for mi, m in enumerate(CANONICAL_21):
            r = by[(op, m)]
            med[oi, mi] = float(r["median_operations_per_s"])
            mn[oi, mi] = float(r["minimum_operations_per_s"])
            mx[oi, mi] = float(r["maximum_operations_per_s"])

    msgs = np.asarray(CANONICAL_21, dtype=float)
    savemat(str(FIG3 / "data_fig3.mat"), {
        "messages": msgs, "median": med, "minimum": mn, "maximum": mx,
    })

    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [FONT], "mathtext.fontset": "stix",
        "font.size": 10.0, "axes.labelsize": 11.0, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.7, "axes.linewidth": 0.9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.95, hspace=0.32, wspace=0.22)
    ticks = np.asarray((16, 64, 256, 1024, 4096, 16384, 65536), dtype=float)
    tick_labels = ("16", "64", "256", "1k", "4k", "16k", "65k")

    boundary = 2098.5
    for oi, axis in enumerate(axes.flat):
        axis.axvspan(14, boundary, color=LIMLEE_FILL, alpha=0.55, zorder=0)
        axis.axvspan(boundary, 76000, color=PIPP_FILL, alpha=0.88, zorder=0)
        axis.axvline(boundary, color="#78968A", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        series = med[oi]
        yerr = np.vstack((series - mn[oi], mx[oi] - series))
        axis.errorbar(
            msgs, series, yerr=yerr, color=COLORS[oi], marker=MARKERS[oi],
            markerfacecolor=COLORS[oi], markeredgecolor=EDGES[oi],
            markersize=4.1, linewidth=1.45, capsize=1.7, elinewidth=0.75, zorder=3,
        )
        axis.set_xscale("log", basex=2) if int(mpl.__version__.split(".")[0]) < 3 else axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlim(14, 76000)
        axis.set_xticks(ticks)
        axis.set_xticklabels(tick_labels)
        axis.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
        axis.text(0.0, 1.035, f"{PANELS[oi]} {OPERATIONS[oi]}", transform=axis.transAxes,
                  fontsize=10.7, fontweight="bold", color=TEXT, va="bottom")
        axis.text(0.975, 0.955, r"$T_{65\,536}$ = " + _compact(series[-1]) + " ops/s",
                  transform=axis.transAxes, ha="right", va="top", fontsize=7.7, color=MUTED,
                  bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.88}, zorder=5)
        for xpos, label, color in (
            (180, "Lim-Lee\nfixed-base", "#805F38"),
            (14000, "10-bit GLV\nPippenger", "#537368"),
        ):
            axis.text(xpos, 0.055, label, transform=axis.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=7.4, color=color, linespacing=0.95)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.3, width=0.85, colors=TEXT)
        axis.grid(axis="y", which="major", color=GRID, linewidth=0.7)
        axis.set_axisbelow(True)

    fig.text(0.55, 0.03, r"Messages per credential, $M$", ha="center", fontsize=11.1, color=TEXT)
    fig.text(0.018, 0.52, "Throughput (operations/s)", va="center", rotation="vertical", fontsize=11.1, color=TEXT)

    out_base = FIG3 / "fig3_message_count_scalability"
    common = {"bbox_inches": "tight", "pad_inches": 0.015}
    save_eps(fig, out_base.with_suffix(".eps"), **common)
    for ext in (".pdf", ".svg"):
        fig.savefig(str(out_base.with_suffix(ext)), **common)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=600, **common)
    plt.close(fig)
    print(f"Rendered {out_base}.{{eps,pdf,png,svg}}")

if __name__ == "__main__":
    main()
