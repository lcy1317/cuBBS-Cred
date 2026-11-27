#!/usr/bin/env python3
"""Build data_fig2.mat from opt-bench CSVs and render Fig 2 (PNG/PDF/SVG/EPS)."""
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
FIG2 = REPO / "paper-figures/fig2"
sys.path.insert(0, str(FIG2.parent))
from eps_export import save_eps

FONT = "Liberation Serif"
COLORS = {
    "r1": "#5F78B5",
    "r2": "#4F978A",
    "r3": "#C78B47",
    "r4": "#C8C3CC",
    "r4_edge": "#716C74",
    "text": "#202630",
    "muted": "#626A74",
    "grid": "#D9DEE7",
}

def _lighten(color: str, amount: float = 0.9):
    rgb = np.asarray(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)

def _style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.6, width=0.9, colors=COLORS["text"])
    axis.xaxis.label.set_color(COLORS["text"])
    axis.yaxis.label.set_color(COLORS["text"])
    axis.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.8)

def _err(med, mn, mx):
    return np.vstack((med - mn, mx - med))

def main():
    r1_rows = list(csv.DictReader((FIG2 / "fig2_route1_batch_summary.csv").open()))
    sh_rows = list(csv.DictReader((FIG2 / "fig2_shape_summary.csv").open()))

    batches = sorted({int(r["batch_size"]) for r in r1_rows})
    r1_by = {(int(r["batch_size"]), r["variant"]): r for r in r1_rows}
    r1_med = np.array([float(r1_by[(b, "r1")]["median"]) for b in batches])
    r1_min = np.array([float(r1_by[(b, "r1")]["minimum"]) for b in batches])
    r1_max = np.array([float(r1_by[(b, "r1")]["maximum"]) for b in batches])
    r1_r4_med = np.array([float(r1_by[(b, "r4")]["median"]) for b in batches])
    r1_r4_min = np.array([float(r1_by[(b, "r4")]["minimum"]) for b in batches])
    r1_r4_max = np.array([float(r1_by[(b, "r4")]["maximum"]) for b in batches])
    r1_sp = r1_med / r1_r4_med

    terms = sorted({int(r["term_count"]) for r in sh_rows})
    msgs = np.array([k - 2 for k in terms], dtype=float)
    sh_by = {(int(r["term_count"]), r["variant"]): r for r in sh_rows}
    c_auto_med = np.array([float(sh_by[(k, "auto")]["median"]) for k in terms])
    c_auto_min = np.array([float(sh_by[(k, "auto")]["minimum"]) for k in terms])
    c_auto_max = np.array([float(sh_by[(k, "auto")]["maximum"]) for k in terms])
    c_auto_rt = np.array([int(sh_by[(k, "auto")]["route_observed"]) for k in terms])
    c_r4_med = np.array([float(sh_by[(k, "r4")]["median"]) for k in terms])
    c_r4_min = np.array([float(sh_by[(k, "r4")]["minimum"]) for k in terms])
    c_r4_max = np.array([float(sh_by[(k, "r4")]["maximum"]) for k in terms])
    c_sp = c_auto_med / c_r4_med

    savemat(str(FIG2 / "data_fig2.mat"), {
        "route1_batches": np.array(batches, dtype=float),
        "route1_median": r1_med, "route1_min": r1_min, "route1_max": r1_max,
        "route1_r4_median": r1_r4_med, "route1_r4_min": r1_r4_min, "route1_r4_max": r1_r4_max,
        "route1_speedup": r1_sp,
        "common_terms": np.array(terms, dtype=float),
        "common_messages": msgs,
        "common_auto_route": c_auto_rt.astype(float),
        "common_auto_median": c_auto_med, "common_auto_min": c_auto_min, "common_auto_max": c_auto_max,
        "common_r4_median": c_r4_med, "common_r4_min": c_r4_min, "common_r4_max": c_r4_max,
        "common_speedup": c_sp,
    })

    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [FONT], "mathtext.fontset": "stix",
        "font.size": 10.3, "axes.labelsize": 11.2, "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0, "legend.fontsize": 8.0, "axes.linewidth": 0.9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1.03, 1.0]})
    fig.subplots_adjust(left=0.092, right=0.985, bottom=0.175, top=0.865, wspace=0.31)

    # Panel (a)
    ax1.axvspan(256, 1.0e5, color=_lighten(COLORS["r1"], 0.93), zorder=0)
    h4 = ax1.errorbar(batches, r1_r4_med, yerr=_err(r1_r4_med, r1_r4_min, r1_r4_max),
                      color=COLORS["r4_edge"], marker="D", markerfacecolor=COLORS["r4"],
                      markeredgecolor=COLORS["r4_edge"], markersize=4.1, linewidth=1.25,
                      linestyle="--", capsize=2.0, label="R4: universal GLV", zorder=2)
    h1 = ax1.errorbar(batches, r1_med, yerr=_err(r1_med, r1_min, r1_max),
                      color=COLORS["r1"], marker="o", markerfacecolor=COLORS["r1"],
                      markeredgecolor="#3E568E", markersize=4.8, linewidth=1.6,
                      capsize=2.0, label="R1: 1t/row GLV short MSM", zorder=3)
    ax1.axvline(256, color="#949BA6", linewidth=0.9, linestyle=":")
    ax1.axvline(8192, color="#AAB0B8", linewidth=0.8, linestyle=":")
    ax1.text(285, 1.8e2, "R1 > R4", fontsize=7.9, color=COLORS["muted"], va="bottom")
    ax1.text(9300, 1.8e2, "saturation", fontsize=7.9, color=COLORS["muted"], va="bottom")
    ax1.text(0.0, 1.08, "(a) Route 1: three-term item-specific MSM",
             transform=ax1.transAxes, fontsize=10.6, fontweight="bold", color=COLORS["text"], va="bottom")
    ax1.text(0.97, 0.48, r"$R_j=\sum_{i=0}^{2}s_{j,i}P_{j,i}$",
             transform=ax1.transAxes, fontsize=9.0, color=COLORS["muted"], ha="right")
    b2i = {b: i for i, b in enumerate(batches)}
    for b in (256, 1024, 16384):
        idx = b2i[b]
        ax1.annotate(f"{r1_sp[idx]:.2f}×", xy=(batches[idx], r1_med[idx]),
                     xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.0, fontweight="bold", color=COLORS["text"])
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(0.72, 1.0e5); ax1.set_ylim(1.0e2, 2.2e5)
    ax1.set_xticks([1, 32, 256, 1024, 8192, 65536])
    ax1.set_xticklabels(["1", "32", "256", "1k", "8k", "65k"])
    ax1.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
    ax1.set_xlabel(r"Batch size, $B$")
    ax1.set_ylabel("Throughput (MSM rows/s)")
    ax1.legend(handles=[h1, h4], loc="upper left", frameon=False, handlelength=1.9)
    _style_axis(ax1)

    # Panel (b)
    r2_mask = c_auto_rt == 2
    r3_mask = c_auto_rt == 3
    c4 = ax2.errorbar(msgs, c_r4_med, yerr=_err(c_r4_med, c_r4_min, c_r4_max),
                      color=COLORS["r4_edge"], marker="D", markerfacecolor=COLORS["r4"],
                      markeredgecolor=COLORS["r4_edge"], markersize=4.1, linewidth=1.25,
                      linestyle="--", capsize=2.0, label="R4: universal GLV", zorder=2)
    c2 = ax2.errorbar(msgs[r2_mask], c_auto_med[r2_mask],
                      yerr=_err(c_auto_med[r2_mask], c_auto_min[r2_mask], c_auto_max[r2_mask]),
                      color=COLORS["r2"], marker="o", markerfacecolor=COLORS["r2"],
                      markeredgecolor="#327064", markersize=4.8, linewidth=1.6,
                      capsize=2.0, label="R2: Lim-Lee (K≤2100)", zorder=3)
    c3 = ax2.errorbar(msgs[r3_mask], c_auto_med[r3_mask],
                      yerr=_err(c_auto_med[r3_mask], c_auto_min[r3_mask], c_auto_max[r3_mask]),
                      color=COLORS["r3"], marker="^", markerfacecolor=COLORS["r3"],
                      markeredgecolor="#8A5B25", markersize=5.1, linewidth=1.6,
                      capsize=2.0, label="R3: 10-bit GLV Pippenger", zorder=3)
    boundary = 2100.0
    ax2.axvline(boundary, color="#949BA6", linewidth=0.9, linestyle=":", zorder=1)
    ax2.text(boundary * 1.07, 4.5e5, "R2→R3\n(K=2100)", fontsize=7.6, color=COLORS["muted"], ha="left", va="top")
    ax2.text(0.0, 1.08, "(b) Routes 2–3: shared-point shape sweep",
             transform=ax2.transAxes, fontsize=10.6, fontweight="bold", color=COLORS["text"], va="bottom")
    ax2.text(130, 5.2e5, f"R2: {c_sp[r2_mask].min():.1f}–{c_sp[r2_mask].max():.1f}×",
             fontsize=8.1, fontweight="bold", color=COLORS["r2"], ha="center")
    ax2.text(5600, 6.2e3, f"R3: {c_sp[r3_mask].min():.1f}–{c_sp[r3_mask].max():.1f}×",
             fontsize=8.1, fontweight="bold", color="#9A642D", ha="center")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(12, 11500); ax2.set_ylim(1.5e2, 9.5e5)
    ax2.set_xticks([16, 64, 256, 1024, 2048, 8192])
    ax2.set_xticklabels(["16", "64", "256", "1k", "2k", "8k"])
    ax2.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
    ax2.set_xlabel(r"BBS message count, $M$  ($K=M+2$)")
    ax2.set_ylabel("Throughput (MSM rows/s)")
    ax2.legend(handles=[c2, c3, c4], loc="lower left", frameon=False, handlelength=1.9)
    _style_axis(ax2)

    out_base = FIG2 / "fig2_msm_route_family_ablation"
    common = {"bbox_inches": "tight", "pad_inches": 0.02}
    save_eps(fig, out_base.with_suffix(".eps"), **common)
    for ext in (".pdf", ".svg"):
        fig.savefig(str(out_base.with_suffix(ext)), **common)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=600, **common)
    plt.close(fig)
    print(f"Rendered {out_base}.{{eps,pdf,png,svg}}")

if __name__ == "__main__":
    main()
