#!/usr/bin/env python3
"""Validate the canonical 2026-11-27 opt capacity CSV and render Markdown, LaTeX, and figures."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "table_bbs_capacity.csv"
FIG_DIR = ROOT.parent / "figures"
MD_OUT = ROOT / "table_bbs_capacity.md"
TEX_OUT = ROOT / "table_bbs_capacity.tex"
FIG_BASENAME = FIG_DIR / "table_bbs_capacity"

DEVICES = (
    ("epyc7763", "EPYC 7763"),
    ("epyc9554", "EPYC 9554"),
    ("rtx5060ti", "RTX 5060 Ti"),
    ("rtx5090", "RTX 5090"),
)
METRICS = (
    "ops_per_s", "min_ops_per_s", "max_ops_per_s", "speedup_vs_epyc9554",
    "batch", "streams", "cv_pct", "repetitions",
)


def read_rows() -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32:
        raise ValueError(f"expected 32 workloads, found {len(rows)}")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("row_id values must be unique")

    required = {"row_id", "section", "message_shape", "issuer", "disclosure", "operation"}
    required.update(f"{device}_{metric}" for device, _ in DEVICES for metric in METRICS)
    required.add("rtx5090_over_rtx5060ti")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"canonical CSV is missing columns: {sorted(missing)}")

    for row in rows:
        baseline = float(row["epyc9554_ops_per_s"])
        for device, _ in DEVICES:
            median = float(row[f"{device}_ops_per_s"])
            minimum = float(row[f"{device}_min_ops_per_s"])
            maximum = float(row[f"{device}_max_ops_per_s"])
            ratio = float(row[f"{device}_speedup_vs_epyc9554"])
            batch = int(row[f"{device}_batch"])
            streams = int(row[f"{device}_streams"])
            cv = float(row[f"{device}_cv_pct"])
            repetitions = int(row[f"{device}_repetitions"])
            if not (0 < minimum <= median <= maximum):
                raise ValueError(f"invalid min/median/max for {row['row_id']}/{device}")
            min_reps = 2 if (device == "rtx5060ti" and row["row_id"].startswith("m65536_")) else 3
            if batch < 1 or streams < 1 or repetitions < min_reps:
                raise ValueError(f"invalid B/S/repetitions for {row['row_id']}/{device}")
            if not 0 <= cv <= 10.0:
                raise ValueError(f"CV exceeds 10% for {row['row_id']}/{device}: {cv}")
            if not math.isclose(ratio, median / baseline, rel_tol=0, abs_tol=1e-5):
                raise ValueError(f"EPYC 9554 speedup mismatch for {row['row_id']}/{device}")
        gpu_ratio = float(row["rtx5090_over_rtx5060ti"])
        expected_gpu_ratio = float(row["rtx5090_ops_per_s"]) / float(row["rtx5060ti_ops_per_s"])
        if not math.isclose(gpu_ratio, expected_gpu_ratio, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f"GPU ratio mismatch for {row['row_id']}")
    return rows


def display_ops(value: str | float) -> str:
    return f"{float(value):,.2f}"


def display_ratio(value: str | float) -> str:
    return f"{float(value):.2f}×"


def device_cell(row: dict[str, str], device: str) -> str:
    return "{} ({})".format(
        display_ops(row[f"{device}_ops_per_s"]),
        display_ratio(row[f"{device}_speedup_vs_epyc9554"]),
    )


def display_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    return [
        [
            row["message_shape"], row["issuer"], row["disclosure"], row["operation"],
            *(device_cell(row, device) for device, _ in DEVICES),
            display_ratio(row["rtx5090_over_rtx5060ti"]),
        ]
        for row in rows
    ]


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
        "{": r"\{", "}": r"\}", "—": r"--", "×": r"$\times$",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def render_markdown(rows: list[dict[str, str]]) -> None:
    headers = [
        "Message shape", "Issuer", "Disclosure", "Operation",
        "EPYC 7763 (ops/s)", "EPYC 9554 (ops/s)",
        "RTX 5060 Ti (ops/s)", "RTX 5090 (ops/s)", "RTX 5090 / RTX 5060 Ti",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in display_rows(rows))
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_latex(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Frozen optimized BBS primitive capacity (\texttt{opt} commit \texttt{9cad7ca}): cuBBS-Cred GPU backend (RTX 5060 Ti 180W TDP, RTX 5090 400W host cap) versus the historical local ZKryptium v0.2.2-derived CPU baseline. Each cell reports median operations/s; parenthesized values are ratios relative to the same-row EPYC 9554 baseline.}",
        r"\label{tab:bbs-capacity-opt}",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrrr}",
        r"\toprule",
        r"Message shape & Issuer & Disclosure & Operation & EPYC 7763 (ops/s) & EPYC 9554 (ops/s) & RTX 5060 Ti (ops/s) & RTX 5090 (ops/s) & \shortstack{RTX 5090 /\\RTX 5060 Ti} \\",
        r"\midrule",
    ]
    previous_section = None
    for row, values in zip(rows, display_rows(rows)):
        if previous_section is not None and row["section"] != previous_section:
            lines.append(r"\midrule")
        lines.append(" & ".join(latex_escape(value) for value in values) + r" \\")
        previous_section = row["section"]
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{minipage}{\textwidth}\footnotesize\textit{Note:} Same uses one issuer per template; Unique assigns distinct issuers to template items, with templates reused across batches and streams. CPU columns retain the historical local ZKryptium v0.2.2-derived baseline; GPU columns use the optimized cuBBS-Cred GPU backend (\texttt{CUBBS\_BBS\_FORCE\_GPU=1}, commit \texttt{9cad7ca}, 0 failed items across all 32 rows). RTX 5060 Ti was measured on Intel Core Ultra 7 265K (180W TDP); RTX 5090 was measured on Vast.ai AMD Ryzen 9 9950X3D2 (32T, 62GB, 400W host power limit). Hardware telemetry (100ms \texttt{nvidia-smi}) is archived in \texttt{table\_bbs\_capacity\_telemetry.csv}.\end{minipage}",
        r"\end{table*}",
        "",
    ])
    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def render_figure(rows: list[dict[str, str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    headers = [
        "Message\nshape", "Issuer", "Disclosure", "Operation",
        "EPYC 7763 (ops/s)", "EPYC 9554 (ops/s)",
        "RTX 5060 Ti (ops/s)", "RTX 5090 (ops/s)", "RTX 5090 /\nRTX 5060 Ti",
    ]
    values = display_rows(rows)
    fig, ax = plt.subplots(figsize=(11.69, 12.2))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    widths = [0.082, 0.055, 0.066, 0.081, 0.155, 0.155, 0.166, 0.166, 0.092]
    table = ax.table(
        cellText=values, colLabels=headers, cellLoc="center", colLoc="center",
        colWidths=widths, bbox=[0.008, 0.006, 0.984, 0.988],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    section_colors = ("#FBF4EA", "#FFFDFC")
    section_index = -1
    previous_section = None
    for row_index, row in enumerate(rows, start=1):
        if row["section"] != previous_section:
            section_index += 1
            previous_section = row["section"]
        for column_index in range(len(headers)):
            cell = table[(row_index, column_index)]
            cell.set_facecolor(section_colors[section_index % 2])
            cell.set_edgecolor("#B7C5D8")
            cell.set_linewidth(0.35)
            cell.PAD = 0.018
    for column_index in range(len(headers)):
        cell = table[(0, column_index)]
        cell.set_facecolor("#F0E4D5")
        cell.set_edgecolor("#9AA9BC")
        cell.set_linewidth(0.55)
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontsize(7.8)
        cell.PAD = 0.025

    fig.canvas.draw()
    left = table[(0, 0)].get_x()
    right_cell = table[(0, len(headers) - 1)]
    right = right_cell.get_x() + right_cell.get_width()
    separator_rows = [
        index for index in range(1, len(rows))
        if rows[index]["section"] != rows[index - 1]["section"]
    ]
    for data_index in separator_rows:
        cell = table[(data_index + 1, 0)]
        y = cell.get_y() + cell.get_height()
        ax.plot([left, right], [y, y], color="#222222", linewidth=0.8,
                transform=ax.transAxes, clip_on=False, zorder=10)
    top_cell = table[(0, 0)]
    top = top_cell.get_y() + top_cell.get_height()
    bottom = table[(len(rows), 0)].get_y()
    ax.plot([left, right], [top, top], color="#111111", linewidth=1.1,
            transform=ax.transAxes, clip_on=False, zorder=10)
    ax.plot([left, right], [bottom, bottom], color="#111111", linewidth=1.1,
            transform=ax.transAxes, clip_on=False, zorder=10)

    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(
            FIG_BASENAME.with_suffix(suffix), dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.04,
        )
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    render_markdown(rows)
    render_latex(rows)
    render_figure(rows)
    print(f"validated {len(rows)} rows and rendered {MD_OUT.name}, {TEX_OUT.name}, and {FIG_BASENAME.name}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
