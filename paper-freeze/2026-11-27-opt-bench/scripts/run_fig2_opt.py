#!/usr/bin/env python3
"""Run complete Figure 2 MSM route ablation on RTX 5060 Ti (opt-bench) and regenerate CSV/MAT/plots."""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/lcy/work/cuBBS-Cred")
BENCH = REPO / "target/native/cuda/msm_route_bench"
FIG2_DIR = REPO / "paper-figures/fig2"
RAW_DIR = FIG2_DIR / "raw/opt-20261127"

ROUTE1_BATCHES = [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
COMMON_TERMS = [18, 66, 130, 258, 514, 1026, 2050, 4098, 8194]
REPEATS = [1, 2, 3]
DUR_R1 = 1.5
DUR_COMMON = 2.0
WARMUP = 2


def run_bench(layout: str, terms: int, route: str, batch: int, duration: float, warmup: int = WARMUP) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        str(BENCH),
        "--layout", layout,
        "--terms", str(terms),
        "--route", route,
        "--batch", str(batch),
        "--duration-secs", str(duration),
        "--warmup", str(warmup),
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    data = json.loads(p.stdout)
    assert data["correctnessPassed"] is True and data["failed"] == 0, f"failed: {data}"
    return data


def summarize(values: list[float]) -> dict:
    med = statistics.median(values)
    mn = min(values)
    mx = max(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = (sd / mean * 100.0) if mean > 0 else 0.0
    return dict(median=med, minimum=mn, maximum=mx, mean=mean, stdev=sd, cv_pct=cv)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    r1_dir = RAW_DIR / "route1-batch-sweep"
    sh_dir = RAW_DIR / "shape-sweep"
    r1_dir.mkdir(parents=True, exist_ok=True)
    sh_dir.mkdir(parents=True, exist_ok=True)

    # Panel (a): Variable K=3 batch sweep (R1 vs R4, 3 reps) + auto-gate check
    print("=== Fig 2 Panel (a): Variable K=3 batch sweep (R1 vs R4) ===", flush=True)
    r1_rows = []
    for rep in REPEATS:
        for b in ROUTE1_BATCHES:
            order = ("1", "4") if rep % 2 == 1 else ("4", "1")
            for rt in order:
                d = run_bench("variable", 3, rt, b, DUR_R1)
                assert d["routeObserved"] == int(rt), f"route mismatch: {d}"
                fname = f"k3-b{b}-r{rt}-rep{rep}.json"
                (r1_dir / fname).write_text(json.dumps(d, indent=2))
                r1_rows.append({
                    "batch_size": b,
                    "variant": f"r{rt}",
                    "repeat": rep,
                    "route_requested": rt,
                    "route_observed": d["routeObserved"],
                    "term_count": 3,
                    "rows_per_sec": float(d["rowsPerSec"]),
                    "elapsed_secs": float(d["elapsedSecs"]),
                    "batches": int(d["batches"]),
                    "raw_file": f"raw/opt-20261127/route1-batch-sweep/{fname}",
                })
        print(f"  Panel (a) rep {rep}/3 complete", flush=True)

    # Auto-gate check
    for b in ROUTE1_BATCHES:
        exp = 1 if b >= 128 else 4
        d = run_bench("variable", 3, "auto", b, 0.05, warmup=1)
        assert d["routeObserved"] == exp, f"auto gate failed B={b}: got {d['routeObserved']} expected {exp}"
    print("  Panel (a) auto-gates verified (R4 < 128, R1 >= 128)", flush=True)

    # Panel (b): Common-point shape sweep B=1024, K=18..8194 (auto R2/R3 vs R4, 3 reps)
    print("=== Fig 2 Panel (b): Common-point shape sweep (B=1024, K=18..8194) ===", flush=True)
    sh_rows = []
    for rep in REPEATS:
        for k in COMMON_TERMS:
            exp_auto = 2 if k <= 2100 else 3
            order = (("auto", exp_auto), ("4", 4)) if rep % 2 == 1 else (("4", 4), ("auto", exp_auto))
            for rt, exp in order:
                variant = "auto" if rt == "auto" else "r4"
                d = run_bench("common", k, rt, 1024, DUR_COMMON)
                assert d["routeObserved"] == exp, f"common K={k} rt={rt}: got {d['routeObserved']} expected {exp}"
                fname = f"common-k{k}-{variant}-r{rep}.json"
                (sh_dir / fname).write_text(json.dumps(d, indent=2))
                sh_rows.append({
                    "layout": "common",
                    "variant": variant,
                    "term_count": k,
                    "message_count": k - 2,
                    "repeat": rep,
                    "route_requested": rt,
                    "route_observed": d["routeObserved"],
                    "rows_per_sec": float(d["rowsPerSec"]),
                    "elapsed_secs": float(d["elapsedSecs"]),
                    "batches": int(d["batches"]),
                    "batch_size": 1024,
                    "raw_file": f"raw/opt-20261127/shape-sweep/{fname}",
                })
        print(f"  Panel (b) rep {rep}/3 complete", flush=True)

    # Write CSVs
    with (FIG2_DIR / "fig2_route1_batch_source_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r1_rows[0].keys()))
        w.writeheader()
        w.writerows(r1_rows)

    r1_summary = []
    for b in ROUTE1_BATCHES:
        for var in ("r1", "r4"):
            sub = [r for r in r1_rows if r["batch_size"] == b and r["variant"] == var]
            s = summarize([r["rows_per_sec"] for r in sub])
            r1_summary.append({
                "batch_size": b, "variant": var,
                "route_observed": sub[0]["route_observed"], "term_count": 3,
                "unit": "MSM rows/s", **s,
            })
    with (FIG2_DIR / "fig2_route1_batch_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r1_summary[0].keys()))
        w.writeheader()
        w.writerows(r1_summary)

    with (FIG2_DIR / "fig2_shape_source_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sh_rows[0].keys()))
        w.writeheader()
        w.writerows(sh_rows)

    sh_summary = []
    for var in ("auto", "r4"):
        for k in COMMON_TERMS:
            sub = [r for r in sh_rows if r["term_count"] == k and r["variant"] == var]
            s = summarize([r["rows_per_sec"] for r in sub])
            sh_summary.append({
                "layout": "common", "variant": var, "term_count": k,
                "message_count": k - 2, "route_observed": sub[0]["route_observed"],
                "unit": "MSM rows/s", **s,
            })
    with (FIG2_DIR / "fig2_shape_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sh_summary[0].keys()))
        w.writeheader()
        w.writerows(sh_summary)

    # Print summary tables
    print("\n--- Panel (a) Summary (K=3 Variable-Base R1 vs R4) ---")
    r1_by = {(r["batch_size"], r["variant"]): r for r in r1_summary}
    for b in ROUTE1_BATCHES:
        a = r1_by[(b, "r1")]["median"]
        c = r1_by[(b, "r4")]["median"]
        print(f"  B={b:<6} R1={a:>10,.1f}  R4={c:>10,.1f}  R1/R4={a/c:.2f}x")

    print("\n--- Panel (b) Summary (Common-Points B=1024 Auto R2/R3 vs R4) ---")
    sh_by = {(r["term_count"], r["variant"]): r for r in sh_summary}
    for k in COMMON_TERMS:
        a = sh_by[(k, "auto")]["median"]
        rt = sh_by[(k, "auto")]["route_observed"]
        c = sh_by[(k, "r4")]["median"]
        print(f"  M={k-2:<5} (K={k:<4}) R{rt}={a:>10,.1f}  R4={c:>10,.1f}  Auto/R4={a/c:.2f}x")


if __name__ == "__main__":
    main()
