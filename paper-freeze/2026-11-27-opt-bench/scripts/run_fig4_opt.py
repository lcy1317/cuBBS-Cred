#!/usr/bin/env python3
"""Run Figure 4 mixed-shape issuer compatibility & Verify pre-B aggregation ablation
on RTX 5060 Ti (opt-bench), keeping archived EPYC 9554 NUMA1 CPU reference,
and write fig4_source_data.csv, fig4_summary.csv, data_fig4.mat."""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import savemat

REPO = Path("/home/lcy/work/cuBBS-Cred")
BIN = REPO / "target/release/cubbs-bench"
LIB = REPO / "target/native/cuda/libcubbs_bbs_cuda.so"
FIG4_DIR = REPO / "paper-figures/fig4"
RAW_DIR = FIG4_DIR / "raw/rtx5060ti-opt-20261127-mixed-ablation"
OLD_SUMMARY_CSV = FIG4_DIR / "fig4_summary.csv"

OPERATIONS = [
    ("sign", "Sign"),
    ("verify-signature", "Verify"),
    ("proof-gen", "ProofGen"),
    ("proof-verify", "ProofVerify"),
]
CONDITIONS = [
    ("production-unique", "GPU / unique issuer", "production", True, True, "factor-gpu"),
    ("production-shared", "GPU / shared issuer", "production", False, True, "factor-gpu"),
    ("serial-shape-shared", "Serial shape buckets / shared issuer", "serial-shape-buckets", False, True, "legacy-dispatch-control"),
    ("serial-shape-unique", "Serial shape buckets / unique issuer", "serial-shape-buckets", True, True, "legacy-dispatch-control"),
    ("aggregate-off-shared", "GPU / shared issuer / pre-B aggregation off", "production", False, False, "verify-policy-control"),
    ("aggregate-off-unique", "GPU / unique issuer / pre-B aggregation off", "production", True, False, "verify-policy-control"),
]
BATCH = 512
STREAMS = 1
DURATION = 5.0
REPEATS = 3


def load_cpu_reference() -> dict[str, dict]:
    """Read archived EPYC 9554 NUMA1 full-core CPU reference rows from existing fig4_summary.csv."""
    ref = {}
    with OLD_SUMMARY_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["condition"] == "cpu-unique":
                ref[row["operation"]] = {
                    "median": float(row["median_operations_per_s"]),
                    "minimum": float(row["minimum_operations_per_s"]),
                    "maximum": float(row["maximum_operations_per_s"]),
                    "cv_pct": float(row["cv_operations_per_s_pct"]),
                    "repetitions": int(row["repetitions"]),
                    "batch_size": int(row["batch_size"]),
                    "streams": int(row["streams"]),
                }
    assert len(ref) == 4, f"expected 4 CPU reference ops, got {ref}"
    return ref


def run_condition(dispatch: str, unique_issuer: bool, agg_enabled: bool, duration: float) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUBBS_BBS_GPU_LIBRARY"] = str(LIB)
    env["CUBBS_BBS_FORCE_GPU"] = "1"
    env["CUBBS_BBS_REQUIRE_HARDWARE_GPU"] = "1"
    env.pop("CUBBS_BBS_VERIFY_AGGREGATE_MIN_ITEMS", None)
    if agg_enabled:
        env.pop("CUBBS_BBS_PAIRING_AGGREGATE", None)
    else:
        env["CUBBS_BBS_PAIRING_AGGREGATE"] = "0"

    cmd = [
        str(BIN),
        "--bbs-throughput-stress",
        "--bbs-backend", "gpu",
        "--require-gpu",
        "--operation", "all",
        "--messages", "16,64,128",
        "--mixed-bbs-shapes",
        "--mixed-bbs-dispatch", dispatch,
        "--bbs-batch", str(BATCH),
        "--streams", str(STREAMS),
        "--disclosed-ratio", "0.5",
        "--no-bbs-caches",
        "--gpu-smi-sample",
        "--duration-secs", str(duration),
    ]
    if unique_issuer:
        cmd.append("--unique-issuer-per-item")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    rep = json.loads(p.stdout)
    assert rep["backendInfo"]["hardwareAcceleration"] is True
    assert rep["backendInfo"]["fallback"] is False
    for m in rep["metrics"]:
        assert m["failed"] == 0 and m["itemsPerSec"] > 0, f"failed metric: {m}"
    return rep


def main():
    cpu_ref = load_cpu_reference()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Preflight warmup
    print("=== Fig 4 preflight warmup ===", flush=True)
    run_condition("production", False, True, 1.0)
    run_condition("production", True, True, 1.0)

    # Formal runs
    raw_rates: dict[tuple[str, str], list[float]] = {}
    source_rows = []
    for rep_idx in range(1, REPEATS + 1):
        for cond_key, cond_label, dispatch, unique_issuer, agg_on, role in CONDITIONS:
            report = run_condition(dispatch, unique_issuer, agg_on, DURATION)
            out_path = RAW_DIR / cond_key / f"r{rep_idx}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2))
            by_op = {m["operation"]: m for m in report["metrics"]}
            for op_key, op_label in OPERATIONS:
                m = by_op[op_key]
                rate = float(m["itemsPerSec"])
                raw_rates.setdefault((cond_key, op_key), []).append(rate)
                source_rows.append({
                    "condition": cond_key,
                    "condition_label": cond_label,
                    "analysis_role": role,
                    "dispatch_mode": dispatch,
                    "unique_issuer_per_item": unique_issuer,
                    "verify_pre_b_aggregate_enabled": agg_on,
                    "batch_size": BATCH,
                    "streams": STREAMS,
                    "duration_seconds": DURATION,
                    "repetition": rep_idx,
                    "operation": op_key,
                    "operation_label": op_label,
                    "items": int(m["items"]),
                    "throughput_operations_per_s": rate,
                    "failed": int(m["failed"]),
                    "raw_file": str(out_path.relative_to(FIG4_DIR)),
                })
            print(
                f"  [r{rep_idx}/{REPEATS}] {cond_key:<22} "
                f"Sign={by_op['sign']['itemsPerSec']:>9,.1f} "
                f"Ver={by_op['verify-signature']['itemsPerSec']:>10,.1f} "
                f"PG={by_op['proof-gen']['itemsPerSec']:>9,.1f} "
                f"PV={by_op['proof-verify']['itemsPerSec']:>9,.1f}",
                flush=True,
            )

    with (FIG4_DIR / "fig4_source_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        w.writeheader()
        w.writerows(source_rows)

    summary_rows = []
    # CPU reference rows first
    for op_key, op_label in OPERATIONS:
        c = cpu_ref[op_key]
        summary_rows.append({
            "analysis_role": "factor-reference",
            "condition": "cpu-unique",
            "condition_label": "CPU / unique issuer (capacity reference)",
            "operation": op_key,
            "operation_label": op_label,
            "batch_size": c["batch_size"],
            "streams": c["streams"],
            "repetitions": c["repetitions"],
            "median_operations_per_s": c["median"],
            "minimum_operations_per_s": c["minimum"],
            "maximum_operations_per_s": c["maximum"],
            "cv_operations_per_s_pct": c["cv_pct"],
        })

    stat_by = {}
    for cond_key, cond_label, _, _, _, role in CONDITIONS:
        for op_key, op_label in OPERATIONS:
            vals = raw_rates[(cond_key, op_key)]
            med = statistics.median(vals)
            mn = min(vals)
            mx = max(vals)
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cv = sd / mean * 100.0 if mean > 0 else 0.0
            stat_by[(cond_key, op_key)] = (med, mn, mx)
            summary_rows.append({
                "analysis_role": role,
                "condition": cond_key,
                "condition_label": cond_label,
                "operation": op_key,
                "operation_label": op_label,
                "batch_size": BATCH,
                "streams": STREAMS,
                "repetitions": REPEATS,
                "median_operations_per_s": med,
                "minimum_operations_per_s": mn,
                "maximum_operations_per_s": mx,
                "cv_operations_per_s_pct": cv,
            })

    with (FIG4_DIR / "fig4_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Build data_fig4.mat matching render_fig4_fullcore_v3.py
    gpu_speedup = np.zeros((4, 2), dtype=float)
    gpu_lower = np.zeros((4, 2), dtype=float)
    gpu_upper = np.zeros((4, 2), dtype=float)
    issuer_uplift = np.zeros(4, dtype=float)
    verify_policy = np.zeros(2, dtype=float)

    for oi, (op_key, _) in enumerate(OPERATIONS):
        c = cpu_ref[op_key]
        u_med, u_min, u_max = stat_by[("production-unique", op_key)]
        s_med, s_min, s_max = stat_by[("production-shared", op_key)]
        gpu_speedup[oi, 0] = u_med / c["median"]
        gpu_lower[oi, 0] = u_min / c["maximum"]
        gpu_upper[oi, 0] = u_max / c["minimum"]
        gpu_speedup[oi, 1] = s_med / c["median"]
        gpu_lower[oi, 1] = s_min / c["maximum"]
        gpu_upper[oi, 1] = s_max / c["minimum"]
        issuer_uplift[oi] = s_med / u_med

    verify_policy[0] = stat_by[("production-shared", "verify-signature")][0] / stat_by[("aggregate-off-shared", "verify-signature")][0]
    verify_policy[1] = stat_by[("production-unique", "verify-signature")][0] / stat_by[("aggregate-off-unique", "verify-signature")][0]

    max_val = float(gpu_upper.max())
    x_max = 1000.0 if max_val > 250.0 else (400.0 if max_val > 120.0 else 150.0)
    ticks = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    ticks = [t for t in ticks if t <= x_max]

    savemat(
        str(FIG4_DIR / "data_fig4.mat"),
        {
            "cpu_reference_speedup_median": gpu_speedup,
            "cpu_reference_speedup_lower": gpu_lower,
            "cpu_reference_speedup_upper": gpu_upper,
            "issuer_sharing_effect_median": issuer_uplift,
            "verify_policy_effect_median": verify_policy,
            "plot_x_min": np.array([0.85]),
            "plot_x_max": np.array([x_max]),
            "plot_x_ticks": np.asarray(ticks, dtype=float),
        },
    )
    print("\n=== Fig 4 Summary ===")
    for oi, (op_key, op_label) in enumerate(OPERATIONS):
        print(
            f"  {op_label:<12} GPU_uniq/CPU={gpu_speedup[oi,0]:>6.1f}x  "
            f"GPU_shared/CPU={gpu_speedup[oi,1]:>6.1f}x  U_issuer={issuer_uplift[oi]:>5.2f}x"
        )
    print(f"  Verify pre-B agg control: shared={verify_policy[0]:.2f}x, unique={verify_policy[1]:.2f}x")


if __name__ == "__main__":
    main()
