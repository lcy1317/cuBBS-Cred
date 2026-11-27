#!/usr/bin/env python3
"""Run Figure 3 scalability sweep on RTX 5060 Ti (opt-bench) across M=16..65536
with B=24, S=8 (plus B=16, S=4 control at M=2099, 4097, 65536), then write
fig3_source_data.csv, fig3_message_length_summary.csv, data_fig3.mat, and render Fig 3."""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

REPO = Path("/home/lcy/work/cuBBS-Cred")
BIN = REPO / "target/release/cubbs-bench"
LIB = REPO / "target/native/cuda/libcubbs_bbs_cuda.so"
FIG3_DIR = REPO / "paper-figures/fig3"
RAW_DIR = FIG3_DIR / "raw/rtx5060ti-opt-20261127-b24s8"

# Original 21 points + 2098/2099 to capture exact Lim-Lee -> 10-bit Pippenger (K=2100) transition
MESSAGES = [
    16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2098, 2099,
    3072, 4095, 4096, 4097, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536,
]
OPERATIONS = [
    ("sign", "Sign"),
    ("verify", "Verify"),
    ("proof-gen", "ProofGen"),
    ("proof-verify", "ProofVerify"),
]
BATCH = 24
STREAMS = 8
REPEATS = 3
DURATION = 3.0  # 3s sustained per op/shape/repeat -> ~15 min total for 23*4*3 runs


def run_one(op_cli: str, m: int, batch: int, streams: int, duration: float) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUBBS_BBS_GPU_LIBRARY"] = str(LIB)
    env["CUBBS_BBS_FORCE_GPU"] = "1"
    env["CUBBS_BBS_REQUIRE_HARDWARE_GPU"] = "1"
    env.pop("CUBBS_BBS_VERIFY_AGGREGATE_MIN_ITEMS", None)
    env.pop("CUBBS_BBS_PAIRING_AGGREGATE", None)
    cmd = [
        str(BIN),
        "--bbs-throughput-stress",
        "--bbs-backend", "gpu",
        "--require-gpu",
        "--operation", op_cli,
        "--messages", str(m),
        "--disclosed-ratio", "0.5",
        "--bbs-batch", str(batch),
        "--streams", str(streams),
        "--duration-secs", str(duration),
        "--no-bbs-caches",
        "--gpu-throughput-max",
        "--gpu-smi-sample",
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    rep = json.loads(p.stdout)
    info = rep["backendInfo"]
    assert info["hardwareAcceleration"] is True and info["fallback"] is False, f"no hw gpu: {info}"
    metric = rep["metrics"][0]
    assert metric["failed"] == 0 and metric["items"] > 0 and metric["itemsPerSec"] > 0, f"bad metric: {metric}"
    return rep


def summarize(vals: list[float]) -> dict:
    med = statistics.median(vals)
    mn = min(vals)
    mx = max(vals)
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    cv = (sd / mean * 100.0) if mean > 0 else 0.0
    return dict(median=med, minimum=mn, maximum=mx, mean=mean, stdev=sd, cv_pct=cv)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = []
    # Quick warmup at M=64 and M=4096
    print("=== Fig 3 warmup ===", flush=True)
    for op_cli, _ in OPERATIONS:
        run_one(op_cli, 64, BATCH, STREAMS, 0.5)

    for rep_idx in range(1, REPEATS + 1):
        m_order = MESSAGES if rep_idx % 2 == 1 else list(reversed(MESSAGES))
        for m in m_order:
            dur = 4.0 if m >= 32768 else DURATION
            for op_cli, op_label in OPERATIONS:
                report = run_one(op_cli, m, BATCH, STREAMS, dur)
                out_path = RAW_DIR / op_cli / f"m{m}" / f"r{rep_idx}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(report, indent=2))
                metric = report["metrics"][0]
                smi = report.get("gpuSmi") or {}
                policy = "limlee-b24-s8" if (m + 2) <= 2100 else "pippenger10-b24-s8"
                source_rows.append({
                    "device": "RTX 5060 Ti",
                    "operation": op_label,
                    "message_count": m,
                    "batch_size": BATCH,
                    "streams": STREAMS,
                    "scheduling_policy": policy,
                    "series_role": "route-aware-capacity",
                    "repeat": rep_idx,
                    "operations_per_s": float(metric["itemsPerSec"]),
                    "items": int(metric["items"]),
                    "elapsed_ms": float(metric["elapsedMs"]),
                    "failed": int(metric["failed"]),
                    "gpu_util_pct": float(smi.get("utilizationGpuMedianPct", 0.0) or 0.0),
                    "power_w": float(smi.get("powerDrawMedianW", 0.0) or 0.0),
                    "memory_mib": float(smi.get("memoryUsedMaxMiB", 0.0) or 0.0),
                    "raw_file": str(out_path.relative_to(FIG3_DIR)),
                })
            # Print compact progress per M
            last4 = {r["operation"]: r["operations_per_s"] for r in source_rows[-4:]}
            print(
                f"  [r{rep_idx}/{REPEATS}] M={m:<5}  Sign={last4['Sign']:>9,.1f}  "
                f"Ver={last4['Verify']:>10,.1f}  PG={last4['ProofGen']:>9,.1f}  PV={last4['ProofVerify']:>9,.1f}",
                flush=True,
            )

    with (FIG3_DIR / "fig3_source_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        w.writeheader()
        w.writerows(source_rows)

    summary_rows = []
    med_mat = np.zeros((4, len(MESSAGES)), dtype=float)
    min_mat = np.zeros((4, len(MESSAGES)), dtype=float)
    max_mat = np.zeros((4, len(MESSAGES)), dtype=float)

    for oi, (_, op_label) in enumerate(OPERATIONS):
        for mi, m in enumerate(MESSAGES):
            sub = [r for r in source_rows if r["operation"] == op_label and r["message_count"] == m]
            s = summarize([r["operations_per_s"] for r in sub])
            med_mat[oi, mi] = s["median"]
            min_mat[oi, mi] = s["minimum"]
            max_mat[oi, mi] = s["maximum"]
            summary_rows.append({
                "device": "RTX 5060 Ti",
                "operation": op_label,
                "message_count": m,
                "batch_size": BATCH,
                "streams": STREAMS,
                "scheduling_policy": sub[0]["scheduling_policy"],
                "series_role": "route-aware-capacity",
                "repetitions": REPEATS,
                "minimum_operations_per_s": s["minimum"],
                "median_operations_per_s": s["median"],
                "maximum_operations_per_s": s["maximum"],
                "mean_operations_per_s": s["mean"],
                "stdev_operations_per_s": s["stdev"],
                "cv_pct": s["cv_pct"],
                "median_message_positions_per_s": s["median"] * m,
                "median_gpu_util_pct": statistics.median([r["gpu_util_pct"] for r in sub]),
                "median_power_w": statistics.median([r["power_w"] for r in sub]),
                "maximum_memory_mib": max(r["memory_mib"] for r in sub),
            })

    with (FIG3_DIR / "fig3_message_length_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    savemat(
        str(FIG3_DIR / "data_fig3.mat"),
        {
            "messages": np.asarray(MESSAGES, dtype=float),
            "median": med_mat,
            "minimum": min_mat,
            "maximum": max_mat,
        },
    )
    print("Wrote Fig 3 CSVs and data_fig3.mat!", flush=True)


if __name__ == "__main__":
    main()
