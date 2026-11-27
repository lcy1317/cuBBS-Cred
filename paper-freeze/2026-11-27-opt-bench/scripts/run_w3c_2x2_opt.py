#!/usr/bin/env python3
"""Run W3C HTTP 2x2 ablation (B0/C0, B0/C1, B1/C0, B1/C1) across all 4 ops
(issue, verify-base, proof-generation, proof-verification) with cubbs-w3c-service + w3c-http-op-load."""
from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OPS = {
    "issue": ("credentials/issue", "issue.json", "document"),
    "verify-base": ("credentials/verify", "verify-base.json", "verified"),
    "proof-generation": ("credentials/derive", "derive.json", "document"),
    "proof-verification": ("credentials/verify", "verify-derived.json", "verified"),
}


def wait_health(port: int, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode())
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"service on port {port} did not become healthy")


def prewarm_one(port: int, route: str, payload_bytes: bytes, expect: str) -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=payload_bytes,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        body = json.loads(resp.read().decode())
    if expect == "document":
        assert body.get("proof", {}).get("proofValue"), f"bad doc response: {body}"
    else:
        assert body.get("verified") is True, f"bad verify response: {body}"


def run_cell_once(
    service_bin: str,
    load_bin: str,
    gpu_lib: str,
    fixtures_dir: Path,
    out_dir: Path,
    op: str,
    batch_on: bool,
    cache_on: bool,
    endpoints: int,
    concurrency_per_ep: int,
    max_batch: int,
    pipelines: int,
    interval_ms: int,
    warmup_secs: int,
    duration_secs: int,
    port_base: int,
    tokio_workers: int,
) -> dict:
    route, payload_name, expect = OPS[op]
    payload_path = fixtures_dir / payload_name
    payload_bytes = payload_path.read_bytes()
    out_dir.mkdir(parents=True, exist_ok=True)

    service_procs = []
    try:
        for i in range(endpoints):
            port = port_base + i
            ep_dir = out_dir / f"e{i}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": os.environ.get("GPU_INDEX", "0"),
                "TOKIO_WORKER_THREADS": str(tokio_workers),
                "RUST_LOG": "warn",
                "CUBBS_PORT": str(port),
                "CUBBS_BBS_BACKEND": "gpu",
                "CUBBS_BBS_GPU_LIBRARY": gpu_lib,
                "CUBBS_BBS_FORCE_GPU": "1",
                "CUBBS_BBS_REQUIRE_HARDWARE_GPU": "1",
                "CUBBS_BBS_FUSED_COMPRESS_WORKERS": str(min(8, max(2, tokio_workers // 2))),
                "CUBBS_BBS_FUSED_COMPRESS_PARALLEL_MIN_ITEMS": "8",
                "CUBBS_BBS_HOST_POOL": "1",
                "CUBBS_BBS_HOST_POOL_WORKERS": str(min(32, max(4, tokio_workers))),
                "CUBBS_W3C_BATCH_INTERVAL_MS": str(interval_ms if batch_on else 1),
                "CUBBS_W3C_QUEUE_CAPACITY": "32768",
                "CUBBS_W3C_PIPELINES": str(pipelines),
                "CUBBS_W3C_ISSUE_MAX_BATCH_SIZE": str(max_batch if batch_on else 1),
                "CUBBS_W3C_DERIVE_MAX_BATCH_SIZE": str(max_batch if batch_on else 1),
                "CUBBS_W3C_VERIFY_BASE_MAX_BATCH_SIZE": str(max_batch if batch_on else 1),
                "CUBBS_W3C_PROOF_VERIFY_MAX_BATCH_SIZE": str(max_batch if batch_on else 1),
                "CUBBS_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS": "32768",
                "CUBBS_W3C_PREPARED_CACHE_CAPACITY": "65536" if cache_on else "0",
                "CUBBS_W3C_CANON_CACHE_CAPACITY": "65536" if cache_on else "0",
                "CUBBS_W3C_LABEL_CACHE_CAPACITY": "16384" if cache_on else "0",
                "CUBBS_W3C_VERIFY_CACHE_CAPACITY": "0",
                "CUBBS_W3C_LISTEN_BACKLOG": "32768",
                "CUBBS_W3C_JSONLD_WORKERS": str(max(16, tokio_workers)),
                "CUBBS_W3C_SHARED_JSONLD_LOADER": "1",
            })
            log_f = (ep_dir / "service.log").open("wb")
            proc = subprocess.Popen([service_bin], env=env, stdout=log_f, stderr=subprocess.STDOUT)
            service_procs.append((proc, log_f))
            h = wait_health(port)
            assert h["status"] == "ok", f"bad health: {h}"
            assert h["bbs_backend_active"] == "cuda-abi-v2-force-gpu", f"bad backend: {h}"
            assert h["bbs_backend_fallback"] is False
            assert h["verification_result_cache"]["capacity"] == 0
            prewarm_one(port, route, payload_bytes, expect)

        load_procs = []
        for i in range(endpoints):
            port = port_base + i
            ep_dir = out_dir / f"e{i}"
            rep_path = ep_dir / "report.json"
            env = os.environ.copy()
            env["TOKIO_WORKER_THREADS"] = str(tokio_workers)
            cmd = [
                load_bin,
                "--operation", op,
                "--url", f"http://127.0.0.1:{port}/{route}",
                "--payload", str(payload_path),
                "--expect", expect,
                "--health-url", f"http://127.0.0.1:{port}/healthz",
                "--concurrency", str(concurrency_per_ep),
                "--warmup-secs", str(warmup_secs),
                "--duration-secs", str(duration_secs),
                "--output", str(rep_path),
            ]
            out_f = (ep_dir / "load.log").open("wb")
            p = subprocess.Popen(cmd, env=env, stdout=out_f, stderr=subprocess.STDOUT)
            load_procs.append((p, out_f, rep_path))

        reports = []
        for p, out_f, rep_path in load_procs:
            rc = p.wait()
            out_f.close()
            if rc != 0:
                raise RuntimeError(f"load generator exited with {rc}: {(rep_path.parent / 'load.log').read_text()[:400]}")
            rep = json.loads(rep_path.read_text())
            assert rep["failed"] == 0 and rep["succeeded"] > 0, f"failed requests: {rep}"
            reports.append(rep)

        total_ops = sum(float(r["operationsPerSecond"]) for r in reports)
        max_p99 = max(float(r["latencyP99Ms"]) for r in reports)
        summary = {
            "operation": op,
            "endpoints": endpoints,
            "totalConcurrency": endpoints * concurrency_per_ep,
            "totalOperationsPerSecond": total_ops,
            "p99Ms": max_p99,
            "totalSucceeded": sum(int(r["succeeded"]) for r in reports),
            "totalFailed": sum(int(r["failed"]) for r in reports),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        for proc, log_f in service_procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            log_f.close()
        time.sleep(0.4)


def main():
    profile = sys.argv[1]  # "rtx5060ti" or "rtx5090"
    out_root = Path(sys.argv[2])
    repo = Path(sys.argv[3])
    fixtures = Path(sys.argv[4])
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    dur = int(sys.argv[6]) if len(sys.argv) > 6 else 10

    service_bin = str(repo / "target/release/cubbs-w3c-service")
    load_bin = str(repo / "target/release/w3c-http-op-load")
    gpu_lib = str(repo / "target/native/cuda/libcubbs_bbs_cuda.so")
    ncpu = min(48, os.cpu_count() or 16)

    if profile == "rtx5060ti":
        # (cell_id, batch_on, cache_on, op_cfg: {op: (endpoints, conc_per_ep, max_batch, pipelines, interval_ms, warmup_s)})
        cells = [
            ("b0_c0", False, False, {
                "issue": (1, 24, 1, 8, 1, 1),
                "verify-base": (1, 64, 1, 8, 1, 1),
                "proof-generation": (1, 24, 1, 8, 1, 1),
                "proof-verification": (1, 24, 1, 8, 1, 1),
            }),
            ("b0_c1", False, True, {
                "issue": (1, 16, 1, 8, 1, 1),
                "verify-base": (1, 16, 1, 8, 1, 1),
                "proof-generation": (1, 16, 1, 8, 1, 1),
                "proof-verification": (1, 16, 1, 8, 1, 1),
            }),
            ("b1_c0", True, False, {
                "issue": (1, 2048, 1024, 8, 5, 2),
                "verify-base": (1, 4096, 2048, 8, 5, 2),
                "proof-generation": (1, 1024, 256, 8, 5, 2),
                "proof-verification": (1, 2048, 1024, 8, 5, 2),
            }),
            ("b1_c1", True, True, {
                "issue": (1, 6144, 2048, 8, 5, 2),
                "verify-base": (1, 14336, 4096, 8, 5, 2),
                "proof-generation": (1, 4096, 512, 8, 5, 2),
                "proof-verification": (1, 10240, 2048, 8, 5, 2),
            }),
        ]
    elif profile == "rtx5090":
        cells = [
            ("b0_c0", False, False, {
                "issue": (1, 64, 1, 8, 1, 1),
                "verify-base": (1, 128, 1, 8, 1, 1),
                "proof-generation": (1, 64, 1, 8, 1, 1),
                "proof-verification": (1, 64, 1, 8, 1, 1),
            }),
            ("b0_c1", False, True, {
                "issue": (1, 64, 1, 8, 1, 1),
                "verify-base": (1, 64, 1, 8, 1, 1),
                "proof-generation": (1, 32, 1, 8, 1, 1),
                "proof-verification": (1, 32, 1, 8, 1, 1),
            }),
            ("b1_c0", True, False, {
                "issue": (1, 4096, 2048, 8, 10, 2),
                "verify-base": (1, 4096, 2048, 8, 10, 2),
                "proof-generation": (1, 2048, 512, 8, 10, 2),
                "proof-verification": (1, 4096, 2048, 8, 10, 2),
            }),
            ("b1_c1", True, True, {
                "issue": (4, 4096, 4096, 4, 15, 3),
                "verify-base": (2, 8192, 4096, 4, 15, 3),
                "proof-generation": (2, 4096, 1024, 8, 5, 3),
                "proof-verification": (2, 8192, 2048, 8, 10, 3),
            }),
        ]
    else:
        raise SystemExit(f"unknown profile {profile}")

    out_root.mkdir(parents=True, exist_ok=True)
    results = {}
    port_cursor = 20100
    for cell_id, batch_on, cache_on, op_cfg in cells:
        for op, (eps, conc, max_b, pipes, int_ms, warm_s) in op_cfg.items():
            rates = []
            p99s = []
            tok_w = max(4, ncpu // max(1, eps))
            for r in range(1, reps + 1):
                run_dir = out_root / cell_id / op / f"r{r}"
                s = run_cell_once(
                    service_bin, load_bin, gpu_lib, fixtures, run_dir,
                    op, batch_on, cache_on, eps, conc, max_b, pipes, int_ms,
                    warm_s, dur, port_cursor, tok_w,
                )
                port_cursor += eps + 2
                rates.append(s["totalOperationsPerSecond"])
                p99s.append(s["p99Ms"])
            med = statistics.median(rates)
            mn = min(rates)
            mx = max(rates)
            mean = statistics.mean(rates)
            sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
            cv = sd / mean * 100.0 if mean > 0 else 0.0
            med_p99 = statistics.median(p99s)
            key = f"{cell_id}:{op}"
            results[key] = {
                "cell_id": cell_id,
                "batch_on": batch_on,
                "cache_on": cache_on,
                "operation": op,
                "endpoints": eps,
                "selected_concurrency": eps * conc,
                "median_ops_s": med,
                "min_ops_s": mn,
                "max_ops_s": mx,
                "cv_pct": cv,
                "median_p99_ms": med_p99,
                "repetitions": reps,
                "duration_s": dur,
            }
            print(
                f"  [{profile} {cell_id} {op:<18}] med={med:>10,.1f} req/s  "
                f"(min={mn:,.1f}, max={mx:,.1f}, cv={cv:.2f}%, p99={med_p99:.1f}ms, c={eps*conc})",
                flush=True,
            )
            (out_root / "summary_all.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
