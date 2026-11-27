#!/usr/bin/env python3
"""High-concurrency, CPU-pinned, GPU-monitored W3C HTTP 2x2 ablation runner for RTX 5090."""
from __future__ import annotations

import json
import os
import re
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


def detect_available_cpus() -> tuple[list[int], int]:
    """Return (sorted usable CPU IDs from sched_getaffinity, effective cgroup CPU quota)."""
    cpus = sorted(os.sched_getaffinity(0))
    eff = len(cpus)
    # Check cgroup v2 cpu.max
    for p in ("/sys/fs/cgroup/cpu.max",):
        try:
            txt = Path(p).read_text().strip().split()
            if len(txt) == 2 and txt[0] != "max":
                q = int(txt[0]) / int(txt[1])
                if q > 0:
                    eff = min(eff, max(1, int(round(q))))
        except Exception:
            pass
    # Check cgroup v1 quota
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if q > 0 and per > 0:
            eff = min(eff, max(1, int(round(q / per))))
    except Exception:
        pass
    return cpus, eff


def detect_numa_nodes() -> list[list[int]]:
    """Return list of CPU ID lists per NUMA node (filtered by sched_getaffinity)."""
    allowed = set(os.sched_getaffinity(0))
    nodes: list[list[int]] = []
    base = Path("/sys/devices/system/node")
    if base.exists():
        for d in sorted(base.glob("node[0-9]*"), key=lambda x: int(x.name[4:])):
            cpulist_f = d / "cpulist"
            if not cpulist_f.exists():
                continue
            ids: list[int] = []
            for part in cpulist_f.read_text().strip().split(","):
                if not part:
                    continue
                if "-" in part:
                    a, b = part.split("-", 1)
                    ids.extend(range(int(a), int(b) + 1))
                else:
                    ids.append(int(part))
            ids = [c for c in ids if c in allowed]
            if ids:
                nodes.append(ids)
    if not nodes:
        nodes = [sorted(allowed)]
    return nodes


def make_cpuset_slices(role_cpus: list[int], endpoints: int) -> list[str]:
    """Partition role_cpus into `endpoints` disjoint CPU-list strings."""
    n = len(role_cpus)
    slices: list[str] = []
    for i in range(endpoints):
        a = (i * n) // endpoints
        b = ((i + 1) * n) // endpoints
        chunk = role_cpus[a:b] if b > a else role_cpus
        slices.append(",".join(str(c) for c in chunk))
    return slices


def wait_health(port: int, timeout_s: float = 25.0) -> dict:
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


def parse_gpu_csv(csv_path: Path, duration_secs: int, sample_ms: int = 200) -> dict:
    if not csv_path.exists():
        return {}
    lines = [l.strip() for l in csv_path.read_text().splitlines() if l.strip()]
    timed_samples = max(1, (duration_secs * 1000 + sample_ms - 1) // sample_ms)
    tail = lines[-timed_samples:] if len(lines) > timed_samples else lines
    utils, powers, clocks = [], [], []
    for l in tail:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 6:
            continue
        try:
            utils.append(float(re.sub(r"[^0-9.]", "", parts[1])))
            powers.append(float(re.sub(r"[^0-9.]", "", parts[4])))
            clocks.append(float(re.sub(r"[^0-9.]", "", parts[5])))
        except ValueError:
            pass
    if not utils:
        return {}
    su = sorted(utils)
    return {
        "samples": len(utils),
        "meanUtilizationPct": round(statistics.mean(utils), 2),
        "medianUtilizationPct": round(statistics.median(utils), 2),
        "p95UtilizationPct": round(su[min(len(su) - 1, int(len(su) * 0.95))], 2),
        "maxUtilizationPct": round(max(utils), 2),
        "meanPowerW": round(statistics.mean(powers), 2),
        "maxPowerW": round(max(powers), 2),
        "meanSmClockMHz": round(statistics.mean(clocks), 1),
    }


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
    service_cpus: list[int],
    load_cpus: list[int],
    workers_per_ep: int,
    host_workers_per_ep: int,
    pin_cpus: bool = True,
) -> dict:
    route, payload_name, expect = OPS[op]
    payload_path = fixtures_dir / payload_name
    payload_bytes = payload_path.read_bytes()
    out_dir.mkdir(parents=True, exist_ok=True)

    svc_slices = make_cpuset_slices(service_cpus, endpoints)
    load_slices = make_cpuset_slices(load_cpus, endpoints)

    service_procs = []
    mon_proc = None
    mon_file = None
    gpu_csv = out_dir / "gpu-timeseries.csv"
    try:
        for i in range(endpoints):
            port = port_base + i
            ep_dir = out_dir / f"e{i}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": os.environ.get("GPU_INDEX", "0"),
                "TOKIO_WORKER_THREADS": str(workers_per_ep),
                "RUST_LOG": "warn",
                "CUBBS_PORT": str(port),
                "CUBBS_BBS_BACKEND": "gpu",
                "CUBBS_BBS_GPU_LIBRARY": gpu_lib,
                "CUBBS_BBS_FORCE_GPU": "1",
                "CUBBS_BBS_REQUIRE_HARDWARE_GPU": "1",
                "CUBBS_BBS_FUSED_COMPRESS_WORKERS": str(host_workers_per_ep),
                "CUBBS_BBS_FUSED_COMPRESS_PARALLEL_MIN_ITEMS": "8",
                "CUBBS_BBS_HOST_POOL": "1",
                "CUBBS_BBS_HOST_POOL_WORKERS": str(host_workers_per_ep),
                "CUBBS_W3C_BATCH_INTERVAL_MS": str(interval_ms if batch_on else 0),
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
                "CUBBS_W3C_JSONLD_WORKERS": str(max(workers_per_ep, host_workers_per_ep)),
                "CUBBS_W3C_SHARED_JSONLD_LOADER": "1",
            })
            cmd = ["taskset", "-c", svc_slices[i], service_bin] if pin_cpus else [service_bin]
            log_f = (ep_dir / "service.log").open("wb")
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            service_procs.append((proc, log_f))
            h = wait_health(port)
            assert h["status"] == "ok", f"bad health: {h}"
            assert h["bbs_backend_active"] == "cuda-abi-v2-force-gpu", f"bad backend: {h}"
            assert h["bbs_backend_fallback"] is False
            assert h["verification_result_cache"]["capacity"] == 0
            prewarm_one(port, route, payload_bytes, expect)

        mon_file = gpu_csv.open("wb")
        mon_proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm",
                "--format=csv,noheader",
                "--loop-ms=200",
            ],
            stdout=mon_file,
            stderr=subprocess.DEVNULL,
        )

        load_procs = []
        for i in range(endpoints):
            port = port_base + i
            ep_dir = out_dir / f"e{i}"
            rep_path = ep_dir / "report.json"
            env = os.environ.copy()
            env["TOKIO_WORKER_THREADS"] = str(workers_per_ep)
            base_cmd = [
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
            cmd = ["taskset", "-c", load_slices[i]] + base_cmd if pin_cpus else base_cmd
            out_f = (ep_dir / "load.log").open("wb")
            p = subprocess.Popen(cmd, env=env, stdout=out_f, stderr=subprocess.STDOUT)
            load_procs.append((p, out_f, rep_path))

        reports = []
        for p, out_f, rep_path in load_procs:
            rc = p.wait()
            out_f.close()
            if rc != 0:
                log_txt = (rep_path.parent / "load.log").read_text(errors="replace")
                raise RuntimeError(f"load generator exited {rc}: {log_txt[:500]}")
            rep = json.loads(rep_path.read_text())
            if rep["failed"] != 0 or rep["succeeded"] <= 0:
                raise RuntimeError(f"failed requests: {rep.get('failed')} firstError={rep.get('firstError')}")
            reports.append(rep)

        if mon_proc is not None:
            mon_proc.terminate()
            try:
                mon_proc.wait(timeout=3)
            except Exception:
                mon_proc.kill()
            mon_proc = None
        if mon_file is not None:
            mon_file.close()
            mon_file = None

        gpu_stats = parse_gpu_csv(gpu_csv, duration_secs, 200)
        total_ops = sum(float(r["operationsPerSecond"]) for r in reports)
        max_p99 = max(float(r["latencyP99Ms"]) for r in reports)
        summary = {
            "operation": op,
            "endpoints": endpoints,
            "concurrencyPerEndpoint": concurrency_per_ep,
            "totalConcurrency": endpoints * concurrency_per_ep,
            "maxBatch": max_batch,
            "pipelines": pipelines,
            "intervalMs": interval_ms if batch_on else 0,
            "totalOperationsPerSecond": total_ops,
            "p99Ms": max_p99,
            "totalSucceeded": sum(int(r["succeeded"]) for r in reports),
            "totalFailed": sum(int(r["failed"]) for r in reports),
            "gpu": gpu_stats,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        if mon_proc is not None:
            try:
                mon_proc.terminate()
                mon_proc.wait(timeout=2)
            except Exception:
                mon_proc.kill()
        if mon_file is not None and not mon_file.closed:
            mon_file.close()
        for proc, log_f in service_procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            log_f.close()
        time.sleep(0.5)


def pick_cpu_pools() -> tuple[list[int], list[int], int]:
    """Split CPUs cleanly across NUMA nodes (or halves) for service vs load."""
    cpus, eff = detect_available_cpus()
    nodes = detect_numa_nodes()
    # Select up to `eff` CPUs (or all if eff >= len(cpus))
    if len(nodes) >= 2 and sum(len(n) for n in nodes) >= eff:
        # Put service on node 0, load on node 1 (capped to eff/2 each if quota < total)
        half = max(4, eff // 2)
        svc = nodes[0][:half]
        load = nodes[1][:half]
        if len(svc) < 4 or len(load) < 4:
            mid = len(cpus) // 2
            svc, load = cpus[:mid], cpus[mid:]
        return svc, load, eff
    # Single NUMA or restricted cpuset: split first half vs second half
    use = cpus[:eff] if eff < len(cpus) else cpus
    mid = max(1, len(use) // 2)
    return use[:mid], use[mid:], eff


def main():
    mode = sys.argv[1]  # "sweep" or "formal" or "all"
    out_root = Path(sys.argv[2])
    bin_dir = Path(sys.argv[3])
    gpu_lib = sys.argv[4]
    fixtures = Path(sys.argv[5])
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    dur = int(sys.argv[7]) if len(sys.argv) > 7 else 15

    service_bin = str(bin_dir / "cubbs-w3c-service")
    load_bin = str(bin_dir / "w3c-http-op-load")
    svc_cpus, load_cpus, eff = pick_cpu_pools()
    print(f"=== CPU Topology: eff={eff}, svc_pool={len(svc_cpus)} ({svc_cpus[0]}..{svc_cpus[-1]}), load_pool={len(load_cpus)} ({load_cpus[0]}..{load_cpus[-1]}) ===", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    port_cursor = 20100

    def workers_for(eps: int, is_b1_c0: bool = False) -> tuple[int, int]:
        per_ep = max(4, min(64, len(svc_cpus) // max(1, eps)))
        host_w = max(2, per_ep // 2) if not is_b1_c0 else per_ep
        return per_ep, host_w

    best_b1c1: dict[str, tuple[int, int, int, int, int]] = {
        "issue":              (8, 8192, 4096, 2, 20),
        "verify-base":        (4, 8192, 4096, 2, 20),
        "proof-generation":   (4, 4096, 1024, 8,  5),
        "proof-verification": (4, 8192, 2048, 8, 10),
    }

    if mode in ("sweep", "all"):
        sweep_cands = {
            "issue": [
                (8,  8192, 4096, 2, 20),
                (8, 12288, 4096, 2, 20),
                (12, 8192, 4096, 2, 20),
                (16, 6144, 4096, 2, 20),
            ],
            "verify-base": [
                (2,  8192, 4096, 1, 20),
                (4,  8192, 4096, 2, 20),
                (8,  8192, 4096, 2, 20),
                (8, 12288, 4096, 2, 20),
            ],
            "proof-generation": [
                (2,  4096, 1024, 8,  5),
                (4,  4096, 1024, 8,  5),
                (4,  8192, 2048, 8, 10),
                (8,  4096, 1024, 4, 10),
            ],
            "proof-verification": [
                (2,  9216, 1024, 8, 10),
                (4,  8192, 2048, 8, 10),
                (8,  4096, 2048, 4, 10),
                (8,  8192, 2048, 4, 15),
            ],
        }
        sweep_log = {}
        print("\n=== Phase 1: High-Concurrency Sweep for b1_c1 (1 rep x 10s, warmup 5s) ===", flush=True)
        for op, cands in sweep_cands.items():
            best_rate = -1.0
            sweep_log[op] = []
            for eps, conc, max_b, pipes, int_ms in cands:
                if eps > len(svc_cpus) // 2:
                    continue
                w, hw = workers_for(eps)
                tag = f"e{eps}_c{conc}_b{max_b}_p{pipes}_i{int_ms}"
                try:
                    s = run_cell_once(
                        service_bin, load_bin, gpu_lib, fixtures,
                        out_root / "sweep_b1c1" / op / tag,
                        op, True, True, eps, conc, max_b, pipes, int_ms,
                        5, 10, port_cursor, svc_cpus, load_cpus, w, hw,
                    )
                    port_cursor += eps + 2
                    rate = s["totalOperationsPerSecond"]
                    g = s.get("gpu", {})
                    print(
                        f"  [SWEEP {op:<18} {tag:<24}] {rate:>10,.1f} req/s  "
                        f"c={eps*conc:<6} p99={s['p99Ms']:6.1f}ms  "
                        f"GPU={g.get('meanUtilizationPct','?')}%/{g.get('meanPowerW','?')}W",
                        flush=True,
                    )
                    sweep_log[op].append({"tag": tag, "cfg": [eps, conc, max_b, pipes, int_ms], "summary": s})
                    if rate > best_rate:
                        best_rate = rate
                        best_b1c1[op] = (eps, conc, max_b, pipes, int_ms)
                except Exception as e:
                    port_cursor += eps + 2
                    print(f"  [SWEEP {op:<18} {tag:<24}] FAILED: {e}", flush=True)
                    sweep_log[op].append({"tag": tag, "cfg": [eps, conc, max_b, pipes, int_ms], "error": str(e)})
        (out_root / "sweep_b1c1_summary.json").write_text(json.dumps({"best": best_b1c1, "sweep": sweep_log}, indent=2))
        print(f"=== Selected b1_c1 winners: {best_b1c1} ===\n", flush=True)

    if mode in ("formal", "all"):
        cells = [
            ("b0_c0", False, False, {
                "issue":              (1,   64, 1, 20, 0, 3),
                "verify-base":        (1,  256, 1, 20, 0, 3),
                "proof-generation":   (1,   64, 1, 20, 0, 3),
                "proof-verification": (1,   64, 1, 20, 0, 3),
            }),
            ("b0_c1", False, True, {
                "issue":              (1,   64, 1, 20, 0, 3),
                "verify-base":        (1,   64, 1, 20, 0, 3),
                "proof-generation":   (1,   32, 1, 20, 0, 3),
                "proof-verification": (1,   32, 1, 20, 0, 3),
            }),
            ("b1_c0", True, False, {
                "issue":              (1, 8192, 4096, 8, 10, 5),
                "verify-base":        (1, 8192, 4096, 8, 10, 5),
                "proof-generation":   (1, 8192, 1024, 8, 10, 5),
                "proof-verification": (1, 8192, 4096, 8, 10, 5),
            }),
            ("b1_c1", True, True, {
                op: (*best_b1c1[op], 8) for op in ("issue", "verify-base", "proof-generation", "proof-verification")
            }),
        ]
        results = {}
        print(f"=== Phase 2: Formal 2x2 Ablation ({reps} reps x {dur}s) ===", flush=True)
        for cell_id, batch_on, cache_on, op_cfg in cells:
            for op, (eps, conc, max_b, pipes, int_ms, warm_s) in op_cfg.items():
                w, hw = workers_for(eps, is_b1_c0=(cell_id == "b1_c0"))
                rates, p99s, gpu_utils, gpu_pwrs = [], [], [], []
                for r in range(1, reps + 1):
                    run_dir = out_root / "formal" / cell_id / op / f"r{r}"
                    s = run_cell_once(
                        service_bin, load_bin, gpu_lib, fixtures, run_dir,
                        op, batch_on, cache_on, eps, conc, max_b, pipes, int_ms,
                        warm_s, dur, port_cursor, svc_cpus, load_cpus, w, hw,
                    )
                    port_cursor += eps + 2
                    if port_cursor > 29000:
                        port_cursor = 20100
                    rates.append(s["totalOperationsPerSecond"])
                    p99s.append(s["p99Ms"])
                    g = s.get("gpu", {})
                    if "meanUtilizationPct" in g:
                        gpu_utils.append(g["meanUtilizationPct"])
                    if "meanPowerW" in g:
                        gpu_pwrs.append(g["meanPowerW"])
                med = statistics.median(rates)
                mn, mx = min(rates), max(rates)
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
                    "concurrency_per_endpoint": conc,
                    "selected_concurrency": eps * conc,
                    "max_batch": max_b,
                    "pipelines": pipes,
                    "interval_ms": int_ms if batch_on else 0,
                    "median_ops_s": med,
                    "min_ops_s": mn,
                    "max_ops_s": mx,
                    "cv_pct": cv,
                    "median_p99_ms": med_p99,
                    "gpu_mean_util_pct": round(statistics.median(gpu_utils), 2) if gpu_utils else None,
                    "gpu_mean_power_w": round(statistics.median(gpu_pwrs), 2) if gpu_pwrs else None,
                    "repetitions": reps,
                    "duration_s": dur,
                    "raw_rates": rates,
                }
                print(
                    f"  [5090 {cell_id} {op:<18}] med={med:>10,.1f} req/s  "
                    f"(min={mn:,.1f}, max={mx:,.1f}, cv={cv:.2f}%, p99={med_p99:.1f}ms, "
                    f"ep={eps}, c={eps*conc}, GPU={results[key]['gpu_mean_util_pct']}%/{results[key]['gpu_mean_power_w']}W)",
                    flush=True,
                )
                (out_root / "summary_all.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
