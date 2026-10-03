#!/usr/bin/env bash
set -euo pipefail

if (( $# != 10 )); then
  echo "usage: $0 PACKAGE FIXTURES OUT OP ENDPOINTS CONCURRENCY_PER_ENDPOINT BATCH PIPELINES INTERVAL_MS DURATION_SECS" >&2
  exit 64
fi

package=$1
fixtures=$2
out=$3
operation=$4
endpoints=$5
concurrency=$6
batch_size=$7
pipelines=$8
interval_ms=$9
duration_secs=${10}

service_bin="$package/bin/cubbs-w3c-service"
load_bin="$package/bin/w3c-http-op-load"
gpu_library="$package/lib/libcubbs_bbs_cuda.so"
gpu_index=${GPU_INDEX:-1}
port_base=${PORT_BASE:-31400}
warmup_secs=${WARMUP_SECS:-5}
sample_ms=${GPU_SAMPLE_MS:-200}
affinity_mode=${AFFINITY_MODE:-split-numa}

case "$operation" in
  issue)
    route=credentials/issue payload=issue.json expect=document batch_key=issue
    ;;
  verify-base)
    route=credentials/verify payload=verify-base.json expect=verified batch_key=verifySignature
    ;;
  proof-generation)
    route=credentials/derive payload=derive.json expect=document batch_key=derive
    ;;
  proof-verification)
    route=credentials/verify payload=verify-derived.json expect=verified batch_key=proofVerify
    ;;
  *) echo "unsupported operation: $operation" >&2; exit 64 ;;
esac

for value in "$endpoints" "$concurrency" "$batch_size" "$pipelines" "$interval_ms" "$duration_secs"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "positive integer required: $value" >&2; exit 64; }
done
(( endpoints <= 8 )) || { echo "at most 8 endpoints are supported" >&2; exit 64; }
[[ "$sample_ms" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ "$warmup_secs" =~ ^[0-9]+$ ]] || exit 64
[[ "$affinity_mode" == split-numa || "$affinity_mode" == unrestricted ]] || exit 64
[[ ! -e "$out" ]] || { echo "refusing to overwrite $out" >&2; exit 1; }
for path in "$service_bin" "$load_bin" "$gpu_library" "$fixtures/$payload"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 66; }
done

mkdir -p "$out/endpoints" "$out/provenance"
ulimit -n "${ULIMIT_N:-1048576}"
workers_per_endpoint=${WORKERS_PER_ENDPOINT:-$((64 / endpoints))}
(( workers_per_endpoint >= 4 )) || workers_per_endpoint=4
host_workers=${HOST_POOL_WORKERS_PER_ENDPOINT:-$((workers_per_endpoint / 2))}
(( host_workers >= 2 )) || host_workers=2

service_pids=()
load_pids=()
monitor_pid=
cleanup() {
  [[ -z "$monitor_pid" ]] || kill -TERM "$monitor_pid" 2>/dev/null || true
  for pid in "${load_pids[@]}" "${service_pids[@]}"; do
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

endpoint_cpuset() {
  local role=$1 index=$2 first last offset
  first=$((index * 32 / endpoints))
  last=$(((index + 1) * 32 / endpoints - 1))
  [[ "$role" == service ]] && offset=0 || offset=32
  printf '%d-%d,%d-%d' "$((first + offset))" "$((last + offset))" \
    "$((first + offset + 64))" "$((last + offset + 64))"
}

printf '%s\n' \
  "operation=$operation" \
  "gpu_index=$gpu_index" \
  "endpoints=$endpoints" \
  "concurrency_per_endpoint=$concurrency" \
  "total_concurrency=$((endpoints * concurrency))" \
  "batch_size=$batch_size" \
  "pipelines_per_endpoint=$pipelines" \
  "interval_ms=$interval_ms" \
  "warmup_secs=$warmup_secs" \
  "duration_secs=$duration_secs" \
  "gpu_sample_ms=$sample_ms" \
  "affinity_mode=$affinity_mode" \
  "workers_per_endpoint=$workers_per_endpoint" \
  "host_workers_per_endpoint=$host_workers" \
  > "$out/provenance/config.env"
sha256sum "$service_bin" "$load_bin" "$gpu_library" "$fixtures/$payload" \
  > "$out/provenance/artifact-sha256.txt"
nvidia-smi -L > "$out/provenance/nvidia-smi-L.txt"
nvidia-smi -i "$gpu_index" --query-gpu=index,name,uuid,driver_version,memory.total,power.limit \
  --format=csv,noheader > "$out/provenance/gpu.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader > "$out/provenance/compute-processes-before.csv" || true
lscpu > "$out/provenance/lscpu.txt"
uname -a > "$out/provenance/uname.txt"
grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor \
  > "$out/provenance/cpu-governors.txt" 2>/dev/null || true

for ((i=0; i<endpoints; i++)); do
  endpoint_dir="$out/endpoints/e$i"
  port=$((port_base + i))
  mkdir -p "$endpoint_dir"
  ! ss -ltn | grep -Eq ":${port}[[:space:]]" || { echo "port in use: $port" >&2; exit 1; }

  service_command=("$service_bin")
  load_command=("$load_bin")
  service_cpuset=unrestricted
  load_cpuset=unrestricted
  if [[ "$affinity_mode" == split-numa ]]; then
    service_cpuset=$(endpoint_cpuset service "$i")
    load_cpuset=$(endpoint_cpuset load "$i")
    service_command=(taskset --cpu-list "$service_cpuset" "$service_bin")
    load_command=(taskset --cpu-list "$load_cpuset" "$load_bin")
  fi
  printf 'service\t%s\nload\t%s\n' "$service_cpuset" "$load_cpuset" \
    > "$endpoint_dir/cpu-affinity.tsv"

  env \
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    TOKIO_WORKER_THREADS="$workers_per_endpoint" \
    RUST_LOG=warn \
    CUBBS_PORT="$port" \
    CUBBS_BBS_BACKEND=gpu \
    CUBBS_BBS_GPU_LIBRARY="$gpu_library" \
    CUBBS_BBS_FORCE_GPU=1 \
    CUBBS_BBS_REQUIRE_HARDWARE_GPU=1 \
    CUBBS_BBS_FUSED_COMPRESS_WORKERS="$host_workers" \
    CUBBS_BBS_FUSED_COMPRESS_PARALLEL_MIN_ITEMS=8 \
    CUBBS_BBS_HOST_POOL=1 \
    CUBBS_BBS_HOST_POOL_WORKERS="$host_workers" \
    CUBBS_W3C_BATCH_INTERVAL_MS="$interval_ms" \
    CUBBS_W3C_QUEUE_CAPACITY=32768 \
    CUBBS_W3C_PIPELINES="$pipelines" \
    CUBBS_W3C_ISSUE_MAX_BATCH_SIZE="$batch_size" \
    CUBBS_W3C_DERIVE_MAX_BATCH_SIZE="$batch_size" \
    CUBBS_W3C_VERIFY_BASE_MAX_BATCH_SIZE="$batch_size" \
    CUBBS_W3C_PROOF_VERIFY_MAX_BATCH_SIZE="$batch_size" \
    CUBBS_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS=32768 \
    CUBBS_W3C_PREPARED_CACHE_CAPACITY=65536 \
    CUBBS_W3C_CANON_CACHE_CAPACITY=65536 \
    CUBBS_W3C_LABEL_CACHE_CAPACITY=16384 \
    CUBBS_W3C_VERIFY_CACHE_CAPACITY=0 \
    CUBBS_W3C_LISTEN_BACKLOG=32768 \
    CUBBS_W3C_JSONLD_WORKERS="$workers_per_endpoint" \
    CUBBS_W3C_SHARED_JSONLD_LOADER=1 \
    "${service_command[@]}" > "$endpoint_dir/service.log" 2>&1 &
  service_pids+=("$!")

  ready=0
  for _ in $(seq 1 200); do
    if curl -fsS "http://127.0.0.1:$port/healthz" > "$endpoint_dir/health-start.json" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 0.1
  done
  (( ready == 1 )) || { echo "service e$i did not become ready" >&2; exit 1; }
  jq -e --arg key "$batch_key" --argjson batch "$batch_size" \
    --argjson pipelines "$pipelines" --argjson interval "$interval_ms" '
      .status == "ok" and
      .bbs_backend_active == "cuda-abi-v2-force-gpu" and
      .bbs_backend_fallback == false and
      .verification_result_cache.capacity == 0 and
      .batching.config.pipelines == $pipelines and
      .batching.operationConfigs[$key].maxBatchSize == $batch and
      .batching.operationConfigs[$key].intervalMs == $interval
    ' "$endpoint_dir/health-start.json" >/dev/null
  curl -fsS -H 'content-type: application/json' --data-binary "@$fixtures/$payload" \
    "http://127.0.0.1:$port/$route" > "$endpoint_dir/prewarm-response.json"
  if [[ "$expect" == document ]]; then
    jq -e '.proof.proofValue | type == "string" and length > 0' \
      "$endpoint_dir/prewarm-response.json" >/dev/null
  else
    jq -e '.verified == true' "$endpoint_dir/prewarm-response.json" >/dev/null
  fi
  printf '%s\0' "${load_command[@]}" > "$endpoint_dir/load-command.nul"
done

nvidia-smi -i "$gpu_index" \
  --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,pstate \
  --format=csv,noheader --loop-ms="$sample_ms" > "$out/gpu-timeseries.csv" &
monitor_pid=$!

for ((i=0; i<endpoints; i++)); do
  endpoint_dir="$out/endpoints/e$i"
  port=$((port_base + i))
  load_command=("$load_bin")
  if [[ "$affinity_mode" == split-numa ]]; then
    load_command=(taskset --cpu-list "$(endpoint_cpuset load "$i")" "$load_bin")
  fi
  env TOKIO_WORKER_THREADS="$workers_per_endpoint" "${load_command[@]}" \
    --operation "$operation" \
    --url "http://127.0.0.1:$port/$route" \
    --payload "$fixtures/$payload" \
    --expect "$expect" \
    --health-url "http://127.0.0.1:$port/healthz" \
    --concurrency "$concurrency" \
    --warmup-secs "$warmup_secs" \
    --duration-secs "$duration_secs" \
    --output "$endpoint_dir/report.json" > "$endpoint_dir/load.stdout" 2> "$endpoint_dir/load.stderr" &
  load_pids+=("$!")
done

load_failed=0
for pid in "${load_pids[@]}"; do
  wait "$pid" || load_failed=1
done
(( load_failed == 0 )) || { echo "one or more load generators failed" >&2; exit 1; }
kill -TERM "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=

for ((i=0; i<endpoints; i++)); do
  jq -e --arg operation "$operation" --argjson concurrency "$concurrency" \
    --argjson duration "$duration_secs" '
      .operation == $operation and .concurrency == $concurrency and
      .targetDurationSeconds == $duration and .failed == 0 and .succeeded > 0 and
      .serviceHealthBefore.bbs_backend_active == "cuda-abi-v2-force-gpu" and
      .serviceHealthAfter.bbs_backend_active == "cuda-abi-v2-force-gpu" and
      .serviceHealthBefore.bbs_backend_fallback == false and
      .serviceHealthAfter.bbs_backend_fallback == false
    ' "$out/endpoints/e$i/report.json" >/dev/null
done

timed_samples=$(((duration_secs * 1000 + sample_ms - 1) / sample_ms))
tail -n "$timed_samples" "$out/gpu-timeseries.csv" > "$out/gpu-timed.csv"
awk -F',' '{v=$2; gsub(/[^0-9.]/,"",v); print v+0}' "$out/gpu-timed.csv" \
  > "$out/gpu-utilization-pct.txt"
awk -F',' '{v=$5; gsub(/[^0-9.]/,"",v); print v+0}' "$out/gpu-timed.csv" \
  > "$out/gpu-power-w.txt"
gpu_samples=$(wc -l < "$out/gpu-utilization-pct.txt")
(( gpu_samples > 0 )) || { echo "no GPU samples" >&2; exit 1; }
gpu_mean=$(awk '{s+=$1} END{printf "%.6f",s/NR}' "$out/gpu-utilization-pct.txt")
gpu_median=$(sort -n "$out/gpu-utilization-pct.txt" | awk -v n="$gpu_samples" \
  'NR==int((n+1)/2){a=$1} NR==int((n+2)/2){printf "%.6f",(a+$1)/2; exit}')
gpu_p95=$(sort -n "$out/gpu-utilization-pct.txt" | awk -v n="$gpu_samples" \
  'NR==int((n*95+99)/100){printf "%.6f",$1; exit}')
gpu_max=$(sort -nr "$out/gpu-utilization-pct.txt" | head -1)
power_mean=$(awk '{s+=$1} END{printf "%.6f",s/NR}' "$out/gpu-power-w.txt")
power_max=$(sort -nr "$out/gpu-power-w.txt" | head -1)
jq -n --argjson samples "$gpu_samples" --argjson mean "$gpu_mean" \
  --argjson median "$gpu_median" --argjson p95 "$gpu_p95" --argjson max "$gpu_max" \
  --argjson power_mean "$power_mean" --argjson power_max "$power_max" '{
    samples: $samples, meanUtilizationPct: $mean, medianUtilizationPct: $median,
    p95UtilizationPct: $p95, maxUtilizationPct: $max,
    meanPowerW: $power_mean, maxPowerW: $power_max
  }' > "$out/gpu-summary.json"

jq -s --arg operation "$operation" --argjson endpoints "$endpoints" \
  --argjson concurrency "$concurrency" --argjson batch "$batch_size" \
  --argjson pipelines "$pipelines" --argjson interval "$interval_ms" '{
    operation: $operation,
    endpointCount: $endpoints,
    concurrencyPerEndpoint: $concurrency,
    totalConcurrency: ($endpoints * $concurrency),
    batchSizePerEndpoint: $batch,
    pipelinesPerEndpoint: $pipelines,
    intervalMs: $interval,
    totalOperationsPerSecond: ([.[].operationsPerSecond] | add),
    totalSucceeded: ([.[].succeeded] | add),
    totalFailed: ([.[].failed] | add),
    endpointReports: map({url, concurrency, operationsPerSecond, succeeded, failed,
      latencyP50Ms, latencyP95Ms, latencyP99Ms, wallSeconds})
  }' "$out"/endpoints/e*/report.json > "$out/aggregate-load.json"
jq --slurpfile gpu "$out/gpu-summary.json" '. + {gpu: $gpu[0]}' \
  "$out/aggregate-load.json" > "$out/summary.json"
sha256sum "$out/summary.json" > "$out/SUMMARY_SHA256"
cat "$out/summary.json"
