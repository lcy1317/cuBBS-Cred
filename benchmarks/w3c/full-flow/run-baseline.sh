#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
service_bin="$repo_root/target/release/cubbs-w3c-service"
load_bin="$repo_root/target/release/w3c-http-load"
dataset_id="http-unique-10k-v1"
dataset_root="$repo_root/benchmarks/w3c/datasets"
results_root="$repo_root/benchmarks/w3c/full-flow/results"
backend=""
host_label=""
run_id=""
port="40443"
concurrency="24"
duration="30"
startup_spread_ms="500"
warmup_rounds="1"
batch_interval_ms="20"
max_batch_size="128"
queue_capacity="16384"
jsonld_workers="16"
canon_cache_capacity="4096"
gpu_library="$repo_root/target/native/cuda/libcubbs_bbs_cuda.so"

while (($#)); do
  case "$1" in
    --backend)
      backend="$2"
      shift 2
      ;;
    --host-label)
      host_label="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    --concurrency)
      concurrency="$2"
      shift 2
      ;;
    --duration-secs)
      duration="$2"
      shift 2
      ;;
    --results-root)
      results_root="$2"
      shift 2
      ;;
    --gpu-library)
      gpu_library="$2"
      shift 2
      ;;
    --service-bin)
      service_bin="$2"
      shift 2
      ;;
    --load-bin)
      load_bin="$2"
      shift 2
      ;;
    --batch-interval-ms)
      batch_interval_ms="$2"
      shift 2
      ;;
    --max-batch-size)
      max_batch_size="$2"
      shift 2
      ;;
    --jsonld-workers)
      jsonld_workers="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '/^## HTTP 全流程回归$/,/^## 外部 W3C HTTP 对照$/p' "$repo_root/docs/benchmarks.md"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$backend" != "cpu" && "$backend" != "gpu" ]]; then
  echo "--backend must be cpu or gpu" >&2
  exit 2
fi
if [[ -z "$host_label" || -z "$run_id" ]]; then
  echo "--host-label and --run-id are required" >&2
  exit 2
fi
if [[ ! -x "$service_bin" || ! -x "$load_bin" ]]; then
  echo "build cubbs-w3c-service and w3c-http-load in release mode first" >&2
  exit 2
fi
if [[ "$backend" == "gpu" && ! -f "$gpu_library" ]]; then
  echo "GPU library not found: $gpu_library" >&2
  exit 2
fi
if ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .; then
  echo "port $port is already in use" >&2
  exit 2
fi

manifest="$dataset_root/$dataset_id/manifest.json"
dataset="$dataset_root/$dataset_id/generated/issue-requests.jsonl"
expected_entries="$(jq -r '.entryCount' "$manifest")"
expected_dataset_sha256="$(jq -r '.generatedSha256' "$manifest")"
manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
result_dir="$results_root/$run_id"
if [[ -e "$result_dir" ]]; then
  echo "refusing to overwrite result directory: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir" "$(dirname "$dataset")"

tmp_dir="$(mktemp -d)"
service_log="$tmp_dir/service.log"
health_start="$tmp_dir/health-start.json"
load_report="$tmp_dir/load-report.json"
service_pid=""
cleanup() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

CUBBS_PORT="$port" \
CUBBS_BBS_BACKEND="$backend" \
CUBBS_BBS_GPU_LIBRARY="$gpu_library" \
CUBBS_W3C_BATCH_INTERVAL_MS="$batch_interval_ms" \
CUBBS_W3C_MAX_BATCH_SIZE="$max_batch_size" \
CUBBS_W3C_QUEUE_CAPACITY="$queue_capacity" \
CUBBS_W3C_JSONLD_WORKERS="$jsonld_workers" \
CUBBS_W3C_SHARED_JSONLD_LOADER=1 \
CUBBS_W3C_CANON_CACHE_CAPACITY="$canon_cache_capacity" \
"$service_bin" >"$service_log" 2>&1 &
service_pid=$!

healthy=0
for _ in $(seq 1 200); do
  if curl -fsS "http://127.0.0.1:$port/id/healthz" >"$health_start"; then
    healthy=1
    break
  fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if ((healthy == 0)); then
  tail -100 "$service_log" >&2
  echo "W3C service did not become healthy" >&2
  exit 1
fi

if [[ "$backend" == "gpu" ]]; then
  jq -e '
    .bbs_backend_requested == "gpu" and
    .bbs_backend_hardware_acceleration == true and
    .bbs_backend_fallback == false
  ' "$health_start" >/dev/null
else
  jq -e '
    .bbs_backend_requested == "cpu" and
    .bbs_backend_active == "zkryptium-cpu" and
    .bbs_backend_hardware_acceleration == false and
    .bbs_backend_fallback == false
  ' "$health_start" >/dev/null
fi

dataset_args=(--dataset "$dataset")
if [[ ! -f "$dataset" ]]; then
  dataset_args=(--generate-dataset "$dataset" --dataset-size "$expected_entries")
fi
"$load_bin" \
  --mode sustained \
  --base-url "http://127.0.0.1:$port/id" \
  --concurrency "$concurrency" \
  --duration-secs "$duration" \
  --warmup-rounds "$warmup_rounds" \
  --startup-spread-ms "$startup_spread_ms" \
  "${dataset_args[@]}" \
  --output "$load_report" >/dev/null

actual_dataset_sha256="$(sha256sum "$dataset" | awk '{print $1}')"
if [[ "$actual_dataset_sha256" != "$expected_dataset_sha256" ]]; then
  echo "dataset digest does not match manifest" >&2
  exit 1
fi

jq -e --arg backend "$backend" --argjson duration "$duration" '
  . as $report |
  .mode == "sustained" and
  .targetDurationSeconds == $duration and
  .uniqueVcInputs == true and
  .datasetEntries == 10000 and
  .totalWorkflows == .succeededWorkflows and
  .failedWorkflows == 0 and
  (.operations | length == 4) and
  all(.operations[];
    .requests == $report.totalWorkflows and .failed == 0) and
  .backendInfo.requested == $backend and
  .backendInfo.fallback == false and
  (if $backend == "gpu"
   then .backendInfo.hardwareAcceleration == true
   else (.backendInfo.active == "zkryptium-cpu" and
         .backendInfo.hardwareAcceleration == false)
   end) and
  all(.batchingDelta[];
    .dispatchedItems == $report.totalWorkflows and
    .averageObservedBatch > 1 and
    .maxObservedBatch > 1)
' "$load_report" >/dev/null

source_commit="$(git -C "$repo_root" rev-parse HEAD)"
source_tree_state="$(
  if git -C "$repo_root" diff --quiet &&
    git -C "$repo_root" diff --cached --quiet; then
    echo clean
  else
    echo task9-working-tree
  fi
)"
service_sha256="$(sha256sum "$service_bin" | awk '{print $1}')"
load_sha256="$(sha256sum "$load_bin" | awk '{print $1}')"
hostname_value="$(hostname)"
architecture="$(uname -m)"
kernel_release="$(uname -r)"
os_pretty_name="$(
  . /etc/os-release
  printf '%s' "$PRETTY_NAME"
)"
cpu_model="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
logical_cpus="$(nproc --all)"
process_affinity="$(awk '/Cpus_allowed_list:/ {print $2}' /proc/self/status)"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
gpu_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)"
service_command="CUBBS_PORT=$port CUBBS_BBS_BACKEND=$backend CUBBS_BBS_GPU_LIBRARY=$gpu_library CUBBS_W3C_BATCH_INTERVAL_MS=$batch_interval_ms CUBBS_W3C_MAX_BATCH_SIZE=$max_batch_size CUBBS_W3C_QUEUE_CAPACITY=$queue_capacity CUBBS_W3C_JSONLD_WORKERS=$jsonld_workers CUBBS_W3C_SHARED_JSONLD_LOADER=1 CUBBS_W3C_CANON_CACHE_CAPACITY=$canon_cache_capacity target/release/cubbs-w3c-service"
load_command="target/release/w3c-http-load --mode sustained --base-url http://127.0.0.1:$port/id --concurrency $concurrency --duration-secs $duration --warmup-rounds $warmup_rounds --startup-spread-ms $startup_spread_ms --dataset benchmarks/w3c/datasets/$dataset_id/generated/issue-requests.jsonl --output <temporary>/load-report.json"

jq -n \
  --slurpfile load "$load_report" \
  --arg run_id "$run_id" \
  --arg dataset_id "$dataset_id" \
  --arg manifest_sha256 "$manifest_sha256" \
  --arg dataset_sha256 "$actual_dataset_sha256" \
  --arg source_commit "$source_commit" \
  --arg source_tree_state "$source_tree_state" \
  --arg service_sha256 "$service_sha256" \
  --arg load_sha256 "$load_sha256" \
  --arg host_label "$host_label" \
  --arg hostname "$hostname_value" \
  --arg architecture "$architecture" \
  --arg kernel "$kernel_release" \
  --arg os "$os_pretty_name" \
  --arg cpu "$cpu_model" \
  --argjson logical_cpus "$logical_cpus" \
  --arg affinity "$process_affinity" \
  --arg gpu "$gpu_name" \
  --arg gpu_driver "$gpu_driver" \
  --arg service_command "$service_command" \
  --arg load_command "$load_command" \
  '{
    schemaVersion: 1,
    tool: "cubbs-w3c-full-flow-baseline",
    runId: $run_id,
    measurement: "sustained-independent-complete-http-workflows",
    dataset: {
      id: $dataset_id,
      manifestSha256: $manifest_sha256,
      generatedSha256: $dataset_sha256
    },
    source: {
      commit: $source_commit,
      treeState: $source_tree_state,
      serviceBinarySha256: $service_sha256,
      loadBinarySha256: $load_sha256
    },
    machine: {
      label: $host_label,
      hostname: $hostname,
      architecture: $architecture,
      osPrettyName: $os,
      kernelRelease: $kernel,
      cpuModel: $cpu,
      logicalCpuCount: $logical_cpus,
      processCpuAffinity: $affinity,
      gpuName: (if $gpu == "" then null else $gpu end),
      gpuDriverVersion: (if $gpu_driver == "" then null else $gpu_driver end)
    },
    serviceCommand: $service_command,
    loadCommand: $load_command,
    loadReport: $load[0],
    accepted: true
  }' >"$result_dir/run.json"

workflows="$(jq -r '.loadReport.totalWorkflows' "$result_dir/run.json")"
workflow_rate="$(jq -r '.loadReport.workflowsPerSecond' "$result_dir/run.json")"
p50="$(jq -r '.loadReport.workflowLatencyP50Ms' "$result_dir/run.json")"
p95="$(jq -r '.loadReport.workflowLatencyP95Ms' "$result_dir/run.json")"
p99="$(jq -r '.loadReport.workflowLatencyP99Ms' "$result_dir/run.json")"
printf '%s\n\n%s\n\n%s\n%s\n%s\n%s\n%s\n' \
  "# W3C full-flow baseline: $run_id" \
  "Backend: \`$(jq -r '.loadReport.backendInfo.active' "$result_dir/run.json")\`" \
  "- workflows: $workflows/$workflows; failed=0" \
  "- workflow throughput: $workflow_rate/s" \
  "- workflow P50/P95/P99: $p50 / $p95 / $p99 ms" \
  "- dataset: \`$dataset_id\` ($expected_entries unique VCs)" \
  "- full start-through-drain accounting; all four queues batched" \
  >"$result_dir/README.md"

cat "$result_dir/run.json"
