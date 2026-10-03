#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_root="${1:-${repo_root}/benchmarks/w3c/results/rtx4090-optimized-discovery}"
service_bin="${SERVICE_BIN:-${repo_root}/target/release/cubbs-w3c-service}"
load_bin="${LOAD_BIN:-${repo_root}/target/release/w3c-http-op-load}"
gpu_library="${GPU_LIBRARY:-${repo_root}/target/native/cuda/libcubbs_bbs_cuda.so}"
fixtures="${FIXTURES:?set FIXTURES to the audited fixtures64 directory}"
port="${PORT:-31450}"
duration_secs="${DISCOVERY_SECS:-5}"
warmup_secs="${WARMUP_SECS:-1}"
cpu_count="$(nproc)"
service_tokio_workers="${SERVICE_TOKIO_WORKERS:-${cpu_count}}"
load_tokio_workers="${LOAD_TOKIO_WORKERS:-${cpu_count}}"
service_cpuset="${SERVICE_CPUSET:-}"
load_cpuset="${LOAD_CPUSET:-}"
gpu_index="${GPU_INDEX:-0}"
fused_compress_workers="${CUBBS_BBS_FUSED_COMPRESS_WORKERS:-8}"
host_pool_workers="${CUBBS_BBS_HOST_POOL_WORKERS:-32}"

service_command=("${service_bin}")
load_command=("${load_bin}")
if [[ -n "${service_cpuset}" ]]; then
  service_command=(taskset --cpu-list "${service_cpuset}" "${service_bin}")
fi
if [[ -n "${load_cpuset}" ]]; then
  load_command=(taskset --cpu-list "${load_cpuset}" "${load_bin}")
fi

ulimit -n "${ULIMIT_N:-1048576}"

issue_concurrencies="${ISSUE_CONCURRENCIES:-1024 2048 4096 6144 8192 12288}"
verify_concurrencies="${VERIFY_CONCURRENCIES:-2048 4096 6144 8192 12288 16384}"
proof_gen_concurrencies="${PROOF_GEN_CONCURRENCIES:-512 1024 1536 2048 3072 4096}"
proof_verify_concurrencies="${PROOF_VERIFY_CONCURRENCIES:-2048 4096 6144 8192 10240 14336}"
read -r -a operations <<< "${OPERATIONS:-issue verify-base proof-generation proof-verification}"

if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite existing output: ${output_root}" >&2
  exit 1
fi
for path in "${service_bin}" "${load_bin}" "${gpu_library}" \
  "${fixtures}/issue.json" "${fixtures}/verify-base.json" \
  "${fixtures}/derive.json" "${fixtures}/verify-derived.json"; do
  [[ -f "${path}" ]] || { echo "missing required file: ${path}" >&2; exit 1; }
done
if ss -ltn | grep -Eq ":${port}[[:space:]]"; then
  echo "discovery port already in use: ${port}" >&2
  exit 1
fi

mkdir -p "${output_root}/provenance" "${output_root}/raw"
sha256sum "${service_bin}" "${load_bin}" "${gpu_library}" \
  > "${output_root}/provenance/artifact-sha256.txt"
sha256sum "${fixtures}/issue.json" "${fixtures}/verify-base.json" \
  "${fixtures}/derive.json" "${fixtures}/verify-derived.json" \
  > "${output_root}/provenance/fixture-sha256.txt"
sha256sum "${BASH_SOURCE[0]}" > "${output_root}/provenance/runner-sha256.txt"
nvidia-smi -L > "${output_root}/provenance/nvidia-smi-L.txt"
nvidia-smi -i "${gpu_index}" --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader > "${output_root}/provenance/gpu.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader > "${output_root}/provenance/gpu-processes-start.csv" 2>/dev/null || true
lscpu > "${output_root}/provenance/lscpu.txt"
uname -a > "${output_root}/provenance/uname.txt"

cat > "${output_root}/provenance/config.env" <<EOF
port=${port}
duration_secs=${duration_secs}
warmup_secs=${warmup_secs}
ulimit_n=$(ulimit -n)
batch_interval_ms=${BATCH_INTERVAL_MS:-5}
queue_capacity=${QUEUE_CAPACITY:-32768}
pipelines=${PIPELINES:-8}
issue_max_batch_size=${ISSUE_MAX_BATCH_SIZE:-1024}
derive_max_batch_size=${DERIVE_MAX_BATCH_SIZE:-128}
verify_base_max_batch_size=${VERIFY_BASE_MAX_BATCH_SIZE:-2048}
proof_verify_max_batch_size=${PROOF_VERIFY_MAX_BATCH_SIZE:-1024}
gpu_verify_max_inflight_items=${GPU_VERIFY_MAX_INFLIGHT_ITEMS:-16384}
issue_concurrencies=${issue_concurrencies}
verify_concurrencies=${verify_concurrencies}
proof_gen_concurrencies=${proof_gen_concurrencies}
proof_verify_concurrencies=${proof_verify_concurrencies}
operations=${operations[*]}
service_tokio_workers=${service_tokio_workers}
load_tokio_workers=${load_tokio_workers}
service_cpuset=${service_cpuset:-unrestricted}
load_cpuset=${load_cpuset:-unrestricted}
gpu_index=${gpu_index}
sign_e_workers=${CUBBS_BBS_SIGN_E_WORKERS:-default}
sign_shape_batching=${CUBBS_BBS_SIGN_SHAPE_BATCHING:-default}
fused_compress_workers=${fused_compress_workers}
host_pool_workers=${host_pool_workers}
EOF

service_pid=""
cleanup() {
  if [[ -n "${service_pid}" ]] && kill -0 "${service_pid}" 2>/dev/null; then
    kill -TERM "${service_pid}" 2>/dev/null || true
    wait "${service_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

env \
  CUDA_VISIBLE_DEVICES="${gpu_index}" \
  TOKIO_WORKER_THREADS="${service_tokio_workers}" \
  RUST_LOG=warn \
  CUBBS_PORT="${port}" \
  CUBBS_BBS_BACKEND=gpu \
  CUBBS_BBS_GPU_LIBRARY="${gpu_library}" \
  CUBBS_BBS_FORCE_GPU=1 \
  CUBBS_BBS_REQUIRE_HARDWARE_GPU=1 \
  CUBBS_BBS_FUSED_COMPRESS_WORKERS="${fused_compress_workers}" \
  CUBBS_BBS_FUSED_COMPRESS_PARALLEL_MIN_ITEMS=8 \
  CUBBS_BBS_HOST_POOL=1 \
  CUBBS_BBS_HOST_POOL_WORKERS="${host_pool_workers}" \
  CUBBS_W3C_BATCH_INTERVAL_MS="${BATCH_INTERVAL_MS:-5}" \
  CUBBS_W3C_QUEUE_CAPACITY="${QUEUE_CAPACITY:-32768}" \
  CUBBS_W3C_PIPELINES="${PIPELINES:-8}" \
  CUBBS_W3C_ISSUE_MAX_BATCH_SIZE="${ISSUE_MAX_BATCH_SIZE:-1024}" \
  CUBBS_W3C_DERIVE_MAX_BATCH_SIZE="${DERIVE_MAX_BATCH_SIZE:-128}" \
  CUBBS_W3C_VERIFY_BASE_MAX_BATCH_SIZE="${VERIFY_BASE_MAX_BATCH_SIZE:-2048}" \
  CUBBS_W3C_PROOF_VERIFY_MAX_BATCH_SIZE="${PROOF_VERIFY_MAX_BATCH_SIZE:-1024}" \
  CUBBS_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS="${GPU_VERIFY_MAX_INFLIGHT_ITEMS:-16384}" \
  CUBBS_W3C_PREPARED_CACHE_CAPACITY=65536 \
  CUBBS_W3C_CANON_CACHE_CAPACITY=65536 \
  CUBBS_W3C_LABEL_CACHE_CAPACITY=16384 \
  CUBBS_W3C_VERIFY_CACHE_CAPACITY=0 \
  CUBBS_W3C_LISTEN_BACKLOG="${LISTEN_BACKLOG:-32768}" \
  CUBBS_W3C_JSONLD_WORKERS="${JSONLD_WORKERS:-64}" \
  CUBBS_W3C_SHARED_JSONLD_LOADER=1 \
  "${service_command[@]}" > "${output_root}/service.log" 2>&1 &
service_pid=$!

ready=0
for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:${port}/healthz" \
    > "${output_root}/health-start.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.1
done
[[ "${ready}" == 1 ]] || { echo "service did not become ready on ${port}" >&2; exit 1; }
jq -e '.status == "ok" and .bbs_backend_active == "cuda-abi-v2-force-gpu" and
  .bbs_backend_fallback == false and .verification_result_cache.capacity == 0' \
  "${output_root}/health-start.json" >/dev/null

declare -A endpoint=(
  [issue]="credentials/issue"
  [verify-base]="credentials/verify"
  [proof-generation]="credentials/derive"
  [proof-verification]="credentials/verify"
)
declare -A payload=(
  [issue]="issue.json"
  [verify-base]="verify-base.json"
  [proof-generation]="derive.json"
  [proof-verification]="verify-derived.json"
)
declare -A expect=(
  [issue]="document"
  [verify-base]="verified"
  [proof-generation]="document"
  [proof-verification]="verified"
)
declare -A concurrency_lists=(
  [issue]="${issue_concurrencies}"
  [verify-base]="${verify_concurrencies}"
  [proof-generation]="${proof_gen_concurrencies}"
  [proof-verification]="${proof_verify_concurrencies}"
)

for operation in "${operations[@]}"; do
  [[ -v "endpoint[${operation}]" ]] || {
    echo "unsupported operation in OPERATIONS: ${operation}" >&2
    exit 1
  }
done

for operation in "${operations[@]}"; do
  operation_root="${output_root}/raw/${operation}"
  mkdir -p "${operation_root}"
  curl -fsS -H 'content-type: application/json' \
    --data-binary "@${fixtures}/${payload[${operation}]}" \
    "http://127.0.0.1:${port}/${endpoint[${operation}]}" \
    > "${operation_root}/prewarm-response.json"
  if [[ "${expect[${operation}]}" == document ]]; then
    jq -e '.proof.proofValue | type == "string" and length > 0' \
      "${operation_root}/prewarm-response.json" >/dev/null
  else
    jq -e '.verified == true' "${operation_root}/prewarm-response.json" >/dev/null
  fi

  for concurrency in ${concurrency_lists[${operation}]}; do
    run_dir="${operation_root}/c${concurrency}"
    mkdir -p "${run_dir}"
    nvidia-smi -i "${gpu_index}" --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
      --format=csv,noheader > "${run_dir}/gpu-before.csv"
    env TOKIO_WORKER_THREADS="${load_tokio_workers}" "${load_command[@]}" \
      --operation "${operation}" \
      --url "http://127.0.0.1:${port}/${endpoint[${operation}]}" \
      --payload "${fixtures}/${payload[${operation}]}" \
      --expect "${expect[${operation}]}" \
      --health-url "http://127.0.0.1:${port}/healthz" \
      --concurrency "${concurrency}" \
      --warmup-secs "${warmup_secs}" \
      --duration-secs "${duration_secs}" \
      --output "${run_dir}/report.json" >/dev/null
    curl -fsS "http://127.0.0.1:${port}/healthz" > "${run_dir}/health-after.json"
    nvidia-smi -i "${gpu_index}" --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
      --format=csv,noheader > "${run_dir}/gpu-after.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
      --format=csv,noheader > "${run_dir}/gpu-processes.csv" 2>/dev/null || true
    jq -e --arg operation "${operation}" --argjson concurrency "${concurrency}" \
      --argjson duration "${duration_secs}" '
      .operation == $operation and .concurrency == $concurrency and
      .targetDurationSeconds == $duration and .failed == 0 and .succeeded > 0 and
      .serviceHealthAfter.bbs_backend_active == "cuda-abi-v2-force-gpu" and
      .serviceHealthAfter.bbs_backend_fallback == false and
      .serviceHealthAfter.verification_result_cache.capacity == 0
    ' "${run_dir}/report.json" >/dev/null
    jq -r '"discovery " + .operation + " c" + (.concurrency|tostring) +
      " rps=" + (.operationsPerSecond|tostring) + " failed=" + (.failed|tostring)' \
      "${run_dir}/report.json"
  done
done

curl -fsS "http://127.0.0.1:${port}/healthz" > "${output_root}/health-end.json"
kill -TERM "${service_pid}" 2>/dev/null || true
wait "${service_pid}" 2>/dev/null || true
service_pid=""
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader > "${output_root}/provenance/gpu-processes-end.csv" 2>/dev/null || true
touch "${output_root}/COMPLETE"
echo "discovery complete: ${output_root}"
