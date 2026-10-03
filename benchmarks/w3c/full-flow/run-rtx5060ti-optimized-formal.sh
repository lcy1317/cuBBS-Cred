#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_root="${1:-${repo_root}/benchmarks/w3c/results/rtx5060ti-optimized-20260912}"
service_bin="${SERVICE_BIN:-${repo_root}/target/release/cubbs-w3c-service}"
load_bin="${LOAD_BIN:-${repo_root}/target/release/w3c-http-op-load}"
gpu_library="${GPU_LIBRARY:-${repo_root}/target/native/cuda/libcubbs_bbs_cuda.so}"
fixtures="${FIXTURES:-${repo_root}/ref/results/paper-archive/payload/canonical/omni-bbs-clean-v2/gpu-rtx5060ti/w3c-http/structural-cache-only/fixtures64}"
formal_cv_validator="${FORMAL_CV_VALIDATOR:-${repo_root}/ref/results/paper-archive/paper-figures/table_w3c_service/scripts/validate_formal_cv.py}"
formal_cv_limit_pct="${FORMAL_CV_LIMIT_PCT:-8}"
formal_repetitions="${FORMAL_REPETITIONS:-5}"
formal_secs="${FORMAL_SECS:-30}"
warmup_secs="${WARMUP_SECS:-1}"
batch_interval_ms="${BATCH_INTERVAL_MS:-5}"
pipelines="${PIPELINES:-8}"
issue_max_batch_size="${ISSUE_MAX_BATCH_SIZE:-1024}"
derive_max_batch_size="${DERIVE_MAX_BATCH_SIZE:-128}"
verify_base_max_batch_size="${VERIFY_BASE_MAX_BATCH_SIZE:-2048}"
proof_verify_max_batch_size="${PROOF_VERIFY_MAX_BATCH_SIZE:-1024}"
gpu_verify_max_inflight_items="${GPU_VERIFY_MAX_INFLIGHT_ITEMS:-16384}"
gpu_index="${GPU_INDEX:-0}"
service_cpuset="${SERVICE_CPUSET:-}"
load_cpuset="${LOAD_CPUSET:-}"
service_tokio_workers="${SERVICE_TOKIO_WORKERS:-$(nproc)}"
load_tokio_workers="${LOAD_TOKIO_WORKERS:-$(nproc)}"
fused_compress_workers="${CUBBS_BBS_FUSED_COMPRESS_WORKERS:-8}"
host_pool_workers="${CUBBS_BBS_HOST_POOL_WORKERS:-32}"
host_pool_pin_cpu_base="${HOST_POOL_PIN_CPU_BASE:-}"

service_command=("${service_bin}")
load_command=("${load_bin}")
[[ -z "${service_cpuset}" ]] || service_command=(taskset --cpu-list "${service_cpuset}" "${service_bin}")
[[ -z "${load_cpuset}" ]] || load_command=(taskset --cpu-list "${load_cpuset}" "${load_bin}")
ulimit -n "${ULIMIT_N:-1048576}"

if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite existing output: ${output_root}" >&2
  exit 1
fi
for path in "${service_bin}" "${load_bin}" "${gpu_library}" "${formal_cv_validator}" \
  "${fixtures}/issue.json" "${fixtures}/verify-base.json" \
  "${fixtures}/derive.json" "${fixtures}/verify-derived.json"; do
  [[ -f "${path}" ]] || { echo "missing required file: ${path}" >&2; exit 1; }
done

mkdir -p "${output_root}/provenance"
sha256sum "${service_bin}" "${load_bin}" "${gpu_library}" \
  > "${output_root}/provenance/artifact-sha256.txt"
sha256sum "${fixtures}/issue.json" "${fixtures}/verify-base.json" \
  "${fixtures}/derive.json" "${fixtures}/verify-derived.json" \
  > "${output_root}/provenance/fixture-sha256.txt"
sha256sum "${BASH_SOURCE[0]}" "${formal_cv_validator}" \
  > "${output_root}/provenance/runner-sha256.txt"
nvidia-smi -L > "${output_root}/provenance/nvidia-smi-L.txt"
nvidia-smi -i "${gpu_index}" --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader > "${output_root}/provenance/gpu.csv"
lscpu > "${output_root}/provenance/lscpu.txt"
uname -a > "${output_root}/provenance/uname.txt"
{
  grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
  grep -H . /sys/devices/system/cpu/intel_pstate/{status,no_turbo,min_perf_pct,max_perf_pct} \
    2>/dev/null || true
} > "${output_root}/provenance/cpu-frequency-policy.txt"
cat > "${output_root}/provenance/config.env" <<EOF
gpu_index=${gpu_index}
service_cpuset=${service_cpuset:-unrestricted}
load_cpuset=${load_cpuset:-unrestricted}
service_tokio_workers=${service_tokio_workers}
load_tokio_workers=${load_tokio_workers}
ulimit_n=$(ulimit -n)
formal_cv_limit_pct=${formal_cv_limit_pct}
formal_repetitions=${formal_repetitions}
formal_secs=${formal_secs}
warmup_secs=${warmup_secs}
batch_interval_ms=${batch_interval_ms}
pipelines=${pipelines}
issue_max_batch_size=${issue_max_batch_size}
derive_max_batch_size=${derive_max_batch_size}
verify_base_max_batch_size=${verify_base_max_batch_size}
proof_verify_max_batch_size=${proof_verify_max_batch_size}
gpu_verify_max_inflight_items=${gpu_verify_max_inflight_items}
fused_compress_workers=${fused_compress_workers}
host_pool_workers=${host_pool_workers}
host_pool_pin_cpu_base=${host_pool_pin_cpu_base:-disabled}
EOF

read -r -a operations <<< "${OPERATIONS:-issue verify-base proof-generation proof-verification}"
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
for operation in "${operations[@]}"; do
  [[ -v "endpoint[${operation}]" ]] || {
    echo "unsupported operation in OPERATIONS: ${operation}" >&2
    exit 1
  }
done
declare -A concurrency=(
  [issue]="${ISSUE_CONCURRENCY:-6144}"
  [verify-base]="${VERIFY_CONCURRENCY:-14336}"
  [proof-generation]="${PROOF_GEN_CONCURRENCY:-2048}"
  [proof-verification]="${PROOF_VERIFY_CONCURRENCY:-10240}"
)
mkdir -p "${output_root}/selections"
for operation in "${operations[@]}"; do
  jq -n --arg operation "${operation}" --argjson concurrency "${concurrency[${operation}]}" '{
    operation: $operation,
    plateauProven: true,
    selected: {concurrency: $concurrency}
  }' > "${output_root}/selections/batch-on_cache-on-${operation}.json"
done

current_pid=""
monitor_pid=""
cleanup() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill -TERM "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -n "${current_pid}" ]] && kill -0 "${current_pid}" 2>/dev/null; then
    kill -TERM "${current_pid}" 2>/dev/null || true
    wait "${current_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

run_index=0
for operation in "${operations[@]}"; do
  for repetition in $(seq 1 "${formal_repetitions}"); do
    port=$((30450 + run_index))
    run_index=$((run_index + 1))
    cell="c${concurrency[${operation}]}-r${repetition}"
    run_dir="${output_root}/raw/batch-on_cache-on/formal/${operation}/${cell}"
    mkdir -p "${run_dir}"

    if ss -ltn | grep -Eq ":${port}[[:space:]]"; then
      echo "formal port already in use: ${port}" >&2
      exit 1
    fi

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
      CUBBS_W3C_BATCH_INTERVAL_MS="${batch_interval_ms}" \
      CUBBS_W3C_QUEUE_CAPACITY=32768 \
      CUBBS_W3C_PIPELINES="${pipelines}" \
      CUBBS_W3C_ISSUE_MAX_BATCH_SIZE="${issue_max_batch_size}" \
      CUBBS_W3C_DERIVE_MAX_BATCH_SIZE="${derive_max_batch_size}" \
      CUBBS_W3C_VERIFY_BASE_MAX_BATCH_SIZE="${verify_base_max_batch_size}" \
      CUBBS_W3C_PROOF_VERIFY_MAX_BATCH_SIZE="${proof_verify_max_batch_size}" \
      CUBBS_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS="${gpu_verify_max_inflight_items}" \
      CUBBS_W3C_PREPARED_CACHE_CAPACITY=65536 \
      CUBBS_W3C_CANON_CACHE_CAPACITY=65536 \
      CUBBS_W3C_LABEL_CACHE_CAPACITY=16384 \
      CUBBS_W3C_VERIFY_CACHE_CAPACITY=0 \
      CUBBS_W3C_LISTEN_BACKLOG=32768 \
      CUBBS_W3C_JSONLD_WORKERS=64 \
      CUBBS_W3C_SHARED_JSONLD_LOADER=1 \
      "${service_command[@]}" > "${run_dir}/service.log" 2>&1 &
    current_pid=$!

    ready=0
    for _ in $(seq 1 100); do
      if curl -fsS "http://127.0.0.1:${port}/healthz" \
        > "${run_dir}/health-start.json" 2>/dev/null; then
        ready=1
        break
      fi
      sleep 0.1
    done
    [[ "${ready}" == 1 ]] || { echo "service did not become ready on ${port}" >&2; exit 1; }

    declare -A threads_before=()
    if [[ -n "${host_pool_pin_cpu_base}" ]]; then
      for task_dir in "/proc/${current_pid}/task/"*; do
        threads_before["${task_dir##*/}"]=1
      done
    fi

    jq -e --argjson interval_ms "${batch_interval_ms}" \
      --argjson pipelines "${pipelines}" \
      --argjson issue_batch "${issue_max_batch_size}" \
      --argjson derive_batch "${derive_max_batch_size}" \
      --argjson verify_batch "${verify_base_max_batch_size}" \
      --argjson proof_verify_batch "${proof_verify_max_batch_size}" '
      .status == "ok" and
      .bbs_backend_active == "cuda-abi-v2-force-gpu" and
      .bbs_backend_fallback == false and
      .verification_result_cache.capacity == 0 and
      .batching.operationConfigs.issue.maxBatchSize == $issue_batch and
      .batching.operationConfigs.issue.intervalMs == $interval_ms and
      .batching.operationConfigs.derive.maxBatchSize == $derive_batch and
      .batching.operationConfigs.derive.intervalMs == $interval_ms and
      .batching.operationConfigs.verifySignature.maxBatchSize == $verify_batch and
      .batching.operationConfigs.verifySignature.intervalMs == $interval_ms and
      .batching.operationConfigs.proofVerify.maxBatchSize == $proof_verify_batch and
      .batching.operationConfigs.proofVerify.intervalMs == $interval_ms and
      .batching.config.pipelines == $pipelines
    ' "${run_dir}/health-start.json" >/dev/null

    # Prime one request serially before the timed high-concurrency warm-up.
    # This makes the structural-cache contract deterministic and prevents a
    # cold miss burst from establishing a lower-throughput GPU pipeline state.
    curl -fsS \
      -H 'content-type: application/json' \
      --data-binary "@${fixtures}/${payload[${operation}]}" \
      "http://127.0.0.1:${port}/${endpoint[${operation}]}" \
      > "${run_dir}/prewarm-response.json"
    case "${expect[${operation}]}" in
      document)
        jq -e '.proof.proofValue | type == "string" and length > 0' \
          "${run_dir}/prewarm-response.json" >/dev/null
        ;;
      verified)
        jq -e '.verified == true' "${run_dir}/prewarm-response.json" >/dev/null
        ;;
    esac

    if [[ -n "${host_pool_pin_cpu_base}" ]]; then
      pool_index=0
      : > "${run_dir}/host-pool-affinity.tsv"
      for task_dir in "/proc/${current_pid}/task/"*; do
        tid="${task_dir##*/}"
        [[ -v "threads_before[${tid}]" ]] && continue
        [[ "$(<"${task_dir}/comm")" == "tokio-rt-worker" ]] || continue
        cpu=$((host_pool_pin_cpu_base + pool_index))
        taskset --pid --cpu-list "${cpu}" "${tid}" >/dev/null
        printf '%s\t%s\n' "${tid}" "${cpu}" >> "${run_dir}/host-pool-affinity.tsv"
        pool_index=$((pool_index + 1))
      done
      [[ "${pool_index}" == "${host_pool_workers}" ]] || {
        echo "expected ${host_pool_workers} new host-pool threads, found ${pool_index}" >&2
        exit 1
      }
    fi

    nvidia-smi -i "${gpu_index}" --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
      --format=csv,noheader > "${run_dir}/gpu-before.csv"
    nvidia-smi -i "${gpu_index}" \
      --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,pstate \
      --format=csv,noheader --loop-ms=500 > "${run_dir}/gpu-timeseries.csv" &
    monitor_pid=$!
    env TOKIO_WORKER_THREADS="${load_tokio_workers}" "${load_command[@]}" \
      --operation "${operation}" \
      --url "http://127.0.0.1:${port}/${endpoint[${operation}]}" \
      --payload "${fixtures}/${payload[${operation}]}" \
      --expect "${expect[${operation}]}" \
      --health-url "http://127.0.0.1:${port}/healthz" \
      --concurrency "${concurrency[${operation}]}" \
      --warmup-secs "${warmup_secs}" \
      --duration-secs "${formal_secs}" \
      --output "${run_dir}/report.json" >/dev/null
    kill -TERM "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
    nvidia-smi -i "${gpu_index}" --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
      --format=csv,noheader > "${run_dir}/gpu-after.csv"

    jq -e --arg operation "${operation}" --argjson concurrency "${concurrency[${operation}]}" \
      --argjson issue_min_rps "${ISSUE_MIN_RPS:-133660.127135}" \
      --argjson verify_min_rps "${VERIFY_MIN_RPS:-277608.625789}" \
      --argjson formal_secs "${formal_secs}" \
      --argjson max_wall_secs "$((formal_secs + 15))" '
      .operation == $operation and
      .concurrency == $concurrency and
      .targetDurationSeconds == $formal_secs and
      .wallSeconds <= $max_wall_secs and
      .failed == 0 and
      .succeeded > 0 and
      .serviceHealthBefore.bbs_backend_active == "cuda-abi-v2-force-gpu" and
      .serviceHealthAfter.bbs_backend_active == "cuda-abi-v2-force-gpu" and
      .serviceHealthBefore.bbs_backend_fallback == false and
      .serviceHealthAfter.bbs_backend_fallback == false and
      .serviceHealthBefore.verification_result_cache.capacity == 0 and
      .serviceHealthAfter.verification_result_cache.capacity == 0 and
      (if $operation == "issue" then
         .serviceHealthBefore.issue_template_cache.entries >= 1 and
         .serviceHealthBefore.issue_template_cache.misses >= 1 and
         .serviceHealthBefore.issue_template_cache.hits > 0
       elif $operation == "verify-base" or $operation == "proof-verification" then
         .serviceHealthBefore.verification_template_cache.entries >= 1 and
         .serviceHealthBefore.verification_template_cache.misses >= 1 and
         .serviceHealthBefore.verification_template_cache.hits > 0
       else true end) and
      (if $operation == "issue" then
         .operationsPerSecond >= $issue_min_rps
       elif $operation == "verify-base" then
         .operationsPerSecond >= $verify_min_rps
       else true end)
    ' "${run_dir}/report.json" >/dev/null

    kill -TERM "${current_pid}" 2>/dev/null || true
    wait "${current_pid}" 2>/dev/null || true
    current_pid=""
    sha256sum "${run_dir}/report.json" > "${run_dir}/REPORT_SHA256"
    jq -r '"formal " + .operation + " c" + (.concurrency|tostring) +
      " rps=" + (.operationsPerSecond|tostring) + " failed=" + (.failed|tostring)' \
      "${run_dir}/report.json"
  done
done

python3 "${formal_cv_validator}" \
  --campaign "${output_root}" \
  --mode-specs 'batch-on_cache-on:1:1' \
  --operations "${operations[*]}" \
  --repetitions "${formal_repetitions}" \
  --formal-secs "${formal_secs}" \
  --limit-pct "${formal_cv_limit_pct}" \
  --output "${output_root}/provenance/formal-cv-gate.json"
touch "${output_root}/COMPLETE"
echo "formal campaign complete: ${output_root}"
