#!/usr/bin/env bash
set -euo pipefail

if (($# != 8)); then
  echo "usage: $0 HOST REMOTE_BINARY BUILD_LABEL OPERATION PROCESSES STREAMS_PER_PROCESS BATCHES_PER_PROCESS OUTPUT_DIR" >&2
  exit 2
fi

host="$1"
binary="$2"
build_label="$3"
operation="$4"
processes="$5"
streams="$6"
batches="$7"
output_dir="$8"
cy_env="${CY_ENV:-/home/lcy/.codex/skills/cy-env/scripts/cy-env}"
messages="${MESSAGES:-16,64,128}"
batch="${BATCH:-32}"
issuer_mode="${ISSUER_MODE:-same}"
disclosed_ratio="${DISCLOSED_RATIO:-0.5}"
memory_node="${CUBBS_MEMORY_NODE:-0}"
physical_cpu_base="${CUBBS_PHYSICAL_CPU_BASE:-0}"
smt_cpu_base="${CUBBS_SMT_CPU_BASE:-128}"
cores_per_numa="${CUBBS_CORES_PER_NUMA:-64}"

case "$issuer_mode" in
  same)
    issuer_flag="--same-issuer"
    expected_unique_issuer=false
    ;;
  unique)
    issuer_flag="--unique-issuer-per-item"
    expected_unique_issuer=true
    ;;
  *)
    echo "ISSUER_MODE must be same or unique" >&2
    exit 2
    ;;
esac
for value in "$processes" "$streams" "$batch" "$batches" "$memory_node" \
  "$physical_cpu_base" "$smt_cpu_base" "$cores_per_numa"; do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "process, stream, batch, and NUMA layout values must be non-negative integers" >&2
    exit 2
  }
done
((processes > 0 && streams > 0 && batch > 0 && batches > 0 && cores_per_numa > 0)) || {
  echo "process, stream, batch, and core counts must be positive" >&2
  exit 2
}

case "$operation" in
  sign)
    batches_flag="--sign-batches"
    streams_flag="--sign-streams"
    ;;
  verify-signature)
    batches_flag="--verify-batches"
    streams_flag="--verify-streams"
    ;;
  proof-gen)
    batches_flag="--proof-gen-batches"
    streams_flag="--proof-gen-streams"
    ;;
  proof-verify)
    batches_flag="--proof-verify-batches"
    streams_flag="--proof-verify-streams"
    ;;
  *)
    echo "unsupported operation: $operation" >&2
    exit 2
    ;;
esac

if ((cores_per_numa % processes != 0)); then
  echo "process count must evenly partition CUBBS_CORES_PER_NUMA" >&2
  exit 2
fi
physical_cores_per_process=$((cores_per_numa / processes))
logical_cpus_per_process=$((physical_cores_per_process * 2))
if ((streams > logical_cpus_per_process)); then
  echo "streams per process exceed the assigned NUMA-local logical CPUs" >&2
  exit 2
fi

if compgen -G "$output_dir/raw/*.json" >/dev/null; then
  echo "OUTPUT_DIR already contains raw JSON reports; use a new directory" >&2
  exit 2
fi

mkdir -p "$output_dir/raw"
pids=()
wall_started="$(date +%s%N)"
for ((process = 0; process < processes; process++)); do
  first_cpu=$((physical_cpu_base + process * physical_cores_per_process))
  last_cpu=$((first_cpu + physical_cores_per_process - 1))
  first_smt_cpu=$((smt_cpu_base + process * physical_cores_per_process))
  last_smt_cpu=$((first_smt_cpu + physical_cores_per_process - 1))
  cpu_set="$first_cpu-$last_cpu,$first_smt_cpu-$last_smt_cpu"
  stem="${build_label}-${operation}-p${processes}-s${streams}-n${batches}-r${process}"
  "$cy_env" run "$host" \
    "CUBBS_BBS_ZKRYPTIUM_REFERENCE=0.6.1 \
     numactl --physcpubind='$cpu_set' --membind='$memory_node' '$binary' \
       --backend cpu --api single --operation '$operation' --messages '$messages' \
       $issuer_flag --disclosed-ratio '$disclosed_ratio' \
       --batch '$batch' '$batches_flag' '$batches' '$streams_flag' '$streams'" \
    >"$output_dir/raw/$stem.json" 2>"$output_dir/raw/$stem.stderr" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
wall_finished="$(date +%s%N)"
((failed == 0))

for report in "$output_dir"/raw/*.json; do
  jq -e \
    --arg messages "$messages" \
    --arg operation "$operation" \
    --argjson expected_unique_issuer "$expected_unique_issuer" \
    --argjson disclosed_ratio "$disclosed_ratio" \
    '(.tool == "cubbs-four-op-reference") and
     (.config.apiMode == "single") and
     (.config.operation == $operation) and
     ((.config.messageCounts | map(tostring) | join(",")) == $messages) and
     (.config.uniqueIssuerPerItem == $expected_unique_issuer) and
     (.config.disclosedRatio == $disclosed_ratio) and
     (.cpuReference == "zkryptium-0.6.1") and
     .validationPassed and
     (.backend.requested == "cpu-reference") and
     (.backend.active == "zkryptium-0.6.1") and
     (.backend.hardwareAcceleration == false) and
     (.backend.fallback == false) and
     (.metrics | length == 1) and
     (.metrics[0].operation == $operation) and
     (.metrics[0].failed == 0)' \
    "$report" >/dev/null
done

wall_ns=$((wall_finished - wall_started))
jq -s \
  --arg host "$host" \
  --arg binary "$binary" \
  --arg build "$build_label" \
  --arg operation "$operation" \
  --arg messages "$messages" \
  --arg issuer_mode "$issuer_mode" \
  --argjson expected_unique_issuer "$expected_unique_issuer" \
  --argjson disclosed_ratio "$disclosed_ratio" \
  --argjson memory_node "$memory_node" \
  --argjson physical_cpu_base "$physical_cpu_base" \
  --argjson smt_cpu_base "$smt_cpu_base" \
  --argjson cores_per_numa "$cores_per_numa" \
  --argjson processes "$processes" \
  --argjson streams_per_process "$streams" \
  --argjson wall_ns "$wall_ns" \
  '{
    schemaVersion: 1,
    host: $host,
    binary: $binary,
    build: $build,
    operation: $operation,
    messages: $messages,
    issuerMode: $issuer_mode,
    disclosedRatio: $disclosed_ratio,
    memoryNode: $memory_node,
    physicalCpuBase: $physical_cpu_base,
    smtCpuBase: $smt_cpu_base,
    coresPerNuma: $cores_per_numa,
    processes: $processes,
    streamsPerProcess: $streams_per_process,
    totalStreams: ($processes * $streams_per_process),
    launcherWallMs: ($wall_ns / 1000000),
    allPassed: all(.[];
      (.tool == "cubbs-four-op-reference") and
      (.config.apiMode == "single") and
      (.config.operation == $operation) and
      ((.config.messageCounts | map(tostring) | join(",")) == $messages) and
      (.config.uniqueIssuerPerItem == $expected_unique_issuer) and
      (.config.disclosedRatio == $disclosed_ratio) and
      (.cpuReference == "zkryptium-0.6.1") and
      .validationPassed and
      (.backend.requested == "cpu-reference") and
      (.backend.active == "zkryptium-0.6.1") and
      (.backend.hardwareAcceleration == false) and
      (.backend.fallback == false) and
      (.metrics | length == 1) and
      (.metrics[0].operation == $operation) and
      (.metrics | all(.failed == 0))),
    totalItems: ([.[].metrics[0].items] | add),
    summedItemsPerSec: ([.[].metrics[0].itemsPerSec] | add),
    maxTimedElapsedMs: ([.[].metrics[0].elapsedMs] | max),
    minTimedElapsedMs: ([.[].metrics[0].elapsedMs] | min),
    aggregateTimedItemsPerSec:
      (([.[].metrics[0].items] | add) / (([.[].metrics[0].elapsedMs] | max) / 1000)),
    equivalentCpuCores: ([.[].metrics[0].equivalentCpuCores] | add),
    failed: ([.[].metrics[0].failed] | add)
  }' "$output_dir"/raw/*.json >"$output_dir/summary.json"

jq -e '.allPassed and (.failed == 0)' "$output_dir/summary.json" >/dev/null
cat "$output_dir/summary.json"
