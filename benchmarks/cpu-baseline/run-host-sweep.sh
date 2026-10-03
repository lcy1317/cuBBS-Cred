#!/usr/bin/env bash
set -euo pipefail

if (($# != 4)); then
  echo "usage: $0 HOST REMOTE_BINARY BUILD_LABEL OUTPUT_DIR" >&2
  exit 2
fi

host="$1"
binary="$2"
build_label="$3"
output_dir="$4"
cy_env="${CY_ENV:-/home/lcy/.codex/skills/cy-env/scripts/cy-env}"
messages="${MESSAGES:-16,64,128}"
batch="${BATCH:-32}"
batches="${BATCHES:-256}"
streams_list="${STREAMS_LIST:-32 64 96 128}"
cpu_set="${CUBBS_CPUSET:-0-63,128-191}"
memory_node="${CUBBS_MEMORY_NODE:-0}"
issuer_mode="${ISSUER_MODE:-same}"
disclosed_ratio="${DISCLOSED_RATIO:-0.5}"

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
for value in "$batch" "$batches" "$memory_node" $streams_list; do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "batch, stream, and memory-node values must be non-negative integers" >&2
    exit 2
  }
done
((batch > 0 && batches > 0)) || {
  echo "batch and batch count must be positive" >&2
  exit 2
}
for streams in $streams_list; do
  ((streams > 0)) || {
    echo "stream counts must be positive" >&2
    exit 2
  }
done

if compgen -G "$output_dir/raw/*.json" >/dev/null; then
  echo "OUTPUT_DIR already contains raw JSON reports; use a new directory" >&2
  exit 2
fi

mkdir -p "$output_dir/raw"

"$cy_env" run "$host" \
  "hostname; uptime; lscpu; numactl --hardware; \
   grep . /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true; \
   sha256sum '$binary'; \
   ps -eo pid,user,ni,psr,pcpu,pmem,etime,comm,args --sort=-pcpu | head -30" \
  >"$output_dir/system.txt"

run_one() {
  local operation="$1"
  local streams="$2"
  local batches_flag
  local streams_flag

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
      return 2
      ;;
  esac

  local stem="${build_label}-${operation}-b${batch}-n${batches}-s${streams}"
  "$cy_env" run "$host" \
    "CUBBS_BBS_ZKRYPTIUM_REFERENCE=0.6.1 \
     numactl --physcpubind='$cpu_set' --membind='$memory_node' '$binary' \
       --backend cpu --api single --operation '$operation' --messages '$messages' \
       $issuer_flag --disclosed-ratio '$disclosed_ratio' \
       --batch '$batch' '$batches_flag' '$batches' '$streams_flag' '$streams'" \
    >"$output_dir/raw/$stem.json" 2>"$output_dir/raw/$stem.stderr"

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
    "$output_dir/raw/$stem.json" >/dev/null
}

for operation in sign verify-signature proof-gen proof-verify; do
  for streams in $streams_list; do
    run_one "$operation" "$streams"
  done
done

jq -s \
  --arg host "$host" \
  --arg binary "$binary" \
  --arg build "$build_label" \
  --arg messages "$messages" \
  --arg issuer_mode "$issuer_mode" \
  --argjson expected_unique_issuer "$expected_unique_issuer" \
  --argjson disclosed_ratio "$disclosed_ratio" \
  --arg cpu_set "$cpu_set" \
  --argjson memory_node "$memory_node" \
  '{
    schemaVersion: 1,
    host: $host,
    binary: $binary,
    build: $build,
    messages: $messages,
    issuerMode: $issuer_mode,
    disclosedRatio: $disclosed_ratio,
    cpuSet: $cpu_set,
    memoryNode: $memory_node,
    allPassed: all(.[];
      (.tool == "cubbs-four-op-reference") and
      (.config.apiMode == "single") and
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
      (.config.operation == .metrics[0].operation) and
      (.metrics | all(.failed == 0))),
    results: map({
      operation: .metrics[0].operation,
      batch: .metrics[0].batchSize,
      batches: .metrics[0].preloadedBatches,
      streams: .metrics[0].streams,
      items: .metrics[0].items,
      elapsedMs: .metrics[0].elapsedMs,
      itemsPerSec: .metrics[0].itemsPerSec,
      equivalentCpuCores: .metrics[0].equivalentCpuCores,
      cpuSecondsPerMillionItems: .metrics[0].cpuSecondsPerMillionItems,
      failed: .metrics[0].failed
    })
  }' "$output_dir"/raw/*.json >"$output_dir/sweep-summary.json"

jq -e '.allPassed' "$output_dir/sweep-summary.json" >/dev/null
jq -r '
  .results
  | sort_by(.operation, .streams)
  | (["operation","streams","items_per_sec","cpu_cores","cpu_s_per_million"] | @tsv),
    (.[] | [.operation,.streams,.itemsPerSec,.equivalentCpuCores,.cpuSecondsPerMillionItems] | @tsv)
' "$output_dir/sweep-summary.json"
