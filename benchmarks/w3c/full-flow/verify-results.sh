#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="$repo_root/benchmarks/w3c/full-flow"
manifest="$repo_root/benchmarks/w3c/datasets/http-unique-10k-v1/manifest.json"
dataset="$repo_root/benchmarks/w3c/datasets/http-unique-10k-v1/generated/issue-requests.jsonl"
manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
dataset_sha256="$(jq -r '.generatedSha256' "$manifest")"

if [[ -f "$dataset" ]] &&
  [[ "$(sha256sum "$dataset" | awk '{print $1}')" != "$dataset_sha256" ]]; then
  echo "W3C generated dataset digest mismatch" >&2
  exit 1
fi

mapfile -t runs < <(find "$root/results" -mindepth 2 -maxdepth 2 -name run.json -print | sort)
if ((${#runs[@]} < 2)); then
  echo "expected at least CPU and GPU full-flow runs" >&2
  exit 1
fi

for run in "${runs[@]}"; do
  jq -e \
    --arg manifest_sha256 "$manifest_sha256" \
    --arg dataset_sha256 "$dataset_sha256" '
    . as $run |
    .schemaVersion == 1 and
    .tool == "cubbs-w3c-full-flow-baseline" and
    .accepted == true and
    .dataset.id == "http-unique-10k-v1" and
    .dataset.manifestSha256 == $manifest_sha256 and
    .dataset.generatedSha256 == $dataset_sha256 and
    (.source.commit | length == 40) and
    (.machine.label | length > 0) and
    .loadReport.mode == "sustained" and
    .loadReport.callPath ==
      "sustained-independent-workflows-single-request-http-buffered-batch" and
    .loadReport.targetDurationSeconds >= 30 and
    .loadReport.startupSpreadMs > 0 and
    .loadReport.datasetEntries == 10000 and
    .loadReport.uniqueVcInputs == true and
    .loadReport.totalWorkflows == .loadReport.succeededWorkflows and
    .loadReport.failedWorkflows == 0 and
    .loadReport.drainSeconds >= 0 and
    (.loadReport.operations | map(.operation) | sort) ==
      ["derive", "issue", "verify-derived", "verify-signature"] and
    all(.loadReport.operations[];
      .requests == $run.loadReport.totalWorkflows and .failed == 0) and
    .loadReport.backendInfo.fallback == false and
    (if .loadReport.backendInfo.requested == "gpu"
     then .loadReport.backendInfo.hardwareAcceleration == true
     else (.loadReport.backendInfo.requested == "cpu" and
           .loadReport.backendInfo.active == "zkryptium-cpu" and
           .loadReport.backendInfo.hardwareAcceleration == false)
     end) and
    (.loadReport.batchingDelta | keys | sort) ==
      ["derive", "issue", "proofVerify", "verifySignature"] and
    all(.loadReport.batchingDelta[];
      .dispatchedItems == $run.loadReport.totalWorkflows and
      .averageObservedBatch > 1 and
      .maxObservedBatch > 1) and
    .loadReport.serviceConfig.batching.intervalMs == 20 and
    .loadReport.serviceConfig.batching.maxBatchSize == 128 and
    .loadReport.serviceConfig.w3cRuntime.jsonldWorkers == 16 and
    .loadReport.serviceConfig.w3cRuntime.sharedJsonldLoader == true and
    .loadReport.serviceConfig.w3cRuntime.canonicalizationCacheCapacity == 4096
  ' "$run" >/dev/null
  echo "PASS $(jq -r '.machine.label' "$run"): $run"
done

comparison="$root/results/local-rtx5060ti-vs-cpu-c24-d30-v1.json"
jq -e '
  .accepted == true and
  .sameMachine == true and
  .sameLoadAndServiceConfig == true and
  .cpuRun.failed == 0 and
  .gpuRun.failed == 0 and
  .cpuRun.backend.requested == "cpu" and
  .gpuRun.backend.requested == "gpu" and
  .gpuRun.backend.hardwareAcceleration == true and
  .gpuRun.backend.fallback == false and
  .gpuVsCpu.throughputSpeedup > 1
' "$comparison" >/dev/null
echo "PASS paired comparison: $comparison"
