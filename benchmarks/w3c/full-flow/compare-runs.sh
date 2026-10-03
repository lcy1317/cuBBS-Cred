#!/usr/bin/env bash
set -euo pipefail

if (($# != 3)); then
  echo "usage: $0 CPU_RUN_JSON GPU_RUN_JSON OUTPUT_JSON" >&2
  exit 2
fi

cpu="$1"
gpu="$2"
output="$3"
mkdir -p "$(dirname "$output")"

jq -n \
  --slurpfile cpu "$cpu" \
  --slurpfile gpu "$gpu" \
  '($cpu[0]) as $c | ($gpu[0]) as $g |
  if $c.dataset != $g.dataset then
    error("dataset identity differs")
  elif ($c.loadReport |
    {mode, concurrency, targetDurationSeconds, startupSpreadMs, serviceConfig}) !=
       ($g.loadReport |
    {mode, concurrency, targetDurationSeconds, startupSpreadMs, serviceConfig}) then
    error("load or service configuration differs")
  else {
    schemaVersion: 1,
    tool: "cubbs-w3c-full-flow-comparison",
    dataset: $c.dataset,
    sameMachine: ($c.machine.hostname == $g.machine.hostname),
    sameLoadAndServiceConfig: true,
    cpuRun: {
      path: $cpu_filename,
      backend: $c.loadReport.backendInfo,
      workflows: $c.loadReport.totalWorkflows,
      failed: $c.loadReport.failedWorkflows,
      workflowsPerSecond: $c.loadReport.workflowsPerSecond,
      p50Ms: $c.loadReport.workflowLatencyP50Ms,
      p95Ms: $c.loadReport.workflowLatencyP95Ms,
      p99Ms: $c.loadReport.workflowLatencyP99Ms
    },
    gpuRun: {
      path: $gpu_filename,
      backend: $g.loadReport.backendInfo,
      workflows: $g.loadReport.totalWorkflows,
      failed: $g.loadReport.failedWorkflows,
      workflowsPerSecond: $g.loadReport.workflowsPerSecond,
      p50Ms: $g.loadReport.workflowLatencyP50Ms,
      p95Ms: $g.loadReport.workflowLatencyP95Ms,
      p99Ms: $g.loadReport.workflowLatencyP99Ms
    },
    gpuVsCpu: {
      throughputSpeedup:
        ($g.loadReport.workflowsPerSecond / $c.loadReport.workflowsPerSecond),
      p50LatencyReduction:
        ($c.loadReport.workflowLatencyP50Ms / $g.loadReport.workflowLatencyP50Ms),
      p95LatencyReduction:
        ($c.loadReport.workflowLatencyP95Ms / $g.loadReport.workflowLatencyP95Ms),
      p99LatencyReduction:
        ($c.loadReport.workflowLatencyP99Ms / $g.loadReport.workflowLatencyP99Ms)
    },
    accepted:
      ($c.loadReport.failedWorkflows == 0 and
       $g.loadReport.failedWorkflows == 0 and
       $g.loadReport.workflowsPerSecond > $c.loadReport.workflowsPerSecond)
  } end' \
  --arg cpu_filename "$cpu" \
  --arg gpu_filename "$gpu" >"$output"

jq -e '.accepted and .sameMachine and .sameLoadAndServiceConfig' "$output" >/dev/null
cat "$output"
