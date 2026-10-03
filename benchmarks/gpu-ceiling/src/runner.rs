use crate::{
    config::Config,
    report::{
        cpu_identity, gpu_identity, process_cpu_time, BackendReport, CeilingReport, MetricReport,
        PreparationReport,
    },
    workload::PreparedWorkload,
};
use cubbs_bbs::{BbsBackend, BbsEngine, BbsMsmTerms, GpuMsmBenchGroupInput};
use std::{
    hint::black_box,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

struct WorkerResult {
    native_calls: u64,
    outputs: u64,
    samples: Vec<Vec<u8>>,
}

pub fn run(
    config: Config,
    engine: &BbsEngine,
    workload: PreparedWorkload,
) -> Result<CeilingReport, String> {
    let backend = engine.backend_info();
    if !backend.hardware_acceleration || backend.fallback {
        return Err("device-ceiling benchmark requires hardware GPU without fallback".to_owned());
    }

    let expected = reference_samples(engine, &workload.groups)?;
    let mut validation_plan = engine
        .prepare_g1_msm_bench_plan(workload.groups.clone())
        .map_err(|error| error.to_string())?;
    validation_plan.run().map_err(|error| error.to_string())?;
    if collect_samples(&validation_plan)? != expected {
        return Err("prepared GPU MSM output differs from the established batch path".to_owned());
    }
    drop(validation_plan);

    let ready = Arc::new(Barrier::new(config.streams + 1));
    let start = Arc::new(Barrier::new(config.streams + 1));
    let warmup = Duration::from_secs_f64(config.warmup_secs);
    let duration = Duration::from_secs_f64(config.duration_secs);
    let per_stream_inputs = (0..config.streams)
        .map(|_| workload.groups.clone())
        .collect::<Vec<_>>();

    let (before, measured_started, worker_results) = thread::scope(|scope| -> Result<_, String> {
        let handles = per_stream_inputs
            .into_iter()
            .map(|inputs| {
                let ready = Arc::clone(&ready);
                let start = Arc::clone(&start);
                scope.spawn(move || worker(inputs, warmup, duration, ready, start))
            })
            .collect::<Vec<_>>();

        ready.wait();
        let before = process_cpu_time()?;
        let measured_started = Instant::now();
        start.wait();
        let worker_results = handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "GPU ceiling worker panicked".to_owned())?
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((before, measured_started, worker_results))
    })?;
    let elapsed = measured_started.elapsed();
    let after = process_cpu_time()?;

    let validation_passed = worker_results
        .iter()
        .all(|result| result.samples == expected);
    if !validation_passed {
        return Err("prepared GPU MSM output changed during measurement".to_owned());
    }

    let native_calls = worker_results
        .iter()
        .map(|result| result.native_calls)
        .sum::<u64>();
    let outputs = worker_results
        .iter()
        .map(|result| result.outputs)
        .sum::<u64>();
    let elapsed_secs = elapsed.as_secs_f64();
    let user_seconds = after.user_seconds - before.user_seconds;
    let system_seconds = after.system_seconds - before.system_seconds;
    let cpu_seconds = user_seconds + system_seconds;

    Ok(CeilingReport {
        tool: "cubbs-gpu-ceiling",
        mode: "prepared-common-points-msm",
        scope: "device-ceiling; excludes request construction, message hashing, protocol finalization, and result materialization",
        config,
        backend: BackendReport {
            requested: backend.requested,
            active: backend.active,
            hardware_acceleration: backend.hardware_acceleration,
            fallback: backend.fallback,
            gpu_abi_version: backend.gpu_abi_version,
            detail: backend.detail,
        },
        preparation: PreparationReport {
            policy: "inputs, packed scalar matrices, ABI descriptors, result slots, and output buffers prepared once per stream",
            scalar_matrix_bytes_per_stream: workload.scalar_matrix_bytes,
            common_point_bytes_per_stream: workload.point_bytes,
            rust_timed_loop_allocations: "zero by design",
            correctness_validation: "all first-row group outputs compared before and after the timed loop",
        },
        metric: MetricReport {
            elapsed_ms: elapsed_secs * 1000.0,
            native_calls,
            outputs,
            outputs_per_sec: outputs as f64 / elapsed_secs,
            native_calls_per_sec: native_calls as f64 / elapsed_secs,
            process_user_seconds: user_seconds,
            process_system_seconds: system_seconds,
            equivalent_cpu_cores: cpu_seconds / elapsed_secs,
            cpu_seconds_per_million_outputs: cpu_seconds * 1_000_000.0 / outputs as f64,
        },
        validation_passed,
        gpu: gpu_identity(),
        cpu: cpu_identity(),
    })
}

fn worker(
    inputs: Vec<GpuMsmBenchGroupInput>,
    warmup: Duration,
    duration: Duration,
    ready: Arc<Barrier>,
    start: Arc<Barrier>,
) -> Result<WorkerResult, String> {
    let engine = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let mut plan = engine
        .prepare_g1_msm_bench_plan(inputs)
        .map_err(|error| error.to_string())?;

    let warmup_started = Instant::now();
    loop {
        plan.run().map_err(|error| error.to_string())?;
        if warmup_started.elapsed() >= warmup {
            break;
        }
    }
    ready.wait();
    start.wait();

    let measured_started = Instant::now();
    let mut native_calls = 0u64;
    let mut outputs = 0u64;
    while measured_started.elapsed() < duration {
        let run = plan.run().map_err(|error| error.to_string())?;
        native_calls += 1;
        outputs += run.output_count as u64;
        black_box(run);
    }
    Ok(WorkerResult {
        native_calls,
        outputs,
        samples: collect_samples(&plan)?,
    })
}

fn collect_samples(plan: &cubbs_bbs::GpuMsmBenchPlan) -> Result<Vec<Vec<u8>>, String> {
    (0..plan.group_count())
        .map(|group| {
            plan.output(group, 0)
                .map(ToOwned::to_owned)
                .ok_or_else(|| format!("missing validation output for group {group}"))
        })
        .collect()
}

fn reference_samples(
    engine: &BbsEngine,
    groups: &[GpuMsmBenchGroupInput],
) -> Result<Vec<Vec<u8>>, String> {
    let terms = groups
        .iter()
        .map(|group| {
            let term_count = group.compressed_points.len();
            let first_row = group
                .scalar_rows_be
                .get(..term_count * 32)
                .ok_or_else(|| "prepared scalar matrix has no first row".to_owned())?;
            Ok(BbsMsmTerms {
                compressed_points: group.compressed_points.clone().into(),
                scalar_be: first_row.chunks_exact(32).map(ToOwned::to_owned).collect(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let reference = engine.g1_msm_batch_for_bench(&terms, true);
    if reference.failed != 0 || reference.results.len() != terms.len() {
        return Err(format!(
            "reference GPU MSM validation failed: attempted={}, succeeded={}, failed={}",
            reference.attempted, reference.succeeded, reference.failed
        ));
    }
    reference
        .results
        .into_iter()
        .map(|item| item.result.map_err(|error| error.to_string()))
        .collect()
}
