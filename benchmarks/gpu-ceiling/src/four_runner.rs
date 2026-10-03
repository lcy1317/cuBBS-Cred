use crate::{
    four_config::FourOpConfig,
    four_workload::FourOpTemplates,
    report::{cpu_identity, gpu_identity, process_cpu_time, BackendReport},
};
use cubbs_bbs::{
    reference::{CpuReferenceBackend, ZkryptiumReference},
    BbsBatchItemResult, BbsBatchRunResult, BbsEngine, ProofGenRequest, ProofVerifyRequest,
    SignRequest, VerifyRequest,
};
use serde::Serialize;
use std::{
    sync::{Arc, Barrier},
    thread,
    time::Instant,
};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FourOpMetric {
    pub operation: &'static str,
    pub batch_size: usize,
    pub preloaded_batches: usize,
    pub streams: usize,
    pub elapsed_ms: f64,
    pub items: usize,
    pub failed: usize,
    pub items_per_sec: f64,
    pub equivalent_cpu_cores: f64,
    pub cpu_seconds_per_million_items: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FourOpReport {
    pub tool: &'static str,
    pub mode: &'static str,
    pub scope: &'static str,
    pub config: FourOpConfig,
    pub backend: BackendReport,
    pub execution_policy: &'static str,
    pub pairing_policy: &'static str,
    pub cpu_reference: String,
    pub gpu_throughput_preset: bool,
    pub per_stream_warmup: bool,
    pub timed_input_clone: bool,
    pub metrics: Vec<FourOpMetric>,
    pub validation_passed: bool,
    pub gpu: Option<String>,
    pub cpu: Option<String>,
}

struct WorkerMetric {
    items: usize,
    failed: usize,
}

fn force_gpu_enabled() -> bool {
    std::env::var("CUBBS_BBS_FORCE_GPU")
        .map(|value| !matches!(value.as_str(), "" | "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(false)
}

fn use_single_proof_verify_host_worker(config: &FourOpConfig) -> bool {
    config.operation == "proof-verify" && config.message_counts.iter().all(|&count| count <= 256)
}

pub fn apply_gpu_throughput_preset(config: &FourOpConfig) {
    for (name, value) in [
        ("CUBBS_BBS_GPU_SPLIT_VARIABLE_PARALLEL_MIN_BATCH", "1"),
        ("CUBBS_BBS_GPU_POINT_GROUP_PARALLEL_MIN_GROUPS", "1"),
        ("CUBBS_BBS_GPU_MIXED_SHAPE_PARALLEL_MIN_GROUPS", "1"),
        ("CUBBS_BBS_PROOF_GEN_MIXED_SHAPE_PARALLEL_STREAMS", "1"),
        ("CUBBS_BBS_PROOF_VERIFY_MIXED_SHAPE_PARALLEL_STREAMS", "16"),
        ("CUBBS_BBS_PROOF_VERIFY_PAIR_MSM_PARALLEL_MIN_BATCH", "1"),
        ("CUBBS_BBS_PROOF_VERIFY_PARALLEL_PAIRING_MIN_BATCH", "1"),
        ("CUBBS_BBS_PROOF_VERIFY_UNIQUE_KEY_WORKERS", "1"),
        ("CUBBS_BBS_SIGN_FINALIZE_PARALLEL_MIN_BATCH", "1"),
        ("CUBBS_BBS_PROOF_GEN_FINALIZE_PARALLEL_MIN_BATCH", "1"),
        ("CUBBS_BBS_PROOF_GPU_PREPARE_PARALLEL_MIN_BATCH", "1"),
        ("CUBBS_BBS_GPU_FUSED_COMMON_POINTS_MSM", "1"),
        ("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI", "1"),
        ("CUBBS_BBS_GPU_NATIVE_MIXED_MSM_GROUPING", "1"),
        ("CUBBS_BBS_GPU_NATIVE_G1_ADD", "1"),
        ("CUBBS_BBS_GPU_SIGN_FINALIZE", "1"),
        ("CUBBS_BBS_GPU_PROOF_VERIFY_SPLIT_MSM", "1"),
        // The preloaded runner intentionally repeats request templates to keep
        // construction and allocation outside the timed interval. Request-
        // result caches would therefore turn the second and later iterations
        // into cache-hit benchmarks instead of measuring cryptographic work.
        // Public-context caches (generators and prepared G2 state) remain on.
        ("CUBBS_BBS_MESSAGE_SCALAR_CACHE", "0"),
        ("CUBBS_BBS_SIGN_PREPARED_CACHE", "0"),
        ("CUBBS_BBS_VERIFY_PREPARED_B_CACHE", "0"),
        ("CUBBS_BBS_PROOF_VERIFY_PREPARED_CACHE", "0"),
        ("CUBBS_BBS_GPU_PAIRING_AGGREGATE_RESULT_CACHE", "0"),
        ("CUBBS_BBS_GPU_PAIRING_MICROBATCH", "1"),
        ("CUBBS_BBS_GPU_PAIRING_MICROBATCH_MIN", "8"),
        ("CUBBS_BBS_GPU_PAIRING_MICROBATCH_MAX", "32"),
        ("CUBBS_BBS_GPU_PAIRING_MICROBATCH_GATHER_US", "750"),
        ("CUBBS_BBS_GPU_PAIRING_AGGREGATE_LANES", "32"),
        ("CUBBS_BBS_FUSED_COMPRESS_WORKERS", "8"),
        ("CUBBS_BBS_FUSED_COMPRESS_PARALLEL_MIN_ITEMS", "8"),
        ("CUBBS_BBS_BLST_MSM_WORKERS", "4"),
        ("CUBBS_BBS_HOST_POOL", "1"),
        (
            "CUBBS_BBS_HASH_NONBLOCKING",
            if config.proof_verify_streams <= 16 {
                "1"
            } else {
                "0"
            },
        ),
    ] {
        if std::env::var_os(name).is_none() {
            std::env::set_var(name, value);
        }
    }
    if use_single_proof_verify_host_worker(config)
        && std::env::var_os("CUBBS_BBS_PROOF_HOST_WORKERS").is_none()
    {
        std::env::set_var("CUBBS_BBS_PROOF_HOST_WORKERS", "1");
    }
    if std::env::var_os("CUBBS_BBS_SIGN_E_WORKERS").is_none() {
        std::env::set_var(
            "CUBBS_BBS_SIGN_E_WORKERS",
            if config.sign_streams >= 4 {
                "1"
            } else if config.sign_streams >= 2 {
                "2"
            } else {
                "4"
            },
        );
    }
    if config.message_counts.len() == 1
        && std::env::var_os("CUBBS_BBS_SIGN_SHAPE_BATCHING").is_none()
    {
        std::env::set_var("CUBBS_BBS_SIGN_SHAPE_BATCHING", "0");
    }
}

pub fn run(
    config: FourOpConfig,
    engine: &BbsEngine,
    templates: FourOpTemplates,
) -> Result<FourOpReport, String> {
    let gpu_backend = config.backend == "gpu";
    let backend = engine.backend_info();
    if gpu_backend && (!backend.hardware_acceleration || backend.fallback) {
        return Err("four-operation benchmark requires hardware GPU without fallback".to_owned());
    }

    validate_templates(engine, &templates)?;
    let single_api = config.api_mode == "single";
    let mut metrics = Vec::with_capacity(4);
    if let Some(template) = &templates.sign {
        metrics.push(measure(
            engine,
            "sign",
            config.batch_size,
            config.sign_batches,
            config.sign_streams,
            template,
            |engine, requests| {
                if single_api {
                    run_single(requests, |request| engine.sign(request))
                } else {
                    engine.sign_batch(requests, true)
                }
            },
            |result| validate_generated(result, 80),
        )?);
    }
    if let Some(template) = &templates.verify {
        metrics.push(measure(
            engine,
            "verify-signature",
            config.batch_size,
            config.verify_batches,
            config.verify_streams,
            template,
            |engine, requests| {
                if single_api {
                    run_single(requests, |request| engine.verify(request))
                } else {
                    engine.verify_batch(requests, true)
                }
            },
            validate_booleans,
        )?);
    }
    if let Some(template) = &templates.proof_gen {
        metrics.push(measure(
            engine,
            "proof-gen",
            config.batch_size,
            config.proof_gen_batches,
            config.proof_gen_streams,
            template,
            |engine, requests| {
                if single_api {
                    run_single(requests, |request| engine.proof_gen(request))
                } else {
                    engine.proof_gen_batch(requests, true)
                }
            },
            |result| validate_generated(result, 1),
        )?);
    }
    if let Some(template) = &templates.proof_verify {
        metrics.push(measure(
            engine,
            "proof-verify",
            config.batch_size,
            config.proof_verify_batches,
            config.proof_verify_streams,
            template,
            |engine, requests| {
                if single_api {
                    run_single(requests, |request| engine.proof_verify(request))
                } else {
                    engine.proof_verify_batch(requests, true)
                }
            },
            validate_booleans,
        )?);
    }

    let validation_passed = metrics.iter().all(|metric| metric.failed == 0);
    Ok(FourOpReport {
        tool: "cubbs-four-op-preloaded",
        mode: "full-protocol-preloaded-input",
        scope: "complete BBS operation API; request construction and benchmark deep-cloning excluded from timing",
        config,
        backend: BackendReport {
            requested: backend.requested,
            active: backend.active,
            hardware_acceleration: backend.hardware_acceleration,
            fallback: backend.fallback,
            gpu_abi_version: backend.gpu_abi_version,
            detail: backend.detail,
        },
        execution_policy: if !gpu_backend {
            "cpu-reference"
        } else if force_gpu_enabled() {
            "force-gpu"
        } else {
            "auto"
        },
        pairing_policy: if !gpu_backend {
            "cpu-reference"
        } else if force_gpu_enabled() {
            "gpu-only-no-cpu-fallback"
        } else {
            "gpu-parallel-primary-blst-single-equation-fallback"
        },
        cpu_reference: if gpu_backend {
            format!(
                "zkryptium-{}",
                std::env::var("CUBBS_BBS_ZKRYPTIUM_REFERENCE")
                    .unwrap_or_else(|_| "0.6.1".to_owned())
            )
        } else {
            "cuBBS-Cred CuBbsProtocolEngine (ZKryptium 0.2.2 derivative)".to_owned()
        },
        gpu_throughput_preset: gpu_backend,
        per_stream_warmup: true,
        timed_input_clone: false,
        metrics,
        validation_passed,
        gpu: gpu_backend.then(gpu_identity).flatten(),
        cpu: cpu_identity(),
    })
}

pub fn run_reference(
    config: FourOpConfig,
    reference: &ZkryptiumReference,
    templates: FourOpTemplates,
) -> Result<FourOpReport, String> {
    if config.backend != "cpu" || config.api_mode != "single" {
        return Err("explicit reference runner requires --backend cpu --api single".to_owned());
    }
    validate_reference_templates(reference, &templates)?;
    let mut metrics = Vec::with_capacity(4);
    if let Some(template) = &templates.sign {
        metrics.push(measure(
            reference,
            "sign",
            config.batch_size,
            config.sign_batches,
            config.sign_streams,
            template,
            |reference, requests| run_single(requests, |request| reference.sign(&request)),
            |result| validate_generated(result, 80),
        )?);
    }
    if let Some(template) = &templates.verify {
        metrics.push(measure(
            reference,
            "verify-signature",
            config.batch_size,
            config.verify_batches,
            config.verify_streams,
            template,
            |reference, requests| run_single(requests, |request| reference.verify(&request)),
            validate_booleans,
        )?);
    }
    if let Some(template) = &templates.proof_gen {
        metrics.push(measure(
            reference,
            "proof-gen",
            config.batch_size,
            config.proof_gen_batches,
            config.proof_gen_streams,
            template,
            |reference, requests| run_single(requests, |request| reference.proof_gen(&request)),
            |result| validate_generated(result, 1),
        )?);
    }
    if let Some(template) = &templates.proof_verify {
        metrics.push(measure(
            reference,
            "proof-verify",
            config.batch_size,
            config.proof_verify_batches,
            config.proof_verify_streams,
            template,
            |reference, requests| run_single(requests, |request| reference.proof_verify(&request)),
            validate_booleans,
        )?);
    }
    let validation_passed = metrics.iter().all(|metric| metric.failed == 0);
    Ok(FourOpReport {
        tool: "cubbs-four-op-reference",
        mode: "full-protocol-preloaded-input",
        scope: "complete BBS operation through explicit ZKryptium public single-item APIs; key and request construction excluded from timing",
        config,
        backend: BackendReport {
            requested: "cpu-reference",
            active: reference.version().name(),
            hardware_acceleration: false,
            fallback: false,
            gpu_abi_version: 0,
            detail: "explicit versioned ZKryptium CPU reference adapter",
        },
        execution_policy: "public-single-item-reference-api",
        pairing_policy: "cpu-reference",
        cpu_reference: reference.version().name().to_owned(),
        gpu_throughput_preset: false,
        per_stream_warmup: true,
        timed_input_clone: false,
        metrics,
        validation_passed,
        gpu: None,
        cpu: cpu_identity(),
    })
}

fn measure<E, I, T, F, C>(
    engine: &E,
    operation: &'static str,
    batch_size: usize,
    batch_count: usize,
    streams: usize,
    template: &[I],
    operation_fn: F,
    check: C,
) -> Result<FourOpMetric, String>
where
    E: Clone + Send + Sync,
    I: Clone + Send + Sync,
    T: Send,
    F: Fn(&E, Vec<I>) -> BbsBatchRunResult<T> + Sync,
    C: Fn(&BbsBatchRunResult<T>) -> bool + Sync,
{
    let stream_batch_counts = (0..streams)
        .map(|stream| batch_count / streams + usize::from(stream < batch_count % streams))
        .collect::<Vec<_>>();
    let ready = Arc::new(Barrier::new(streams + 1));
    let start = Arc::new(Barrier::new(streams + 1));
    let (before, started, workers) = thread::scope(|scope| -> Result<_, String> {
        let handles = stream_batch_counts
            .into_iter()
            .map(|local_batch_count| {
                let engine = engine.clone();
                let ready = Arc::clone(&ready);
                let start = Arc::clone(&start);
                let operation_fn = &operation_fn;
                let check = &check;
                scope.spawn(move || {
                    // Allocate and deep-clone on the consumer thread before
                    // the CPU/time snapshot. This avoids measuring both clone
                    // work and cross-thread allocator reclamation.
                    let batches = (0..local_batch_count)
                        .map(|_| template.to_vec())
                        .collect::<Vec<_>>();
                    let warmup = batches
                        .first()
                        .expect("each stream must own at least one batch")
                        .clone();
                    let warmup_result = operation_fn(&engine, warmup);
                    assert!(check(&warmup_result), "{operation} stream warmup failed");
                    drop(warmup_result);
                    ready.wait();
                    start.wait();
                    let mut items = 0usize;
                    let mut failed = 0usize;
                    for batch in batches {
                        let result = operation_fn(&engine, batch);
                        if !check(&result) {
                            failed += result.attempted.max(1);
                        } else {
                            failed += result.failed;
                            items += result.succeeded;
                        }
                    }
                    WorkerMetric { items, failed }
                })
            })
            .collect::<Vec<_>>();
        ready.wait();
        let before = process_cpu_time()?;
        let started = Instant::now();
        start.wait();
        let workers = handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| format!("{operation} worker panicked"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((before, started, workers))
    })?;
    let elapsed = started.elapsed();
    let after = process_cpu_time()?;
    let items = workers.iter().map(|worker| worker.items).sum::<usize>();
    let failed = workers.iter().map(|worker| worker.failed).sum::<usize>();
    let elapsed_secs = elapsed.as_secs_f64();
    let cpu_seconds =
        (after.user_seconds - before.user_seconds) + (after.system_seconds - before.system_seconds);
    Ok(FourOpMetric {
        operation,
        batch_size,
        preloaded_batches: batch_count,
        streams,
        elapsed_ms: elapsed_secs * 1000.0,
        items,
        failed,
        items_per_sec: items as f64 / elapsed_secs,
        equivalent_cpu_cores: cpu_seconds / elapsed_secs,
        cpu_seconds_per_million_items: cpu_seconds * 1_000_000.0 / items as f64,
    })
}

fn run_single<I, T, F>(requests: Vec<I>, operation: F) -> BbsBatchRunResult<T>
where
    F: Fn(I) -> cubbs_bbs::Result<T>,
{
    let attempted = requests.len();
    let mut succeeded = 0usize;
    let results = requests
        .into_iter()
        .enumerate()
        .map(|(index, request)| {
            let result = operation(request);
            succeeded += usize::from(result.is_ok());
            BbsBatchItemResult { index, result }
        })
        .collect();
    BbsBatchRunResult {
        attempted,
        succeeded,
        failed: attempted - succeeded,
        results,
    }
}

fn validate_generated(result: &BbsBatchRunResult<Vec<u8>>, minimum_len: usize) -> bool {
    result.failed == 0
        && result.succeeded == result.attempted
        && result.results.iter().all(|item| {
            item.result
                .as_ref()
                .is_ok_and(|value| value.len() >= minimum_len)
        })
}

fn validate_booleans(result: &BbsBatchRunResult<bool>) -> bool {
    result.failed == 0
        && result.succeeded == result.attempted
        && result
            .results
            .iter()
            .all(|item| item.result.as_ref().is_ok_and(|value| *value))
}

fn validate_templates(engine: &BbsEngine, templates: &FourOpTemplates) -> Result<(), String> {
    let valid =
        templates.sign.as_ref().is_none_or(|requests| {
            validate_generated(&engine.sign_batch(requests.clone(), true), 80)
        }) && templates
            .verify
            .as_ref()
            .is_none_or(|requests| validate_booleans(&engine.verify_batch(requests.clone(), true)))
            && templates.proof_gen.as_ref().is_none_or(|requests| {
                validate_generated(&engine.proof_gen_batch(requests.clone(), true), 1)
            })
            && templates.proof_verify.as_ref().is_none_or(|requests| {
                validate_booleans(&engine.proof_verify_batch(requests.clone(), true))
            });
    if !valid {
        Err("four-operation template validation failed".to_owned())
    } else {
        Ok(())
    }
}

fn validate_reference_templates(
    reference: &ZkryptiumReference,
    templates: &FourOpTemplates,
) -> Result<(), String> {
    let valid = templates.sign.as_ref().is_none_or(|requests| {
        validate_generated(
            &run_single(requests.clone(), |request| reference.sign(&request)),
            80,
        )
    }) && templates.verify.as_ref().is_none_or(|requests| {
        validate_booleans(&run_single(requests.clone(), |request| {
            reference.verify(&request)
        }))
    }) && templates.proof_gen.as_ref().is_none_or(|requests| {
        validate_generated(
            &run_single(requests.clone(), |request| reference.proof_gen(&request)),
            1,
        )
    }) && templates.proof_verify.as_ref().is_none_or(|requests| {
        validate_booleans(&run_single(requests.clone(), |request| {
            reference.proof_verify(&request)
        }))
    });
    if valid {
        Ok(())
    } else {
        Err("explicit reference template validation failed".to_owned())
    }
}

#[allow(dead_code)]
fn _request_types_are_send(
    _: SignRequest,
    _: VerifyRequest,
    _: ProofGenRequest,
    _: ProofVerifyRequest,
) {
}

#[cfg(test)]
mod tests {
    use super::{use_single_proof_verify_host_worker, FourOpConfig};

    #[test]
    fn single_proof_verify_host_worker_is_limited_to_short_shapes() {
        let mut config = FourOpConfig {
            operation: "proof-verify".to_owned(),
            message_counts: vec![64, 128, 256],
            ..FourOpConfig::default()
        };
        assert!(use_single_proof_verify_host_worker(&config));

        config.message_counts = vec![2_048];
        assert!(!use_single_proof_verify_host_worker(&config));
    }
}
