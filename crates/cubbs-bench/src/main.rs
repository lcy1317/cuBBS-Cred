use std::{
    collections::{BTreeMap, VecDeque},
    ffi::OsString,
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use cubbs_bbs::{
    BbsBackend, BbsBatchItemResult, BbsBatchRunResult, BbsEngine, BbsMsmRouteSummary,
    BbsMsmShapeSummary, BbsMsmTerms, ProofGenRequest, ProofVerifyRequest, SignRequest,
    VerifyRequest,
};
use cubbs_bbs_proofvalue::{parse_base, parse_derived};
use cubbs_vc_di_bbs::{
    build_proof_configuration, canonicalize_and_group_jsonld, canonicalize_jsonld,
    default_bbs_2023_proof_options, derive_disclosure_proof, derive_jsonld_disclosure_proof,
    derive_jsonld_disclosure_proof_batch, hash_joined_strings, issue_base_proof,
    issue_jsonld_base_proof, issue_jsonld_base_proof_batch, public_key_from_verification_method,
    relabel_and_sort_canonical_nquads, verification_method_from_public_key,
    verify_disclosure_proof, verify_jsonld_disclosure_proof, verify_jsonld_disclosure_proof_batch,
    BatchRunResult, DeriveJsonLdDisclosureProofBatchRequest, DeriveJsonLdDisclosureProofRequest,
    DisclosureGroups, IssueBaseProofRequest, IssueJsonLdBaseProofBatchRequest,
    IssueJsonLdBaseProofRequest, PreparedDisclosureRequest, PreparedDisclosureVerification,
    VerifyJsonLdDisclosureProofBatchRequest, HMAC_KEY_LEN,
};
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
struct BenchCase {
    name: &'static str,
    mandatory_pointers: Vec<String>,
    selective_pointers: Vec<String>,
    template: fn(usize) -> Value,
}

#[derive(Debug, Clone)]
struct BenchConfig {
    profile: BenchProfile,
    batch_size: usize,
    workers: usize,
    cases: Vec<String>,
    backend: BbsBackend,
    w3c_batch_api: bool,
    mixed_w3c_batch_api: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchProfile {
    Quick,
    Standard,
    Stress,
}

/// Benchmark-only mixed-workload dispatch control. The serial mode retains
/// the production BBS batch API inside each homogeneous shape bucket, but
/// invokes the buckets one after another. It is therefore a paired dispatcher
/// control, not a broad "all optimizations off" baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MixedBbsDispatchMode {
    Production,
    SerialShapeBuckets,
}

impl MixedBbsDispatchMode {
    fn label(self) -> &'static str {
        match self {
            Self::Production => "production-compatibility-aware",
            Self::SerialShapeBuckets => "serial-shape-buckets",
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchReport {
    tool: &'static str,
    profile: &'static str,
    batch_size: usize,
    workers: usize,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    call_path: &'static str,
    metrics: Vec<BenchMetric>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchBackendInfo {
    requested: &'static str,
    active: &'static str,
    hardware_acceleration: bool,
    fallback: bool,
    gpu_abi_version: u32,
    execution_backend: &'static str,
    kernel_ownership: &'static str,
    foundational_libraries: Vec<&'static str>,
    open_source_kernel_use: &'static str,
    detail: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchMetric {
    case: String,
    operation: &'static str,
    batch_size: usize,
    workers: usize,
    attempted: usize,
    succeeded: usize,
    failed: usize,
    total_ms: f64,
    ops_per_sec: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    avg_proof_value_chars: Option<f64>,
}

#[derive(Debug)]
struct ItemMeasurement<T> {
    elapsed_ms: f64,
    result: Result<T, String>,
}

#[derive(Debug)]
struct TimedBatch<T> {
    elapsed_ms: f64,
    items: Vec<ItemMeasurement<T>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsVsW3cReport {
    tool: &'static str,
    mode: &'static str,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    pure_bbs: PureBbsReport,
    w3c_e2e: BenchReport,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PureBbsReport {
    iterations: usize,
    batch_size: usize,
    call_path: &'static str,
    disclosed_ratio: f64,
    message_counts: Vec<usize>,
    metrics: Vec<PureBbsMetric>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PureBbsMetric {
    operation: &'static str,
    message_count: usize,
    disclosed_count: Option<usize>,
    iterations: usize,
    batch_size: usize,
    total_ms: f64,
    ops_per_sec: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    signature_bytes: Option<usize>,
    proof_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuSpeedupGateReport {
    tool: &'static str,
    mode: &'static str,
    target_reduction: f64,
    passed: bool,
    backend_info: BenchBackendInfo,
    cpu: PureBbsReport,
    gpu: PureBbsReport,
    comparisons: Vec<GpuSpeedupComparison>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuSpeedupComparison {
    operation: &'static str,
    message_count: usize,
    cpu_p50_ms: f64,
    gpu_p50_ms: f64,
    reduction: f64,
    target_reduction: f64,
    passed: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsSignProfileReport {
    tool: &'static str,
    mode: &'static str,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    iterations: usize,
    message_counts: Vec<usize>,
    metrics: Vec<BbsSignProfileMetric>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsSignProfileMetric {
    message_count: usize,
    iterations: usize,
    total_p50_ms: f64,
    total_p95_ms: f64,
    total_p99_ms: f64,
    message_scalar_p50_ms: f64,
    generator_p50_ms: f64,
    domain_p50_ms: f64,
    e_scalar_p50_ms: f64,
    msm_and_finalize_p50_ms: f64,
    message_scalar_share: f64,
    generator_share: f64,
    domain_share: f64,
    e_scalar_share: f64,
    msm_and_finalize_share: f64,
    signature_bytes: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsProofProfileReport {
    tool: &'static str,
    mode: &'static str,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    iterations: usize,
    batch_size: usize,
    mixed_shapes: bool,
    disclosed_ratio: f64,
    message_counts: Vec<usize>,
    metrics: Vec<BbsProofProfileMetric>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsProofProfileMetric {
    operation: &'static str,
    message_count: usize,
    disclosed_count: usize,
    iterations: usize,
    batch_size: usize,
    total_p50_ms: f64,
    total_p95_ms: f64,
    total_p99_ms: f64,
    message_scalar_p50_ms: f64,
    b_terms_p50_ms: Option<f64>,
    b_msm_p50_ms: Option<f64>,
    t2_prepare_p50_ms: Option<f64>,
    t2_msm_p50_ms: Option<f64>,
    msm_terms_p50_ms: Option<f64>,
    msm_p50_ms: Option<f64>,
    challenge_pairing_input_p50_ms: Option<f64>,
    pairing_p50_ms: Option<f64>,
    finalize_p50_ms: f64,
    proof_bytes: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PrepareBVector {
    tool: &'static str,
    mode: &'static str,
    ciphersuite: &'static str,
    message_count: usize,
    secret_key_hex: String,
    public_key_hex: String,
    header_hex: String,
    message_hex: Vec<String>,
    message_scalar_hex: Vec<String>,
    msm_point_hex: Vec<String>,
    msm_scalar_hex: Vec<String>,
    prepared_b_hex: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsThroughputStressReport {
    tool: &'static str,
    mode: &'static str,
    scheduling_profile: Option<&'static str>,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    duration_secs: f64,
    warmup_secs: f64,
    batch_size: usize,
    streams: usize,
    mixed_shapes: bool,
    unique_issuer_per_item: bool,
    shared_repeated_requests: bool,
    mixed_dispatch_mode: &'static str,
    serial_shape_buckets: Option<Vec<BbsSerialShapeBucket>>,
    verify_pre_b_aggregate_enabled: bool,
    verify_pre_b_aggregate_min_items: usize,
    state_cache_mode: &'static str,
    message_count: usize,
    message_counts: Vec<usize>,
    disclosed_count: usize,
    disclosed_counts: Vec<usize>,
    disclosed_ratio: f64,
    gpu_smi: Option<GpuSmiSummary>,
    efficiency: Vec<BbsThroughputEfficiencyMetric>,
    metrics: Vec<BbsThroughputStressMetric>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsSerialShapeBucket {
    message_count: usize,
    item_count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsCommonPointsWorkerSweepReport {
    tool: &'static str,
    mode: &'static str,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    repeat_count: usize,
    best_by_operation: Vec<BbsCommonPointsWorkerBestReport>,
    common_points_worker_variants: Vec<BbsCommonPointsWorkerVariantReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsCommonPointsWorkerVariantReport {
    label: String,
    common_points_workers: Option<usize>,
    operation: String,
    report: BbsThroughputStressReport,
    repeat_reports: Vec<BbsThroughputStressReport>,
    summary: BbsCommonPointsWorkerVariantSummary,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsCommonPointsWorkerVariantSummary {
    sample_count: usize,
    mean_items_per_sec: f64,
    best_items_per_sec: f64,
    worst_items_per_sec: f64,
    stddev_items_per_sec: f64,
    relative_range: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsCommonPointsWorkerBestReport {
    operation: String,
    best_label: String,
    best_common_points_workers: Option<usize>,
    best_mean_items_per_sec: f64,
    default_mean_items_per_sec: Option<f64>,
    speedup_vs_default: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsCommonPointsAbiSweepReport {
    tool: &'static str,
    mode: &'static str,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    repeat_count: usize,
    best_by_operation: Vec<BbsCommonPointsAbiBestReport>,
    common_points_abi_variants: Vec<BbsCommonPointsAbiVariantReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsCommonPointsAbiVariantReport {
    label: String,
    common_points_msm_abi: Option<bool>,
    operation: String,
    report: BbsThroughputStressReport,
    repeat_reports: Vec<BbsThroughputStressReport>,
    summary: BbsCommonPointsWorkerVariantSummary,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsCommonPointsAbiBestReport {
    operation: String,
    best_label: String,
    best_common_points_msm_abi: Option<bool>,
    best_mean_items_per_sec: f64,
    default_mean_items_per_sec: Option<f64>,
    speedup_vs_default: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuSmiSummary {
    sample_count: usize,
    errors: usize,
    avg_gpu_util_percent: f64,
    max_gpu_util_percent: f64,
    avg_power_watts: f64,
    max_power_watts: f64,
    avg_memory_used_mib: f64,
    max_memory_used_mib: f64,
    memory_total_mib: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct GpuSmiSample {
    gpu_util_percent: f64,
    power_watts: f64,
    memory_used_mib: f64,
    memory_total_mib: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsThroughputStressMetric {
    operation: &'static str,
    streams: usize,
    elapsed_ms: f64,
    batches: usize,
    items: usize,
    failed: usize,
    items_per_sec: f64,
    batches_per_sec: f64,
    batch_p50_ms: f64,
    batch_p95_ms: f64,
    batch_p99_ms: f64,
    batch_max_ms: f64,
    item_p50_ms: f64,
    item_p95_ms: f64,
    item_p99_ms: f64,
    signature_bytes: Option<usize>,
    proof_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsThroughputEfficiencyMetric {
    operation: &'static str,
    items_per_sec: f64,
    target_items_per_sec: Option<f64>,
    target_ratio: Option<f64>,
    required_speedup: Option<f64>,
    avg_gpu_util_percent: Option<f64>,
    max_gpu_util_percent: Option<f64>,
    items_per_sec_per_gpu_util_percent: Option<f64>,
    avg_power_watts: Option<f64>,
    items_per_sec_per_watt: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsG1MsmStressReport {
    tool: &'static str,
    mode: &'static str,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    duration_secs: f64,
    batch_size: usize,
    streams: usize,
    message_count: usize,
    term_count: usize,
    common_points: bool,
    metric: BbsThroughputStressMetric,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsProofMsmShapesReport {
    tool: &'static str,
    mode: &'static str,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    batch_size: usize,
    mixed_shapes: bool,
    message_counts: Vec<usize>,
    disclosed_ratio: f64,
    shapes: Vec<BbsMsmShapeMetric>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsProofMsmStressReport {
    tool: &'static str,
    mode: &'static str,
    backend: &'static str,
    backend_info: BenchBackendInfo,
    duration_secs: f64,
    target_proofs_per_sec: f64,
    batch_size: usize,
    streams: usize,
    mixed_shapes: bool,
    message_counts: Vec<usize>,
    disclosed_ratio: f64,
    gpu_smi: Option<GpuSmiSummary>,
    gpu_route_gate: Option<BbsGpuRouteGateReport>,
    fused_route_gate: Option<BbsFusedRouteGateReport>,
    throughput_gate: Option<BbsThroughputGateReport>,
    metrics: Vec<BbsProofMsmStressMetric>,
    stage_group_metrics: Vec<BbsProofMsmStageGroupMetric>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsProofMsmStressMetric {
    shape: BbsMsmShapeMetric,
    routes: BbsMsmRouteMetric,
    throughput: BbsThroughputStressMetric,
    fused_target: BbsFusedMsmTargetMetric,
    dispatch_pressure: BbsMsmDispatchPressureMetric,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsProofMsmStageGroupMetric {
    stage: &'static str,
    group_label: String,
    group_kind: &'static str,
    shape: BbsMsmShapeMetric,
    routes: BbsMsmRouteMetric,
    throughput: BbsThroughputStressMetric,
    dispatch_pressure: BbsMsmDispatchPressureMetric,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsFusedMsmTargetMetric {
    target_proofs_per_sec: f64,
    stage_msm_per_proof: f64,
    required_msm_per_sec: f64,
    current_msm_per_sec: f64,
    current_vs_required: f64,
    required_speedup: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsMsmDispatchPressureMetric {
    row_msm_per_proof: f64,
    row_msm_per_batch: usize,
    gpu_row_msm_per_batch: usize,
    fused_row_msm_per_batch: usize,
    gpu_route_ratio: f64,
    fused_route_ratio: f64,
    current_row_msm_per_sec: f64,
    required_row_msm_per_sec: f64,
    current_vs_required: f64,
    required_speedup: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsGpuRouteGateReport {
    min_gpu_route_ratio: f64,
    passed: bool,
    failing_stages: Vec<BbsGpuRouteGateFailure>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsGpuRouteGateFailure {
    scope: &'static str,
    stage: &'static str,
    group_label: Option<String>,
    gpu_route_ratio: f64,
    gpu_row_msm_per_batch: usize,
    row_msm_per_batch: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsFusedRouteGateReport {
    min_fused_route_ratio: f64,
    passed: bool,
    failing_stages: Vec<BbsFusedRouteGateFailure>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsFusedRouteGateFailure {
    scope: &'static str,
    stage: &'static str,
    group_label: Option<String>,
    fused_route_ratio: f64,
    fused_row_msm_per_batch: usize,
    row_msm_per_batch: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsThroughputGateReport {
    target_proofs_per_sec: f64,
    passed: bool,
    failing_stages: Vec<BbsThroughputGateFailure>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsThroughputGateFailure {
    scope: &'static str,
    stage: &'static str,
    group_label: Option<String>,
    current_row_msm_per_sec: f64,
    required_row_msm_per_sec: f64,
    current_vs_required: f64,
    required_speedup: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsMsmRouteMetric {
    unknown: usize,
    blst_item: usize,
    blst_common_points: usize,
    sppark_item: usize,
    sppark_common_points: usize,
    fused_item: usize,
    fused_common_points: usize,
    cubbs_large: usize,
    cubbs_generic: usize,
    total: usize,
}

impl From<BbsMsmRouteSummary> for BbsMsmRouteMetric {
    fn from(value: BbsMsmRouteSummary) -> Self {
        Self {
            unknown: value.unknown,
            blst_item: value.blst_item,
            blst_common_points: value.blst_common_points,
            sppark_item: value.sppark_item,
            sppark_common_points: value.sppark_common_points,
            fused_item: value.fused_item,
            fused_common_points: value.fused_common_points,
            cubbs_large: value.cubbs_large,
            cubbs_generic: value.cubbs_generic,
            total: value.total(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BbsMsmShapeMetric {
    label: &'static str,
    item_count: usize,
    min_term_count: usize,
    max_term_count: usize,
    common_point_groups: usize,
    largest_common_point_group: usize,
    single_variable_point_index: Option<usize>,
    term_count_groups: usize,
    split_candidate_groups: usize,
    largest_split_candidate_group: usize,
}

impl From<BbsMsmShapeSummary> for BbsMsmShapeMetric {
    fn from(value: BbsMsmShapeSummary) -> Self {
        Self {
            label: value.label,
            item_count: value.item_count,
            min_term_count: value.min_term_count,
            max_term_count: value.max_term_count,
            common_point_groups: value.common_point_groups,
            largest_common_point_group: value.largest_common_point_group,
            single_variable_point_index: value.single_variable_point_index,
            term_count_groups: value.term_count_groups,
            split_candidate_groups: value.split_candidate_groups,
            largest_split_candidate_group: value.largest_split_candidate_group,
        }
    }
}

fn summarize_terms_for_bench(label: &'static str, terms: &[BbsMsmTerms]) -> BbsMsmShapeSummary {
    let min_term_count = terms
        .iter()
        .map(|term| term.compressed_points.len())
        .min()
        .unwrap_or_default();
    let max_term_count = terms
        .iter()
        .map(|term| term.compressed_points.len())
        .max()
        .unwrap_or_default();
    let mut point_groups = BTreeMap::<Arc<Vec<Vec<u8>>>, usize>::new();
    let mut term_count_groups = BTreeMap::<usize, Vec<&BbsMsmTerms>>::new();
    for term in terms {
        *point_groups
            .entry(term.compressed_points.clone())
            .or_default() += 1;
        term_count_groups
            .entry(term.compressed_points.len())
            .or_default()
            .push(term);
    }
    let largest_common_point_group = point_groups.values().copied().max().unwrap_or_default();
    let single_variable_point_index = single_variable_point_index_for_bench(terms);
    let split_candidate_groups = term_count_groups
        .values()
        .filter(|group| {
            group.len() >= 2
                && single_variable_point_index_for_bench(
                    &group.iter().copied().cloned().collect::<Vec<_>>(),
                )
                .is_some()
        })
        .count();
    let largest_split_candidate_group = term_count_groups
        .values()
        .filter(|group| {
            group.len() >= 2
                && single_variable_point_index_for_bench(
                    &group.iter().copied().cloned().collect::<Vec<_>>(),
                )
                .is_some()
        })
        .map(Vec::len)
        .max()
        .unwrap_or_default();

    BbsMsmShapeSummary {
        label,
        item_count: terms.len(),
        min_term_count,
        max_term_count,
        common_point_groups: point_groups.len(),
        largest_common_point_group,
        single_variable_point_index,
        term_count_groups: term_count_groups.len(),
        split_candidate_groups,
        largest_split_candidate_group,
    }
}

fn single_variable_point_index_for_bench(terms: &[BbsMsmTerms]) -> Option<usize> {
    let first = terms.first()?;
    let term_count = first.compressed_points.len();
    if terms.len() < 2 || term_count == 0 {
        return None;
    }
    if terms
        .iter()
        .any(|term| term.compressed_points.len() != term_count)
    {
        return None;
    }
    let variable_indexes = (0..term_count)
        .filter(|&point_index| {
            let base = &first.compressed_points[point_index];
            terms
                .iter()
                .skip(1)
                .any(|term| term.compressed_points[point_index] != *base)
        })
        .collect::<Vec<_>>();
    match variable_indexes.as_slice() {
        [index] => Some(*index),
        _ => None,
    }
}

fn main() {
    async_std::task::block_on(async {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        if args.iter().any(|arg| arg == "--help" || arg == "-h") {
            print_help();
        } else if args.iter().any(|arg| arg == "--backend-info-only") {
            let backend = parse_backend_arg(&args);
            println!(
                "{}",
                serde_json::to_string_pretty(&backend_info(backend)).unwrap()
            );
        } else if args.iter().any(|arg| arg == "--prepare-b-vector") {
            let report = run_prepare_b_vector(&args);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args.iter().any(|arg| arg == "--bbs-sign-profile") {
            let report = run_bbs_sign_profile(&args);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args.iter().any(|arg| arg == "--bbs-proof-profile") {
            let report = run_bbs_proof_profile(&args);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args
            .iter()
            .any(|arg| arg == "--bbs-common-points-worker-sweep")
        {
            let report = run_bbs_common_points_worker_sweep(&args);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args
            .iter()
            .any(|arg| arg == "--bbs-common-points-abi-sweep")
        {
            let report = run_bbs_common_points_abi_sweep(&args);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args.iter().any(|arg| arg == "--bbs-throughput-stress") {
            let report = run_bbs_throughput_stress(&args);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args.iter().any(|arg| arg == "--bbs-g1-msm-stress") {
            let report = run_bbs_g1_msm_stress(&args);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args.iter().any(|arg| arg == "--bbs-proof-msm-shapes") {
            let report = run_bbs_proof_msm_shapes(&args);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args.iter().any(|arg| arg == "--bbs-gpu-msm-acceptance") {
            let acceptance_args = bbs_gpu_msm_acceptance_args(&args);
            let report = run_bbs_proof_msm_stress(&acceptance_args);
            print_bbs_proof_msm_stress_report(report);
        } else if args.iter().any(|arg| arg == "--bbs-proof-msm-stress") {
            let report = run_bbs_proof_msm_stress(&args);
            print_bbs_proof_msm_stress_report(report);
        } else if args.iter().any(|arg| arg == "--profile-program") {
            let report = run_program_profile(&args).await;
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else if args.iter().any(|arg| arg == "--gpu-sign-speedup-gate") {
            let report = run_gpu_sign_speedup_gate(&args);
            let passed = report.passed;
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            if !passed {
                std::process::exit(1);
            }
        } else if args.iter().any(|arg| arg == "--gpu-speedup-gate") {
            let report = run_gpu_speedup_gate(&args);
            let passed = report.passed;
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            if !passed {
                std::process::exit(1);
            }
        } else if args.iter().any(|arg| arg == "--bbs-vs-w3c") {
            let report = run_bbs_vs_w3c(&args).await;
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else {
            let config = BenchConfig::from_args();
            let report = run_bench(config).await;
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
    });
}

fn print_help() {
    println!(
        r#"cubbs-bench

目标 / Goal
  展示两个速度指标，不混入内部 profile 子阶段。
  Show two speed layers only, without internal profile phases.

核心模式 / Main mode
  --bbs-vs-w3c
      同一份 JSON 报告里输出 pureBbs 和 w3cE2e。
      Emit pureBbs and w3cE2e in one JSON report.

诊断模式 / Diagnostic modes
  --bbs-sign-profile
      分阶段输出底层 BBS 签名耗时。
      Break down low-level BBS signature latency by phase.

  --bbs-proof-profile
      分阶段输出底层 BBS proof-gen/proof-verify batch 耗时。
      Break down low-level BBS proof-gen/proof-verify batch latency by phase.

  --bbs-throughput-stress
      指定 batch 和持续时间，持续压测底层 BBS primitive 吞吐。
      Run a timed low-level BBS primitive throughput stress test.

  --bbs-common-points-worker-sweep
      扫描 common-points BLST worker 配置，比较 proof-gen/proof-verify 端到端吞吐。
      Sweep common-points BLST worker settings and compare proof-gen/proof-verify throughput.

  --bbs-common-points-abi-sweep
      扫描 common-points packed ABI 开关，比较不同 operation/batch/VC shape 是否适合启用。
      Sweep the common-points packed ABI switch for operation/batch/VC-shape scheduling.

  --bbs-g1-msm-stress
      只压测底层 G1 MSM batch，隔离 common-points MSM 的原始吞吐。
      Stress only low-level G1 MSM batches to isolate common-points MSM throughput.

  --bbs-proof-msm-shapes
      输出 proof-gen/proof-verify 真实 MSM terms 形态，用于设计 fused kernel。
      Print real proof-gen/proof-verify MSM term shapes for fused-kernel planning.

  --bbs-proof-msm-stress
      分阶段压测 proof-gen/proof-verify 真实 MSM terms 的原始吞吐。
      Stress real proof-gen/proof-verify MSM term batches by stage.

  --bbs-gpu-msm-acceptance
      运行 5060Ti fused MSM 验收门槛：mixed VC、stage groups、fused route、row-MSM target。
      Run the 5060Ti fused-MSM acceptance gate with mixed VC, stage groups, fused route, and row-MSM target checks.

  --gpu-throughput-max
      启用高吞吐 GPU 预设，强制打开常见加速开关并放松并行阈值。
      Enable a throughput-focused GPU preset that enables common acceleration knobs and lowers parallelism thresholds.

速度层 / Speed layers
  pureBbs
      底层 BBS primitive 速度，直接调用 cubbs-bbs。
      Measures low-level BBS primitive calls through cubbs-bbs.

      Operations:
        sign                    BBS 签名 / BBS signature generation
        verify-signature        BBS 签名验证 / BBS signature verification
        proof-gen               BBS 选择性披露证明生成 / BBS selective disclosure proof generation
        proof-verify            BBS 选择性披露证明验证 / BBS selective disclosure proof verification
        *-batch-api             同上，但通过 batch API 均摊每个 item 的时延 / same operations amortized per item through batch API

  w3cE2e
      W3C/Data Integrity 全流程速度，包含 JSON-LD canonicalization、proof configuration、
      N-Quads 分组、proofValue 编解码和底层 BBS 调用。
      Measures the full W3C/Data Integrity path, including JSON-LD canonicalization,
      proof configuration, N-Quads grouping, proofValue encoding/decoding, and BBS calls.

      Operations:
        issue                   签发 base proof / issue a base proof
        derive                  生成 VP/derived proof / derive a presentation proof
        verify-derived          验证 VP/derived proof / verify a derived proof
        *-batch-api             同上，但通过 W3C batch API 均摊每个 item 的时延 / same operations amortized per item through W3C batch API

推荐命令 / Recommended commands
  CPU:
    cargo run --release -p cubbs-bench -- --bbs-vs-w3c \
      --pure-bbs-batch-api --w3c-batch-api \
      --bbs-backend cpu --iterations 8 --messages 16,64,128 \
      --batch 8 --workers 1 --profile quick

  GPU requested:
    CUBBS_BBS_GPU_LIBRARY=$PWD/target/native/cuda/libcubbs_bbs_cuda.so \
    cargo run --release -p cubbs-bench -- --bbs-vs-w3c \
      --pure-bbs-batch-api --w3c-batch-api \
      --bbs-backend gpu --iterations 8 --messages 16,64,128 \
      --batch 8 --workers 1 --profile quick

  GPU MSM acceptance:
    CUBBS_BBS_GPU_LIBRARY=$PWD/target/native/cuda/libcubbs_bbs_cuda.so \
    target/release/cubbs-bench --bbs-gpu-msm-acceptance

常用参数 / Common options
  --bbs-backend cpu|gpu          选择 BBS 后端 / select BBS backend
  --messages 16,64,128          pureBbs 消息数量 / pureBbs message counts
  --disclosed-ratio 0.5         pureBbs 披露比例 / pureBbs disclosed message ratio
  --batch 8                     W3C batch size, also default pureBbs batch size
  --bbs-batch 32                单独设置 pureBbs batch size / override pureBbs batch size
  --iterations 64               pureBbs 重复轮数 / pureBbs repeated rounds
  --workers 1                   W3C single-item worker count / W3C single-item workers
  --profile quick|standard|stress
                                W3C case profile / W3C case profile
  --cases tiny-flat,array-heavy  只跑指定 W3C case / run selected W3C cases only
  --pure-bbs-batch-api          pureBbs 使用 batch API / use pureBbs batch API
  --w3c-batch-api               W3C 使用 batch API / use W3C batch API
  --mixed-w3c-batch-api         W3C batch 内混合不同 VC/VP case / mix VC/VP cases inside one W3C batch
  --operation proof-gen         stress 操作: sign|verify|proof-gen|proof-verify|all
                                stress operation: sign|verify|proof-gen|proof-verify|all
  --duration-secs 10            stress 每个操作持续秒数 / stress duration per operation
  --streams 4                   stress 并发提交 batch 的 stream 数 / concurrent batch submit streams
  --gpu-smi-sample              stress 时采样 nvidia-smi，输出 GPU util/power/memory
                                sample nvidia-smi during stress and report GPU util/power/memory
  --gpu-smi-interval-ms 500     nvidia-smi 采样间隔 / nvidia-smi sample interval
  --target-items-per-sec 20000  stress 输出 efficiency.targetRatio/requiredSpeedup 的目标产出
                                target throughput used for efficiency.targetRatio/requiredSpeedup
  --mixed-bbs-shapes            stress batch 内轮转 --messages 的多个长度，模拟不同 VC 形状
                                rotate multiple --messages lengths inside one stress batch to model mixed VC shapes
  --mixed-bbs-dispatch production|serial-shape-buckets
                                mixed-shape dispatch control: production scheduler or exact-shape buckets executed serially
  --large-bbs-shapes            使用 256,512,1024,2048,4096 messages 的 large-shape 兼容性预设
                                use the 256,512,1024,2048,4096-message large-shape compatibility preset
  --ultra-large-bbs-shapes      使用 4097,8192,16384,65536 messages 的 cuBBS-Cred generic route 4 兼容性预设
                                use the 4097,8192,16384,65536-message cuBBS-Cred generic route 4 compatibility preset
  --unique-issuer-per-item      stress batch 内每条 VC/VP 使用独立 issuer key，模拟托管方全异构输入
                                use a distinct issuer key for every VC/VP in the stress batch
  --shared-repeated            stress 直接复用一个结构化请求，匹配 HTTP prepared-cache 的 shared batch 路径
                                reuse one structured request through the HTTP prepared-cache shared batch path
  --no-bbs-caches              关闭跨调用 BBS 状态缓存，测量不依赖重复输入的 kernel/管线吞吐
                                disable cross-call BBS state caches for cache-independent kernel/pipeline throughput
  --common-points-workers default,6,10
                                --bbs-common-points-worker-sweep 的 worker 候选
                                worker candidates for --bbs-common-points-worker-sweep
  --common-points-abi default,packed
                                --bbs-common-points-abi-sweep 的 ABI 候选
                                ABI candidates for --bbs-common-points-abi-sweep
  --sweep-repeats 3             sweep 中每个 worker/operation 重复次数，用 mean 选 best
                                repeat count per worker/operation; best is selected by mean
  --target-proofs-per-sec 20000 --bbs-proof-msm-stress 的 fused MSM 目标 proof 吞吐
                                target proof throughput for fused-MSM planning in --bbs-proof-msm-stress
  --msm-stage-groups            逐组压测 proof MSM stage 内的 common/split group
                                stress common/split groups inside each proof MSM stage
  --require-gpu-route-ratio 0.9 要求每个 MSM stage 至少该比例 row-MSM 走 GPU 路由
                                require each MSM stage to route at least this row-MSM ratio through GPU
                                with --msm-stage-groups, group metrics are gated too
  --require-fused-route-ratio 0.9
                                要求每个 MSM stage 至少该比例 row-MSM 走 cuBBS-Cred route 1/2/3/4
                                require each MSM stage to route at least this row-MSM ratio through cuBBS-Cred routes 1/2/3/4
  --require-dispatch-target     要求 row-MSM/s 达到 --target-proofs-per-sec 推导的目标
                                require row-MSM/s to satisfy the --target-proofs-per-sec target
  --gpu-throughput-max         在 throughput/g1/proof-msm 压测中启用高吞吐 GPU 预设 / enable high-throughput GPU preset for throughput, g1-msm and proof-msm stress
  --gpu-throughput-auto        按四操作自动选择饱和 batch/streams，并隐含 --gpu-throughput-max
                                choose operation-aware saturation batch/streams; implies --gpu-throughput-max

W3C cases / W3C 用例
  tiny-flat
      小型扁平 VC，少量 claim，基础路径。
      Small flat VC with a few claims, baseline path.

  nested-profile
      嵌套 profile 字段，覆盖嵌套 JSON-LD 和指针选择。
      Nested profile fields, covering nested JSON-LD and pointer selection.

  array-heavy
      数组字段较重，覆盖数组 canonicalization 和披露。
      Array-heavy credential, covering array canonicalization and disclosure.

  wide-claims-100
      100 个横向 claim，覆盖多消息 BBS 和较大的 derived proof。
      100 wide claims, covering larger BBS message vectors and derived proofs.

  large-array-100
      100 条数组记录，仅 standard/stress profile。
      100 array records, standard/stress profiles only.

  multi-subject
      多 credentialSubject，仅 standard/stress profile。
      Multiple credential subjects, standard/stress profiles only.

读数方式 / How to read
  pureBbs 表示底层 BBS 速度。
  pureBbs is the bottom-layer BBS speed.

  w3cE2e 表示 cuBBS-Cred W3C 全流程速度。
  w3cE2e is the full cuBBS-Cred W3C path speed.

  不建议把 w3cE2e - pureBbs 当成精确封装开销，因为 W3C case 的 canonical statement
  数量、JSON-LD 结构、proofValue 大小和 BBS message count 并不一定一一对应。
  Do not treat w3cE2e - pureBbs as exact wrapper overhead, because W3C cases
  differ in canonical statement count, JSON-LD shape, proofValue size, and BBS
  message count.
"#
    );
}

fn run_prepare_b_vector(args: &[String]) -> PrepareBVector {
    let message_count = arg_value(args, "--messages")
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let engine = BbsEngine::new(BbsBackend::ZkryptiumCpu);
    let mut rng = StdRng::seed_from_u64(0x0bb5_2023_0000_0001);
    let keys = engine.generate_keypair(&mut rng).expect("key generation");
    let header = deterministic_header(message_count);
    let messages = deterministic_bbs_messages(message_count);
    let req = SignRequest {
        secret_key: keys.secret_key.clone(),
        public_key: keys.public_key.clone(),
        header: header.clone(),
        messages: messages.clone(),
    };
    let message_scalars = engine
        .messages_to_scalar_bytes(&messages)
        .expect("message scalars");
    let prepared_b = engine.prepare_b_for_sign(&req).expect("prepare b");
    let msm_terms = engine
        .prepare_b_msm_terms_for_sign(&req)
        .expect("prepare b msm terms");

    PrepareBVector {
        tool: "cubbs-bench",
        mode: "prepare-b-vector",
        ciphersuite: cubbs_bbs::CIPHERSUITE_ID,
        message_count,
        secret_key_hex: hex_string(&keys.secret_key),
        public_key_hex: hex_string(&keys.public_key),
        header_hex: hex_string(&header),
        message_hex: messages.iter().map(|message| hex_string(message)).collect(),
        message_scalar_hex: message_scalars
            .iter()
            .map(|scalar| hex_string(scalar))
            .collect(),
        msm_point_hex: msm_terms
            .compressed_points
            .iter()
            .map(|point| hex_string(point))
            .collect(),
        msm_scalar_hex: msm_terms
            .scalar_be
            .iter()
            .map(|scalar| hex_string(scalar))
            .collect(),
        prepared_b_hex: hex_string(&prepared_b.compressed_b),
    }
}

impl BenchConfig {
    fn from_args() -> Self {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let profile = arg_value(&args, "--profile")
            .as_deref()
            .map(BenchProfile::parse)
            .unwrap_or_else(|| {
                if args.iter().any(|arg| arg == "--full") {
                    BenchProfile::Standard
                } else {
                    BenchProfile::Quick
                }
            });
        let batch_size = arg_value(&args, "--batch")
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| profile.default_batch_size());
        let workers = arg_value(&args, "--workers")
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| profile.default_workers());
        let cases = arg_value(&args, "--cases")
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let backend = parse_backend_arg(&args);
        let w3c_batch_api = args.iter().any(|arg| arg == "--w3c-batch-api");
        let mixed_w3c_batch_api = args.iter().any(|arg| arg == "--mixed-w3c-batch-api");
        Self {
            profile,
            batch_size,
            workers: workers.max(1),
            cases,
            backend,
            w3c_batch_api,
            mixed_w3c_batch_api,
        }
    }
}

impl BenchProfile {
    fn parse(value: &str) -> Self {
        match value {
            "quick" => Self::Quick,
            "standard" | "full" => Self::Standard,
            "stress" => Self::Stress,
            _ => panic!("unknown profile: {value}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Standard => "standard",
            Self::Stress => "stress",
        }
    }

    fn default_batch_size(self) -> usize {
        match self {
            Self::Quick => 8,
            Self::Standard => 64,
            Self::Stress => 256,
        }
    }

    fn default_workers(self) -> usize {
        match self {
            Self::Quick => 1,
            Self::Standard => 4,
            Self::Stress => 8,
        }
    }
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| (window[0] == name).then(|| window[1].clone()))
}

fn parse_backend_arg(args: &[String]) -> BbsBackend {
    let backend = arg_value(args, "--bbs-backend")
        .as_deref()
        .map(BbsBackend::parse)
        .transpose()
        .expect("valid --bbs-backend")
        .unwrap_or(BbsBackend::ZkryptiumCpu);
    if args.iter().any(|arg| arg == "--require-gpu") {
        if let Err(error) = backend.ensure_hardware_available() {
            eprintln!("hardware GPU BBS backend required: {error}");
            std::process::exit(2);
        }
    }
    backend
}

#[derive(Debug)]
struct EnvVarSnapshot {
    original: Vec<(String, Option<OsString>)>,
    applied: bool,
}

impl EnvVarSnapshot {
    fn new() -> Self {
        Self {
            original: Vec::new(),
            applied: false,
        }
    }

    fn set(&mut self, key: &str, value: &str) {
        if !self.applied {
            self.original.push((key.to_owned(), std::env::var_os(key)));
        }
        std::env::set_var(key, value);
    }

    fn finalize(mut self) -> Self {
        self.applied = true;
        self
    }
}

impl Drop for EnvVarSnapshot {
    fn drop(&mut self) {
        if !self.applied {
            return;
        }
        for (key, original) in self.original.drain(..).rev() {
            match original {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

fn maybe_gpu_throughput_max_preset(args: &[String]) -> Option<EnvVarSnapshot> {
    if !args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--gpu-throughput-max" | "--gpu-throughput-auto"
        )
    }) {
        return None;
    }

    let mut env = EnvVarSnapshot::new();
    // Route pressure tuning: reduce GPU scheduling barriers so mixed VC shapes and
    // variable-point batches can be partitioned and sent to GPU earlier.
    env.set("CUBBS_BBS_GPU_SPLIT_VARIABLE_PARALLEL_MIN_BATCH", "1");
    env.set("CUBBS_BBS_GPU_POINT_GROUP_PARALLEL_MIN_GROUPS", "1");
    env.set("CUBBS_BBS_GPU_MIXED_SHAPE_PARALLEL_MIN_GROUPS", "1");
    env.set("CUBBS_BBS_PROOF_MIXED_SHAPE_PARALLEL_STREAMS", "16");
    env.set("CUBBS_BBS_PROOF_VERIFY_PAIR_MSM_PARALLEL_MIN_BATCH", "1");
    env.set("CUBBS_BBS_PROOF_VERIFY_PARALLEL_PAIRING_MIN_BATCH", "1");
    env.set("CUBBS_BBS_SIGN_FINALIZE_PARALLEL_MIN_BATCH", "1");
    env.set("CUBBS_BBS_PROOF_GEN_FINALIZE_PARALLEL_MIN_BATCH", "1");
    env.set("CUBBS_BBS_PROOF_GPU_PREPARE_PARALLEL_MIN_BATCH", "1");

    // Force high-throughput paths for BBS proof pipeline.
    // The fused multi-output common-points MSM kernel
    // (CUBBS_BBS_GPU_FUSED_COMMON_POINTS_MSM) is the production common-points
    // path: it runs the window8 Pippenger kernel at high occupancy on a
    // per-point-set CUDA stream and compresses outputs with a parallel blst
    // pool. This supersedes the per-output SPPARK invoke loop, so the preset
    // no longer forces SPPARK routing.
    //
    // Flag selection was tuned by bisecting each knob on the RTX 5060 Ti:
    //  - GPU_SIGN_FINALIZE helps sign (+14%); kept.
    //  - GPU_PRECOMPUTE_E regressed sign 6x (extra hash kernel launch on the
    //    sign hot path); removed.
    //  - GPU_CHALLENGE_SCALAR regressed proof-verify 20%; removed.
    //  - PROOF_GEN_OVERLAP_DOMAIN regressed proof-gen both when it overlapped
    //    only domain precompute AND when extended to overlap random-scalar
    //    pre-generation (the real cost is the 6 B-dependent G1 muls in
    //    t2Prepare, which cannot overlap with B-MSM); removed.
    //  - GPU_PROOF_VERIFY_SPLIT_MSM helps proof-verify (+5%); kept.
    env.set("CUBBS_BBS_GPU_FUSED_COMMON_POINTS_MSM", "1");
    env.set("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI", "1");
    env.set("CUBBS_BBS_GPU_NATIVE_MIXED_MSM_GROUPING", "1");
    env.set("CUBBS_BBS_GPU_NATIVE_G1_ADD", "1");
    env.set("CUBBS_BBS_GPU_SIGN_FINALIZE", "1");
    env.set("CUBBS_BBS_GPU_PROOF_VERIFY_SPLIT_MSM", "1");

    // Every GPU-selected MSM shape stays on an cuBBS-Cred kernel. Specialized
    // routes handle common BBS shapes and route 4 is the bounded universal
    // common/variable fallback.
    env.set("CUBBS_BBS_FUSED_COMPRESS_WORKERS", "8");
    env.set("CUBBS_BBS_FUSED_COMPRESS_PARALLEL_MIN_ITEMS", "8");
    // Keep a bounded total host-feed budget. Low-stream verify runs need more
    // inner decode parallelism, while mixed high-stream runs would otherwise
    // multiply the host workers far beyond the available cores.
    let operation = arg_value(args, "--operation").unwrap_or_default();
    let (batch, streams) = bbs_throughput_batch_and_streams(args);
    let unique_issuer_per_item = args.iter().any(|arg| arg == "--unique-issuer-per-item");
    let decode_workers = match operation.as_str() {
        "verify" | "verify-signature" if streams <= 4 => Some(16),
        "sign" if batch <= 512 => Some(2),
        "sign" => Some(5),
        // `all` contains four different host-feed patterns. Keep each
        // backend's adaptive default instead of forcing one worker count onto
        // every operation in the coverage matrix.
        "all" => None,
        _ => Some(4),
    };
    if let Some(decode_workers) = decode_workers {
        env.set("CUBBS_BBS_BLST_MSM_WORKERS", &decode_workers.to_string());
    }
    if matches!(operation.as_str(), "sign" | "all") {
        let e_workers = if streams >= 4 {
            1
        } else if streams >= 2 {
            2
        } else {
            4
        };
        env.set("CUBBS_BBS_SIGN_E_WORKERS", &e_workers.to_string());
    }
    if matches!(operation.as_str(), "sign" | "all")
        && (batch <= 128 || (unique_issuer_per_item && batch <= 512))
    {
        env.set("CUBBS_BBS_SIGN_SHAPE_BATCHING", "0");
    }
    // Throughput batches create many short-lived host decode/compress jobs,
    // especially after mixed-shape and independent-issuer bucketing. Reuse a
    // bounded process-wide pool once queueing is amortized. Proof verification
    // alone already has a balanced host/device pipeline, so keep its direct
    // worker path; `all` benefits overall across the other three operations.
    if operation != "proof-verify" && batch >= 128 {
        env.set("CUBBS_BBS_HOST_POOL", "1");
    }
    if matches!(operation.as_str(), "proof-verify" | "all") {
        env.set(
            "CUBBS_BBS_HASH_NONBLOCKING",
            if streams <= 16 { "1" } else { "0" },
        );
    }
    Some(env.finalize())
}

fn bbs_throughput_auto_config(args: &[String]) -> Option<(usize, usize)> {
    if !args.iter().any(|arg| arg == "--gpu-throughput-auto") {
        return None;
    }
    let operation = arg_value(args, "--operation").unwrap_or_else(|| "proof-gen".to_owned());
    Some(match operation.as_str() {
        "sign" => (16_384, 4),
        "verify" | "verify-signature" => (8_192, 20),
        "proof-gen" => (4_096, 4),
        "proof-verify" => (2_048, 10),
        "all" => panic!(
            "--gpu-throughput-auto requires one operation per process; run sign, \
             verify-signature, proof-gen, and proof-verify separately"
        ),
        other => panic!("unsupported --operation for --gpu-throughput-auto: {other}"),
    })
}

fn bbs_throughput_batch_and_streams(args: &[String]) -> (usize, usize) {
    let auto = bbs_throughput_auto_config(args);
    let batch = arg_value(args, "--bbs-batch")
        .or_else(|| arg_value(args, "--batch"))
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| auto.map(|config| config.0))
        .unwrap_or(512);
    let streams = arg_value(args, "--stress-streams")
        .or_else(|| arg_value(args, "--streams"))
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| auto.map(|config| config.1))
        .unwrap_or(1)
        .max(1);
    (batch, streams)
}

fn backend_label(backend: BbsBackend) -> &'static str {
    match backend {
        BbsBackend::ZkryptiumCpu => "cuBBS-Cred protocol / BLS12-381 SHA-256 / CPU",
        BbsBackend::GpuRequestedCpuFallback => {
            "cuBBS-Cred protocol / BLS12-381 SHA-256 / GPU requested"
        }
    }
}

fn backend_info(backend: BbsBackend) -> BenchBackendInfo {
    let info = BbsEngine::new(backend).backend_info();
    let (execution_backend, kernel_ownership, foundational_libraries, open_source_kernel_use) =
        if info.hardware_acceleration {
            (
                "hybrid: cuBBS-Cred CUDA + SPPARK primitives + BLST",
                "BBS-specialized CUDA kernels and routing are implemented in cuBBS-Cred",
                vec![
                    "SPPARK BLS12-381 field/curve primitives",
                    "BLST point encoding and pairing final checks",
                    "cuBBS-Cred modified ZKryptium-derived BBS protocol layer",
                ],
                "the normal production router has no BLST/SPPARK MSM call edge and uses cuBBS-Cred routes 1/2/3/4; optional SPPARK routes 7/8 are linked only in an explicit reference benchmark build",
            )
        } else if info.fallback {
            (
                "CPU fallback: versioned ZKryptium reference + bls12_381_plus",
                "no cuBBS-Cred CUDA kernel is active",
                vec![
                    "registry ZKryptium reference (0.6.1 by default)",
                    "bls12_381_plus BLS12-381 primitives",
                ],
                "all cryptographic execution uses the selected upstream CPU fallback",
            )
        } else if info.active == "zkryptium-cpu" {
            (
                "CPU: cuBBS-Cred protocol engine + bls12_381_plus",
                "no cuBBS-Cred CUDA kernel is active",
                vec![
                    "cuBBS-Cred modified ZKryptium-derived BBS protocol layer",
                    "bls12_381_plus BLS12-381 primitives",
                ],
                "all cryptographic execution uses the local CPU protocol engine",
            )
        } else {
            (
                "GPU ABI test/incomplete backend",
                "no hardware kernel ownership claim",
                Vec::new(),
                "not a GPU performance backend",
            )
        };
    BenchBackendInfo {
        requested: info.requested,
        active: info.active,
        hardware_acceleration: info.hardware_acceleration,
        fallback: info.fallback,
        gpu_abi_version: info.gpu_abi_version,
        execution_backend,
        kernel_ownership,
        foundational_libraries,
        open_source_kernel_use,
        detail: info.detail,
    }
}

async fn run_bench(config: BenchConfig) -> BenchReport {
    if config.mixed_w3c_batch_api {
        return run_bench_w3c_mixed_batch_api(config).await;
    }
    if config.w3c_batch_api {
        return run_bench_w3c_batch_api(config).await;
    }

    std::env::set_var("CUBBS_BBS_BACKEND", config.backend.name());
    let all_cases = bench_cases(config.profile);
    let cases = if config.cases.is_empty() {
        all_cases
    } else {
        all_cases
            .into_iter()
            .filter(|case| config.cases.iter().any(|name| name == case.name))
            .collect()
    };
    let engine = BbsEngine::new(config.backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let verification_method =
        verification_method_from_public_key(&keys.public_key).expect("verification method");
    let proof_options = default_bbs_2023_proof_options(verification_method, "assertionMethod");
    let mut metrics = Vec::new();

    for case in cases {
        let issue_items = (0..config.batch_size)
            .map(|i| IssueJsonLdBaseProofRequest {
                document: (case.template)(i),
                proof_options: proof_options.clone(),
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                mandatory_pointers: case.mandatory_pointers.clone(),
                hmac_key: deterministic_hmac_key(i),
            })
            .collect::<Vec<_>>();
        let issued = measure_items(issue_items, config.workers, |item| async move {
            issue_jsonld_base_proof(item)
                .await
                .map_err(|error| error.to_string())
        });
        let signed = collect_success_values(&issued);
        metrics.push(metric_from_value_batch(
            case.name,
            "issue",
            config.batch_size,
            config.workers,
            &issued,
        ));

        let derive_items = signed
            .into_iter()
            .enumerate()
            .map(|(i, signed_document)| DeriveJsonLdDisclosureProofRequest {
                signed_document,
                selective_pointers: case.selective_pointers.clone(),
                presentation_header: format!("cubbs-bench-{}-{i}", case.name).into_bytes(),
            })
            .collect::<Vec<_>>();
        let derived = measure_items(derive_items, config.workers, |item| async move {
            derive_jsonld_disclosure_proof(item)
                .await
                .map_err(|error| error.to_string())
        });
        let revealed = collect_success_values(&derived);
        metrics.push(metric_from_value_batch(
            case.name,
            "derive",
            config.batch_size,
            config.workers,
            &derived,
        ));

        let verify_items = revealed
            .into_iter()
            .map(|document| (document, None))
            .collect::<Vec<_>>();
        let verified = measure_items(
            verify_items,
            config.workers,
            |(document, expected)| async move {
                verify_jsonld_disclosure_proof(&document, expected)
                    .await
                    .map_err(|error| error.to_string())
            },
        );
        metrics.push(metric_from_bool_batch(
            case.name,
            "verify-derived",
            config.batch_size,
            config.workers,
            &verified,
        ));
    }

    BenchReport {
        tool: "cubbs-bench",
        profile: config.profile.name(),
        batch_size: config.batch_size,
        workers: config.workers,
        backend: backend_label(config.backend),
        backend_info: backend_info(config.backend),
        call_path: "single-item-functions",
        metrics,
    }
}

async fn run_bench_w3c_batch_api(config: BenchConfig) -> BenchReport {
    std::env::set_var("CUBBS_BBS_BACKEND", config.backend.name());
    let all_cases = bench_cases(config.profile);
    let cases = if config.cases.is_empty() {
        all_cases
    } else {
        all_cases
            .into_iter()
            .filter(|case| config.cases.iter().any(|name| name == case.name))
            .collect()
    };
    let engine = BbsEngine::new(config.backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let verification_method =
        verification_method_from_public_key(&keys.public_key).expect("verification method");
    let proof_options = default_bbs_2023_proof_options(verification_method, "assertionMethod");
    let mut metrics = Vec::new();

    for case in cases {
        let issue_items = (0..config.batch_size)
            .map(|i| IssueJsonLdBaseProofRequest {
                document: (case.template)(i),
                proof_options: proof_options.clone(),
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                mandatory_pointers: case.mandatory_pointers.clone(),
                hmac_key: deterministic_hmac_key(i),
            })
            .collect::<Vec<_>>();
        let issued = measure_batch_api(|| async {
            issue_jsonld_base_proof_batch(IssueJsonLdBaseProofBatchRequest {
                items: issue_items,
                fail_fast: false,
            })
            .await
        })
        .await;
        let signed = collect_success_values_from_api_batch(&issued);
        metrics.push(metric_from_api_value_batch(
            case.name,
            "issue-batch-api",
            config.batch_size,
            &issued,
        ));

        let derive_items = signed
            .into_iter()
            .enumerate()
            .map(|(i, signed_document)| DeriveJsonLdDisclosureProofRequest {
                signed_document,
                selective_pointers: case.selective_pointers.clone(),
                presentation_header: format!("cubbs-bench-{}-{i}", case.name).into_bytes(),
            })
            .collect::<Vec<_>>();
        let derived = measure_batch_api(|| async {
            derive_jsonld_disclosure_proof_batch(DeriveJsonLdDisclosureProofBatchRequest {
                items: derive_items,
                fail_fast: false,
            })
            .await
        })
        .await;
        let revealed = collect_success_values_from_api_batch(&derived);
        metrics.push(metric_from_api_value_batch(
            case.name,
            "derive-batch-api",
            config.batch_size,
            &derived,
        ));

        let verify_items = revealed
            .into_iter()
            .map(|document| (document, None))
            .collect::<Vec<_>>();
        let verified = measure_batch_api(|| async {
            verify_jsonld_disclosure_proof_batch(VerifyJsonLdDisclosureProofBatchRequest {
                items: verify_items,
                fail_fast: false,
            })
            .await
        })
        .await;
        metrics.push(metric_from_api_bool_batch(
            case.name,
            "verify-derived-batch-api",
            config.batch_size,
            &verified,
        ));
    }

    BenchReport {
        tool: "cubbs-bench",
        profile: config.profile.name(),
        batch_size: config.batch_size,
        workers: 1,
        backend: backend_label(config.backend),
        backend_info: backend_info(config.backend),
        call_path: "w3c-batch-api",
        metrics,
    }
}

async fn run_bench_w3c_mixed_batch_api(config: BenchConfig) -> BenchReport {
    std::env::set_var("CUBBS_BBS_BACKEND", config.backend.name());
    let all_cases = bench_cases(config.profile);
    let cases = if config.cases.is_empty() {
        all_cases
    } else {
        all_cases
            .into_iter()
            .filter(|case| config.cases.iter().any(|name| name == case.name))
            .collect()
    };
    assert!(!cases.is_empty(), "mixed W3C batch needs at least one case");
    let engine = BbsEngine::new(config.backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let verification_method =
        verification_method_from_public_key(&keys.public_key).expect("verification method");
    let proof_options = default_bbs_2023_proof_options(verification_method, "assertionMethod");

    let issue_plan = (0..config.batch_size)
        .map(|i| cases[i % cases.len()].clone())
        .collect::<Vec<_>>();
    let issue_items = issue_plan
        .iter()
        .enumerate()
        .map(|(i, case)| IssueJsonLdBaseProofRequest {
            document: (case.template)(i),
            proof_options: proof_options.clone(),
            secret_key: keys.secret_key.clone(),
            public_key: keys.public_key.clone(),
            mandatory_pointers: case.mandatory_pointers.clone(),
            hmac_key: deterministic_hmac_key(i),
        })
        .collect::<Vec<_>>();

    let issued = measure_batch_api(|| async {
        issue_jsonld_base_proof_batch(IssueJsonLdBaseProofBatchRequest {
            items: issue_items,
            fail_fast: false,
        })
        .await
    })
    .await;
    let signed = collect_success_values_from_api_batch(&issued);

    let derive_items = signed
        .into_iter()
        .enumerate()
        .map(|(i, signed_document)| {
            let case = &issue_plan[i];
            DeriveJsonLdDisclosureProofRequest {
                signed_document,
                selective_pointers: case.selective_pointers.clone(),
                presentation_header: format!("cubbs-bench-mixed-{}-{i}", case.name).into_bytes(),
            }
        })
        .collect::<Vec<_>>();
    let derived = measure_batch_api(|| async {
        derive_jsonld_disclosure_proof_batch(DeriveJsonLdDisclosureProofBatchRequest {
            items: derive_items,
            fail_fast: false,
        })
        .await
    })
    .await;
    let revealed = collect_success_values_from_api_batch(&derived);

    let verify_items = revealed
        .into_iter()
        .map(|document| (document, None))
        .collect::<Vec<_>>();
    let verified = measure_batch_api(|| async {
        verify_jsonld_disclosure_proof_batch(VerifyJsonLdDisclosureProofBatchRequest {
            items: verify_items,
            fail_fast: false,
        })
        .await
    })
    .await;

    let case_label = format!(
        "mixed-vc-vp:{}",
        cases
            .iter()
            .map(|case| case.name)
            .collect::<Vec<_>>()
            .join("+")
    );
    let metrics = vec![
        metric_from_api_value_batch(
            &case_label,
            "issue-mixed-batch-api",
            config.batch_size,
            &issued,
        ),
        metric_from_api_value_batch(
            &case_label,
            "derive-mixed-batch-api",
            config.batch_size,
            &derived,
        ),
        metric_from_api_bool_batch(
            &case_label,
            "verify-derived-mixed-batch-api",
            config.batch_size,
            &verified,
        ),
    ];

    BenchReport {
        tool: "cubbs-bench",
        profile: config.profile.name(),
        batch_size: config.batch_size,
        workers: 1,
        backend: backend_label(config.backend),
        backend_info: backend_info(config.backend),
        call_path: "w3c-mixed-batch-api",
        metrics,
    }
}

async fn run_bbs_vs_w3c(args: &[String]) -> BbsVsW3cReport {
    let iterations = arg_value(args, "--iterations")
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let message_counts = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 64, 128, 256]);
    let disclosed_ratio = arg_value(args, "--disclosed-ratio")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let pure_bbs_batch_size = arg_value(args, "--bbs-batch")
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| {
            arg_value(args, "--batch")
                .and_then(|value| value.parse().ok())
                .unwrap_or(16)
        });
    let backend = parse_backend_arg(args);
    let pure_bbs = if args.iter().any(|arg| arg == "--pure-bbs-batch-api") {
        run_pure_bbs_batch_bench(
            iterations,
            pure_bbs_batch_size,
            &message_counts,
            disclosed_ratio,
            backend,
        )
    } else {
        run_pure_bbs_bench(iterations, &message_counts, disclosed_ratio, backend)
    };

    let w3c_config = BenchConfig {
        profile: arg_value(args, "--profile")
            .as_deref()
            .map(BenchProfile::parse)
            .unwrap_or(BenchProfile::Stress),
        batch_size: arg_value(args, "--batch")
            .and_then(|value| value.parse().ok())
            .unwrap_or(16),
        workers: arg_value(args, "--workers")
            .and_then(|value| value.parse().ok())
            .unwrap_or(1),
        cases: arg_value(args, "--cases")
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        backend,
        w3c_batch_api: args.iter().any(|arg| arg == "--w3c-batch-api"),
        mixed_w3c_batch_api: args.iter().any(|arg| arg == "--mixed-w3c-batch-api"),
    };
    let w3c_e2e = run_bench(w3c_config).await;

    BbsVsW3cReport {
        tool: "cubbs-bench",
        mode: "bbs-vs-w3c",
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        pure_bbs,
        w3c_e2e,
    }
}

fn run_gpu_speedup_gate(args: &[String]) -> GpuSpeedupGateReport {
    let iterations = arg_value(args, "--iterations")
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    let batch_size = arg_value(args, "--bbs-batch")
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    let message_counts = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![64, 128, 256]);
    let disclosed_ratio = arg_value(args, "--disclosed-ratio")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let target_reduction = arg_value(args, "--target-reduction")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.90);

    let cpu = run_pure_bbs_batch_bench(
        iterations,
        batch_size,
        &message_counts,
        disclosed_ratio,
        BbsBackend::ZkryptiumCpu,
    );
    let gpu_backend = BbsBackend::GpuRequestedCpuFallback;
    let gpu = run_pure_bbs_batch_bench(
        iterations,
        batch_size,
        &message_counts,
        disclosed_ratio,
        gpu_backend,
    );
    let comparisons = compare_speedup_metrics(&cpu, &gpu, target_reduction);
    let backend_info = backend_info(gpu_backend);
    let passed = backend_info.hardware_acceleration
        && !backend_info.fallback
        && comparisons.iter().all(|comparison| comparison.passed);

    GpuSpeedupGateReport {
        tool: "cubbs-bench",
        mode: "gpu-speedup-gate",
        target_reduction,
        passed,
        backend_info,
        cpu,
        gpu,
        comparisons,
    }
}

fn run_gpu_sign_speedup_gate(args: &[String]) -> GpuSpeedupGateReport {
    let iterations = arg_value(args, "--iterations")
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    let batch_size = arg_value(args, "--bbs-batch")
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    let message_counts = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![64, 128, 256]);
    let target_reduction = arg_value(args, "--target-reduction")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.90);

    let cpu = run_pure_bbs_batch_sign_bench(
        iterations,
        batch_size,
        &message_counts,
        BbsBackend::ZkryptiumCpu,
    );
    let gpu_backend = BbsBackend::GpuRequestedCpuFallback;
    let gpu = run_pure_bbs_batch_sign_bench(iterations, batch_size, &message_counts, gpu_backend);
    let comparisons = compare_speedup_metrics(&cpu, &gpu, target_reduction);
    let backend_info = backend_info(gpu_backend);
    let passed = !backend_info.fallback && comparisons.iter().all(|comparison| comparison.passed);

    GpuSpeedupGateReport {
        tool: "cubbs-bench",
        mode: "gpu-sign-speedup-gate",
        target_reduction,
        passed,
        backend_info,
        cpu,
        gpu,
        comparisons,
    }
}

fn compare_speedup_metrics(
    cpu: &PureBbsReport,
    gpu: &PureBbsReport,
    target_reduction: f64,
) -> Vec<GpuSpeedupComparison> {
    const GATED_OPERATIONS: &[&str] = &[
        "sign-batch-api",
        "verify-signature-batch-api",
        "proof-gen-batch-api",
        "proof-verify-batch-api",
    ];
    let mut comparisons = Vec::new();
    for cpu_metric in &cpu.metrics {
        if !GATED_OPERATIONS.contains(&cpu_metric.operation) {
            continue;
        }
        let Some(gpu_metric) = gpu.metrics.iter().find(|metric| {
            metric.operation == cpu_metric.operation
                && metric.message_count == cpu_metric.message_count
                && metric.disclosed_count == cpu_metric.disclosed_count
        }) else {
            continue;
        };
        let reduction = if cpu_metric.p50_ms == 0.0 {
            0.0
        } else {
            1.0 - gpu_metric.p50_ms / cpu_metric.p50_ms
        };
        comparisons.push(GpuSpeedupComparison {
            operation: cpu_metric.operation,
            message_count: cpu_metric.message_count,
            cpu_p50_ms: cpu_metric.p50_ms,
            gpu_p50_ms: gpu_metric.p50_ms,
            reduction,
            target_reduction,
            passed: reduction >= target_reduction,
        });
    }
    comparisons
}

fn run_bbs_sign_profile(args: &[String]) -> BbsSignProfileReport {
    let iterations = arg_value(args, "--iterations")
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    let message_counts = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![16, 64, 128, 256]);
    let backend = parse_backend_arg(args);
    let engine = BbsEngine::new(backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let mut metrics = Vec::new();
    for &message_count in &message_counts {
        let header = deterministic_header(message_count);
        let messages = deterministic_bbs_messages(message_count);
        let mut total = Vec::with_capacity(iterations);
        let mut message_scalar = Vec::with_capacity(iterations);
        let mut generator = Vec::with_capacity(iterations);
        let mut domain = Vec::with_capacity(iterations);
        let mut e_scalar = Vec::with_capacity(iterations);
        let mut msm_and_finalize = Vec::with_capacity(iterations);
        let mut signature_bytes = 0usize;

        for _ in 0..iterations {
            let profile = engine
                .profile_sign(SignRequest {
                    secret_key: keys.secret_key.clone(),
                    public_key: keys.public_key.clone(),
                    header: header.clone(),
                    messages: messages.clone(),
                })
                .expect("profile sign");
            signature_bytes = profile.signature_bytes;
            total.push(profile.total_ms);
            message_scalar.push(profile.message_scalar_ms);
            generator.push(profile.generator_ms);
            domain.push(profile.domain_ms);
            e_scalar.push(profile.e_scalar_ms);
            msm_and_finalize.push(profile.msm_and_finalize_ms);
        }

        let total_p50_ms = percentile(&total, 50.0);
        let share = |value: f64| {
            if total_p50_ms == 0.0 {
                0.0
            } else {
                value / total_p50_ms
            }
        };
        let message_scalar_p50_ms = percentile(&message_scalar, 50.0);
        let generator_p50_ms = percentile(&generator, 50.0);
        let domain_p50_ms = percentile(&domain, 50.0);
        let e_scalar_p50_ms = percentile(&e_scalar, 50.0);
        let msm_and_finalize_p50_ms = percentile(&msm_and_finalize, 50.0);

        metrics.push(BbsSignProfileMetric {
            message_count,
            iterations,
            total_p50_ms,
            total_p95_ms: percentile(&total, 95.0),
            total_p99_ms: percentile(&total, 99.0),
            message_scalar_p50_ms,
            generator_p50_ms,
            domain_p50_ms,
            e_scalar_p50_ms,
            msm_and_finalize_p50_ms,
            message_scalar_share: share(message_scalar_p50_ms),
            generator_share: share(generator_p50_ms),
            domain_share: share(domain_p50_ms),
            e_scalar_share: share(e_scalar_p50_ms),
            msm_and_finalize_share: share(msm_and_finalize_p50_ms),
            signature_bytes,
        });
    }

    BbsSignProfileReport {
        tool: "cubbs-bench",
        mode: "bbs-sign-profile",
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        iterations,
        message_counts,
        metrics,
    }
}

fn run_bbs_proof_profile(args: &[String]) -> BbsProofProfileReport {
    let iterations = arg_value(args, "--iterations")
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    let batch_size = arg_value(args, "--bbs-batch")
        .or_else(|| arg_value(args, "--batch"))
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let message_counts = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![16, 64, 128]);
    let disclosed_ratio = arg_value(args, "--disclosed-ratio")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let backend = parse_backend_arg(args);
    let engine = BbsEngine::new(backend);
    let shared_keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("shared key generation");
    let mut metrics = Vec::new();
    let mixed_shapes =
        args.iter().any(|arg| arg == "--mixed-bbs-shapes") || message_counts.len() > 1;

    if mixed_shapes {
        let unique_issuer_per_item = args.iter().any(|arg| arg == "--unique-issuer-per-item");
        let item_message_counts = (0..batch_size)
            .map(|index| message_counts[index % message_counts.len()])
            .collect::<Vec<_>>();
        let headers = item_message_counts
            .iter()
            .map(|&message_count| deterministic_header(message_count))
            .collect::<Vec<_>>();
        let disclosed_counts = item_message_counts
            .iter()
            .map(|&message_count| disclosed_count(message_count, disclosed_ratio))
            .collect::<Vec<_>>();
        let disclosed_indexes = item_message_counts
            .iter()
            .zip(disclosed_counts.iter())
            .map(|(&message_count, &disclosed_count)| {
                disclosed_indexes(message_count, disclosed_count)
            })
            .collect::<Vec<_>>();
        let presentation_header = b"omniid-pure-bbs-proof-profile-mixed".to_vec();
        let message_batches = (0..batch_size)
            .map(|batch_index| {
                deterministic_bbs_messages_for_item(item_message_counts[batch_index], batch_index)
            })
            .collect::<Vec<_>>();
        let item_keys = if unique_issuer_per_item {
            (0..batch_size)
                .map(|_| {
                    engine
                        .generate_keypair(&mut rand::thread_rng())
                        .expect("profile per-item key generation")
                })
                .collect::<Vec<_>>()
        } else {
            (0..batch_size)
                .map(|_| shared_keys.clone())
                .collect::<Vec<_>>()
        };
        let signatures = message_batches
            .iter()
            .zip(headers.iter())
            .zip(item_keys.iter())
            .map(|((messages, header), keys)| {
                engine
                    .sign(SignRequest {
                        secret_key: keys.secret_key.clone(),
                        public_key: keys.public_key.clone(),
                        header: header.clone(),
                        messages: messages.clone(),
                    })
                    .expect("mixed warm signature")
            })
            .collect::<Vec<_>>();
        let proofs = message_batches
            .iter()
            .zip(signatures.iter())
            .zip(headers.iter())
            .zip(disclosed_indexes.iter())
            .zip(item_keys.iter())
            .map(
                |((((messages, signature), header), disclosed_indexes), keys)| {
                    engine
                        .proof_gen(ProofGenRequest {
                            public_key: keys.public_key.clone(),
                            signature: signature.clone(),
                            header: header.clone(),
                            presentation_header: presentation_header.clone(),
                            messages: messages.clone(),
                            disclosed_indexes: disclosed_indexes.clone(),
                        })
                        .expect("mixed warm proof")
                },
            )
            .collect::<Vec<_>>();
        let disclosed_message_batches = message_batches
            .iter()
            .zip(disclosed_indexes.iter())
            .map(|(messages, disclosed_indexes)| {
                disclosed_indexes
                    .iter()
                    .map(|&index| messages[index].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut gen_total = Vec::with_capacity(iterations);
        let mut gen_message_scalar = Vec::with_capacity(iterations);
        let mut gen_b_terms = Vec::with_capacity(iterations);
        let mut gen_b_msm = Vec::with_capacity(iterations);
        let mut gen_t2_prepare = Vec::with_capacity(iterations);
        let mut gen_t2_msm = Vec::with_capacity(iterations);
        let mut gen_finalize = Vec::with_capacity(iterations);
        let mut gen_proof_bytes = proofs.iter().map(Vec::len).max().unwrap_or_default();

        for _ in 0..iterations {
            let reqs = message_batches
                .iter()
                .zip(signatures.iter())
                .zip(headers.iter())
                .zip(disclosed_indexes.iter())
                .zip(item_keys.iter())
                .map(
                    |((((messages, signature), header), disclosed_indexes), keys)| {
                        ProofGenRequest {
                            public_key: keys.public_key.clone(),
                            signature: signature.clone(),
                            header: header.clone(),
                            presentation_header: presentation_header.clone(),
                            messages: messages.clone(),
                            disclosed_indexes: disclosed_indexes.clone(),
                        }
                    },
                )
                .collect::<Vec<_>>();
            let profile = engine.profile_proof_gen_batch(reqs, true);
            let proofs = profile
                .result
                .results
                .into_iter()
                .map(|item| item.result.expect("mixed proof-gen profile item"))
                .collect::<Vec<_>>();
            assert_eq!(proofs.len(), batch_size);
            gen_proof_bytes = proofs.iter().map(Vec::len).max().unwrap_or(gen_proof_bytes);
            gen_total.push(profile.total_ms / batch_size as f64);
            gen_message_scalar.push(profile.message_scalar_ms / batch_size as f64);
            gen_b_terms.push(profile.b_terms_ms / batch_size as f64);
            gen_b_msm.push(profile.b_msm_ms / batch_size as f64);
            gen_t2_prepare.push(profile.t2_prepare_ms / batch_size as f64);
            gen_t2_msm.push(profile.t2_msm_ms / batch_size as f64);
            gen_finalize.push(profile.finalize_ms / batch_size as f64);
        }
        metrics.push(BbsProofProfileMetric {
            operation: "proof-gen-mixed-batch-api",
            message_count: 0,
            disclosed_count: 0,
            iterations,
            batch_size,
            total_p50_ms: percentile(&gen_total, 50.0),
            total_p95_ms: percentile(&gen_total, 95.0),
            total_p99_ms: percentile(&gen_total, 99.0),
            message_scalar_p50_ms: percentile(&gen_message_scalar, 50.0),
            b_terms_p50_ms: Some(percentile(&gen_b_terms, 50.0)),
            b_msm_p50_ms: Some(percentile(&gen_b_msm, 50.0)),
            t2_prepare_p50_ms: Some(percentile(&gen_t2_prepare, 50.0)),
            t2_msm_p50_ms: Some(percentile(&gen_t2_msm, 50.0)),
            msm_terms_p50_ms: None,
            msm_p50_ms: None,
            challenge_pairing_input_p50_ms: None,
            pairing_p50_ms: None,
            finalize_p50_ms: percentile(&gen_finalize, 50.0),
            proof_bytes: gen_proof_bytes,
        });

        let mut verify_total = Vec::with_capacity(iterations);
        let mut verify_message_scalar = Vec::with_capacity(iterations);
        let mut verify_msm_terms = Vec::with_capacity(iterations);
        let mut verify_msm = Vec::with_capacity(iterations);
        let mut verify_challenge_pairing_input = Vec::with_capacity(iterations);
        let mut verify_pairing = Vec::with_capacity(iterations);
        let mut verify_finalize = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let reqs = proofs
                .iter()
                .zip(disclosed_message_batches.iter())
                .zip(headers.iter())
                .zip(disclosed_indexes.iter())
                .zip(item_message_counts.iter())
                .zip(item_keys.iter())
                .map(
                    |(
                        (
                            (((proof, disclosed_messages), header), disclosed_indexes),
                            &message_count,
                        ),
                        keys,
                    )| {
                        ProofVerifyRequest {
                            public_key: keys.public_key.clone(),
                            proof: proof.clone(),
                            header: header.clone(),
                            presentation_header: presentation_header.clone(),
                            disclosed_messages: disclosed_messages.clone(),
                            disclosed_indexes: disclosed_indexes.clone(),
                            total_message_count: message_count,
                        }
                    },
                )
                .collect::<Vec<_>>();
            let profile = engine.profile_proof_verify_batch(reqs, true);
            let oks = profile
                .result
                .results
                .into_iter()
                .map(|item| item.result.expect("mixed proof-verify profile item"))
                .collect::<Vec<_>>();
            assert_eq!(oks.len(), batch_size);
            assert!(oks.iter().all(|ok| *ok));
            verify_total.push(profile.total_ms / batch_size as f64);
            verify_message_scalar.push(profile.message_scalar_ms / batch_size as f64);
            verify_msm_terms.push(profile.msm_terms_ms / batch_size as f64);
            verify_msm.push(profile.msm_ms / batch_size as f64);
            verify_challenge_pairing_input
                .push(profile.challenge_pairing_input_ms / batch_size as f64);
            verify_pairing.push(profile.pairing_ms / batch_size as f64);
            verify_finalize.push(profile.fallback_finalize_ms / batch_size as f64);
        }
        metrics.push(BbsProofProfileMetric {
            operation: "proof-verify-mixed-batch-api",
            message_count: 0,
            disclosed_count: 0,
            iterations,
            batch_size,
            total_p50_ms: percentile(&verify_total, 50.0),
            total_p95_ms: percentile(&verify_total, 95.0),
            total_p99_ms: percentile(&verify_total, 99.0),
            message_scalar_p50_ms: percentile(&verify_message_scalar, 50.0),
            b_terms_p50_ms: None,
            b_msm_p50_ms: None,
            t2_prepare_p50_ms: None,
            t2_msm_p50_ms: None,
            msm_terms_p50_ms: Some(percentile(&verify_msm_terms, 50.0)),
            msm_p50_ms: Some(percentile(&verify_msm, 50.0)),
            challenge_pairing_input_p50_ms: Some(percentile(&verify_challenge_pairing_input, 50.0)),
            pairing_p50_ms: Some(percentile(&verify_pairing, 50.0)),
            finalize_p50_ms: percentile(&verify_finalize, 50.0),
            proof_bytes: proofs.iter().map(Vec::len).max().unwrap_or_default(),
        });

        return BbsProofProfileReport {
            tool: "cubbs-bench",
            mode: "bbs-proof-profile",
            backend: backend_label(backend),
            backend_info: backend_info(backend),
            iterations,
            batch_size,
            mixed_shapes,
            disclosed_ratio,
            message_counts,
            metrics,
        };
    }

    let keys = shared_keys;
    for &message_count in &message_counts {
        let header = deterministic_header(message_count);
        let disclosed_count = disclosed_count(message_count, disclosed_ratio);
        let disclosed_indexes = disclosed_indexes(message_count, disclosed_count);
        let presentation_header = b"omniid-pure-bbs-proof-profile".to_vec();
        let message_batches = (0..batch_size)
            .map(|batch_index| deterministic_bbs_messages_for_item(message_count, batch_index))
            .collect::<Vec<_>>();
        let signatures = message_batches
            .iter()
            .map(|messages| {
                engine
                    .sign(SignRequest {
                        secret_key: keys.secret_key.clone(),
                        public_key: keys.public_key.clone(),
                        header: header.clone(),
                        messages: messages.clone(),
                    })
                    .expect("warm signature")
            })
            .collect::<Vec<_>>();
        let proofs = message_batches
            .iter()
            .zip(signatures.iter())
            .map(|(messages, signature)| {
                engine
                    .proof_gen(ProofGenRequest {
                        public_key: keys.public_key.clone(),
                        signature: signature.clone(),
                        header: header.clone(),
                        presentation_header: presentation_header.clone(),
                        messages: messages.clone(),
                        disclosed_indexes: disclosed_indexes.clone(),
                    })
                    .expect("warm proof")
            })
            .collect::<Vec<_>>();
        let disclosed_message_batches = message_batches
            .iter()
            .map(|messages| {
                disclosed_indexes
                    .iter()
                    .map(|&index| messages[index].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut gen_total = Vec::with_capacity(iterations);
        let mut gen_message_scalar = Vec::with_capacity(iterations);
        let mut gen_b_terms = Vec::with_capacity(iterations);
        let mut gen_b_msm = Vec::with_capacity(iterations);
        let mut gen_t2_prepare = Vec::with_capacity(iterations);
        let mut gen_t2_msm = Vec::with_capacity(iterations);
        let mut gen_finalize = Vec::with_capacity(iterations);
        let mut gen_proof_bytes = proofs[0].len();

        for _ in 0..iterations {
            let reqs = message_batches
                .iter()
                .zip(signatures.iter())
                .map(|(messages, signature)| ProofGenRequest {
                    public_key: keys.public_key.clone(),
                    signature: signature.clone(),
                    header: header.clone(),
                    presentation_header: presentation_header.clone(),
                    messages: messages.clone(),
                    disclosed_indexes: disclosed_indexes.clone(),
                })
                .collect::<Vec<_>>();
            let profile = engine.profile_proof_gen_batch(reqs, true);
            let proofs = profile
                .result
                .results
                .into_iter()
                .map(|item| item.result.expect("proof-gen profile item"))
                .collect::<Vec<_>>();
            assert_eq!(proofs.len(), batch_size);
            gen_proof_bytes = proofs[0].len();
            gen_total.push(profile.total_ms / batch_size as f64);
            gen_message_scalar.push(profile.message_scalar_ms / batch_size as f64);
            gen_b_terms.push(profile.b_terms_ms / batch_size as f64);
            gen_b_msm.push(profile.b_msm_ms / batch_size as f64);
            gen_t2_prepare.push(profile.t2_prepare_ms / batch_size as f64);
            gen_t2_msm.push(profile.t2_msm_ms / batch_size as f64);
            gen_finalize.push(profile.finalize_ms / batch_size as f64);
        }
        metrics.push(BbsProofProfileMetric {
            operation: "proof-gen-batch-api",
            message_count,
            disclosed_count,
            iterations,
            batch_size,
            total_p50_ms: percentile(&gen_total, 50.0),
            total_p95_ms: percentile(&gen_total, 95.0),
            total_p99_ms: percentile(&gen_total, 99.0),
            message_scalar_p50_ms: percentile(&gen_message_scalar, 50.0),
            b_terms_p50_ms: Some(percentile(&gen_b_terms, 50.0)),
            b_msm_p50_ms: Some(percentile(&gen_b_msm, 50.0)),
            t2_prepare_p50_ms: Some(percentile(&gen_t2_prepare, 50.0)),
            t2_msm_p50_ms: Some(percentile(&gen_t2_msm, 50.0)),
            msm_terms_p50_ms: None,
            msm_p50_ms: None,
            challenge_pairing_input_p50_ms: None,
            pairing_p50_ms: None,
            finalize_p50_ms: percentile(&gen_finalize, 50.0),
            proof_bytes: gen_proof_bytes,
        });

        let mut verify_total = Vec::with_capacity(iterations);
        let mut verify_message_scalar = Vec::with_capacity(iterations);
        let mut verify_msm_terms = Vec::with_capacity(iterations);
        let mut verify_msm = Vec::with_capacity(iterations);
        let mut verify_challenge_pairing_input = Vec::with_capacity(iterations);
        let mut verify_pairing = Vec::with_capacity(iterations);
        let mut verify_finalize = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let reqs = proofs
                .iter()
                .zip(disclosed_message_batches.iter())
                .map(|(proof, disclosed_messages)| ProofVerifyRequest {
                    public_key: keys.public_key.clone(),
                    proof: proof.clone(),
                    header: header.clone(),
                    presentation_header: presentation_header.clone(),
                    disclosed_messages: disclosed_messages.clone(),
                    disclosed_indexes: disclosed_indexes.clone(),
                    total_message_count: message_count,
                })
                .collect::<Vec<_>>();
            let profile = engine.profile_proof_verify_batch(reqs, true);
            let oks = profile
                .result
                .results
                .into_iter()
                .map(|item| item.result.expect("proof-verify profile item"))
                .collect::<Vec<_>>();
            assert_eq!(oks.len(), batch_size);
            assert!(oks.iter().all(|ok| *ok));
            verify_total.push(profile.total_ms / batch_size as f64);
            verify_message_scalar.push(profile.message_scalar_ms / batch_size as f64);
            verify_msm_terms.push(profile.msm_terms_ms / batch_size as f64);
            verify_msm.push(profile.msm_ms / batch_size as f64);
            verify_challenge_pairing_input
                .push(profile.challenge_pairing_input_ms / batch_size as f64);
            verify_pairing.push(profile.pairing_ms / batch_size as f64);
            verify_finalize.push(profile.fallback_finalize_ms / batch_size as f64);
        }
        metrics.push(BbsProofProfileMetric {
            operation: "proof-verify-batch-api",
            message_count,
            disclosed_count,
            iterations,
            batch_size,
            total_p50_ms: percentile(&verify_total, 50.0),
            total_p95_ms: percentile(&verify_total, 95.0),
            total_p99_ms: percentile(&verify_total, 99.0),
            message_scalar_p50_ms: percentile(&verify_message_scalar, 50.0),
            b_terms_p50_ms: None,
            b_msm_p50_ms: None,
            t2_prepare_p50_ms: None,
            t2_msm_p50_ms: None,
            msm_terms_p50_ms: Some(percentile(&verify_msm_terms, 50.0)),
            msm_p50_ms: Some(percentile(&verify_msm, 50.0)),
            challenge_pairing_input_p50_ms: Some(percentile(&verify_challenge_pairing_input, 50.0)),
            pairing_p50_ms: Some(percentile(&verify_pairing, 50.0)),
            finalize_p50_ms: percentile(&verify_finalize, 50.0),
            proof_bytes: proofs[0].len(),
        });
    }

    BbsProofProfileReport {
        tool: "cubbs-bench",
        mode: "bbs-proof-profile",
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        iterations,
        batch_size,
        mixed_shapes,
        disclosed_ratio,
        message_counts,
        metrics,
    }
}

fn run_bbs_throughput_stress(args: &[String]) -> BbsThroughputStressReport {
    let _throughput_preset = maybe_gpu_throughput_max_preset(args);
    let no_bbs_caches = args.iter().any(|arg| arg == "--no-bbs-caches");
    if no_bbs_caches {
        for name in [
            "CUBBS_BBS_MESSAGE_SCALAR_CACHE",
            "CUBBS_BBS_SIGN_PREPARED_CACHE",
            "CUBBS_BBS_VERIFY_PREPARED_B_CACHE",
            "CUBBS_BBS_PROOF_VERIFY_PREPARED_CACHE",
        ] {
            std::env::set_var(name, "0");
        }
    }
    let duration_secs: f64 = arg_value(args, "--duration-secs")
        .and_then(|value| value.parse().ok())
        .unwrap_or(10.0);
    let duration = Duration::from_secs_f64(duration_secs.max(0.001));
    let warmup_secs: f64 = arg_value(args, "--warmup-secs")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.0);
    let warmup = Duration::from_secs_f64(warmup_secs.max(0.0));
    let (batch_size, streams) = bbs_throughput_batch_and_streams(args);
    let large_shape_preset = args.iter().any(|arg| arg == "--large-bbs-shapes");
    let ultra_large_shape_preset = args.iter().any(|arg| arg == "--ultra-large-bbs-shapes");
    let requested_message_counts = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| {
            if ultra_large_shape_preset {
                vec![4097, 8192, 16384, 65536]
            } else if large_shape_preset {
                vec![256, 512, 1024, 2048, 4096]
            } else {
                vec![64]
            }
        });
    let mixed_shapes =
        args.iter().any(|arg| arg == "--mixed-bbs-shapes") || requested_message_counts.len() > 1;
    let mixed_dispatch_mode = parse_mixed_bbs_dispatch_mode(args);
    if mixed_dispatch_mode == MixedBbsDispatchMode::SerialShapeBuckets && !mixed_shapes {
        panic!("--mixed-bbs-dispatch serial-shape-buckets requires mixed BBS shapes");
    }
    let unique_issuer_per_item = args.iter().any(|arg| arg == "--unique-issuer-per-item");
    let shared_repeated_requests = args.iter().any(|arg| arg == "--shared-repeated");
    let message_count = requested_message_counts[0];
    let disclosed_ratio = arg_value(args, "--disclosed-ratio")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let item_message_counts = (0..batch_size)
        .map(|index| {
            if mixed_shapes {
                requested_message_counts[index % requested_message_counts.len()]
            } else {
                message_count
            }
        })
        .collect::<Vec<_>>();
    let item_disclosed_counts = item_message_counts
        .iter()
        .map(|&message_count| disclosed_count(message_count, disclosed_ratio))
        .collect::<Vec<_>>();
    let item_disclosed_indexes = item_message_counts
        .iter()
        .zip(item_disclosed_counts.iter())
        .map(|(&message_count, &disclosed_count)| disclosed_indexes(message_count, disclosed_count))
        .collect::<Vec<_>>();
    let serial_shape_buckets = (mixed_dispatch_mode == MixedBbsDispatchMode::SerialShapeBuckets)
        .then(|| serial_shape_bucket_summary(&item_message_counts));
    let requested_disclosed_counts = requested_message_counts
        .iter()
        .map(|&message_count| disclosed_count(message_count, disclosed_ratio))
        .collect::<Vec<_>>();
    let disclosed_count = item_disclosed_counts[0];
    let operation = arg_value(args, "--operation").unwrap_or_else(|| "proof-gen".to_owned());
    let operations = stress_operations(&operation);
    if shared_repeated_requests
        && operations
            .iter()
            .any(|operation| !matches!(*operation, "sign" | "verify" | "verify-signature"))
    {
        panic!("--shared-repeated supports only sign and verify operations");
    }
    let sample_gpu_smi = args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--gpu-smi-sample" | "--nvidia-smi-sample" | "--sample-gpu"
        )
    });
    let gpu_smi_interval = arg_value(args, "--gpu-smi-interval-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(500));
    let target_items_per_sec = arg_value(args, "--target-items-per-sec")
        .or_else(|| arg_value(args, "--target-proofs-per-sec"))
        .or_else(|| arg_value(args, "--target-proof-per-sec"))
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0);
    let backend = parse_backend_arg(args);
    let engine = BbsEngine::new(backend);
    let shared_keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("shared key generation");
    let item_keys = if unique_issuer_per_item {
        (0..batch_size)
            .map(|_| {
                engine
                    .generate_keypair(&mut rand::thread_rng())
                    .expect("per-item key generation")
            })
            .collect::<Vec<_>>()
    } else {
        (0..batch_size)
            .map(|_| shared_keys.clone())
            .collect::<Vec<_>>()
    };
    let headers = item_message_counts
        .iter()
        .map(|&message_count| deterministic_header(message_count))
        .collect::<Vec<_>>();
    let presentation_header = b"omniid-bbs-throughput-stress".to_vec();
    let message_batches = (0..batch_size)
        .map(|batch_index| {
            deterministic_bbs_messages_for_item(item_message_counts[batch_index], batch_index)
        })
        .collect::<Vec<_>>();

    let sign_reqs = message_batches
        .iter()
        .zip(headers.iter())
        .zip(item_keys.iter())
        .map(|((messages, header), keys)| SignRequest {
            secret_key: keys.secret_key.clone(),
            public_key: keys.public_key.clone(),
            header: header.clone(),
            messages: messages.clone(),
        })
        .collect::<Vec<_>>();
    let signatures = engine
        .sign_batch(sign_reqs.clone(), true)
        .results
        .into_iter()
        .map(|item| item.result.expect("stress warm signature"))
        .collect::<Vec<_>>();
    assert_eq!(signatures.len(), batch_size);

    let verify_reqs = message_batches
        .iter()
        .zip(signatures.iter())
        .zip(headers.iter())
        .zip(item_keys.iter())
        .map(|(((messages, signature), header), keys)| VerifyRequest {
            public_key: keys.public_key.clone(),
            signature: signature.clone(),
            header: header.clone(),
            messages: messages.clone(),
        })
        .collect::<Vec<_>>();
    let shared_sign_reqs = shared_repeated_requests.then(|| {
        let request = Arc::new(sign_reqs[0].clone());
        vec![request; batch_size]
    });
    let shared_verify_reqs = shared_repeated_requests.then(|| {
        let request = Arc::new(verify_reqs[0].clone());
        vec![request; batch_size]
    });
    let proof_gen_reqs = message_batches
        .iter()
        .zip(signatures.iter())
        .zip(headers.iter())
        .zip(item_disclosed_indexes.iter())
        .zip(item_keys.iter())
        .map(
            |((((messages, signature), header), disclosed_indexes), keys)| ProofGenRequest {
                public_key: keys.public_key.clone(),
                signature: signature.clone(),
                header: header.clone(),
                presentation_header: presentation_header.clone(),
                messages: messages.clone(),
                disclosed_indexes: disclosed_indexes.clone(),
            },
        )
        .collect::<Vec<_>>();
    let proofs = engine
        .proof_gen_batch(proof_gen_reqs.clone(), true)
        .results
        .into_iter()
        .map(|item| item.result.expect("stress warm proof"))
        .collect::<Vec<_>>();
    assert_eq!(proofs.len(), batch_size);
    let disclosed_message_batches = message_batches
        .iter()
        .zip(item_disclosed_indexes.iter())
        .map(|(messages, disclosed_indexes)| {
            disclosed_indexes
                .iter()
                .map(|&index| messages[index].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let proof_verify_reqs = proofs
        .iter()
        .zip(disclosed_message_batches.iter())
        .zip(headers.iter())
        .zip(item_disclosed_indexes.iter())
        .zip(item_message_counts.iter())
        .zip(item_keys.iter())
        .map(
            |(
                ((((proof, disclosed_messages), header), disclosed_indexes), &message_count),
                keys,
            )| {
                ProofVerifyRequest {
                    public_key: keys.public_key.clone(),
                    proof: proof.clone(),
                    header: header.clone(),
                    presentation_header: presentation_header.clone(),
                    disclosed_messages: disclosed_messages.clone(),
                    disclosed_indexes: disclosed_indexes.clone(),
                    total_message_count: message_count,
                }
            },
        )
        .collect::<Vec<_>>();

    let signature_bytes = signatures.first().map(Vec::len);
    let proof_bytes = if mixed_shapes {
        None
    } else {
        proofs.first().map(Vec::len)
    };
    let gpu_smi_sampler = sample_gpu_smi.then(|| start_gpu_smi_sampler(gpu_smi_interval));
    let mut metrics = Vec::new();
    for operation in operations {
        let metric = match operation {
            "sign" => stress_batch_operation_with_warmup(
                "sign",
                duration,
                warmup,
                batch_size,
                streams,
                signature_bytes,
                None,
                || {
                    if let Some(reqs) = &shared_sign_reqs {
                        engine.sign_batch_shared(reqs, false)
                    } else {
                        dispatch_mixed_bbs_batch(
                            mixed_dispatch_mode,
                            &sign_reqs,
                            |req| req.messages.len(),
                            |batch| engine.sign_batch(batch, true),
                        )
                    }
                },
                |values| {
                    values
                        .iter()
                        .all(|signature| signature.len() == signature_bytes.unwrap())
                },
            ),
            "verify" | "verify-signature" => stress_batch_operation_with_warmup(
                "verify-signature",
                duration,
                warmup,
                batch_size,
                streams,
                signature_bytes,
                None,
                || {
                    if let Some(reqs) = &shared_verify_reqs {
                        engine.verify_batch_shared(reqs, false)
                    } else if mixed_dispatch_mode == MixedBbsDispatchMode::Production {
                        engine.verify_batch_borrowed(&verify_reqs, true)
                    } else {
                        dispatch_mixed_bbs_batch(
                            mixed_dispatch_mode,
                            &verify_reqs,
                            |req| req.messages.len(),
                            |batch| engine.verify_batch(batch, true),
                        )
                    }
                },
                |values| values.iter().all(|ok| *ok),
            ),
            "proof-gen" => stress_batch_operation_with_warmup(
                "proof-gen",
                duration,
                warmup,
                batch_size,
                streams,
                signature_bytes,
                proof_bytes,
                || {
                    dispatch_mixed_bbs_batch(
                        mixed_dispatch_mode,
                        &proof_gen_reqs,
                        |req| req.messages.len(),
                        |batch| engine.proof_gen_batch(batch, true),
                    )
                },
                |values| {
                    values.iter().all(|proof| {
                        proof_bytes
                            .map(|expected| proof.len() == expected)
                            .unwrap_or(!proof.is_empty())
                    })
                },
            ),
            "proof-verify" => stress_batch_operation_with_warmup(
                "proof-verify",
                duration,
                warmup,
                batch_size,
                streams,
                signature_bytes,
                proof_bytes,
                || {
                    dispatch_mixed_bbs_batch(
                        mixed_dispatch_mode,
                        &proof_verify_reqs,
                        |req| req.total_message_count,
                        |batch| engine.proof_verify_batch(batch, true),
                    )
                },
                |values| values.iter().all(|ok| *ok),
            ),
            _ => unreachable!("stress operation parser filters operations"),
        };
        metrics.push(metric);
    }
    let gpu_smi = gpu_smi_sampler.map(finish_gpu_smi_sampler);
    let efficiency =
        throughput_efficiency_metrics(&metrics, gpu_smi.as_ref(), target_items_per_sec);

    BbsThroughputStressReport {
        tool: "cubbs-bench",
        mode: "bbs-throughput-stress",
        scheduling_profile: args
            .iter()
            .any(|arg| arg == "--gpu-throughput-auto")
            .then_some("operation-aware-gpu-saturation-v1"),
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        duration_secs,
        warmup_secs,
        batch_size,
        streams,
        mixed_shapes,
        unique_issuer_per_item,
        shared_repeated_requests,
        mixed_dispatch_mode: mixed_dispatch_mode.label(),
        serial_shape_buckets,
        verify_pre_b_aggregate_enabled: verify_pre_b_aggregate_enabled_for_bench(),
        verify_pre_b_aggregate_min_items: verify_pre_b_aggregate_min_items_for_bench(),
        state_cache_mode: if no_bbs_caches {
            "disabled"
        } else {
            "production-default"
        },
        message_count,
        message_counts: requested_message_counts,
        disclosed_count,
        disclosed_counts: requested_disclosed_counts,
        disclosed_ratio,
        gpu_smi,
        efficiency,
        metrics,
    }
}

fn run_bbs_common_points_worker_sweep(args: &[String]) -> BbsCommonPointsWorkerSweepReport {
    let variants = arg_value(args, "--common-points-workers")
        .or_else(|| arg_value(args, "--common-workers"))
        .map(|value| parse_common_points_worker_variants(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![("default".to_owned(), None), ("6".to_owned(), Some(6))]);
    let operations = arg_value(args, "--operation")
        .map(|operation| stress_operations(&operation))
        .unwrap_or_else(|| vec!["proof-gen", "proof-verify"]);
    let repeat_count = arg_value(args, "--sweep-repeats")
        .or_else(|| arg_value(args, "--repeats"))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let backend = parse_backend_arg(args);
    let original_common_workers = std::env::var_os("CUBBS_BBS_BLST_COMMON_POINTS_MSM_WORKERS");
    let mut reports = Vec::new();

    for (label, workers) in variants {
        match workers {
            Some(workers) => {
                std::env::set_var(
                    "CUBBS_BBS_BLST_COMMON_POINTS_MSM_WORKERS",
                    workers.to_string(),
                );
            }
            None => {
                std::env::remove_var("CUBBS_BBS_BLST_COMMON_POINTS_MSM_WORKERS");
            }
        }
        for operation in &operations {
            let operation_args = args_with_replaced_arg(args, "--operation", operation);
            let mut run_reports = (0..repeat_count)
                .map(|_| run_bbs_throughput_stress(&operation_args))
                .collect::<Vec<_>>();
            let summary = common_points_worker_variant_summary(&run_reports);
            let report = run_reports.remove(0);
            reports.push(BbsCommonPointsWorkerVariantReport {
                label: label.clone(),
                common_points_workers: workers,
                operation: (*operation).to_owned(),
                report,
                repeat_reports: run_reports,
                summary,
            });
        }
    }

    match original_common_workers {
        Some(value) => std::env::set_var("CUBBS_BBS_BLST_COMMON_POINTS_MSM_WORKERS", value),
        None => std::env::remove_var("CUBBS_BBS_BLST_COMMON_POINTS_MSM_WORKERS"),
    }

    let best_by_operation = common_points_worker_best_by_operation(&reports);

    BbsCommonPointsWorkerSweepReport {
        tool: "cubbs-bench",
        mode: "bbs-common-points-worker-sweep",
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        repeat_count,
        best_by_operation,
        common_points_worker_variants: reports,
    }
}

fn run_bbs_common_points_abi_sweep(args: &[String]) -> BbsCommonPointsAbiSweepReport {
    let variants = arg_value(args, "--common-points-abi")
        .or_else(|| arg_value(args, "--common-abi"))
        .map(|value| parse_common_points_abi_variants(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| {
            vec![
                ("default".to_owned(), None),
                ("packed".to_owned(), Some(true)),
            ]
        });
    let operations = arg_value(args, "--operation")
        .map(|operation| stress_operations(&operation))
        .unwrap_or_else(|| vec!["proof-gen", "proof-verify"]);
    let repeat_count = arg_value(args, "--sweep-repeats")
        .or_else(|| arg_value(args, "--repeats"))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let backend = parse_backend_arg(args);
    let original_common_points_abi = std::env::var_os("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI");
    let mut reports = Vec::new();

    for (label, enabled) in variants {
        match enabled {
            Some(true) => std::env::set_var("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI", "1"),
            Some(false) => std::env::set_var("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI", "0"),
            None => std::env::remove_var("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI"),
        }
        for operation in &operations {
            let operation_args = args_with_replaced_arg(args, "--operation", operation);
            let mut run_reports = (0..repeat_count)
                .map(|_| run_bbs_throughput_stress(&operation_args))
                .collect::<Vec<_>>();
            let summary = common_points_worker_variant_summary(&run_reports);
            let report = run_reports.remove(0);
            reports.push(BbsCommonPointsAbiVariantReport {
                label: label.clone(),
                common_points_msm_abi: enabled,
                operation: (*operation).to_owned(),
                report,
                repeat_reports: run_reports,
                summary,
            });
        }
    }

    match original_common_points_abi {
        Some(value) => std::env::set_var("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI", value),
        None => std::env::remove_var("CUBBS_BBS_GPU_COMMON_POINTS_MSM_ABI"),
    }

    let best_by_operation = common_points_abi_best_by_operation(&reports);

    BbsCommonPointsAbiSweepReport {
        tool: "cubbs-bench",
        mode: "bbs-common-points-abi-sweep",
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        repeat_count,
        best_by_operation,
        common_points_abi_variants: reports,
    }
}

fn common_points_worker_variant_summary(
    reports: &[BbsThroughputStressReport],
) -> BbsCommonPointsWorkerVariantSummary {
    let items_per_sec = reports
        .iter()
        .filter_map(|report| report.metrics.first().map(|metric| metric.items_per_sec))
        .collect::<Vec<_>>();
    if items_per_sec.is_empty() {
        return BbsCommonPointsWorkerVariantSummary {
            sample_count: 0,
            mean_items_per_sec: 0.0,
            best_items_per_sec: 0.0,
            worst_items_per_sec: 0.0,
            stddev_items_per_sec: 0.0,
            relative_range: 0.0,
        };
    }
    let sample_count = items_per_sec.len();
    let mean_items_per_sec = items_per_sec.iter().sum::<f64>() / sample_count as f64;
    let best_items_per_sec = items_per_sec
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let worst_items_per_sec = items_per_sec.iter().copied().fold(f64::INFINITY, f64::min);
    let variance = items_per_sec
        .iter()
        .map(|value| {
            let delta = value - mean_items_per_sec;
            delta * delta
        })
        .sum::<f64>()
        / sample_count as f64;
    let stddev_items_per_sec = variance.sqrt();
    let relative_range = if mean_items_per_sec == 0.0 {
        0.0
    } else {
        (best_items_per_sec - worst_items_per_sec) / mean_items_per_sec
    };
    BbsCommonPointsWorkerVariantSummary {
        sample_count,
        mean_items_per_sec,
        best_items_per_sec,
        worst_items_per_sec,
        stddev_items_per_sec,
        relative_range,
    }
}

fn common_points_abi_best_by_operation(
    reports: &[BbsCommonPointsAbiVariantReport],
) -> Vec<BbsCommonPointsAbiBestReport> {
    let mut operations = Vec::<String>::new();
    for report in reports {
        if !operations
            .iter()
            .any(|operation| operation == &report.operation)
        {
            operations.push(report.operation.clone());
        }
    }

    operations
        .into_iter()
        .filter_map(|operation| {
            let candidates = reports
                .iter()
                .filter(|report| report.operation == operation)
                .map(|report| {
                    (
                        report.label.clone(),
                        report.common_points_msm_abi,
                        report.summary.mean_items_per_sec,
                    )
                })
                .collect::<Vec<_>>();
            let (best_label, best_common_points_msm_abi, best_mean_items_per_sec) = candidates
                .iter()
                .cloned()
                .max_by(|lhs, rhs| lhs.2.total_cmp(&rhs.2))?;
            let default_mean_items_per_sec = candidates
                .iter()
                .find(|(label, enabled, _)| enabled.is_none() || label == "default")
                .map(|(_, _, items_per_sec)| *items_per_sec);
            let speedup_vs_default = default_mean_items_per_sec
                .and_then(|default| (default > 0.0).then_some(best_mean_items_per_sec / default));
            Some(BbsCommonPointsAbiBestReport {
                operation,
                best_label,
                best_common_points_msm_abi,
                best_mean_items_per_sec,
                default_mean_items_per_sec,
                speedup_vs_default,
            })
        })
        .collect()
}

fn common_points_worker_best_by_operation(
    reports: &[BbsCommonPointsWorkerVariantReport],
) -> Vec<BbsCommonPointsWorkerBestReport> {
    let mut operations = Vec::<String>::new();
    for report in reports {
        if !operations
            .iter()
            .any(|operation| operation == &report.operation)
        {
            operations.push(report.operation.clone());
        }
    }

    operations
        .into_iter()
        .filter_map(|operation| {
            let candidates = reports
                .iter()
                .filter(|report| report.operation == operation)
                .map(|report| {
                    (
                        report.label.clone(),
                        report.common_points_workers,
                        report.summary.mean_items_per_sec,
                    )
                })
                .collect::<Vec<_>>();
            let (best_label, best_common_points_workers, best_mean_items_per_sec) = candidates
                .iter()
                .cloned()
                .max_by(|lhs, rhs| lhs.2.total_cmp(&rhs.2))?;
            let default_mean_items_per_sec = candidates
                .iter()
                .find(|(label, workers, _)| workers.is_none() || label == "default")
                .map(|(_, _, items_per_sec)| *items_per_sec);
            let speedup_vs_default = default_mean_items_per_sec
                .and_then(|default| (default > 0.0).then_some(best_mean_items_per_sec / default));
            Some(BbsCommonPointsWorkerBestReport {
                operation,
                best_label,
                best_common_points_workers,
                best_mean_items_per_sec,
                default_mean_items_per_sec,
                speedup_vs_default,
            })
        })
        .collect()
}

fn parse_common_points_worker_variants(value: &str) -> Vec<(String, Option<usize>)> {
    value
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                None
            } else if matches!(trimmed, "default" | "auto" | "unset") {
                Some((trimmed.to_owned(), None))
            } else {
                trimmed
                    .parse::<usize>()
                    .ok()
                    .map(|workers| (trimmed.to_owned(), Some(workers)))
            }
        })
        .collect()
}

fn parse_common_points_abi_variants(value: &str) -> Vec<(String, Option<bool>)> {
    value
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                None
            } else if matches!(trimmed, "default" | "auto" | "unset") {
                Some((trimmed.to_owned(), None))
            } else if matches!(trimmed, "packed" | "on" | "true" | "1") {
                Some((trimmed.to_owned(), Some(true)))
            } else if matches!(trimmed, "off" | "false" | "0") {
                Some((trimmed.to_owned(), Some(false)))
            } else {
                None
            }
        })
        .collect()
}

fn args_with_replaced_arg(args: &[String], name: &str, value: &str) -> Vec<String> {
    let mut replaced = Vec::with_capacity(args.len() + 2);
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == name {
            index += 2;
        } else {
            replaced.push(args[index].clone());
            index += 1;
        }
    }
    replaced.push(name.to_owned());
    replaced.push(value.to_owned());
    replaced
}

fn bbs_gpu_msm_acceptance_args(args: &[String]) -> Vec<String> {
    let mut expanded = args
        .iter()
        .filter(|arg| arg.as_str() != "--bbs-gpu-msm-acceptance")
        .cloned()
        .collect::<Vec<_>>();
    ensure_arg_value(&mut expanded, "--bbs-backend", "gpu");
    ensure_arg_value(&mut expanded, "--messages", "16,64,128");
    ensure_arg_value(&mut expanded, "--batch", "512");
    ensure_arg_value(&mut expanded, "--streams", "4");
    ensure_arg_value(&mut expanded, "--duration-secs", "3");
    ensure_arg_value(&mut expanded, "--target-proofs-per-sec", "20000");
    ensure_arg_value(&mut expanded, "--require-gpu-route-ratio", "0.9");
    ensure_arg_value(&mut expanded, "--require-fused-route-ratio", "1.0");
    ensure_flag(&mut expanded, "--mixed-bbs-shapes");
    ensure_flag(&mut expanded, "--msm-stage-groups");
    ensure_flag(&mut expanded, "--require-dispatch-target");
    expanded
}

fn ensure_arg_value(args: &mut Vec<String>, name: &str, value: &str) {
    if !args.iter().any(|arg| arg == name) {
        args.push(name.to_owned());
        args.push(value.to_owned());
    }
}

fn ensure_flag(args: &mut Vec<String>, name: &str) {
    if !args.iter().any(|arg| arg == name) {
        args.push(name.to_owned());
    }
}

fn print_bbs_proof_msm_stress_report(report: BbsProofMsmStressReport) {
    let route_passed = report
        .gpu_route_gate
        .as_ref()
        .map(|gate| gate.passed)
        .unwrap_or(true);
    let fused_route_passed = report
        .fused_route_gate
        .as_ref()
        .map(|gate| gate.passed)
        .unwrap_or(true);
    let throughput_passed = report
        .throughput_gate
        .as_ref()
        .map(|gate| gate.passed)
        .unwrap_or(true);
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !(route_passed && fused_route_passed && throughput_passed) {
        std::process::exit(1);
    }
}

fn stress_operations(operation: &str) -> Vec<&'static str> {
    match operation {
        "all" => vec!["sign", "verify-signature", "proof-gen", "proof-verify"],
        "sign" => vec!["sign"],
        "verify" | "verify-signature" => vec!["verify-signature"],
        "proof-gen" => vec!["proof-gen"],
        "proof-verify" => vec!["proof-verify"],
        other => panic!("unknown --operation {other}; use sign|verify|proof-gen|proof-verify|all"),
    }
}

struct GpuSmiSampler {
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<GpuSmiSample>>>,
    errors: Arc<AtomicUsize>,
    handle: thread::JoinHandle<()>,
}

fn start_gpu_smi_sampler(interval: Duration) -> GpuSmiSampler {
    let stop = Arc::new(AtomicBool::new(false));
    let samples = Arc::new(Mutex::new(Vec::new()));
    let errors = Arc::new(AtomicUsize::new(0));
    let thread_stop = Arc::clone(&stop);
    let thread_samples = Arc::clone(&samples);
    let thread_errors = Arc::clone(&errors);
    let interval = interval.max(Duration::from_millis(100));
    let handle = thread::spawn(move || {
        while !thread_stop.load(Ordering::Relaxed) {
            match read_gpu_smi_sample() {
                Some(sample) => thread_samples
                    .lock()
                    .expect("gpu smi sample lock poisoned")
                    .push(sample),
                None => {
                    thread_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            thread::sleep(interval);
        }
    });
    GpuSmiSampler {
        stop,
        samples,
        errors,
        handle,
    }
}

fn finish_gpu_smi_sampler(sampler: GpuSmiSampler) -> GpuSmiSummary {
    sampler.stop.store(true, Ordering::Relaxed);
    let _ = sampler.handle.join();
    let samples = sampler
        .samples
        .lock()
        .expect("gpu smi sample lock poisoned")
        .clone();
    summarize_gpu_smi_samples(samples, sampler.errors.load(Ordering::Relaxed))
}

fn read_gpu_smi_sample() -> Option<GpuSmiSample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let line = stdout.lines().next()?;
    let values = line
        .split(',')
        .map(|value| value.trim().parse::<f64>().ok())
        .collect::<Option<Vec<_>>>()?;
    if values.len() < 4 {
        return None;
    }
    Some(GpuSmiSample {
        gpu_util_percent: values[0],
        power_watts: values[1],
        memory_used_mib: values[2],
        memory_total_mib: values[3],
    })
}

fn summarize_gpu_smi_samples(samples: Vec<GpuSmiSample>, errors: usize) -> GpuSmiSummary {
    if samples.is_empty() {
        return GpuSmiSummary {
            sample_count: 0,
            errors,
            avg_gpu_util_percent: 0.0,
            max_gpu_util_percent: 0.0,
            avg_power_watts: 0.0,
            max_power_watts: 0.0,
            avg_memory_used_mib: 0.0,
            max_memory_used_mib: 0.0,
            memory_total_mib: None,
        };
    }

    let sample_count = samples.len();
    let sum_util = samples
        .iter()
        .map(|sample| sample.gpu_util_percent)
        .sum::<f64>();
    let sum_power = samples.iter().map(|sample| sample.power_watts).sum::<f64>();
    let sum_memory = samples
        .iter()
        .map(|sample| sample.memory_used_mib)
        .sum::<f64>();
    GpuSmiSummary {
        sample_count,
        errors,
        avg_gpu_util_percent: sum_util / sample_count as f64,
        max_gpu_util_percent: samples
            .iter()
            .map(|sample| sample.gpu_util_percent)
            .fold(0.0, f64::max),
        avg_power_watts: sum_power / sample_count as f64,
        max_power_watts: samples
            .iter()
            .map(|sample| sample.power_watts)
            .fold(0.0, f64::max),
        avg_memory_used_mib: sum_memory / sample_count as f64,
        max_memory_used_mib: samples
            .iter()
            .map(|sample| sample.memory_used_mib)
            .fold(0.0, f64::max),
        memory_total_mib: samples.first().map(|sample| sample.memory_total_mib),
    }
}

fn throughput_efficiency_metrics(
    metrics: &[BbsThroughputStressMetric],
    gpu_smi: Option<&GpuSmiSummary>,
    target_items_per_sec: Option<f64>,
) -> Vec<BbsThroughputEfficiencyMetric> {
    metrics
        .iter()
        .map(|metric| {
            let target_ratio = target_items_per_sec.map(|target| metric.items_per_sec / target);
            let required_speedup = target_items_per_sec.map(|target| {
                if metric.items_per_sec > 0.0 {
                    target / metric.items_per_sec
                } else {
                    f64::INFINITY
                }
            });
            let avg_gpu_util_percent = gpu_smi.map(|summary| summary.avg_gpu_util_percent);
            let max_gpu_util_percent = gpu_smi.map(|summary| summary.max_gpu_util_percent);
            let items_per_sec_per_gpu_util_percent = avg_gpu_util_percent
                .and_then(|util| (util > 0.0).then_some(metric.items_per_sec / util));
            let avg_power_watts = gpu_smi.map(|summary| summary.avg_power_watts);
            let items_per_sec_per_watt = avg_power_watts
                .and_then(|watts| (watts > 0.0).then_some(metric.items_per_sec / watts));
            BbsThroughputEfficiencyMetric {
                operation: metric.operation,
                items_per_sec: metric.items_per_sec,
                target_items_per_sec,
                target_ratio,
                required_speedup,
                avg_gpu_util_percent,
                max_gpu_util_percent,
                items_per_sec_per_gpu_util_percent,
                avg_power_watts,
                items_per_sec_per_watt,
            }
        })
        .collect()
}

fn run_bbs_g1_msm_stress(args: &[String]) -> BbsG1MsmStressReport {
    let _throughput_preset = maybe_gpu_throughput_max_preset(args);
    let duration_secs: f64 = arg_value(args, "--duration-secs")
        .and_then(|value| value.parse().ok())
        .unwrap_or(10.0);
    let duration = Duration::from_secs_f64(duration_secs.max(0.001));
    let batch_size = arg_value(args, "--bbs-batch")
        .or_else(|| arg_value(args, "--batch"))
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);
    let streams = arg_value(args, "--stress-streams")
        .or_else(|| arg_value(args, "--streams"))
        .and_then(|value| value.parse().ok())
        .unwrap_or(1usize)
        .max(1);
    let message_count = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .and_then(|values| values.into_iter().next())
        .unwrap_or(64);
    let backend = parse_backend_arg(args);
    let engine = BbsEngine::new(backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let header = deterministic_header(message_count);
    let template_req = SignRequest {
        secret_key: keys.secret_key.clone(),
        public_key: keys.public_key.clone(),
        header: header.clone(),
        messages: deterministic_bbs_messages_for_item(message_count, 0),
    };
    let template_terms = engine
        .prepare_b_msm_terms_for_sign(&template_req)
        .expect("prepare G1 MSM template");
    let terms = (0..batch_size)
        .map(|batch_index| {
            let req = SignRequest {
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                header: header.clone(),
                messages: deterministic_bbs_messages_for_item(message_count, batch_index),
            };
            engine
                .prepare_b_msm_terms_for_sign(&req)
                .expect("G1 MSM stress terms")
        })
        .collect::<Vec<_>>();
    let term_count = template_terms.compressed_points.len();

    let metric = stress_batch_operation(
        "g1-msm",
        duration,
        batch_size,
        streams,
        None,
        None,
        || engine.g1_msm_batch_for_bench(&terms, true),
        |values| values.iter().all(|point| point.len() == 48),
    );

    BbsG1MsmStressReport {
        tool: "cubbs-bench",
        mode: "bbs-g1-msm-stress",
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        duration_secs,
        batch_size,
        streams,
        message_count,
        term_count,
        common_points: true,
        metric,
    }
}

fn run_bbs_proof_msm_shapes(args: &[String]) -> BbsProofMsmShapesReport {
    let batch_size = arg_value(args, "--bbs-batch")
        .or_else(|| arg_value(args, "--batch"))
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);
    let message_counts = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![64]);
    let mixed_shapes =
        args.iter().any(|arg| arg == "--mixed-bbs-shapes") || message_counts.len() > 1;
    let disclosed_ratio = arg_value(args, "--disclosed-ratio")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let backend = parse_backend_arg(args);
    let engine = BbsEngine::new(backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let item_message_counts = (0..batch_size)
        .map(|index| {
            if mixed_shapes {
                message_counts[index % message_counts.len()]
            } else {
                message_counts[0]
            }
        })
        .collect::<Vec<_>>();
    let headers = item_message_counts
        .iter()
        .map(|&message_count| deterministic_header(message_count))
        .collect::<Vec<_>>();
    let disclosed_indexes = item_message_counts
        .iter()
        .map(|&message_count| {
            disclosed_indexes(
                message_count,
                disclosed_count(message_count, disclosed_ratio),
            )
        })
        .collect::<Vec<_>>();
    let presentation_header = b"omniid-bbs-proof-msm-shapes".to_vec();
    let message_batches = (0..batch_size)
        .map(|index| deterministic_bbs_messages_for_item(item_message_counts[index], index))
        .collect::<Vec<_>>();
    let signatures = message_batches
        .iter()
        .zip(headers.iter())
        .map(|(messages, header)| {
            engine
                .sign(SignRequest {
                    secret_key: keys.secret_key.clone(),
                    public_key: keys.public_key.clone(),
                    header: header.clone(),
                    messages: messages.clone(),
                })
                .expect("proof MSM shape signature")
        })
        .collect::<Vec<_>>();
    let proof_gen_reqs = message_batches
        .iter()
        .zip(signatures.iter())
        .zip(headers.iter())
        .zip(disclosed_indexes.iter())
        .map(
            |(((messages, signature), header), disclosed_indexes)| ProofGenRequest {
                public_key: keys.public_key.clone(),
                signature: signature.clone(),
                header: header.clone(),
                presentation_header: presentation_header.clone(),
                messages: messages.clone(),
                disclosed_indexes: disclosed_indexes.clone(),
            },
        )
        .collect::<Vec<_>>();
    let proofs = engine
        .proof_gen_batch(proof_gen_reqs.clone(), true)
        .results
        .into_iter()
        .map(|item| item.result.expect("proof MSM shape proof"))
        .collect::<Vec<_>>();
    let disclosed_message_batches = message_batches
        .iter()
        .zip(disclosed_indexes.iter())
        .map(|(messages, disclosed_indexes)| {
            disclosed_indexes
                .iter()
                .map(|&index| messages[index].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let proof_verify_reqs = proofs
        .iter()
        .zip(disclosed_message_batches.iter())
        .zip(headers.iter())
        .zip(disclosed_indexes.iter())
        .zip(item_message_counts.iter())
        .map(
            |((((proof, disclosed_messages), header), disclosed_indexes), &message_count)| {
                ProofVerifyRequest {
                    public_key: keys.public_key.clone(),
                    proof: proof.clone(),
                    header: header.clone(),
                    presentation_header: presentation_header.clone(),
                    disclosed_messages: disclosed_messages.clone(),
                    disclosed_indexes: disclosed_indexes.clone(),
                    total_message_count: message_count,
                }
            },
        )
        .collect::<Vec<_>>();
    let shapes = engine
        .proof_msm_shapes_for_bench(proof_gen_reqs, proof_verify_reqs, true)
        .expect("proof MSM shapes")
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>();

    BbsProofMsmShapesReport {
        tool: "cubbs-bench",
        mode: "bbs-proof-msm-shapes",
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        batch_size,
        mixed_shapes,
        message_counts,
        disclosed_ratio,
        shapes,
    }
}

fn run_bbs_proof_msm_stress(args: &[String]) -> BbsProofMsmStressReport {
    let _throughput_preset = maybe_gpu_throughput_max_preset(args);
    let duration_secs: f64 = arg_value(args, "--duration-secs")
        .and_then(|value| value.parse().ok())
        .unwrap_or(10.0);
    let duration = Duration::from_secs_f64(duration_secs.max(0.001));
    let batch_size = arg_value(args, "--bbs-batch")
        .or_else(|| arg_value(args, "--batch"))
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);
    let streams = arg_value(args, "--stress-streams")
        .or_else(|| arg_value(args, "--streams"))
        .and_then(|value| value.parse().ok())
        .unwrap_or(1usize)
        .max(1);
    let message_counts = arg_value(args, "--messages")
        .map(|value| parse_usize_list(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![64]);
    let mixed_shapes =
        args.iter().any(|arg| arg == "--mixed-bbs-shapes") || message_counts.len() > 1;
    let disclosed_ratio = arg_value(args, "--disclosed-ratio")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let sample_gpu_smi = args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--gpu-smi-sample" | "--nvidia-smi-sample" | "--sample-gpu"
        )
    });
    let include_stage_groups = args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--msm-stage-groups" | "--group-msm-stages"));
    let gpu_smi_interval = arg_value(args, "--gpu-smi-interval-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(500));
    let backend = parse_backend_arg(args);
    let engine = BbsEngine::new(backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let item_message_counts = (0..batch_size)
        .map(|index| {
            if mixed_shapes {
                message_counts[index % message_counts.len()]
            } else {
                message_counts[0]
            }
        })
        .collect::<Vec<_>>();
    let headers = item_message_counts
        .iter()
        .map(|&message_count| deterministic_header(message_count))
        .collect::<Vec<_>>();
    let disclosed_indexes = item_message_counts
        .iter()
        .map(|&message_count| {
            disclosed_indexes(
                message_count,
                disclosed_count(message_count, disclosed_ratio),
            )
        })
        .collect::<Vec<_>>();
    let presentation_header = b"omniid-bbs-proof-msm-stress".to_vec();
    let message_batches = (0..batch_size)
        .map(|index| deterministic_bbs_messages_for_item(item_message_counts[index], index))
        .collect::<Vec<_>>();
    let signatures = message_batches
        .iter()
        .zip(headers.iter())
        .map(|(messages, header)| {
            engine
                .sign(SignRequest {
                    secret_key: keys.secret_key.clone(),
                    public_key: keys.public_key.clone(),
                    header: header.clone(),
                    messages: messages.clone(),
                })
                .expect("proof MSM stress signature")
        })
        .collect::<Vec<_>>();
    let proof_gen_reqs = message_batches
        .iter()
        .zip(signatures.iter())
        .zip(headers.iter())
        .zip(disclosed_indexes.iter())
        .map(
            |(((messages, signature), header), disclosed_indexes)| ProofGenRequest {
                public_key: keys.public_key.clone(),
                signature: signature.clone(),
                header: header.clone(),
                presentation_header: presentation_header.clone(),
                messages: messages.clone(),
                disclosed_indexes: disclosed_indexes.clone(),
            },
        )
        .collect::<Vec<_>>();
    let proofs = engine
        .proof_gen_batch(proof_gen_reqs.clone(), true)
        .results
        .into_iter()
        .map(|item| item.result.expect("proof MSM stress proof"))
        .collect::<Vec<_>>();
    let disclosed_message_batches = message_batches
        .iter()
        .zip(disclosed_indexes.iter())
        .map(|(messages, disclosed_indexes)| {
            disclosed_indexes
                .iter()
                .map(|&index| messages[index].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let proof_verify_reqs = proofs
        .iter()
        .zip(disclosed_message_batches.iter())
        .zip(headers.iter())
        .zip(disclosed_indexes.iter())
        .zip(item_message_counts.iter())
        .map(
            |((((proof, disclosed_messages), header), disclosed_indexes), &message_count)| {
                ProofVerifyRequest {
                    public_key: keys.public_key.clone(),
                    proof: proof.clone(),
                    header: header.clone(),
                    presentation_header: presentation_header.clone(),
                    disclosed_messages: disclosed_messages.clone(),
                    disclosed_indexes: disclosed_indexes.clone(),
                    total_message_count: message_count,
                }
            },
        )
        .collect::<Vec<_>>();
    let target_proofs_per_sec = arg_value(args, "--target-proofs-per-sec")
        .or_else(|| arg_value(args, "--target-proof-per-sec"))
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(20_000.0);
    let min_gpu_route_ratio = arg_value(args, "--require-gpu-route-ratio")
        .or_else(|| arg_value(args, "--min-gpu-route-ratio"))
        .and_then(|value| value.parse::<f64>().ok());
    let min_fused_route_ratio = arg_value(args, "--require-fused-route-ratio")
        .or_else(|| arg_value(args, "--min-fused-route-ratio"))
        .and_then(|value| value.parse::<f64>().ok());
    let require_dispatch_target = args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--require-dispatch-target" | "--require-row-msm-target"
        )
    });
    let stage_terms = engine
        .proof_msm_terms_for_bench(proof_gen_reqs, proof_verify_reqs, true)
        .expect("proof MSM stress terms");
    let gpu_smi_sampler = sample_gpu_smi.then(|| start_gpu_smi_sampler(gpu_smi_interval));
    let mut stage_group_metrics = Vec::new();
    let metrics = stage_terms
        .iter()
        .map(|batch| {
            let shape =
                BbsMsmShapeMetric::from(summarize_terms_for_bench(batch.label, &batch.terms));
            let use_split = batch.label.ends_with("-t2");
            let routes = engine
                .g1_msm_route_summary_for_bench(&batch.terms, use_split, true)
                .expect("proof MSM route summary")
                .into();
            let throughput = stress_batch_operation(
                batch.label,
                duration,
                batch.terms.len(),
                streams,
                None,
                None,
                || {
                    if use_split {
                        engine.g1_msm_batch_split_for_bench(&batch.terms, true)
                    } else {
                        engine.g1_msm_batch_for_bench(&batch.terms, true)
                    }
                },
                |values| values.iter().all(|point| point.len() == 48),
            );
            let fused_target = fused_msm_target_metric(
                batch.terms.len(),
                batch_size,
                target_proofs_per_sec,
                &throughput,
            );
            let dispatch_pressure = msm_dispatch_pressure_metric(
                routes,
                batch_size,
                target_proofs_per_sec,
                &throughput,
            );
            BbsProofMsmStressMetric {
                shape,
                routes,
                throughput,
                fused_target,
                dispatch_pressure,
            }
        })
        .collect::<Vec<_>>();
    if include_stage_groups {
        for batch in &stage_terms {
            stage_group_metrics.extend(run_bbs_proof_msm_stage_group_stress(
                &engine,
                batch.label,
                &batch.terms,
                duration,
                streams,
                batch_size,
                target_proofs_per_sec,
            ));
        }
    }
    let gpu_smi = gpu_smi_sampler.map(finish_gpu_smi_sampler);
    let gpu_route_gate = min_gpu_route_ratio
        .map(|min_ratio| gpu_route_gate_report(min_ratio, &metrics, &stage_group_metrics));
    let fused_route_gate = min_fused_route_ratio
        .map(|min_ratio| fused_route_gate_report(min_ratio, &metrics, &stage_group_metrics));
    let throughput_gate = require_dispatch_target
        .then(|| throughput_gate_report(target_proofs_per_sec, &metrics, &stage_group_metrics));

    BbsProofMsmStressReport {
        tool: "cubbs-bench",
        mode: "bbs-proof-msm-stress",
        backend: backend_label(backend),
        backend_info: backend_info(backend),
        duration_secs,
        target_proofs_per_sec,
        batch_size,
        streams,
        mixed_shapes,
        message_counts,
        disclosed_ratio,
        gpu_smi,
        gpu_route_gate,
        fused_route_gate,
        throughput_gate,
        metrics,
        stage_group_metrics,
    }
}

struct BbsProofMsmStageGroup {
    label: String,
    kind: &'static str,
    terms: Vec<BbsMsmTerms>,
}

fn run_bbs_proof_msm_stage_group_stress(
    engine: &BbsEngine,
    stage: &'static str,
    terms: &[BbsMsmTerms],
    duration: Duration,
    streams: usize,
    proof_batch_size: usize,
    target_proofs_per_sec: f64,
) -> Vec<BbsProofMsmStageGroupMetric> {
    proof_msm_stage_groups_for_bench(stage, terms)
        .into_iter()
        .filter(|group| !group.terms.is_empty())
        .map(|group| {
            let operation = Box::leak(format!("{}:{}", stage, group.label).into_boxed_str());
            let routes = engine
                .g1_msm_route_summary_for_bench(&group.terms, false, true)
                .expect("proof MSM group route summary")
                .into();
            let throughput = stress_batch_operation(
                operation,
                duration,
                group.terms.len(),
                streams,
                None,
                None,
                || engine.g1_msm_batch_for_bench(&group.terms, true),
                |values| values.iter().all(|point| point.len() == 48),
            );
            let dispatch_pressure = msm_dispatch_pressure_metric(
                routes,
                proof_batch_size,
                target_proofs_per_sec,
                &throughput,
            );
            BbsProofMsmStageGroupMetric {
                stage,
                group_label: group.label,
                group_kind: group.kind,
                shape: BbsMsmShapeMetric::from(summarize_terms_for_bench(stage, &group.terms)),
                routes,
                throughput,
                dispatch_pressure,
            }
        })
        .collect()
}

fn proof_msm_stage_groups_for_bench(
    stage: &'static str,
    terms: &[BbsMsmTerms],
) -> Vec<BbsProofMsmStageGroup> {
    let mut groups = common_point_groups_for_bench(terms)
        .into_iter()
        .enumerate()
        .map(|(index, group_terms)| BbsProofMsmStageGroup {
            label: format!(
                "common-points-{}-terms{}-items{}",
                index,
                group_terms
                    .first()
                    .map(|term| term.compressed_points.len())
                    .unwrap_or_default(),
                group_terms.len()
            ),
            kind: "common-points",
            terms: group_terms,
        })
        .collect::<Vec<_>>();

    if stage.ends_with("-t2") {
        for (term_count, term_group) in terms_grouped_by_term_count_for_bench(terms) {
            if term_group.len() < 2 {
                continue;
            }
            let Some(variable_index) = single_variable_point_index_for_bench(&term_group) else {
                continue;
            };
            let Some((common_terms, variable_terms)) =
                split_variable_terms_for_bench(&term_group, variable_index)
            else {
                continue;
            };
            groups.push(BbsProofMsmStageGroup {
                label: format!(
                    "split-common-terms{}-var{}-items{}",
                    term_count,
                    variable_index,
                    common_terms.len()
                ),
                kind: "split-common",
                terms: common_terms,
            });
            groups.push(BbsProofMsmStageGroup {
                label: format!(
                    "split-variable-terms{}-var{}-items{}",
                    term_count,
                    variable_index,
                    variable_terms.len()
                ),
                kind: "split-variable",
                terms: variable_terms,
            });
        }
    }

    groups
}

fn common_point_groups_for_bench(terms: &[BbsMsmTerms]) -> Vec<Vec<BbsMsmTerms>> {
    let mut order = Vec::<Arc<Vec<Vec<u8>>>>::new();
    let mut groups = BTreeMap::<Arc<Vec<Vec<u8>>>, Vec<BbsMsmTerms>>::new();
    for term in terms {
        let key = term.compressed_points.clone();
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(term.clone());
    }
    order
        .into_iter()
        .filter_map(|key| groups.remove(&key))
        .filter(|group| group.len() >= 2)
        .collect()
}

fn terms_grouped_by_term_count_for_bench(terms: &[BbsMsmTerms]) -> Vec<(usize, Vec<BbsMsmTerms>)> {
    let mut order = Vec::<usize>::new();
    let mut groups = BTreeMap::<usize, Vec<BbsMsmTerms>>::new();
    for term in terms {
        let key = term.compressed_points.len();
        if !groups.contains_key(&key) {
            order.push(key);
        }
        groups.entry(key).or_default().push(term.clone());
    }
    order
        .into_iter()
        .map(|key| (key, groups.remove(&key).unwrap_or_default()))
        .collect()
}

fn split_variable_terms_for_bench(
    terms: &[BbsMsmTerms],
    variable_index: usize,
) -> Option<(Vec<BbsMsmTerms>, Vec<BbsMsmTerms>)> {
    let first = terms.first()?;
    if variable_index >= first.compressed_points.len()
        || first.compressed_points.len() != first.scalar_be.len()
        || first.compressed_points.len() <= 1
    {
        return None;
    }
    if terms.iter().any(|term| {
        term.compressed_points.len() != first.compressed_points.len()
            || term.scalar_be.len() != first.scalar_be.len()
    }) {
        return None;
    }

    let mut common_terms = Vec::with_capacity(terms.len());
    let mut variable_terms = Vec::with_capacity(terms.len());
    for term in terms {
        let mut common_points = Vec::with_capacity(term.compressed_points.len() - 1);
        let mut common_scalars = Vec::with_capacity(term.scalar_be.len() - 1);
        for (index, (point, scalar)) in term
            .compressed_points
            .iter()
            .zip(term.scalar_be.iter())
            .enumerate()
        {
            if index == variable_index {
                continue;
            }
            common_points.push(point.clone());
            common_scalars.push(scalar.clone());
        }
        common_terms.push(BbsMsmTerms {
            compressed_points: common_points.into(),
            scalar_be: common_scalars,
        });
        variable_terms.push(BbsMsmTerms {
            compressed_points: vec![term.compressed_points[variable_index].clone()].into(),
            scalar_be: vec![term.scalar_be[variable_index].clone()],
        });
    }
    Some((common_terms, variable_terms))
}

fn stress_batch_operation<T, F, C>(
    operation: &'static str,
    duration: Duration,
    batch_size: usize,
    streams: usize,
    signature_bytes: Option<usize>,
    proof_bytes: Option<usize>,
    f: F,
    check: C,
) -> BbsThroughputStressMetric
where
    T: Send,
    F: Fn() -> cubbs_bbs::BbsBatchRunResult<T> + Sync,
    C: Fn(&[T]) -> bool + Sync,
{
    stress_batch_operation_with_warmup(
        operation,
        duration,
        Duration::ZERO,
        batch_size,
        streams,
        signature_bytes,
        proof_bytes,
        f,
        check,
    )
}

fn stress_batch_operation_with_warmup<T, F, C>(
    operation: &'static str,
    duration: Duration,
    warmup: Duration,
    batch_size: usize,
    streams: usize,
    signature_bytes: Option<usize>,
    proof_bytes: Option<usize>,
    f: F,
    check: C,
) -> BbsThroughputStressMetric
where
    T: Send,
    F: Fn() -> cubbs_bbs::BbsBatchRunResult<T> + Sync,
    C: Fn(&[T]) -> bool + Sync,
{
    if !warmup.is_zero() {
        let warmup_started = Instant::now();
        while warmup_started.elapsed() < warmup {
            let result = f();
            let values = result
                .results
                .into_iter()
                .map(|item| item.result.expect("stress warm-up batch failed"))
                .collect::<Vec<_>>();
            assert!(check(&values), "stress warm-up result check failed");
        }
    }

    let started = Instant::now();
    let stream_results = thread::scope(|scope| {
        (0..streams)
            .map(|_| {
                let f = &f;
                let check = &check;
                scope.spawn(move || {
                    let mut batch_latencies = Vec::new();
                    let mut items = 0usize;
                    let mut failed = 0usize;

                    while started.elapsed() < duration {
                        let batch_started = Instant::now();
                        let result = f();
                        let elapsed_ms = batch_started.elapsed().as_secs_f64() * 1000.0;
                        let mut values = Vec::with_capacity(result.results.len());
                        for item in result.results {
                            match item.result {
                                Ok(value) => values.push(value),
                                Err(_) => failed += 1,
                            }
                        }
                        // Stress reports must expose partial failures in a
                        // heterogeneous batch instead of aborting the whole
                        // benchmark. This is essential for discovering
                        // unsupported mixed VC/issuer paths.
                        if !check(&values) {
                            failed += values.len();
                            values.clear();
                        }
                        items += values.len();
                        batch_latencies.push(elapsed_ms);
                    }

                    (batch_latencies, items, failed)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("stress stream panicked"))
            .collect::<Vec<_>>()
    });

    let mut batch_latencies = Vec::new();
    let mut items = 0usize;
    let mut failed = 0usize;
    for (mut latencies, stream_items, stream_failed) in stream_results {
        batch_latencies.append(&mut latencies);
        items += stream_items;
        failed += stream_failed;
    }

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let batches = batch_latencies.len();
    let item_latencies = batch_latencies
        .iter()
        .map(|latency| latency / batch_size as f64)
        .collect::<Vec<_>>();
    BbsThroughputStressMetric {
        operation,
        streams,
        elapsed_ms,
        batches,
        items,
        failed,
        items_per_sec: ops_per_sec(items, elapsed_ms),
        batches_per_sec: ops_per_sec(batches, elapsed_ms),
        batch_p50_ms: percentile(&batch_latencies, 50.0),
        batch_p95_ms: percentile(&batch_latencies, 95.0),
        batch_p99_ms: percentile(&batch_latencies, 99.0),
        batch_max_ms: batch_latencies.iter().copied().fold(0.0, f64::max),
        item_p50_ms: percentile(&item_latencies, 50.0),
        item_p95_ms: percentile(&item_latencies, 95.0),
        item_p99_ms: percentile(&item_latencies, 99.0),
        signature_bytes,
        proof_bytes,
    }
}

fn parse_mixed_bbs_dispatch_mode(args: &[String]) -> MixedBbsDispatchMode {
    match arg_value(args, "--mixed-bbs-dispatch")
        .as_deref()
        .unwrap_or("production")
    {
        "production" | "production-compatibility-aware" => MixedBbsDispatchMode::Production,
        "serial-shape-buckets" => MixedBbsDispatchMode::SerialShapeBuckets,
        other => {
            panic!("unknown --mixed-bbs-dispatch {other}; use production|serial-shape-buckets")
        }
    }
}

fn verify_pre_b_aggregate_enabled_for_bench() -> bool {
    std::env::var("CUBBS_BBS_PAIRING_AGGREGATE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn verify_pre_b_aggregate_min_items_for_bench() -> usize {
    std::env::var("CUBBS_BBS_VERIFY_AGGREGATE_MIN_ITEMS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
}

fn serial_shape_bucket_summary(message_counts: &[usize]) -> Vec<BbsSerialShapeBucket> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for &message_count in message_counts {
        *counts.entry(message_count).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(message_count, item_count)| BbsSerialShapeBucket {
            message_count,
            item_count,
        })
        .collect()
}

fn dispatch_mixed_bbs_batch<T, R, K, F>(
    mode: MixedBbsDispatchMode,
    reqs: &[R],
    shape_key: K,
    run_batch: F,
) -> BbsBatchRunResult<T>
where
    R: Clone,
    K: Fn(&R) -> usize,
    F: Fn(Vec<R>) -> BbsBatchRunResult<T>,
{
    if mode == MixedBbsDispatchMode::Production {
        return run_batch(reqs.to_vec());
    }

    let attempted = reqs.len();
    let mut groups = BTreeMap::<usize, Vec<(usize, R)>>::new();
    for (index, req) in reqs.iter().cloned().enumerate() {
        groups
            .entry(shape_key(&req))
            .or_default()
            .push((index, req));
    }

    let mut results = Vec::with_capacity(attempted);
    for (_shape, group) in groups {
        let original_indices = group.iter().map(|(index, _)| *index).collect::<Vec<_>>();
        let batch = group.into_iter().map(|(_, req)| req).collect::<Vec<_>>();
        let bucket = run_batch(batch);
        assert_eq!(
            bucket.attempted,
            original_indices.len(),
            "serial-shape dispatcher received an inconsistent bucket result"
        );
        assert_eq!(
            bucket.results.len(),
            original_indices.len(),
            "serial-shape dispatcher requires a complete bucket result"
        );
        for item in bucket.results {
            let original_index = *original_indices
                .get(item.index)
                .expect("serial-shape bucket returned an out-of-range item index");
            results.push(BbsBatchItemResult {
                index: original_index,
                result: item.result,
            });
        }
    }
    results.sort_by_key(|item| item.index);
    assert_eq!(
        results.len(),
        attempted,
        "serial-shape dispatcher did not return every request"
    );
    assert!(
        results
            .iter()
            .enumerate()
            .all(|(expected, item)| item.index == expected),
        "serial-shape dispatcher changed request order"
    );
    let succeeded = results.iter().filter(|item| item.result.is_ok()).count();
    BbsBatchRunResult {
        attempted,
        succeeded,
        failed: attempted - succeeded,
        results,
    }
}

fn fused_msm_target_metric(
    stage_item_count: usize,
    proof_batch_size: usize,
    target_proofs_per_sec: f64,
    throughput: &BbsThroughputStressMetric,
) -> BbsFusedMsmTargetMetric {
    let stage_msm_per_proof = if proof_batch_size == 0 {
        0.0
    } else {
        stage_item_count as f64 / proof_batch_size as f64
    };
    let required_msm_per_sec = target_proofs_per_sec * stage_msm_per_proof;
    let current_msm_per_sec = throughput.items_per_sec;
    let current_vs_required = if required_msm_per_sec == 0.0 {
        1.0
    } else {
        current_msm_per_sec / required_msm_per_sec
    };
    let required_speedup = if current_msm_per_sec == 0.0 {
        f64::INFINITY
    } else {
        required_msm_per_sec / current_msm_per_sec
    };
    BbsFusedMsmTargetMetric {
        target_proofs_per_sec,
        stage_msm_per_proof,
        required_msm_per_sec,
        current_msm_per_sec,
        current_vs_required,
        required_speedup,
    }
}

fn msm_dispatch_pressure_metric(
    routes: BbsMsmRouteMetric,
    proof_batch_size: usize,
    target_proofs_per_sec: f64,
    throughput: &BbsThroughputStressMetric,
) -> BbsMsmDispatchPressureMetric {
    let row_msm_per_batch = routes.total;
    let gpu_row_msm_per_batch =
        routes.fused_item + routes.fused_common_points + routes.cubbs_large + routes.cubbs_generic;
    let fused_row_msm_per_batch =
        routes.fused_item + routes.fused_common_points + routes.cubbs_large + routes.cubbs_generic;
    let gpu_route_ratio = if row_msm_per_batch == 0 {
        0.0
    } else {
        gpu_row_msm_per_batch as f64 / row_msm_per_batch as f64
    };
    let fused_route_ratio = if row_msm_per_batch == 0 {
        0.0
    } else {
        fused_row_msm_per_batch as f64 / row_msm_per_batch as f64
    };
    let row_msm_per_proof = if proof_batch_size == 0 {
        0.0
    } else {
        row_msm_per_batch as f64 / proof_batch_size as f64
    };
    let current_row_msm_per_sec = throughput.batches_per_sec * row_msm_per_batch as f64;
    let required_row_msm_per_sec = target_proofs_per_sec * row_msm_per_proof;
    let current_vs_required = if required_row_msm_per_sec == 0.0 {
        1.0
    } else {
        current_row_msm_per_sec / required_row_msm_per_sec
    };
    let required_speedup = if current_row_msm_per_sec == 0.0 {
        f64::INFINITY
    } else {
        required_row_msm_per_sec / current_row_msm_per_sec
    };
    BbsMsmDispatchPressureMetric {
        row_msm_per_proof,
        row_msm_per_batch,
        gpu_row_msm_per_batch,
        fused_row_msm_per_batch,
        gpu_route_ratio,
        fused_route_ratio,
        current_row_msm_per_sec,
        required_row_msm_per_sec,
        current_vs_required,
        required_speedup,
    }
}

fn fused_route_gate_report(
    min_fused_route_ratio: f64,
    metrics: &[BbsProofMsmStressMetric],
    stage_group_metrics: &[BbsProofMsmStageGroupMetric],
) -> BbsFusedRouteGateReport {
    let mut failing_stages = metrics
        .iter()
        .filter(|metric| metric.dispatch_pressure.fused_route_ratio < min_fused_route_ratio)
        .map(|metric| BbsFusedRouteGateFailure {
            scope: "stage",
            stage: metric.shape.label,
            group_label: None,
            fused_route_ratio: metric.dispatch_pressure.fused_route_ratio,
            fused_row_msm_per_batch: metric.dispatch_pressure.fused_row_msm_per_batch,
            row_msm_per_batch: metric.dispatch_pressure.row_msm_per_batch,
        })
        .collect::<Vec<_>>();
    failing_stages.extend(
        stage_group_metrics
            .iter()
            .filter(|metric| metric.dispatch_pressure.fused_route_ratio < min_fused_route_ratio)
            .map(|metric| BbsFusedRouteGateFailure {
                scope: "stage-group",
                stage: metric.stage,
                group_label: Some(metric.group_label.clone()),
                fused_route_ratio: metric.dispatch_pressure.fused_route_ratio,
                fused_row_msm_per_batch: metric.dispatch_pressure.fused_row_msm_per_batch,
                row_msm_per_batch: metric.dispatch_pressure.row_msm_per_batch,
            }),
    );
    BbsFusedRouteGateReport {
        min_fused_route_ratio,
        passed: failing_stages.is_empty(),
        failing_stages,
    }
}

fn gpu_route_gate_report(
    min_gpu_route_ratio: f64,
    metrics: &[BbsProofMsmStressMetric],
    stage_group_metrics: &[BbsProofMsmStageGroupMetric],
) -> BbsGpuRouteGateReport {
    let mut failing_stages = metrics
        .iter()
        .filter(|metric| metric.dispatch_pressure.gpu_route_ratio < min_gpu_route_ratio)
        .map(|metric| BbsGpuRouteGateFailure {
            scope: "stage",
            stage: metric.shape.label,
            group_label: None,
            gpu_route_ratio: metric.dispatch_pressure.gpu_route_ratio,
            gpu_row_msm_per_batch: metric.dispatch_pressure.gpu_row_msm_per_batch,
            row_msm_per_batch: metric.dispatch_pressure.row_msm_per_batch,
        })
        .collect::<Vec<_>>();
    failing_stages.extend(
        stage_group_metrics
            .iter()
            .filter(|metric| metric.dispatch_pressure.gpu_route_ratio < min_gpu_route_ratio)
            .map(|metric| BbsGpuRouteGateFailure {
                scope: "stage-group",
                stage: metric.stage,
                group_label: Some(metric.group_label.clone()),
                gpu_route_ratio: metric.dispatch_pressure.gpu_route_ratio,
                gpu_row_msm_per_batch: metric.dispatch_pressure.gpu_row_msm_per_batch,
                row_msm_per_batch: metric.dispatch_pressure.row_msm_per_batch,
            }),
    );
    BbsGpuRouteGateReport {
        min_gpu_route_ratio,
        passed: failing_stages.is_empty(),
        failing_stages,
    }
}

fn throughput_gate_report(
    target_proofs_per_sec: f64,
    metrics: &[BbsProofMsmStressMetric],
    stage_group_metrics: &[BbsProofMsmStageGroupMetric],
) -> BbsThroughputGateReport {
    let mut failing_stages = metrics
        .iter()
        .filter(|metric| metric.dispatch_pressure.current_vs_required < 1.0)
        .map(|metric| {
            throughput_gate_failure("stage", metric.shape.label, None, &metric.dispatch_pressure)
        })
        .collect::<Vec<_>>();
    failing_stages.extend(
        stage_group_metrics
            .iter()
            .filter(|metric| metric.dispatch_pressure.current_vs_required < 1.0)
            .map(|metric| {
                throughput_gate_failure(
                    "stage-group",
                    metric.stage,
                    Some(metric.group_label.clone()),
                    &metric.dispatch_pressure,
                )
            }),
    );
    BbsThroughputGateReport {
        target_proofs_per_sec,
        passed: failing_stages.is_empty(),
        failing_stages,
    }
}

fn throughput_gate_failure(
    scope: &'static str,
    stage: &'static str,
    group_label: Option<String>,
    dispatch: &BbsMsmDispatchPressureMetric,
) -> BbsThroughputGateFailure {
    BbsThroughputGateFailure {
        scope,
        stage,
        group_label,
        current_row_msm_per_sec: dispatch.current_row_msm_per_sec,
        required_row_msm_per_sec: dispatch.required_row_msm_per_sec,
        current_vs_required: dispatch.current_vs_required,
        required_speedup: dispatch.required_speedup,
    }
}

fn run_pure_bbs_bench(
    iterations: usize,
    message_counts: &[usize],
    disclosed_ratio: f64,
    backend: BbsBackend,
) -> PureBbsReport {
    let engine = BbsEngine::new(backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let mut metrics = Vec::new();

    for &message_count in message_counts {
        let header = deterministic_header(message_count);
        let messages = deterministic_bbs_messages(message_count);
        let disclosed_count = disclosed_count(message_count, disclosed_ratio);
        let disclosed_indexes = disclosed_indexes(message_count, disclosed_count);
        let disclosed_messages = disclosed_indexes
            .iter()
            .map(|&index| messages[index].clone())
            .collect::<Vec<_>>();
        let signature = engine
            .sign(SignRequest {
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                header: header.clone(),
                messages: messages.clone(),
            })
            .expect("warm signature");
        let proof = engine
            .proof_gen(ProofGenRequest {
                public_key: keys.public_key.clone(),
                signature: signature.clone(),
                header: header.clone(),
                presentation_header: b"omniid-pure-bbs-bench".to_vec(),
                messages: messages.clone(),
                disclosed_indexes: disclosed_indexes.clone(),
            })
            .expect("warm proof");

        metrics.push(measure_pure_bbs_operation(
            "sign",
            message_count,
            None,
            iterations,
            1,
            Some(signature.len()),
            None,
            || {
                engine
                    .sign(SignRequest {
                        secret_key: keys.secret_key.clone(),
                        public_key: keys.public_key.clone(),
                        header: header.clone(),
                        messages: messages.clone(),
                    })
                    .map(|_| ())
            },
        ));
        metrics.push(measure_pure_bbs_operation(
            "verify-signature",
            message_count,
            None,
            iterations,
            1,
            Some(signature.len()),
            None,
            || {
                engine
                    .verify(VerifyRequest {
                        public_key: keys.public_key.clone(),
                        signature: signature.clone(),
                        header: header.clone(),
                        messages: messages.clone(),
                    })
                    .and_then(|ok| ok.then_some(()).ok_or(cubbs_bbs::Error::InvalidSignature))
            },
        ));
        metrics.push(measure_pure_bbs_operation(
            "proof-gen",
            message_count,
            Some(disclosed_count),
            iterations,
            1,
            Some(signature.len()),
            Some(proof.len()),
            || {
                engine
                    .proof_gen(ProofGenRequest {
                        public_key: keys.public_key.clone(),
                        signature: signature.clone(),
                        header: header.clone(),
                        presentation_header: b"omniid-pure-bbs-bench".to_vec(),
                        messages: messages.clone(),
                        disclosed_indexes: disclosed_indexes.clone(),
                    })
                    .map(|_| ())
            },
        ));
        metrics.push(measure_pure_bbs_operation(
            "proof-verify",
            message_count,
            Some(disclosed_count),
            iterations,
            1,
            Some(signature.len()),
            Some(proof.len()),
            || {
                engine
                    .proof_verify(ProofVerifyRequest {
                        public_key: keys.public_key.clone(),
                        proof: proof.clone(),
                        header: header.clone(),
                        presentation_header: b"omniid-pure-bbs-bench".to_vec(),
                        disclosed_messages: disclosed_messages.clone(),
                        disclosed_indexes: disclosed_indexes.clone(),
                        total_message_count: message_count,
                    })
                    .and_then(|ok| ok.then_some(()).ok_or(cubbs_bbs::Error::InvalidProof))
            },
        ));
    }

    PureBbsReport {
        iterations,
        batch_size: 1,
        call_path: "single-item-functions",
        disclosed_ratio,
        message_counts: message_counts.to_vec(),
        metrics,
    }
}

fn run_pure_bbs_batch_bench(
    iterations: usize,
    batch_size: usize,
    message_counts: &[usize],
    disclosed_ratio: f64,
    backend: BbsBackend,
) -> PureBbsReport {
    let engine = BbsEngine::new(backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let mut metrics = Vec::new();

    for &message_count in message_counts {
        let header = deterministic_header(message_count);
        let disclosed_count = disclosed_count(message_count, disclosed_ratio);
        let disclosed_indexes = disclosed_indexes(message_count, disclosed_count);
        let presentation_header = b"omniid-pure-bbs-batch-bench".to_vec();
        let message_batches = (0..batch_size)
            .map(|batch_index| deterministic_bbs_messages_for_item(message_count, batch_index))
            .collect::<Vec<_>>();
        let signatures = message_batches
            .iter()
            .map(|messages| {
                engine
                    .sign(SignRequest {
                        secret_key: keys.secret_key.clone(),
                        public_key: keys.public_key.clone(),
                        header: header.clone(),
                        messages: messages.clone(),
                    })
                    .expect("warm signature")
            })
            .collect::<Vec<_>>();
        let proofs = message_batches
            .iter()
            .zip(signatures.iter())
            .map(|(messages, signature)| {
                engine
                    .proof_gen(ProofGenRequest {
                        public_key: keys.public_key.clone(),
                        signature: signature.clone(),
                        header: header.clone(),
                        presentation_header: presentation_header.clone(),
                        messages: messages.clone(),
                        disclosed_indexes: disclosed_indexes.clone(),
                    })
                    .expect("warm proof")
            })
            .collect::<Vec<_>>();
        let disclosed_message_batches = message_batches
            .iter()
            .map(|messages| {
                disclosed_indexes
                    .iter()
                    .map(|&index| messages[index].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        metrics.push(measure_pure_bbs_batch_operation(
            "sign-batch-api",
            message_count,
            None,
            iterations,
            batch_size,
            Some(signatures[0].len()),
            None,
            || {
                let reqs = message_batches
                    .iter()
                    .map(|messages| SignRequest {
                        secret_key: keys.secret_key.clone(),
                        public_key: keys.public_key.clone(),
                        header: header.clone(),
                        messages: messages.clone(),
                    })
                    .collect();
                engine.sign_batch(reqs, true)
            },
            |result| {
                result
                    .iter()
                    .all(|signature| signature.len() == signatures[0].len())
            },
        ));
        metrics.push(measure_pure_bbs_batch_operation(
            "verify-signature-batch-api",
            message_count,
            None,
            iterations,
            batch_size,
            Some(signatures[0].len()),
            None,
            || {
                let reqs = message_batches
                    .iter()
                    .zip(signatures.iter())
                    .map(|(messages, signature)| VerifyRequest {
                        public_key: keys.public_key.clone(),
                        signature: signature.clone(),
                        header: header.clone(),
                        messages: messages.clone(),
                    })
                    .collect();
                engine.verify_batch(reqs, true)
            },
            |result| result.iter().all(|ok| *ok),
        ));
        metrics.push(measure_pure_bbs_batch_operation(
            "proof-gen-batch-api",
            message_count,
            Some(disclosed_count),
            iterations,
            batch_size,
            Some(signatures[0].len()),
            Some(proofs[0].len()),
            || {
                let reqs = message_batches
                    .iter()
                    .zip(signatures.iter())
                    .map(|(messages, signature)| ProofGenRequest {
                        public_key: keys.public_key.clone(),
                        signature: signature.clone(),
                        header: header.clone(),
                        presentation_header: presentation_header.clone(),
                        messages: messages.clone(),
                        disclosed_indexes: disclosed_indexes.clone(),
                    })
                    .collect();
                engine.proof_gen_batch(reqs, true)
            },
            |result| result.iter().all(|proof| proof.len() == proofs[0].len()),
        ));
        metrics.push(measure_pure_bbs_batch_operation(
            "proof-verify-batch-api",
            message_count,
            Some(disclosed_count),
            iterations,
            batch_size,
            Some(signatures[0].len()),
            Some(proofs[0].len()),
            || {
                let reqs = proofs
                    .iter()
                    .zip(disclosed_message_batches.iter())
                    .map(|(proof, disclosed_messages)| ProofVerifyRequest {
                        public_key: keys.public_key.clone(),
                        proof: proof.clone(),
                        header: header.clone(),
                        presentation_header: presentation_header.clone(),
                        disclosed_messages: disclosed_messages.clone(),
                        disclosed_indexes: disclosed_indexes.clone(),
                        total_message_count: message_count,
                    })
                    .collect();
                engine.proof_verify_batch(reqs, true)
            },
            |result| result.iter().all(|ok| *ok),
        ));
    }

    PureBbsReport {
        iterations,
        batch_size,
        call_path: "bbs-batch-api",
        disclosed_ratio,
        message_counts: message_counts.to_vec(),
        metrics,
    }
}

fn measure_pure_bbs_operation<F>(
    operation: &'static str,
    message_count: usize,
    disclosed_count: Option<usize>,
    iterations: usize,
    batch_size: usize,
    signature_bytes: Option<usize>,
    proof_bytes: Option<usize>,
    mut f: F,
) -> PureBbsMetric
where
    F: FnMut() -> cubbs_bbs::Result<()>,
{
    let started = Instant::now();
    let mut latencies = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let item_started = Instant::now();
        f().expect("pure BBS operation");
        latencies.push(item_started.elapsed().as_secs_f64() * 1000.0);
    }
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    PureBbsMetric {
        operation,
        message_count,
        disclosed_count,
        iterations,
        batch_size,
        total_ms,
        ops_per_sec: ops_per_sec(iterations, total_ms),
        p50_ms: percentile(&latencies, 50.0),
        p95_ms: percentile(&latencies, 95.0),
        p99_ms: percentile(&latencies, 99.0),
        max_ms: latencies.iter().copied().fold(0.0, f64::max),
        signature_bytes,
        proof_bytes,
    }
}

fn run_pure_bbs_batch_sign_bench(
    iterations: usize,
    batch_size: usize,
    message_counts: &[usize],
    backend: BbsBackend,
) -> PureBbsReport {
    let engine = BbsEngine::new(backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let mut metrics = Vec::new();

    for &message_count in message_counts {
        let header = deterministic_header(message_count);
        let message_batches = (0..batch_size)
            .map(|batch_index| deterministic_bbs_messages_for_item(message_count, batch_index))
            .collect::<Vec<_>>();
        let warm_signature = engine
            .sign(SignRequest {
                secret_key: keys.secret_key.clone(),
                public_key: keys.public_key.clone(),
                header: header.clone(),
                messages: message_batches[0].clone(),
            })
            .expect("warm signature");

        metrics.push(measure_pure_bbs_batch_operation(
            "sign-batch-api",
            message_count,
            None,
            iterations,
            batch_size,
            Some(warm_signature.len()),
            None,
            || {
                let reqs = message_batches
                    .iter()
                    .map(|messages| SignRequest {
                        secret_key: keys.secret_key.clone(),
                        public_key: keys.public_key.clone(),
                        header: header.clone(),
                        messages: messages.clone(),
                    })
                    .collect();
                engine.sign_batch(reqs, true)
            },
            |result| {
                result
                    .iter()
                    .all(|signature| signature.len() == warm_signature.len())
            },
        ));
    }

    PureBbsReport {
        iterations,
        batch_size,
        call_path: "bbs-sign-batch-api",
        disclosed_ratio: 0.0,
        message_counts: message_counts.to_vec(),
        metrics,
    }
}

fn measure_pure_bbs_batch_operation<T, F, C>(
    operation: &'static str,
    message_count: usize,
    disclosed_count: Option<usize>,
    iterations: usize,
    batch_size: usize,
    signature_bytes: Option<usize>,
    proof_bytes: Option<usize>,
    mut f: F,
    check: C,
) -> PureBbsMetric
where
    F: FnMut() -> cubbs_bbs::BbsBatchRunResult<T>,
    C: Fn(&[T]) -> bool,
{
    let started = Instant::now();
    let mut latencies = Vec::with_capacity(iterations);
    let mut succeeded = 0;
    for _ in 0..iterations {
        let item_started = Instant::now();
        let result = f();
        let elapsed_ms = item_started.elapsed().as_secs_f64() * 1000.0;
        let values = result
            .results
            .into_iter()
            .map(|item| item.result.expect("pure BBS batch item"))
            .collect::<Vec<_>>();
        assert_eq!(values.len(), batch_size);
        assert!(check(&values));
        succeeded += values.len();
        latencies.push(elapsed_ms / values.len() as f64);
    }
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    PureBbsMetric {
        operation,
        message_count,
        disclosed_count,
        iterations,
        batch_size,
        total_ms,
        ops_per_sec: ops_per_sec(succeeded, total_ms),
        p50_ms: percentile(&latencies, 50.0),
        p95_ms: percentile(&latencies, 95.0),
        p99_ms: percentile(&latencies, 99.0),
        max_ms: latencies.iter().copied().fold(0.0, f64::max),
        signature_bytes,
        proof_bytes,
    }
}

async fn measure_batch_api<T, F, Fut>(f: F) -> TimedBatch<T>
where
    F: FnOnce() -> Fut,
    Fut: core::future::Future<Output = BatchRunResult<T>>,
{
    let started = Instant::now();
    let result = f().await;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let item_count = result.results.len().max(1);
    let amortized_item_ms = elapsed_ms / item_count as f64;
    TimedBatch {
        elapsed_ms,
        items: result
            .results
            .into_iter()
            .map(|item| ItemMeasurement {
                elapsed_ms: amortized_item_ms,
                result: item.result,
            })
            .collect(),
    }
}

fn parse_usize_list(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse().expect("usize list item"))
        .collect()
}

fn deterministic_header(seed: usize) -> Vec<u8> {
    (0..64)
        .map(|i| (seed as u8).wrapping_mul(17).wrapping_add(i as u8))
        .collect()
}

fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn deterministic_bbs_messages(count: usize) -> Vec<Vec<u8>> {
    deterministic_bbs_messages_for_item(count, 0)
}

fn deterministic_bbs_messages_for_item(count: usize, item_index: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|index| {
            format!(
                "cubbs-bbs-message-{count}-{item_index:04}-{index:04}-{}",
                (item_index + 1) * (index + 1) * 7919
            )
            .into_bytes()
        })
        .collect()
}

fn disclosed_count(message_count: usize, ratio: f64) -> usize {
    if message_count == 0 {
        return 0;
    }
    ((message_count as f64 * ratio.clamp(0.0, 1.0)).round() as usize)
        .max(1)
        .min(message_count)
}

fn disclosed_indexes(message_count: usize, disclosed_count: usize) -> Vec<usize> {
    if disclosed_count >= message_count {
        return (0..message_count).collect();
    }
    (0..disclosed_count)
        .map(|i| i * message_count / disclosed_count)
        .collect()
}

fn measure_items<I, T, F, Fut>(items: Vec<I>, workers: usize, f: F) -> TimedBatch<T>
where
    I: Send,
    T: Send,
    F: Fn(I) -> Fut + Sync,
    Fut: core::future::Future<Output = Result<T, String>>,
{
    let attempted = items.len();
    let queue = Arc::new(Mutex::new(
        items.into_iter().enumerate().collect::<VecDeque<_>>(),
    ));
    let results = Arc::new(Mutex::new(Vec::with_capacity(attempted)));
    let started = Instant::now();

    std::thread::scope(|scope| {
        for _ in 0..workers.min(attempted.max(1)) {
            let queue = Arc::clone(&queue);
            let results = Arc::clone(&results);
            let f = &f;
            scope.spawn(move || loop {
                let Some((index, item)) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                let item_started = Instant::now();
                let result = async_std::task::block_on(f(item));
                let elapsed_ms = item_started.elapsed().as_secs_f64() * 1000.0;
                results
                    .lock()
                    .unwrap()
                    .push((index, ItemMeasurement { elapsed_ms, result }));
            });
        }
    });

    let mut indexed = match Arc::try_unwrap(results) {
        Ok(results) => results.into_inner().expect("results lock"),
        Err(_) => panic!("all workers joined"),
    };
    indexed.sort_by_key(|(index, _)| *index);

    TimedBatch {
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        items: indexed.into_iter().map(|(_, item)| item).collect(),
    }
}

fn collect_success_values(batch: &TimedBatch<Value>) -> Vec<Value> {
    batch
        .items
        .iter()
        .map(|item| {
            item.result
                .as_ref()
                .expect("previous operation success")
                .clone()
        })
        .collect()
}

fn collect_success_values_from_api_batch(batch: &TimedBatch<Value>) -> Vec<Value> {
    collect_success_values(batch)
}

fn metric_from_value_batch(
    case: &str,
    operation: &'static str,
    batch_size: usize,
    workers: usize,
    batch: &TimedBatch<Value>,
) -> BenchMetric {
    metric_base(
        case,
        operation,
        batch_size,
        workers,
        batch,
        average_proof_value_chars(&batch.items),
    )
}

fn metric_from_bool_batch(
    case: &str,
    operation: &'static str,
    batch_size: usize,
    workers: usize,
    batch: &TimedBatch<bool>,
) -> BenchMetric {
    metric_base(case, operation, batch_size, workers, batch, None)
}

fn metric_from_api_value_batch(
    case: &str,
    operation: &'static str,
    batch_size: usize,
    batch: &TimedBatch<Value>,
) -> BenchMetric {
    metric_base(
        case,
        operation,
        batch_size,
        1,
        batch,
        average_proof_value_chars(&batch.items),
    )
}

fn metric_from_api_bool_batch(
    case: &str,
    operation: &'static str,
    batch_size: usize,
    batch: &TimedBatch<bool>,
) -> BenchMetric {
    metric_base(case, operation, batch_size, 1, batch, None)
}

fn metric_base<T>(
    case: &str,
    operation: &'static str,
    batch_size: usize,
    workers: usize,
    batch: &TimedBatch<T>,
    avg_proof_value_chars: Option<f64>,
) -> BenchMetric {
    let attempted = batch.items.len();
    let succeeded = batch
        .items
        .iter()
        .filter(|item| item.result.is_ok())
        .count();
    let failed = attempted - succeeded;
    let latencies = batch
        .items
        .iter()
        .map(|item| item.elapsed_ms)
        .collect::<Vec<_>>();
    BenchMetric {
        case: case.to_owned(),
        operation,
        batch_size,
        workers,
        attempted,
        succeeded,
        failed,
        total_ms: batch.elapsed_ms,
        ops_per_sec: ops_per_sec(succeeded, batch.elapsed_ms),
        p50_ms: percentile(&latencies, 50.0),
        p95_ms: percentile(&latencies, 95.0),
        p99_ms: percentile(&latencies, 99.0),
        max_ms: latencies.iter().copied().fold(0.0, f64::max),
        avg_proof_value_chars,
    }
}

fn ops_per_sec(succeeded: usize, elapsed_ms: f64) -> f64 {
    if elapsed_ms == 0.0 {
        return 0.0;
    }
    succeeded as f64 / (elapsed_ms / 1000.0)
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((percentile / 100.0) * (sorted.len() as f64 - 1.0)).ceil() as usize;
    sorted[index]
}

fn average_proof_value_chars(results: &[ItemMeasurement<Value>]) -> Option<f64> {
    let lengths = results
        .iter()
        .filter_map(|item| item.result.as_ref().ok())
        .filter_map(|document| {
            document
                .get("proof")
                .and_then(|proof| proof.get("proofValue"))
                .and_then(Value::as_str)
                .map(str::len)
        })
        .collect::<Vec<_>>();
    if lengths.is_empty() {
        return None;
    }
    Some(lengths.iter().sum::<usize>() as f64 / lengths.len() as f64)
}

fn deterministic_hmac_key(index: usize) -> Vec<u8> {
    let mut key = vec![0u8; HMAC_KEY_LEN];
    for (i, byte) in key.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_add(i as u8);
    }
    key
}

fn bench_cases(profile: BenchProfile) -> Vec<BenchCase> {
    let mut cases = vec![
        BenchCase {
            name: "tiny-flat",
            mandatory_pointers: vec!["/issuer".to_owned()],
            selective_pointers: vec!["/credentialSubject/id".to_owned()],
            template: tiny_flat,
        },
        BenchCase {
            name: "nested-profile",
            mandatory_pointers: vec!["/issuer".to_owned(), "/credentialSubject/id".to_owned()],
            selective_pointers: vec!["/credentialSubject/profile/email".to_owned()],
            template: nested_profile,
        },
        BenchCase {
            name: "array-heavy",
            mandatory_pointers: vec!["/issuer".to_owned()],
            selective_pointers: vec!["/credentialSubject/records".to_owned()],
            template: array_heavy,
        },
        BenchCase {
            name: "wide-claims-100",
            mandatory_pointers: vec!["/issuer".to_owned(), "/credentialSubject/id".to_owned()],
            selective_pointers: vec![
                "/credentialSubject/claim000".to_owned(),
                "/credentialSubject/claim050".to_owned(),
                "/credentialSubject/claim099".to_owned(),
            ],
            template: wide_claims_100,
        },
    ];
    if profile != BenchProfile::Quick {
        cases.extend([
            BenchCase {
                name: "large-array-100",
                mandatory_pointers: vec!["/issuer".to_owned()],
                selective_pointers: vec!["/credentialSubject/records".to_owned()],
                template: large_array_100,
            },
            BenchCase {
                name: "multi-subject",
                mandatory_pointers: vec!["/issuer".to_owned()],
                selective_pointers: vec!["/credentialSubject".to_owned()],
                template: multi_subject,
            },
        ]);
    }
    cases
}

fn context() -> Value {
    json!({
        "PersonCredential": "https://example.com/PersonCredential",
        "ProfileCredential": "https://example.com/ProfileCredential",
        "RecordCredential": "https://example.com/RecordCredential",
        "ClaimsCredential": "https://example.com/ClaimsCredential",
        "MultiSubjectCredential": "https://example.com/MultiSubjectCredential",
        "name": "https://schema.org/name",
        "email": "https://schema.org/email",
        "age": "https://schema.org/age",
        "profile": "https://example.com/profile",
        "department": "https://example.com/department",
        "level": "https://example.com/level",
        "records": "https://example.com/records",
        "score": "https://example.com/score",
        "tag": "https://example.com/tag"
    })
}

fn tiny_flat(index: usize) -> Value {
    json!({
        "@context": ["https://www.w3.org/2018/credentials/v1", context()],
        "id": format!("urn:cubbs-bench:tiny:{index}"),
        "type": ["VerifiableCredential", "PersonCredential"],
        "issuer": "did:example:issuer",
        "credentialSubject": {
            "id": format!("did:example:holder:{index}"),
            "name": format!("Holder {index}"),
            "age": 30 + (index % 10)
        }
    })
}

fn nested_profile(index: usize) -> Value {
    json!({
        "@context": ["https://www.w3.org/ns/credentials/v2", context()],
        "id": format!("urn:cubbs-bench:nested:{index}"),
        "type": ["VerifiableCredential", "ProfileCredential"],
        "issuer": "did:example:issuer",
        "credentialSubject": {
            "id": format!("did:example:holder:{index}"),
            "profile": {
                "name": format!("Holder {index}"),
                "email": format!("holder{index}@example.com"),
                "department": "research",
                "level": format!("L{}", index % 5)
            }
        }
    })
}

fn array_heavy(index: usize) -> Value {
    records_credential("array", index, 24)
}

fn large_array_100(index: usize) -> Value {
    records_credential("large-array", index, 100)
}

fn records_credential(name: &str, index: usize, count: usize) -> Value {
    let records = (0..count)
        .map(|record| {
            json!({
                "name": format!("record-{record}"),
                "score": ((index + record) % 100),
                "tag": format!("tag-{}", record % 4)
            })
        })
        .collect::<Vec<_>>();
    json!({
        "@context": ["https://www.w3.org/ns/credentials/v2", context()],
        "id": format!("urn:cubbs-bench:{name}:{index}"),
        "type": ["VerifiableCredential", "RecordCredential"],
        "issuer": "did:example:issuer",
        "credentialSubject": {
            "id": format!("did:example:holder:{index}"),
            "records": records
        }
    })
}

fn wide_claims_100(index: usize) -> Value {
    let mut subject = serde_json::Map::new();
    subject.insert(
        "id".to_owned(),
        Value::String(format!("did:example:holder:{index}")),
    );
    for i in 0..100 {
        subject.insert(format!("claim{i:03}"), json!(format!("value-{index}-{i}")));
    }
    let mut context = context();
    if let Value::Object(map) = &mut context {
        for i in 0..100 {
            map.insert(
                format!("claim{i:03}"),
                Value::String(format!("https://example.com/claim/{i:03}")),
            );
        }
    }
    json!({
        "@context": ["https://www.w3.org/ns/credentials/v2", context],
        "id": format!("urn:cubbs-bench:wide:{index}"),
        "type": ["VerifiableCredential", "ClaimsCredential"],
        "issuer": "did:example:issuer",
        "credentialSubject": Value::Object(subject)
    })
}

fn multi_subject(index: usize) -> Value {
    let subjects = (0..16)
        .map(|subject| {
            json!({
                "id": format!("did:example:holder:{index}:{subject}"),
                "name": format!("Subject {index}-{subject}"),
                "profile": {
                    "email": format!("subject{index}-{subject}@example.com"),
                    "department": format!("dept-{}", subject % 4),
                    "level": format!("L{}", subject % 6)
                }
            })
        })
        .collect::<Vec<_>>();
    json!({
        "@context": ["https://www.w3.org/ns/credentials/v2", context()],
        "id": format!("urn:cubbs-bench:multi-subject:{index}"),
        "type": ["VerifiableCredential", "MultiSubjectCredential"],
        "issuer": "did:example:issuer",
        "credentialSubject": subjects
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProgramProfileReport {
    tool: &'static str,
    mode: &'static str,
    iterations: usize,
    backend: &'static str,
    cases: Vec<ProgramCaseProfile>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProgramCaseProfile {
    case: String,
    issue: OperationProfile,
    derive: OperationProfile,
    verify_derived: OperationProfile,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OperationProfile {
    total_avg_ms: f64,
    stages: Vec<StageProfile>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StageProfile {
    name: &'static str,
    avg_ms: f64,
    percent_of_operation: f64,
}

#[derive(Debug, Default)]
struct StageTotals {
    values: BTreeMap<&'static str, f64>,
}

struct ProfileMaterial {
    signed: Value,
    derived: Value,
}

async fn run_program_profile(args: &[String]) -> ProgramProfileReport {
    let backend = parse_backend_arg(args);
    std::env::set_var("CUBBS_BBS_BACKEND", backend.name());
    let iterations = arg_value(args, "--iterations")
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let requested_cases = arg_value(args, "--cases")
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let cases = bench_cases(BenchProfile::Stress)
        .into_iter()
        .filter(|case| requested_cases.is_empty() || requested_cases.iter().any(|v| v == case.name))
        .collect::<Vec<_>>();
    let engine = BbsEngine::new(backend);
    let keys = engine
        .generate_keypair(&mut rand::thread_rng())
        .expect("key generation");
    let verification_method =
        verification_method_from_public_key(&keys.public_key).expect("verification method");
    let proof_options = default_bbs_2023_proof_options(verification_method, "assertionMethod");

    let mut profiles = Vec::new();
    for case in cases {
        let mut issue = StageTotals::default();
        let mut derive = StageTotals::default();
        let mut verify = StageTotals::default();

        for i in 0..iterations {
            let material = profile_one_iteration(
                &case,
                i,
                &keys.secret_key,
                &keys.public_key,
                &proof_options,
                &mut issue,
                &mut derive,
                &mut verify,
            )
            .await;
            assert!(material.signed.get("proof").is_some());
            assert!(material.derived.get("proof").is_some());
        }

        profiles.push(ProgramCaseProfile {
            case: case.name.to_owned(),
            issue: issue.operation_profile(iterations),
            derive: derive.operation_profile(iterations),
            verify_derived: verify.operation_profile(iterations),
        });
    }

    ProgramProfileReport {
        tool: "cubbs-bench",
        mode: "program-profile",
        iterations,
        backend: backend_label(backend),
        cases: profiles,
    }
}

async fn profile_one_iteration(
    case: &BenchCase,
    index: usize,
    secret_key: &[u8],
    public_key: &[u8],
    proof_options: &Value,
    issue: &mut StageTotals,
    derive: &mut StageTotals,
    verify: &mut StageTotals,
) -> ProfileMaterial {
    let document = ensure_data_integrity_context((case.template)(index));
    let hmac_key = deterministic_hmac_key(index);

    let started = Instant::now();
    let proof_config = build_proof_configuration(&document, proof_options).expect("proof config");
    issue.add("proof_config", started);

    let started = Instant::now();
    let proof_hash = hash_joined_strings(
        &canonicalize_jsonld(&proof_config)
            .await
            .expect("canonical proof config")
            .lines,
    );
    issue.add("canonicalize_proof_config", started);

    let started = Instant::now();
    let groups = BTreeMap::from([("mandatory".to_owned(), case.mandatory_pointers.clone())]);
    let grouped = canonicalize_and_group_jsonld(&document, &groups, &hmac_key)
        .await
        .expect("canonicalize and group issue document");
    let mandatory = grouped.groups["mandatory"]
        .group
        .matching
        .values()
        .cloned()
        .collect();
    let non_mandatory = grouped.groups["mandatory"]
        .group
        .non_matching
        .values()
        .cloned()
        .collect();
    issue.add("canonicalize_and_group_document", started);

    let started = Instant::now();
    let proof_value = issue_base_proof(IssueBaseProofRequest {
        secret_key: secret_key.to_vec(),
        public_key: public_key.to_vec(),
        proof_hash,
        mandatory_pointers: case.mandatory_pointers.clone(),
        mandatory,
        non_mandatory,
        hmac_key,
    })
    .expect("issue base proof");
    issue.add("bbs_sign_and_proof_value", started);

    let signed = attach_proof(document, proof_options.clone(), proof_value);
    let (unsigned, base_proof) = detach_proof(&signed);

    let started = Instant::now();
    let base_proof_value = base_proof
        .get("proofValue")
        .and_then(Value::as_str)
        .expect("base proof value")
        .to_owned();
    let parsed_base = parse_base(&base_proof_value).expect("parse base proof");
    derive.add("parse_base_proof_value", started);

    let started = Instant::now();
    let combined_pointers = parsed_base
        .mandatory_pointers
        .iter()
        .cloned()
        .chain(case.selective_pointers.iter().cloned())
        .collect::<Vec<_>>();
    let groups = BTreeMap::from([
        (
            "mandatory".to_owned(),
            parsed_base.mandatory_pointers.clone(),
        ),
        ("selective".to_owned(), case.selective_pointers.clone()),
        ("combined".to_owned(), combined_pointers),
    ]);
    let grouped = canonicalize_and_group_jsonld(&unsigned, &groups, &parsed_base.hmac_key)
        .await
        .expect("canonicalize and group derive document");
    derive.add("canonicalize_and_group_signed_document", started);

    let started = Instant::now();
    let derived_material = derive_disclosure_proof(PreparedDisclosureRequest {
        base_proof_value: base_proof_value.clone(),
        groups: DisclosureGroups {
            mandatory: grouped.groups["mandatory"].group.clone(),
            selective: grouped.groups["selective"].group.clone(),
            combined: grouped.groups["combined"].group.clone(),
            verifier_label_map: grouped.label_map,
        },
        presentation_header: format!("cubbs-profile-{}-{index}", case.name).into_bytes(),
    })
    .expect("derive disclosure proof");
    derive.add("bbs_proof_gen_and_proof_value", started);
    let derived = attach_proof(
        grouped.groups["combined"].selected_document.clone(),
        base_proof,
        derived_material.proof_value,
    );

    let (revealed_unsigned, derived_proof) = detach_proof(&derived);

    let started = Instant::now();
    let proof_config =
        build_proof_configuration(&revealed_unsigned, &derived_proof).expect("verify config");
    verify.add("proof_config", started);

    let started = Instant::now();
    let proof_hash = hash_joined_strings(
        &canonicalize_jsonld(&proof_config)
            .await
            .expect("canonical proof config")
            .lines,
    );
    verify.add("canonicalize_proof_config", started);

    let started = Instant::now();
    let proof_value = derived_proof
        .get("proofValue")
        .and_then(Value::as_str)
        .expect("derived proof value");
    let parsed = parse_derived(proof_value).expect("parse derived proof");
    verify.add("parse_derived_proof_value", started);

    let started = Instant::now();
    let canonicalized = canonicalize_jsonld(&revealed_unsigned)
        .await
        .expect("canonicalize revealed document");
    let canonical_nquads =
        relabel_and_sort_canonical_nquads(&canonicalized.lines, &parsed.label_map);
    verify.add("canonicalize_revealed_document", started);

    let started = Instant::now();
    let verification_method = derived_proof
        .get("verificationMethod")
        .and_then(Value::as_str)
        .expect("verification method");
    let public_key = public_key_from_verification_method(verification_method).expect("public key");
    let ok = verify_disclosure_proof(PreparedDisclosureVerification {
        public_key,
        proof_hash,
        canonical_nquads,
        proof_value: proof_value.to_owned(),
        expected_presentation_header: None,
    })
    .expect("verify disclosure proof");
    assert!(ok);
    verify.add("bbs_proof_verify", started);

    ProfileMaterial { signed, derived }
}

impl StageTotals {
    fn add(&mut self, name: &'static str, started: Instant) {
        *self.values.entry(name).or_insert(0.0) += started.elapsed().as_secs_f64() * 1000.0;
    }

    fn operation_profile(&self, iterations: usize) -> OperationProfile {
        let total: f64 = self.values.values().sum();
        let total_avg_ms = total / iterations as f64;
        let stages = self
            .values
            .iter()
            .map(|(&name, total_stage_ms)| StageProfile {
                name,
                avg_ms: total_stage_ms / iterations as f64,
                percent_of_operation: if total == 0.0 {
                    0.0
                } else {
                    total_stage_ms / total * 100.0
                },
            })
            .collect();
        OperationProfile {
            total_avg_ms,
            stages,
        }
    }
}

fn attach_proof(mut document: Value, mut proof: Value, proof_value: String) -> Value {
    proof
        .as_object_mut()
        .expect("proof object")
        .insert("proofValue".to_owned(), Value::String(proof_value));
    document
        .as_object_mut()
        .expect("document object")
        .insert("proof".to_owned(), proof);
    document
}

fn detach_proof(document: &Value) -> (Value, Value) {
    let mut unsigned = document.as_object().expect("document object").clone();
    let proof = unsigned.remove("proof").expect("proof");
    (Value::Object(unsigned), proof)
}

fn ensure_data_integrity_context(mut document: Value) -> Value {
    const DATA_INTEGRITY_CONTEXT_V2: &str = "https://w3id.org/security/data-integrity/v2";
    let object = document.as_object_mut().expect("document object");
    if object
        .get("@context")
        .is_some_and(|context| context_defines_data_integrity_terms(context))
    {
        return document;
    }
    match object.get_mut("@context") {
        Some(Value::Array(contexts)) => {
            if !contexts
                .iter()
                .any(|context| context.as_str() == Some(DATA_INTEGRITY_CONTEXT_V2))
            {
                contexts.push(Value::String(DATA_INTEGRITY_CONTEXT_V2.to_owned()));
            }
        }
        Some(Value::String(context)) if context == DATA_INTEGRITY_CONTEXT_V2 => {}
        Some(context) => {
            let existing = context.clone();
            *context = Value::Array(vec![
                existing,
                Value::String(DATA_INTEGRITY_CONTEXT_V2.to_owned()),
            ]);
        }
        None => {
            object.insert(
                "@context".to_owned(),
                Value::Array(vec![Value::String(DATA_INTEGRITY_CONTEXT_V2.to_owned())]),
            );
        }
    }
    document
}

fn context_defines_data_integrity_terms(context: &Value) -> bool {
    const DATA_INTEGRITY_CONTEXT_V2: &str = "https://w3id.org/security/data-integrity/v2";
    const VC_CONTEXT_V2: &str = "https://www.w3.org/ns/credentials/v2";
    match context {
        Value::String(context) => context == DATA_INTEGRITY_CONTEXT_V2 || context == VC_CONTEXT_V2,
        Value::Array(contexts) => contexts.iter().any(context_defines_data_integrity_terms),
        Value::Object(context) => {
            context.contains_key("DataIntegrityProof")
                && context.contains_key("cryptosuite")
                && context.contains_key("verificationMethod")
                && context.contains_key("proofPurpose")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn gpu_throughput_auto_uses_operation_specific_saturation_profiles() {
        for (operation, expected) in [
            ("sign", (16_384, 4)),
            ("verify-signature", (8_192, 20)),
            ("proof-gen", (4_096, 4)),
            ("proof-verify", (2_048, 10)),
        ] {
            let values = args(&["--gpu-throughput-auto", "--operation", operation]);
            assert_eq!(bbs_throughput_batch_and_streams(&values), expected);
        }
    }

    #[test]
    fn explicit_batch_and_streams_override_gpu_throughput_auto() {
        let values = args(&[
            "--gpu-throughput-auto",
            "--operation",
            "proof-gen",
            "--bbs-batch",
            "24",
            "--streams",
            "2",
        ]);
        assert_eq!(bbs_throughput_batch_and_streams(&values), (24, 2));
    }

    #[test]
    fn mixed_bbs_dispatch_defaults_to_production_and_accepts_serial_control() {
        assert_eq!(
            parse_mixed_bbs_dispatch_mode(&args(&[])),
            MixedBbsDispatchMode::Production
        );
        assert_eq!(
            parse_mixed_bbs_dispatch_mode(&args(
                &["--mixed-bbs-dispatch", "serial-shape-buckets",]
            )),
            MixedBbsDispatchMode::SerialShapeBuckets
        );
    }

    #[test]
    fn serial_shape_dispatch_preserves_order_and_uses_sorted_shape_buckets() {
        let seen = Arc::new(Mutex::new(Vec::<Vec<usize>>::new()));
        let seen_by_runner = Arc::clone(&seen);
        let result = dispatch_mixed_bbs_batch(
            MixedBbsDispatchMode::SerialShapeBuckets,
            &[64usize, 16, 64, 128, 16],
            |item| *item,
            move |batch| {
                seen_by_runner.lock().unwrap().push(batch.clone());
                BbsBatchRunResult {
                    attempted: batch.len(),
                    succeeded: batch.len(),
                    failed: 0,
                    results: batch
                        .into_iter()
                        .enumerate()
                        .map(|(index, item)| BbsBatchItemResult {
                            index,
                            result: Ok(item),
                        })
                        .collect(),
                }
            },
        );
        assert_eq!(
            *seen.lock().unwrap(),
            vec![vec![16, 16], vec![64, 64], vec![128]]
        );
        assert_eq!(
            result
                .results
                .into_iter()
                .map(|item| item.result.unwrap())
                .collect::<Vec<_>>(),
            vec![64, 16, 64, 128, 16]
        );
    }
}
