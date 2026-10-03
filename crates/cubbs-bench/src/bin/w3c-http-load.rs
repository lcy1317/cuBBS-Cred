use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug)]
struct Config {
    base_url: String,
    mode: LoadMode,
    concurrency: usize,
    rounds: usize,
    warmup_rounds: usize,
    duration: Duration,
    startup_spread: Duration,
    output: Option<PathBuf>,
    dataset: Option<PathBuf>,
    generate_dataset: Option<PathBuf>,
    dataset_size: usize,
    verify_base: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadMode {
    Barrier,
    Sustained,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    tool: &'static str,
    call_path: &'static str,
    mode: &'static str,
    base_url: String,
    concurrency: usize,
    rounds: Option<usize>,
    target_duration_seconds: Option<f64>,
    startup_spread_ms: Option<u64>,
    dataset_source: String,
    dataset_entries: usize,
    unique_vc_inputs: bool,
    drain_seconds: f64,
    total_workflows: usize,
    succeeded_workflows: usize,
    failed_workflows: usize,
    total_seconds: f64,
    workflows_per_second: f64,
    workflow_latency_p50_ms: f64,
    workflow_latency_p95_ms: f64,
    workflow_latency_p99_ms: f64,
    operations: Vec<OperationMetric>,
    backend_info: Value,
    service_config: Value,
    service_metrics_before: Value,
    service_metrics_after: Value,
    batching_before: Value,
    batching_after: Value,
    batching_delta: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OperationMetric {
    operation: &'static str,
    requests: usize,
    failed: usize,
    wall_seconds: f64,
    requests_per_second: f64,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    latency_p99_ms: f64,
}

struct PhaseResult {
    values: Vec<Value>,
    latencies_ms: Vec<f64>,
    wall_seconds: f64,
    failed: usize,
}

#[derive(Default)]
struct OperationSamples {
    latencies_ms: Vec<f64>,
    failed: usize,
}

#[derive(Default)]
struct SustainedResult {
    workflows: usize,
    succeeded: usize,
    failed: usize,
    workflow_latencies_ms: Vec<f64>,
    issue: OperationSamples,
    base_verify: OperationSamples,
    derive: OperationSamples,
    proof_verify: OperationSamples,
}

struct RequestSample {
    value: Result<Value, String>,
    latency_ms: f64,
}

#[tokio::main]
async fn main() {
    let config = Config::parse();
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return;
    }
    let (dataset, dataset_source) = load_or_generate_dataset(&config);
    let dataset = Arc::new(dataset);
    let client = Client::builder()
        .pool_max_idle_per_host(config.concurrency.max(8))
        .timeout(Duration::from_secs(120))
        .build()
        .expect("HTTP client");

    wait_until_healthy(&client, &config.base_url).await;
    for round in 0..config.warmup_rounds {
        run_round(&client, &config, &dataset, round, false).await;
    }

    let health_before = get_health(&client, &config.base_url).await;
    let report = match config.mode {
        LoadMode::Barrier => {
            run_barrier_load(
                &client,
                &config,
                Arc::clone(&dataset),
                dataset_source,
                health_before,
            )
            .await
        }
        LoadMode::Sustained => {
            run_sustained_load(
                &client,
                &config,
                Arc::clone(&dataset),
                dataset_source,
                health_before,
            )
            .await
        }
    };
    let serialized = serde_json::to_string_pretty(&report).expect("serialize report");
    if let Some(path) = &config.output {
        std::fs::write(path, &serialized)
            .unwrap_or_else(|error| panic!("write report {}: {error}", path.display()));
    }
    println!("{serialized}");
}

async fn run_barrier_load(
    client: &Client,
    config: &Config,
    dataset: Arc<Vec<Value>>,
    dataset_source: String,
    health_before: Value,
) -> Report {
    let started = Instant::now();
    let mut issue_latencies = Vec::new();
    let mut base_verify_latencies = Vec::new();
    let mut derive_latencies = Vec::new();
    let mut verify_latencies = Vec::new();
    let mut issue_wall = 0.0;
    let mut base_verify_wall = 0.0;
    let mut derive_wall = 0.0;
    let mut verify_wall = 0.0;
    let mut issue_failed = 0;
    let mut base_verify_failed = 0;
    let mut derive_failed = 0;
    let mut verify_failed = 0;

    for round in 0..config.rounds {
        let dataset_round = config.warmup_rounds + round;
        let (issue, base_verify, derive, verify) =
            run_round(client, config, &dataset, dataset_round, true).await;
        issue_latencies.extend(issue.latencies_ms);
        base_verify_latencies.extend(base_verify.latencies_ms);
        derive_latencies.extend(derive.latencies_ms);
        verify_latencies.extend(verify.latencies_ms);
        issue_wall += issue.wall_seconds;
        base_verify_wall += base_verify.wall_seconds;
        derive_wall += derive.wall_seconds;
        verify_wall += verify.wall_seconds;
        issue_failed += issue.failed;
        base_verify_failed += base_verify.failed;
        derive_failed += derive.failed;
        verify_failed += verify.failed;
    }

    let total_seconds = started.elapsed().as_secs_f64();
    let health_after = get_health(client, &config.base_url).await;
    let service_config = service_config(&health_after);
    let service_metrics_before = service_metrics(&health_before);
    let service_metrics_after = service_metrics(&health_after);
    let mut batching_before = health_before["batching"].clone();
    let mut batching_after = health_after["batching"].clone();
    let total_workflows = config.concurrency * config.rounds;
    let batching_delta = batching_delta(&batching_before, &batching_after);
    let operation_counts = [total_workflows; 4];
    validate_batching_delta(
        &batching_delta,
        &batching_after,
        operation_counts,
        config.concurrency,
    );
    strip_batch_histograms(&mut batching_before);
    strip_batch_histograms(&mut batching_after);
    Report {
        tool: "w3c-http-load",
        call_path: "phase-barrier-single-request-http-buffered-batch",
        mode: "barrier",
        base_url: config.base_url.clone(),
        rounds: Some(config.rounds),
        target_duration_seconds: None,
        startup_spread_ms: None,
        dataset_source,
        dataset_entries: dataset.len(),
        unique_vc_inputs: true,
        drain_seconds: 0.0,
        concurrency: config.concurrency,
        total_workflows,
        succeeded_workflows: total_workflows,
        failed_workflows: 0,
        total_seconds,
        workflows_per_second: total_workflows as f64 / total_seconds.max(f64::EPSILON),
        workflow_latency_p50_ms: 0.0,
        workflow_latency_p95_ms: 0.0,
        workflow_latency_p99_ms: 0.0,
        operations: vec![
            metric("issue", issue_latencies, issue_wall, issue_failed),
            metric(
                "verify-signature",
                base_verify_latencies,
                base_verify_wall,
                base_verify_failed,
            ),
            metric("derive", derive_latencies, derive_wall, derive_failed),
            metric(
                "verify-derived",
                verify_latencies,
                verify_wall,
                verify_failed,
            ),
        ],
        backend_info: backend_info(&health_after),
        service_config,
        service_metrics_before,
        service_metrics_after,
        batching_delta,
        batching_before,
        batching_after,
    }
}

impl Config {
    fn parse() -> Self {
        let args = std::env::args().collect::<Vec<_>>();
        let mode = match arg_value(&args, "--mode").as_deref() {
            Some("barrier") => LoadMode::Barrier,
            Some("sustained") => LoadMode::Sustained,
            None if arg_value(&args, "--duration-secs").is_some() => LoadMode::Sustained,
            None => LoadMode::Barrier,
            Some(value) => panic!("unsupported --mode {value}; use barrier or sustained"),
        };
        Self {
            base_url: arg_value(&args, "--base-url")
                .unwrap_or_else(|| "http://127.0.0.1:40443/id".to_owned())
                .trim_end_matches('/')
                .to_owned(),
            mode,
            concurrency: parse_usize(&args, "--concurrency", 24).max(1),
            rounds: parse_usize(&args, "--rounds", 4).max(1),
            warmup_rounds: parse_usize(&args, "--warmup-rounds", 1),
            duration: Duration::from_secs_f64(parse_f64(&args, "--duration-secs", 10.0).max(0.001)),
            startup_spread: Duration::from_millis(
                parse_usize(&args, "--startup-spread-ms", 250) as u64
            ),
            output: arg_value(&args, "--output").map(PathBuf::from),
            dataset: arg_value(&args, "--dataset").map(PathBuf::from),
            generate_dataset: arg_value(&args, "--generate-dataset").map(PathBuf::from),
            dataset_size: parse_usize(&args, "--dataset-size", 16_384).max(1),
            verify_base: !args.iter().any(|arg| arg == "--skip-base-verify"),
        }
    }
}

async fn run_sustained_load(
    client: &Client,
    config: &Config,
    dataset: Arc<Vec<Value>>,
    dataset_source: String,
    health_before: Value,
) -> Report {
    let started = Instant::now();
    let deadline = started + config.duration;
    let verify_base = config.verify_base;
    let next_workflow = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::with_capacity(config.concurrency);
    for worker in 0..config.concurrency {
        let client = client.clone();
        let base_url = config.base_url.clone();
        let next_workflow = Arc::clone(&next_workflow);
        let dataset = Arc::clone(&dataset);
        let dataset_offset = config.warmup_rounds * config.concurrency;
        let startup_delay = startup_delay(config.startup_spread, worker, config.concurrency);
        tasks.push(tokio::spawn(async move {
            if !startup_delay.is_zero() {
                tokio::time::sleep(startup_delay).await;
            }
            let mut result = SustainedResult::default();
            while Instant::now() < deadline {
                let workflow_id = next_workflow.fetch_add(1, Ordering::Relaxed);
                let dataset_index = dataset_offset + workflow_id;
                let Some(issue_payload) = dataset.get(dataset_index).cloned() else {
                    break;
                };
                run_single_workflow(
                    &client,
                    &base_url,
                    workflow_id,
                    issue_payload,
                    verify_base,
                    &mut result,
                )
                .await;
            }
            result
        }));
    }

    let mut combined = SustainedResult::default();
    for task in tasks {
        merge_sustained_result(
            &mut combined,
            task.await.expect("sustained workflow worker"),
        );
    }
    let total_seconds = started.elapsed().as_secs_f64();
    let drain_seconds = (total_seconds - config.duration.as_secs_f64()).max(0.0);
    assert!(
        started.elapsed() >= config.duration,
        "VC dataset exhausted before target duration; increase --dataset-size or provide a larger --dataset"
    );
    let health_after = get_health(client, &config.base_url).await;
    let service_config = service_config(&health_after);
    let service_metrics_before = service_metrics(&health_before);
    let service_metrics_after = service_metrics(&health_after);
    let mut batching_before = health_before["batching"].clone();
    let mut batching_after = health_after["batching"].clone();
    let batching_delta = batching_delta(&batching_before, &batching_after);
    let operation_counts = [
        combined.issue.latencies_ms.len(),
        if config.verify_base {
            combined.base_verify.latencies_ms.len()
        } else {
            0
        },
        combined.derive.latencies_ms.len(),
        combined.proof_verify.latencies_ms.len(),
    ];
    validate_batching_delta(
        &batching_delta,
        &batching_after,
        operation_counts,
        config.concurrency,
    );
    strip_batch_histograms(&mut batching_before);
    strip_batch_histograms(&mut batching_after);

    combined.workflow_latencies_ms.sort_by(f64::total_cmp);
    let mut operations = vec![sustained_metric("issue", combined.issue, total_seconds)];
    if config.verify_base {
        operations.push(sustained_metric(
            "verify-signature",
            combined.base_verify,
            total_seconds,
        ));
    }
    operations.push(sustained_metric("derive", combined.derive, total_seconds));
    operations.push(sustained_metric(
        "verify-derived",
        combined.proof_verify,
        total_seconds,
    ));
    Report {
        tool: "w3c-http-load",
        call_path: if config.verify_base {
            "sustained-independent-four-step-workflows-single-request-http-buffered-batch"
        } else {
            "sustained-independent-three-step-workflows-single-request-http-buffered-batch"
        },
        mode: "sustained",
        base_url: config.base_url.clone(),
        concurrency: config.concurrency,
        rounds: None,
        target_duration_seconds: Some(config.duration.as_secs_f64()),
        startup_spread_ms: Some(config.startup_spread.as_millis().min(u64::MAX as u128) as u64),
        dataset_source,
        dataset_entries: dataset.len(),
        unique_vc_inputs: true,
        drain_seconds,
        total_workflows: combined.workflows,
        succeeded_workflows: combined.succeeded,
        failed_workflows: combined.failed,
        total_seconds,
        workflows_per_second: combined.workflows as f64 / total_seconds.max(f64::EPSILON),
        workflow_latency_p50_ms: percentile(&combined.workflow_latencies_ms, 0.50),
        workflow_latency_p95_ms: percentile(&combined.workflow_latencies_ms, 0.95),
        workflow_latency_p99_ms: percentile(&combined.workflow_latencies_ms, 0.99),
        operations,
        backend_info: backend_info(&health_after),
        service_config,
        service_metrics_before,
        service_metrics_after,
        batching_delta,
        batching_before,
        batching_after,
    }
}

async fn run_single_workflow(
    client: &Client,
    base_url: &str,
    workflow_id: usize,
    issue_payload: Value,
    verify_base: bool,
    result: &mut SustainedResult,
) {
    result.workflows += 1;
    let started = Instant::now();
    let issue = post_one(
        client,
        format!("{base_url}/credentials/issue"),
        issue_payload,
    )
    .await;
    record_request(&mut result.issue, &issue);
    let credential = match issue.value {
        Ok(value) => value,
        Err(error) => {
            eprintln!("workflow {workflow_id} issue failed: {error}");
            finish_failed_workflow(result, started);
            return;
        }
    };

    if verify_base {
        let base_verify = post_one(
            client,
            format!("{base_url}/credentials/verify"),
            json!({
                "verifiableCredential": &credential,
                "options": {}
            }),
        )
        .await;
        record_request(&mut result.base_verify, &base_verify);
        match base_verify.value {
            Ok(value) if value["verified"] == Value::Bool(true) => {}
            Ok(_) => {
                eprintln!("workflow {workflow_id} base verification returned false");
                finish_failed_workflow(result, started);
                return;
            }
            Err(error) => {
                eprintln!("workflow {workflow_id} base verification failed: {error}");
                finish_failed_workflow(result, started);
                return;
            }
        }
    }

    let derive = post_one(
        client,
        format!("{base_url}/credentials/derive"),
        json!({
            "verifiableCredential": &credential,
            "options": {
                "selectivePointers": ["/credentialSubject/name"],
                "presentationHeader": [111, 109, 110, 105, 105, 100]
            }
        }),
    )
    .await;
    record_request(&mut result.derive, &derive);
    let derived = match derive.value {
        Ok(value) => value,
        Err(error) => {
            eprintln!("workflow {workflow_id} derive failed: {error}");
            finish_failed_workflow(result, started);
            return;
        }
    };

    let proof_verify = post_one(
        client,
        format!("{base_url}/credentials/verify"),
        json!({
            "verifiableCredential": derived,
            "options": {}
        }),
    )
    .await;
    record_request(&mut result.proof_verify, &proof_verify);
    match proof_verify.value {
        Ok(value) if value["verified"] == Value::Bool(true) => {
            result.succeeded += 1;
            result
                .workflow_latencies_ms
                .push(started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(_) => {
            eprintln!("workflow {workflow_id} derived verification returned false");
            finish_failed_workflow(result, started);
        }
        Err(error) => {
            eprintln!("workflow {workflow_id} derived verification failed: {error}");
            finish_failed_workflow(result, started);
        }
    }
}

async fn post_one(client: &Client, url: String, payload: Value) -> RequestSample {
    let started = Instant::now();
    let value = async {
        let response = client
            .post(url)
            .json(&payload)
            .send()
            .await
            .map_err(|error| error.to_string())?;
        let status = response.status();
        let body = response.text().await.map_err(|error| error.to_string())?;
        if !status.is_success() {
            return Err(format!("HTTP {status}: {body}"));
        }
        serde_json::from_str(&body).map_err(|error| error.to_string())
    }
    .await;
    RequestSample {
        value,
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn record_request(samples: &mut OperationSamples, sample: &RequestSample) {
    samples.latencies_ms.push(sample.latency_ms);
    if sample.value.is_err() {
        samples.failed += 1;
    }
}

fn finish_failed_workflow(result: &mut SustainedResult, started: Instant) {
    result.failed += 1;
    result
        .workflow_latencies_ms
        .push(started.elapsed().as_secs_f64() * 1000.0);
}

fn merge_sustained_result(combined: &mut SustainedResult, mut worker: SustainedResult) {
    combined.workflows += worker.workflows;
    combined.succeeded += worker.succeeded;
    combined.failed += worker.failed;
    combined
        .workflow_latencies_ms
        .append(&mut worker.workflow_latencies_ms);
    merge_operation_samples(&mut combined.issue, &mut worker.issue);
    merge_operation_samples(&mut combined.base_verify, &mut worker.base_verify);
    merge_operation_samples(&mut combined.derive, &mut worker.derive);
    merge_operation_samples(&mut combined.proof_verify, &mut worker.proof_verify);
}

fn merge_operation_samples(combined: &mut OperationSamples, worker: &mut OperationSamples) {
    combined.latencies_ms.append(&mut worker.latencies_ms);
    combined.failed += worker.failed;
}

fn sustained_metric(
    operation: &'static str,
    mut samples: OperationSamples,
    total_seconds: f64,
) -> OperationMetric {
    samples.latencies_ms.sort_by(f64::total_cmp);
    let requests = samples.latencies_ms.len();
    OperationMetric {
        operation,
        requests,
        failed: samples.failed,
        wall_seconds: total_seconds,
        requests_per_second: requests as f64 / total_seconds.max(f64::EPSILON),
        latency_p50_ms: percentile(&samples.latencies_ms, 0.50),
        latency_p95_ms: percentile(&samples.latencies_ms, 0.95),
        latency_p99_ms: percentile(&samples.latencies_ms, 0.99),
    }
}

fn startup_delay(spread: Duration, worker: usize, concurrency: usize) -> Duration {
    if concurrency <= 1 {
        return Duration::ZERO;
    }
    spread.mul_f64(worker as f64 / (concurrency - 1) as f64)
}

async fn run_round(
    client: &Client,
    config: &Config,
    dataset: &[Value],
    round: usize,
    require_success: bool,
) -> (PhaseResult, PhaseResult, PhaseResult, PhaseResult) {
    let issue_payloads = (0..config.concurrency)
        .map(|index| {
            let dataset_index = round * config.concurrency + index;
            dataset.get(dataset_index).cloned().unwrap_or_else(|| {
                panic!(
                    "VC dataset exhausted at entry {dataset_index}; need at least {} entries",
                    (config.warmup_rounds + config.rounds) * config.concurrency
                )
            })
        })
        .collect();
    let issue = post_many(
        client,
        format!("{}/credentials/issue", config.base_url),
        issue_payloads,
    )
    .await;
    if require_success {
        assert_eq!(issue.failed, 0, "issue phase returned failures");
    }

    let base_verify_payloads = issue
        .values
        .iter()
        .map(|credential| {
            json!({
                "verifiableCredential": credential,
                "options": {}
            })
        })
        .collect();
    let base_verify = post_many(
        client,
        format!("{}/credentials/verify", config.base_url),
        base_verify_payloads,
    )
    .await;
    if require_success {
        assert_eq!(
            base_verify.failed, 0,
            "base verification phase returned failures"
        );
        assert!(
            base_verify
                .values
                .iter()
                .all(|value| value["verified"] == Value::Bool(true)),
            "base verification response was not true"
        );
    }

    let derive_payloads = issue
        .values
        .iter()
        .map(|credential| {
            json!({
                "verifiableCredential": credential,
                "options": {
                    "selectivePointers": ["/credentialSubject/name"],
                    "presentationHeader": [111, 109, 110, 105, 105, 100]
                }
            })
        })
        .collect();
    let derive = post_many(
        client,
        format!("{}/credentials/derive", config.base_url),
        derive_payloads,
    )
    .await;
    if require_success {
        assert_eq!(derive.failed, 0, "derive phase returned failures");
    }

    let verify_payloads = derive
        .values
        .iter()
        .map(|credential| {
            json!({
                "verifiableCredential": credential,
                "options": {}
            })
        })
        .collect();
    let verify = post_many(
        client,
        format!("{}/credentials/verify", config.base_url),
        verify_payloads,
    )
    .await;
    if require_success {
        assert_eq!(verify.failed, 0, "verify phase returned failures");
        assert!(
            verify
                .values
                .iter()
                .all(|value| value["verified"] == Value::Bool(true)),
            "verification response was not true"
        );
    }
    (issue, base_verify, derive, verify)
}

async fn post_many(client: &Client, url: String, payloads: Vec<Value>) -> PhaseResult {
    let started = Instant::now();
    let mut tasks = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let client = client.clone();
        let url = url.clone();
        tasks.push(tokio::spawn(async move {
            let request_started = Instant::now();
            let response = client
                .post(url)
                .json(&payload)
                .send()
                .await
                .map_err(|error| error.to_string())?;
            let status = response.status();
            let body = response.text().await.map_err(|error| error.to_string())?;
            if !status.is_success() {
                return Err(format!("HTTP {status}: {body}"));
            }
            let value = serde_json::from_str(&body).map_err(|error| error.to_string())?;
            Ok((value, request_started.elapsed().as_secs_f64() * 1000.0))
        }));
    }

    let mut values = Vec::new();
    let mut latencies_ms = Vec::new();
    let mut failed = 0;
    for task in tasks {
        match task.await {
            Ok(Ok((value, latency_ms))) => {
                values.push(value);
                latencies_ms.push(latency_ms);
            }
            Ok(Err(error)) => {
                failed += 1;
                eprintln!("{error}");
            }
            Err(error) => {
                failed += 1;
                eprintln!("request task failed: {error}");
            }
        }
    }
    PhaseResult {
        values,
        latencies_ms,
        wall_seconds: started.elapsed().as_secs_f64(),
        failed,
    }
}

fn generate_issue_payload(index: usize) -> Value {
    let cohort = index % 17;
    let region = ["apac", "emea", "amer"][index % 3];
    json!({
        "credential": {
            "@context": [
                "https://www.w3.org/ns/credentials/v2",
                {
                    "ExampleCredential": "https://example.com/ExampleCredential",
                    "name": "https://schema.org/name",
                    "sequence": "https://schema.org/identifier",
                    "cohort": "https://example.com/cohort",
                    "region": "https://example.com/region",
                    "active": "https://schema.org/Boolean"
                }
            ],
            "id": format!("urn:uuid:cubbs-http-dataset-{index}"),
            "type": ["VerifiableCredential", "ExampleCredential"],
            "issuer": "did:example:placeholder",
            "credentialSubject": {
                "id": format!("did:example:subject-{index}"),
                "name": format!("Dataset Holder {index}"),
                "sequence": index,
                "cohort": cohort,
                "region": region,
                "active": index % 5 != 0
            }
        },
        "options": {
            "mandatoryPointers": ["/issuer"]
        }
    })
}

fn load_or_generate_dataset(config: &Config) -> (Vec<Value>, String) {
    assert!(
        config.dataset.is_none() || config.generate_dataset.is_none(),
        "use only one of --dataset and --generate-dataset"
    );
    if let Some(path) = &config.dataset {
        let entries = read_jsonl_dataset(path);
        validate_unique_dataset(&entries);
        return (entries, path.display().to_string());
    }

    let entries = (0..config.dataset_size)
        .map(generate_issue_payload)
        .collect::<Vec<_>>();
    validate_unique_dataset(&entries);
    if let Some(path) = &config.generate_dataset {
        write_jsonl_dataset(path, &entries);
        (entries, path.display().to_string())
    } else {
        (
            entries,
            format!("generated-in-memory:{}", config.dataset_size),
        )
    }
}

fn validate_unique_dataset(entries: &[Value]) {
    let mut ids = HashSet::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let id = entry["credential"]["id"]
            .as_str()
            .unwrap_or_else(|| panic!("dataset entry {index} is missing credential.id"));
        assert!(
            ids.insert(id.to_owned()),
            "dataset credential.id is not unique: {id}"
        );
    }
}

fn read_jsonl_dataset(path: &PathBuf) -> Vec<Value> {
    let file =
        File::open(path).unwrap_or_else(|error| panic!("open dataset {}: {error}", path.display()));
    let entries = BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line =
                line.unwrap_or_else(|error| panic!("read dataset {}: {error}", path.display()));
            (!line.trim().is_empty()).then(|| {
                serde_json::from_str(&line).unwrap_or_else(|error| {
                    panic!(
                        "parse dataset {} line {}: {error}",
                        path.display(),
                        index + 1
                    )
                })
            })
        })
        .collect::<Vec<_>>();
    assert!(!entries.is_empty(), "dataset {} is empty", path.display());
    entries
}

fn write_jsonl_dataset(path: &PathBuf, entries: &[Value]) {
    let file = File::create(path)
        .unwrap_or_else(|error| panic!("create dataset {}: {error}", path.display()));
    let mut writer = BufWriter::new(file);
    for entry in entries {
        serde_json::to_writer(&mut writer, entry)
            .unwrap_or_else(|error| panic!("serialize dataset {}: {error}", path.display()));
        writer
            .write_all(b"\n")
            .unwrap_or_else(|error| panic!("write dataset {}: {error}", path.display()));
    }
    writer
        .flush()
        .unwrap_or_else(|error| panic!("flush dataset {}: {error}", path.display()));
}

fn metric(
    operation: &'static str,
    mut latencies_ms: Vec<f64>,
    wall_seconds: f64,
    failed: usize,
) -> OperationMetric {
    latencies_ms.sort_by(f64::total_cmp);
    let requests = latencies_ms.len() + failed;
    OperationMetric {
        operation,
        requests,
        failed,
        wall_seconds,
        requests_per_second: requests as f64 / wall_seconds.max(f64::EPSILON),
        latency_p50_ms: percentile(&latencies_ms, 0.50),
        latency_p95_ms: percentile(&latencies_ms, 0.95),
        latency_p99_ms: percentile(&latencies_ms, 0.99),
    }
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values[index.min(values.len() - 1)]
}

async fn wait_until_healthy(client: &Client, base_url: &str) {
    for _ in 0..100 {
        if client
            .get(format!("{base_url}/healthz"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("W3C service did not become healthy at {base_url}");
}

async fn get_health(client: &Client, base_url: &str) -> Value {
    client
        .get(format!("{base_url}/healthz"))
        .send()
        .await
        .expect("health request")
        .error_for_status()
        .expect("healthy status")
        .json()
        .await
        .expect("health JSON")
}

fn batching_delta(before: &Value, after: &Value) -> Value {
    let mut queues = serde_json::Map::new();
    for name in ["issue", "verifySignature", "derive", "proofVerify"] {
        let before_queue = &before[name];
        let after_queue = &after[name];
        let dispatches = delta_u64(before_queue, after_queue, "dispatches");
        let items = delta_u64(before_queue, after_queue, "dispatchedItems");
        let queue_wait_ms = delta_f64(before_queue, after_queue, "totalQueueWaitMs");
        let batch_size_counts = batch_size_counts_delta(before_queue, after_queue);
        let max_observed_batch = batch_size_counts
            .iter()
            .rev()
            .find_map(|(size, count)| (*count > 0).then_some(*size))
            .unwrap_or(0);
        let observed_batch_size_counts = batch_size_counts
            .into_iter()
            .filter(|(_, count)| *count > 0)
            .map(|(size, count)| (size.to_string(), Value::from(count)))
            .collect::<serde_json::Map<_, _>>();
        queues.insert(
            name.to_owned(),
            json!({
                "dispatches": dispatches,
                "dispatchedItems": items,
                "averageObservedBatch": if dispatches == 0 {
                    0.0
                } else {
                    items as f64 / dispatches as f64
                },
                "maxObservedBatch": max_observed_batch,
                "maxObservedBatchSinceServiceStart": after_queue["maxObservedBatch"],
                "observedBatchSizeCounts": observed_batch_size_counts,
                "lastBatchSize": after_queue["lastBatchSize"],
                "averageQueueWaitMs": if items == 0 {
                    0.0
                } else {
                    queue_wait_ms / items as f64
                }
            }),
        );
    }
    Value::Object(queues)
}

fn batch_size_counts_delta(before: &Value, after: &Value) -> Vec<(usize, u64)> {
    let before = before["batchSizeCounts"].as_array();
    let after = after["batchSizeCounts"].as_array();
    let len = after.map_or(0, Vec::len);
    (1..len)
        .map(|size| {
            let before_count = before
                .and_then(|counts| counts.get(size))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let after_count = after
                .and_then(|counts| counts.get(size))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            (size, after_count.saturating_sub(before_count))
        })
        .collect()
}

fn strip_batch_histograms(batching: &mut Value) {
    for name in ["issue", "verifySignature", "derive", "proofVerify"] {
        if let Some(queue) = batching[name].as_object_mut() {
            queue.remove("batchSizeCounts");
        }
    }
}

fn validate_batching_delta(
    delta: &Value,
    batching: &Value,
    expected_per_operation: [usize; 4],
    concurrency: usize,
) {
    for (name, expected) in ["issue", "verifySignature", "derive", "proofVerify"]
        .into_iter()
        .zip(expected_per_operation)
    {
        assert_eq!(
            delta[name]["dispatchedItems"].as_u64(),
            Some(expected as u64),
            "{name} queue did not account for every submitted request"
        );
        let interval_ms = batching["config"]["intervalMs"].as_u64().unwrap_or(0);
        if interval_ms > 0 && concurrency > 1 && expected > 1 {
            assert!(
                delta[name]["averageObservedBatch"].as_f64().unwrap_or(0.0) > 1.0,
                "{name} queue did not coalesce concurrent HTTP requests"
            );
        }
    }
}

fn delta_u64(before: &Value, after: &Value, field: &str) -> u64 {
    after[field]
        .as_u64()
        .unwrap_or(0)
        .saturating_sub(before[field].as_u64().unwrap_or(0))
}

fn delta_f64(before: &Value, after: &Value, field: &str) -> f64 {
    (after[field].as_f64().unwrap_or(0.0) - before[field].as_f64().unwrap_or(0.0)).max(0.0)
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn parse_usize(args: &[String], flag: &str, default: usize) -> usize {
    arg_value(args, flag)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_f64(args: &[String], flag: &str, default: f64) -> f64 {
    arg_value(args, flag)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn backend_info(health: &Value) -> Value {
    json!({
        "requested": health["bbs_backend_requested"],
        "active": health["bbs_backend_active"],
        "hardwareAcceleration": health["bbs_backend_hardware_acceleration"],
        "fallback": health["bbs_backend_fallback"],
        "gpuAbiVersion": health["bbs_backend_gpu_abi_version"]
    })
}

fn service_config(health: &Value) -> Value {
    json!({
        "batching": health["batching"]["config"],
        "w3cRuntime": health["w3c_runtime"]
    })
}

fn service_metrics(health: &Value) -> Value {
    json!({
        "canonicalizationCache": health["canonicalization_cache"],
        "verificationResultCache": health["verification_result_cache"],
        "w3cBatchTiming": health["w3c_batch_timing"]
    })
}

fn print_help() {
    println!(
        "\
w3c-http-load

Modes:
  --mode barrier
      Compatibility mode. Sends all issue requests, waits, then all base
      verifies, derives, and derived-proof verifies.

  --mode sustained
      Saturation mode. Runs independent complete workflows continuously so
      issue, verify, derive, and proof-verify traffic can overlap.

Common options:
  --base-url URL               default: http://127.0.0.1:40443/id
  --concurrency N              in-flight HTTP workflows; default: 24
  --warmup-rounds N            unmeasured phase-barrier warm-up; default: 1
  --output PATH                also write the JSON report to PATH
  --dataset PATH               read unique issue-request payloads from JSONL
  --generate-dataset PATH      generate JSONL locally, then use it for this run
  --dataset-size N             generated unique VC count; default: 16384
  --skip-base-verify           sustained mode: issue, derive and verify-derived
                               only, for three-step reference comparisons

Barrier options:
  --rounds N                   measured rounds; default: 4

Sustained options:
  --duration-secs N            target launch duration; default: 10
  --startup-spread-ms N        spread worker starts to avoid phase lockstep;
                               default: 250

Supplying --duration-secs without --mode selects sustained mode.
The sustained report includes final drain time, workflow latency, per-operation
latency/throughput, backend identity, and exact /healthz batching deltas."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_nearest_rank_ceiling() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.50), 3.0);
        assert_eq!(percentile(&values, 0.95), 5.0);
        assert_eq!(percentile(&[], 0.99), 0.0);
    }

    #[test]
    fn startup_delay_spreads_workers_across_window() {
        let spread = Duration::from_millis(500);
        assert_eq!(startup_delay(spread, 0, 3), Duration::ZERO);
        assert_eq!(startup_delay(spread, 1, 3), Duration::from_millis(250));
        assert_eq!(startup_delay(spread, 2, 3), spread);
        assert_eq!(startup_delay(spread, 0, 1), Duration::ZERO);
    }

    #[test]
    fn batch_histogram_delta_is_run_local() {
        let before = json!({"batchSizeCounts": [0, 3, 2, 0, 1]});
        let after = json!({"batchSizeCounts": [0, 4, 5, 0, 1]});
        assert_eq!(
            batch_size_counts_delta(&before, &after),
            vec![(1, 1), (2, 3), (3, 0), (4, 0)]
        );
    }
}
