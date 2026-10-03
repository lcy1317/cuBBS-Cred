use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use bytes::Bytes;
use reqwest::{header::CONTENT_TYPE, Client, Request};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug)]
enum Expect {
    Document,
    Verified,
    BatchDocument,
    BatchVerified,
}

#[derive(Deserialize)]
struct DocumentResponse {
    proof: ProofResponse,
}

#[derive(Deserialize)]
struct ProofResponse {
    #[serde(rename = "proofValue")]
    proof_value: String,
}

#[derive(Deserialize)]
struct VerifiedResponse {
    verified: bool,
}

#[derive(Debug)]
struct Config {
    operation: String,
    url: String,
    health_url: Option<String>,
    payload: PathBuf,
    expect: Expect,
    concurrency: usize,
    warmup: Duration,
    duration: Duration,
    output: Option<PathBuf>,
}

#[derive(Default)]
struct Samples {
    http_requests: usize,
    succeeded: usize,
    failed: usize,
    latency_us: Vec<u64>,
    first_error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    tool: &'static str,
    contract: &'static str,
    operation: String,
    url: String,
    payload_bytes: usize,
    concurrency: usize,
    target_duration_seconds: f64,
    wall_seconds: f64,
    http_requests: usize,
    succeeded: usize,
    failed: usize,
    operations_per_second: f64,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    latency_p99_ms: f64,
    first_error: Option<String>,
    service_health_before: Option<Value>,
    service_health_after: Option<Value>,
}

#[tokio::main]
async fn main() {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return;
    }
    let config = Config::parse();
    let payload_bytes = std::fs::read(&config.payload)
        .unwrap_or_else(|error| panic!("read {}: {error}", config.payload.display()));
    let payload: Value = serde_json::from_slice(&payload_bytes)
        .unwrap_or_else(|error| panic!("parse {}: {error}", config.payload.display()));
    let payload = Bytes::from(serde_json::to_vec(&payload).expect("serialize request payload"));
    let client = Client::builder()
        .pool_max_idle_per_host(config.concurrency.max(8))
        .tcp_nodelay(true)
        .timeout(Duration::from_secs(120))
        .build()
        .expect("HTTP client");
    let request = client
        .post(&config.url)
        .header(CONTENT_TYPE, "application/json")
        .body(payload.clone())
        .build()
        .expect("build HTTP request template");

    if !config.warmup.is_zero() {
        let warmup = run_phase(
            &client,
            &request,
            config.expect,
            config.concurrency,
            config.warmup,
        )
        .await;
        assert_eq!(warmup.failed, 0, "warmup failed: {:?}", warmup.first_error);
    }
    let health_before = get_health(&client, config.health_url.as_deref()).await;
    let started = Instant::now();
    let mut samples = run_phase(
        &client,
        &request,
        config.expect,
        config.concurrency,
        config.duration,
    )
    .await;
    let wall_seconds = started.elapsed().as_secs_f64();
    let health_after = get_health(&client, config.health_url.as_deref()).await;
    samples.latency_us.sort_unstable();
    let report = Report {
        tool: "w3c-http-op-load",
        contract: match config.expect {
            Expect::Document | Expect::Verified => "one-http-request-equals-one-w3c-operation",
            Expect::BatchDocument | Expect::BatchVerified => {
                "one-http-request-carries-multiple-w3c-operation-items"
            }
        },
        operation: config.operation,
        url: config.url,
        payload_bytes: payload.len(),
        concurrency: config.concurrency,
        target_duration_seconds: config.duration.as_secs_f64(),
        wall_seconds,
        http_requests: samples.http_requests,
        succeeded: samples.succeeded,
        failed: samples.failed,
        operations_per_second: samples.succeeded as f64 / wall_seconds.max(f64::EPSILON),
        latency_p50_ms: percentile_ms(&samples.latency_us, 0.50),
        latency_p95_ms: percentile_ms(&samples.latency_us, 0.95),
        latency_p99_ms: percentile_ms(&samples.latency_us, 0.99),
        first_error: samples.first_error,
        service_health_before: health_before,
        service_health_after: health_after,
    };
    let serialized = serde_json::to_string_pretty(&report).expect("serialize report");
    if let Some(path) = config.output {
        std::fs::write(&path, &serialized)
            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    }
    println!("{serialized}");
}

async fn run_phase(
    client: &Client,
    request: &Request,
    expect: Expect,
    concurrency: usize,
    duration: Duration,
) -> Samples {
    let deadline = Instant::now() + duration;
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let client = client.clone();
        let request = request.try_clone().expect("clone HTTP request template");
        tasks.push(tokio::spawn(async move {
            let mut samples = Samples::default();
            while Instant::now() < deadline {
                let started = Instant::now();
                let result = request_one(&client, &request, expect).await;
                samples
                    .latency_us
                    .push(started.elapsed().as_micros().min(u64::MAX as u128) as u64);
                samples.http_requests += 1;
                match result {
                    Ok(count) => samples.succeeded += count,
                    Err(error) => {
                        samples.failed += 1;
                        if samples.first_error.is_none() {
                            samples.first_error = Some(error);
                        }
                    }
                }
            }
            samples
        }));
    }
    let mut combined = Samples::default();
    for task in tasks {
        let mut worker = task.await.expect("HTTP load worker");
        combined.http_requests += worker.http_requests;
        combined.succeeded += worker.succeeded;
        combined.failed += worker.failed;
        combined.latency_us.append(&mut worker.latency_us);
        if combined.first_error.is_none() {
            combined.first_error = worker.first_error.take();
        }
    }
    combined
}

async fn request_one(client: &Client, request: &Request, expect: Expect) -> Result<usize, String> {
    let request = request
        .try_clone()
        .ok_or_else(|| "request-template: body is not replayable".to_owned())?;
    let response = client
        .execute(request)
        .await
        // Keep the error origin explicit.  The capacity runner may retain a
        // discovery warm-up rejection as HTTP-admission evidence, but only
        // when it is unquestionably below the HTTP/application boundary.
        // A status response or an invalid response body is therefore never
        // allowed to look like a harmless connection rejection.
        .map_err(|error| format!("transport: {error}"))?;
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("transport: response body read failed: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "http-status: HTTP {status}: {}",
            String::from_utf8_lossy(&bytes)
        ));
    }
    validate_response(&bytes, expect)
}

fn validate_response(bytes: &[u8], expect: Expect) -> Result<usize, String> {
    match expect {
        Expect::Document => serde_json::from_slice::<DocumentResponse>(bytes)
            .map_err(|error| format!("response-semantics: invalid document response: {error}"))
            .and_then(|response| {
                if response.proof.proof_value.is_empty() {
                    Err(
                        "response-semantics: response contains an empty proof.proofValue"
                            .to_owned(),
                    )
                } else {
                    Ok(1)
                }
            }),
        Expect::Verified => serde_json::from_slice::<VerifiedResponse>(bytes)
            .map_err(|error| format!("response-semantics: invalid verification response: {error}"))
            .and_then(|response| {
                if response.verified {
                    Ok(1)
                } else {
                    Err("response-semantics: response did not contain verified=true".to_owned())
                }
            }),
        Expect::BatchDocument | Expect::BatchVerified => {
            let value: Value = serde_json::from_slice(bytes)
                .map_err(|error| format!("response-semantics: invalid JSON response: {error}"))?;
            validate_batch_response(
                &value,
                if matches!(expect, Expect::BatchDocument) {
                    Expect::Document
                } else {
                    Expect::Verified
                },
            )
        }
    }
}

fn validate_batch_response(value: &Value, item_expect: Expect) -> Result<usize, String> {
    if value.get("ok").and_then(Value::as_bool) != Some(true) {
        return Err("batch response did not contain ok=true".to_owned());
    }
    match item_expect {
        Expect::Verified => {
            if let Some(verified) = value.get("verified").and_then(Value::as_array) {
                if verified.iter().all(|item| item.as_bool() == Some(true)) {
                    return validate_batch_counts(value, verified.len());
                }
                return Err("compact verify batch contains false".to_owned());
            }
        }
        Expect::Document => {
            if let Some(proof_values) = value.get("proofValues").and_then(Value::as_array) {
                if proof_values.iter().all(|item| item.as_str().is_some()) {
                    return validate_batch_counts(value, proof_values.len());
                }
                return Err("compact derive batch contains a missing proofValue".to_owned());
            }
        }
        Expect::BatchDocument | Expect::BatchVerified => unreachable!(),
    }
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| "batch response is missing results".to_owned())?;
    for item in results {
        if item.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err("batch response contains a failed item".to_owned());
        }
        let output = item
            .get("value")
            .ok_or_else(|| "batch item is missing value".to_owned())?;
        match item_expect {
            Expect::Document
                if output
                    .pointer("/proof/proofValue")
                    .and_then(Value::as_str)
                    .is_some() => {}
            Expect::Verified if output.get("verified").and_then(Value::as_bool) == Some(true) => {}
            Expect::Document => return Err("batch document item is invalid".to_owned()),
            Expect::Verified => return Err("batch verify item is invalid".to_owned()),
            Expect::BatchDocument | Expect::BatchVerified => unreachable!(),
        }
    }
    validate_batch_counts(value, results.len())
}

fn validate_batch_counts(value: &Value, item_count: usize) -> Result<usize, String> {
    if item_count == 0 {
        return Err("batch response contains no operation items".to_owned());
    }
    let attempted = value
        .get("attempted")
        .and_then(Value::as_u64)
        .ok_or_else(|| "batch response is missing attempted".to_owned())?;
    let succeeded = value
        .get("succeeded")
        .and_then(Value::as_u64)
        .ok_or_else(|| "batch response is missing succeeded".to_owned())?;
    let failed = value
        .get("failed")
        .and_then(Value::as_u64)
        .ok_or_else(|| "batch response is missing failed".to_owned())?;
    if attempted != item_count as u64 || succeeded != item_count as u64 || failed != 0 {
        return Err(format!(
            "batch counters do not match {item_count} successful items"
        ));
    }
    Ok(item_count)
}

async fn get_health(client: &Client, url: Option<&str>) -> Option<Value> {
    let url = url?;
    client.get(url).send().await.ok()?.json().await.ok()
}

fn percentile_ms(sorted: &[u64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index] as f64 / 1_000.0
}

impl Config {
    fn parse() -> Self {
        let args = std::env::args().collect::<Vec<_>>();
        Self {
            operation: required(&args, "--operation"),
            url: required(&args, "--url"),
            health_url: arg_value(&args, "--health-url"),
            payload: PathBuf::from(required(&args, "--payload")),
            expect: match required(&args, "--expect").as_str() {
                "document" => Expect::Document,
                "verified" => Expect::Verified,
                "batch-document" => Expect::BatchDocument,
                "batch-verified" => Expect::BatchVerified,
                value => panic!("unsupported --expect {value}; use document, verified, batch-document, or batch-verified"),
            },
            concurrency: parse_usize(&args, "--concurrency", 1).max(1),
            warmup: Duration::from_secs_f64(parse_f64(&args, "--warmup-secs", 1.0).max(0.0)),
            duration: Duration::from_secs_f64(parse_f64(&args, "--duration-secs", 5.0).max(0.001)),
            output: arg_value(&args, "--output").map(PathBuf::from),
        }
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn required(args: &[String], flag: &str) -> String {
    arg_value(args, flag).unwrap_or_else(|| panic!("missing required {flag}"))
}

fn parse_usize(args: &[String], flag: &str, default: usize) -> usize {
    arg_value(args, flag)
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {flag}")))
        .unwrap_or(default)
}

fn parse_f64(args: &[String], flag: &str, default: f64) -> f64 {
    arg_value(args, flag)
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {flag}")))
        .unwrap_or(default)
}

fn print_help() {
    println!(
        "Usage: w3c-http-op-load --operation NAME --url URL --payload FILE \\\n         --expect document|verified|batch-document|batch-verified [--health-url URL] [--concurrency N] \\\n         [--warmup-secs S] [--duration-secs S] [--output FILE]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_single_operation_responses_without_materializing_unknown_fields() {
        assert_eq!(
            validate_response(
                br#"{"credentialSubject":{"claims":[1,2,3]},"proof":{"proofValue":"uProof"}}"#,
                Expect::Document,
            )
            .unwrap(),
            1
        );
        assert_eq!(
            validate_response(
                br#"{"verified":true,"ignored":{"large":[1,2,3]}}"#,
                Expect::Verified
            )
            .unwrap(),
            1
        );
        assert!(validate_response(br#"{"proof":{}}"#, Expect::Document).is_err());
        assert!(validate_response(br#"{"verified":false}"#, Expect::Verified).is_err());
    }
}
