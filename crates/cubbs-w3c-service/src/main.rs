mod batcher;
mod cache;
use cache::*;
mod managed;
mod response_template;
use response_template::ProofResponseTemplate;

use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    time::Instant,
};

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use batcher::{BatchQueue, BatchReceiver, BatcherConfig, BatcherSnapshot};
use cubbs_bbs::{
    BbsBackend, BbsEngine, ProofVerifyRequest, SignRequest, VerifyRequest as BbsVerifyRequest,
    SECRET_KEY_LEN,
};
use cubbs_vc_di_bbs::{
    acquire_w3c_verify_crypto_admission_for_backend, canonicalization_cache_snapshot,
    default_bbs_2023_proof_options, derive_jsonld_disclosure_proof_batch, did_key_from_public_key,
    issue_jsonld_base_proof_batch, json_content_digest, jsonld_batch_workers,
    lower_base_verification, lower_disclosure_verification, prepare_jsonld_base_issue,
    prepare_jsonld_base_verification, prepare_jsonld_disclosure_verification,
    prepared_cache_snapshot, shared_jsonld_loader_enabled, verification_method_from_public_key,
    w3c_batch_timing_snapshot, w3c_verify_crypto_admission_snapshot_for_backend, BatchRunResult,
    CanonicalizationCacheSnapshot, DeriveJsonLdDisclosureProofBatchRequest,
    DeriveJsonLdDisclosureProofRequest, IssueJsonLdBaseProofBatchRequest,
    IssueJsonLdBaseProofRequest, PreparedCacheSnapshot, PreparedJsonLdBaseIssue,
    W3cBatchTimingSnapshot, W3cVerifyCryptoAdmissionSnapshot, HMAC_KEY_LEN,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{net::TcpSocket, sync::Semaphore};

use managed::{
    ManagedDeriveEntry, ManagedDeriveLookupRequest, ManagedLookupRequest, ManagedRegisterRequest,
    ManagedRegisterResponse, ManagedVcSnapshot, ManagedVcStore, ManagedVerification,
};

#[derive(Clone)]
struct AppState {
    managed_engine: Arc<BbsEngine>,
    secret_key: Vec<u8>,
    public_key: Vec<u8>,
    did: String,
    verification_method: String,
    bbs_backend: String,
    bbs_backend_requested: String,
    bbs_backend_active: String,
    bbs_backend_hardware_acceleration: bool,
    bbs_backend_fallback: bool,
    bbs_backend_gpu_abi_version: u32,
    bbs_backend_detail: String,
    bbs_backend_config: BbsBackend,
    listen_backlog: u32,
    batcher_config: BatcherConfig,
    issue_queue: BatchQueue<IssueBatchItem, IssueBatchOutput>,
    derive_queue: BatchQueue<DeriveJsonLdDisclosureProofRequest, Value>,
    verify_signature_queue: BatchQueue<Arc<BbsVerifyRequest>, bool>,
    proof_verify_queue: BatchQueue<Arc<ProofVerifyRequest>, bool>,
    issue_template_cache: Arc<RawTemplateCache<PreparedIssueTemplate>>,
    verification_template_cache: Arc<RawTemplateCache<PreparedVerification>>,
    verification_cache: Arc<VerificationResultCache>,
    managed_vc_store: Arc<ManagedVcStore>,
    managed_derive_queue: BatchQueue<Arc<ManagedDeriveEntry>, Value>,
    managed_verify_base_queue: BatchQueue<Arc<BbsVerifyRequest>, Value>,
    managed_proof_verify_queue: BatchQueue<Arc<ProofVerifyRequest>, Value>,
    managed_bulk_slots: Arc<Semaphore>,
    managed_bulk_pipelines: usize,
    managed_bulk_max_items: usize,
}

#[derive(Debug)]
struct PreparedIssueTemplate {
    prepared: PreparedJsonLdBaseIssue,
    response: ProofResponseTemplate,
}

impl PreparedIssueTemplate {
    fn new(prepared: PreparedJsonLdBaseIssue) -> Result<Self, String> {
        let response = prepared
            .assemble_document(String::new())
            .map_err(|error| error.to_string())?;
        Ok(Self {
            prepared,
            response: ProofResponseTemplate::new(response)?,
        })
    }

    fn sign_request(&self) -> Arc<SignRequest> {
        Arc::clone(self.prepared.sign_request())
    }

    fn finish(&self, signature: Vec<u8>, hmac_key: &[u8]) -> Result<Bytes, String> {
        let proof_value = self
            .prepared
            .proof_value(signature, hmac_key)
            .map_err(|error| error.to_string())?;
        self.response.finish(&proof_value)
    }
}

#[derive(Debug, Clone)]
enum IssueBatchItem {
    Prepared {
        template: Arc<PreparedIssueTemplate>,
        hmac_key: [u8; HMAC_KEY_LEN],
    },
    General(IssueJsonLdBaseProofRequest),
}

#[derive(Debug, Clone)]
enum IssueBatchOutput {
    Serialized(Bytes),
    Value(Value),
}

#[derive(Debug, Clone)]
enum PreparedVerification {
    Signature {
        document: Arc<Value>,
        request: Arc<BbsVerifyRequest>,
    },
    Proof {
        document: Arc<Value>,
        request: Arc<ProofVerifyRequest>,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct IssueRequest {
    credential: Value,
    #[serde(default)]
    options: IssueOptions,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IssueOptions {
    #[serde(default)]
    mandatory_pointers: Vec<String>,
    verification_method: Option<String>,
    proof_purpose: Option<String>,
    feature_option: Option<String>,
    #[serde(alias = "commitment_with_proof")]
    commitment_with_proof: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerifyRequest {
    verifiable_credential: Value,
    #[serde(default)]
    options: VerifyOptions,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerifyOptions {
    expected_proof_purpose: Option<String>,
    domain: Option<String>,
    challenge: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeriveRequest {
    verifiable_credential: Value,
    #[serde(default)]
    options: DeriveOptions,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeriveOptions {
    #[serde(default)]
    selective_pointers: Vec<String>,
    #[serde(default)]
    presentation_header: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchRequest<T> {
    items: Vec<T>,
    #[serde(default)]
    fail_fast: bool,
    #[serde(default)]
    compact: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiBatchResult<T> {
    ok: bool,
    attempted: usize,
    succeeded: usize,
    failed: usize,
    results: Vec<ApiBatchItem<T>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiBatchItem<T> {
    index: usize,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    did: String,
    verification_method: String,
    bbs_backend: String,
    bbs_backend_requested: String,
    bbs_backend_active: String,
    bbs_backend_hardware_acceleration: bool,
    bbs_backend_fallback: bool,
    bbs_backend_gpu_abi_version: u32,
    bbs_backend_detail: String,
    w3c_runtime: W3cRuntimeConfig,
    canonicalization_cache: CanonicalizationCacheSnapshot,
    prepared_cache: PreparedCacheSnapshot,
    issue_template_cache: RawTemplateCacheSnapshot,
    verification_template_cache: RawTemplateCacheSnapshot,
    verification_result_cache: VerificationResultCacheSnapshot,
    managed_vc: ManagedVcSnapshot,
    w3c_batch_timing: W3cBatchTimingSnapshot,
    verify_crypto_admission: W3cVerifyCryptoAdmissionSnapshot,
    batching: BatchingHealth,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct W3cRuntimeConfig {
    jsonld_workers: usize,
    shared_jsonld_loader: bool,
    canonicalization_cache_capacity: usize,
    prepared_cache_capacity: usize,
    verification_result_cache_capacity: usize,
    listen_backlog: u32,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct OperationBatchingConfigs {
    issue: BatcherConfig,
    derive: BatcherConfig,
    verify_signature: BatcherConfig,
    proof_verify: BatcherConfig,
    managed_derive: BatcherConfig,
    managed_verify_base: BatcherConfig,
    managed_proof_verify: BatcherConfig,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchingHealth {
    enabled: bool,
    config: BatcherConfig,
    operation_configs: OperationBatchingConfigs,
    issue: BatcherSnapshot,
    derive: BatcherSnapshot,
    verify_signature: BatcherSnapshot,
    proof_verify: BatcherSnapshot,
    managed_derive: BatcherSnapshot,
    managed_verify_base: BatcherSnapshot,
    managed_proof_verify: BatcherSnapshot,
    managed_bulk_pipelines: usize,
    managed_bulk_max_items: usize,
}

#[tokio::main]
async fn main() {
    let state = Arc::new(create_state());
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/id/healthz", get(healthz))
        .route("/credentials/issue", post(issue))
        .route("/id/credentials/issue", post(issue))
        .route("/credentials/issue-batch", post(issue_batch))
        .route("/id/credentials/issue-batch", post(issue_batch))
        .route("/credentials/verify", post(verify))
        .route("/id/credentials/verify", post(verify))
        .route("/credentials/verify-batch", post(verify_batch))
        .route("/id/credentials/verify-batch", post(verify_batch))
        .route("/credentials/derive", post(derive))
        .route("/id/credentials/derive", post(derive))
        .route("/credentials/derive-batch", post(derive_batch))
        .route("/id/credentials/derive-batch", post(derive_batch))
        .route("/managed/credentials/register", post(managed_register))
        .route("/id/managed/credentials/register", post(managed_register))
        .route("/managed/credentials/derive", post(managed_derive))
        .route("/id/managed/credentials/derive", post(managed_derive))
        .route(
            "/managed/credentials/derive-batch",
            post(managed_derive_batch),
        )
        .route(
            "/id/managed/credentials/derive-batch",
            post(managed_derive_batch),
        )
        .route("/managed/credentials/verify", post(managed_verify))
        .route("/id/managed/credentials/verify", post(managed_verify))
        .route(
            "/managed/credentials/verify-batch",
            post(managed_verify_batch),
        )
        .route(
            "/id/managed/credentials/verify-batch",
            post(managed_verify_batch),
        )
        .with_state(state.clone());

    let port = env::var("CUBBS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(40443);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    println!("cuBBS-Cred W3C service listening on http://{addr}/id");
    println!("issuer id: {}", state.did);
    println!("verificationMethod: {}", state.verification_method);
    println!(
        "BBS backend: requested={} active={} hardware_acceleration={} fallback={}",
        state.bbs_backend_requested,
        state.bbs_backend_active,
        state.bbs_backend_hardware_acceleration,
        state.bbs_backend_fallback,
    );
    println!("BBS backend detail: {}", state.bbs_backend_detail);
    println!(
        "W3C shared pipeline slots: {} (queue capacity: {})",
        state.batcher_config.pipelines, state.batcher_config.queue_capacity
    );
    for (operation, config) in [
        ("issue", state.issue_queue.config()),
        ("derive", state.derive_queue.config()),
        ("verify-base", state.verify_signature_queue.config()),
        ("proof-verify", state.proof_verify_queue.config()),
    ] {
        println!(
            "W3C {operation} queue: interval={}ms max_batch={} pipelines={}",
            config.interval_ms, config.max_batch_size, config.pipelines
        );
    }
    println!("W3C localConfig endpoint baseUrl: http://localhost:{port}/id");
    println!(
        "W3C managed VC mode: {} (capacity: {}, bulk pipelines: {})",
        state.managed_vc_store.enabled(),
        state.managed_vc_store.snapshot().capacity,
        state.managed_bulk_pipelines
    );

    // A high client-concurrency wave must be admitted to the bounded W3C
    // queue, not dropped by a small implicit listener backlog before the
    // request reaches application-level admission control.
    let socket = TcpSocket::new_v4().expect("create IPv4 TCP socket");
    socket.set_reuseaddr(true).expect("set TCP reuseaddr");
    socket.bind(addr).expect("bind W3C TCP socket");
    let listener = socket
        .listen(state.listen_backlog)
        .expect("listen on W3C TCP socket");
    axum::serve(listener, app).await.unwrap();
}

fn create_state() -> AppState {
    let backend = env::var("CUBBS_BBS_BACKEND")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| BbsBackend::parse(value.trim()))
        .transpose()
        .expect("valid CUBBS_BBS_BACKEND")
        .unwrap_or(BbsBackend::ZkryptiumCpu);
    if require_hardware_gpu() {
        backend
            .ensure_hardware_available()
            .expect("hardware GPU BBS backend required");
    }
    let engine = Arc::new(BbsEngine::new(backend));
    let backend_info = engine.backend_info();
    let keys = match env::var("CUBBS_SECRET_KEY") {
        Ok(secret_key) => {
            let secret_key = decode_secret_key(&secret_key).expect("valid CUBBS_SECRET_KEY");
            let public_key = engine
                .public_key_from_secret(&secret_key)
                .expect("public key derivation must succeed");
            cubbs_bbs::KeyPair {
                secret_key,
                public_key,
            }
        }
        Err(_) => engine
            .generate_keypair(&mut rand::thread_rng())
            .expect("BBS key generation must succeed"),
    };
    let did = did_key_from_public_key(&keys.public_key).expect("valid generated public key");
    let verification_method =
        verification_method_from_public_key(&keys.public_key).expect("valid generated public key");
    let listen_backlog = w3c_listen_backlog();
    let batcher_config = BatcherConfig::from_env();
    let issue_config = BatcherConfig::from_env_for("issue", 128, 20);
    let derive_config = BatcherConfig::from_env_for("derive", 128, 20);
    let verify_signature_config = BatcherConfig::from_env_for("verify-base", 512, 5);
    let proof_verify_config = BatcherConfig::from_env_for("proof-verify", 128, 20);
    let managed_derive_config = BatcherConfig::from_env_for("managed-derive", 4096, 1);
    let managed_verify_base_config = BatcherConfig::from_env_for("managed-verify-base", 4096, 1);
    let managed_proof_verify_config = BatcherConfig::from_env_for("managed-proof-verify", 4096, 1);
    let verify_crypto_admission = w3c_verify_crypto_admission_snapshot_for_backend(backend);
    if verify_crypto_admission.enabled {
        let item_budget = verify_crypto_admission
            .item_budget
            .expect("enabled GPU Verify admission must expose an item budget");
        for (operation, max_batch_size) in [
            ("VERIFY_BASE", verify_signature_config.max_batch_size),
            ("PROOF_VERIFY", proof_verify_config.max_batch_size),
        ] {
            assert!(
                max_batch_size <= item_budget,
                "CUBBS_W3C_{operation}_MAX_BATCH_SIZE={max_batch_size} exceeds CUBBS_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS={item_budget}",
            );
        }
    }
    let pipeline_slots = Arc::new(Semaphore::new(batcher_config.pipelines));
    let verification_cache = Arc::new(VerificationResultCache::from_env());
    let issue_engine = Arc::clone(&engine);
    let issue_queue = BatchQueue::spawn_with_slots(issue_config, Arc::clone(&pipeline_slots), {
        let verification_cache = Arc::clone(&verification_cache);
        move |items| {
            process_issue_batch(
                Arc::clone(&issue_engine),
                items,
                Arc::clone(&verification_cache),
            )
        }
    });
    let derive_queue = BatchQueue::spawn_with_slots(derive_config, Arc::clone(&pipeline_slots), {
        let verification_cache = Arc::clone(&verification_cache);
        move |items| process_derive_batch(items, Arc::clone(&verification_cache))
    });
    let verify_engine = Arc::clone(&engine);
    let verify_backend = backend;
    let verify_signature_queue = BatchQueue::spawn_with_slots(
        verify_signature_config,
        Arc::clone(&pipeline_slots),
        move |items| {
            let engine = Arc::clone(&verify_engine);
            async move { process_prepared_base_verification(engine, verify_backend, items).await }
        },
    );
    let proof_verify_engine = Arc::clone(&engine);
    let proof_verify_backend = backend;
    let proof_verify_queue = BatchQueue::spawn_with_slots(
        proof_verify_config,
        Arc::clone(&pipeline_slots),
        move |items| {
            let engine = Arc::clone(&proof_verify_engine);
            async move { process_prepared_proof_verification(engine, proof_verify_backend, items).await }
        },
    );
    let issue_template_cache = Arc::new(RawTemplateCache::from_env());
    let verification_template_cache = Arc::new(RawTemplateCache::from_env());
    let managed_vc_store = Arc::new(ManagedVcStore::from_env());
    let managed_bulk_pipelines = managed_bulk_pipelines();
    let managed_bulk_max_items = managed_bulk_max_items();
    let managed_bulk_slots = Arc::new(Semaphore::new(managed_bulk_pipelines));
    let managed_derive_engine = Arc::clone(&engine);
    let managed_derive_queue = BatchQueue::spawn_with_slots(
        managed_derive_config,
        Arc::clone(&pipeline_slots),
        move |items| {
            let engine = Arc::clone(&managed_derive_engine);
            async move {
                let count = items.len();
                tokio::task::spawn_blocking(move || {
                    managed::process_derivation(engine.as_ref(), items)
                })
                .await
                .unwrap_or_else(|error| {
                    (0..count)
                        .map(|_| Err(format!("BBS Derive worker failed: {error}")))
                        .collect()
                })
            }
        },
    );
    let managed_verify_base_engine = Arc::clone(&engine);
    let managed_verify_base_queue = BatchQueue::spawn_with_slots(
        managed_verify_base_config,
        Arc::clone(&pipeline_slots),
        move |items| {
            let engine = Arc::clone(&managed_verify_base_engine);
            async move {
                managed_verification_values(
                    process_prepared_base_verification(engine, backend, items).await,
                )
            }
        },
    );
    let managed_proof_verify_engine = Arc::clone(&engine);
    let managed_proof_verify_queue =
        BatchQueue::spawn_with_slots(managed_proof_verify_config, pipeline_slots, move |items| {
            let engine = Arc::clone(&managed_proof_verify_engine);
            async move {
                managed_verification_values(
                    process_prepared_proof_verification(engine, backend, items).await,
                )
            }
        });
    AppState {
        managed_engine: Arc::clone(&engine),
        secret_key: keys.secret_key,
        public_key: keys.public_key,
        did,
        verification_method,
        bbs_backend: engine.backend_name().to_owned(),
        bbs_backend_requested: backend_info.requested.to_owned(),
        bbs_backend_active: backend_info.active.to_owned(),
        bbs_backend_hardware_acceleration: backend_info.hardware_acceleration,
        bbs_backend_fallback: backend_info.fallback,
        bbs_backend_gpu_abi_version: backend_info.gpu_abi_version,
        bbs_backend_detail: backend_info.detail.to_owned(),
        bbs_backend_config: backend,
        listen_backlog,
        batcher_config,
        issue_queue,
        derive_queue,
        verify_signature_queue,
        proof_verify_queue,
        issue_template_cache,
        verification_template_cache,
        verification_cache,
        managed_vc_store,
        managed_derive_queue,
        managed_verify_base_queue,
        managed_proof_verify_queue,
        managed_bulk_slots,
        managed_bulk_pipelines,
        managed_bulk_max_items,
    }
}

fn w3c_listen_backlog() -> u32 {
    match env::var("CUBBS_W3C_LISTEN_BACKLOG") {
        Ok(value) => value
            .parse::<u32>()
            .ok()
            .filter(|backlog| *backlog > 0)
            .expect("CUBBS_W3C_LISTEN_BACKLOG must be a positive u32"),
        Err(_) => 4096,
    }
}

fn managed_bulk_pipelines() -> usize {
    env::var("CUBBS_W3C_MANAGED_BULK_PIPELINES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6)
        .max(1)
}

fn managed_bulk_max_items() -> usize {
    env::var("CUBBS_W3C_MANAGED_BULK_MAX_ITEMS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096)
        .max(1)
}

fn require_hardware_gpu() -> bool {
    env::var("CUBBS_BBS_REQUIRE_HARDWARE_GPU")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn decode_secret_key(value: &str) -> Result<Vec<u8>, String> {
    let trimmed = value.trim();
    let bytes = if let Some(hex) = trimmed.strip_prefix("hex:") {
        decode_hex(hex)?
    } else if let Some(encoded) = trimmed.strip_prefix("base64url:") {
        URL_SAFE_NO_PAD.decode(encoded).map_err(|e| e.to_string())?
    } else if let Some(encoded) = trimmed.strip_prefix("z") {
        bs58::decode(encoded)
            .into_vec()
            .map_err(|e| e.to_string())?
    } else if trimmed.len() == SECRET_KEY_LEN * 2 && trimmed.chars().all(|c| c.is_ascii_hexdigit())
    {
        decode_hex(trimmed)?
    } else {
        URL_SAFE_NO_PAD.decode(trimmed).map_err(|e| e.to_string())?
    };
    if bytes.len() != SECRET_KEY_LEN {
        return Err(format!(
            "secret key must be {SECRET_KEY_LEN} bytes, got {}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn decode_hex(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err("hex secret key must have an even number of characters".to_owned());
    }
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let byte = std::str::from_utf8(pair).map_err(|e| e.to_string())?;
            u8::from_str_radix(byte, 16).map_err(|e| e.to_string())
        })
        .collect()
}

async fn healthz(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let canonicalization_cache = canonicalization_cache_snapshot();
    let prepared_cache = prepared_cache_snapshot();
    Json(HealthResponse {
        status: "ok".to_owned(),
        did: state.did.clone(),
        verification_method: state.verification_method.clone(),
        bbs_backend: state.bbs_backend.clone(),
        bbs_backend_requested: state.bbs_backend_requested.clone(),
        bbs_backend_active: state.bbs_backend_active.clone(),
        bbs_backend_hardware_acceleration: state.bbs_backend_hardware_acceleration,
        bbs_backend_fallback: state.bbs_backend_fallback,
        bbs_backend_gpu_abi_version: state.bbs_backend_gpu_abi_version,
        bbs_backend_detail: state.bbs_backend_detail.clone(),
        w3c_runtime: W3cRuntimeConfig {
            jsonld_workers: jsonld_batch_workers(),
            shared_jsonld_loader: shared_jsonld_loader_enabled(),
            canonicalization_cache_capacity: canonicalization_cache.capacity,
            prepared_cache_capacity: prepared_cache.capacity_per_operation,
            verification_result_cache_capacity: state.verification_cache.capacity,
            listen_backlog: state.listen_backlog,
        },
        canonicalization_cache,
        prepared_cache,
        issue_template_cache: state.issue_template_cache.snapshot(),
        verification_template_cache: state.verification_template_cache.snapshot(),
        verification_result_cache: state.verification_cache.snapshot(),
        managed_vc: state.managed_vc_store.snapshot(),
        w3c_batch_timing: w3c_batch_timing_snapshot(),
        verify_crypto_admission: w3c_verify_crypto_admission_snapshot_for_backend(
            state.bbs_backend_config,
        ),
        batching: BatchingHealth {
            enabled: true,
            config: state.batcher_config,
            operation_configs: OperationBatchingConfigs {
                issue: state.issue_queue.config(),
                derive: state.derive_queue.config(),
                verify_signature: state.verify_signature_queue.config(),
                proof_verify: state.proof_verify_queue.config(),
                managed_derive: state.managed_derive_queue.config(),
                managed_verify_base: state.managed_verify_base_queue.config(),
                managed_proof_verify: state.managed_proof_verify_queue.config(),
            },
            issue: state.issue_queue.snapshot(),
            derive: state.derive_queue.snapshot(),
            verify_signature: state.verify_signature_queue.snapshot(),
            proof_verify: state.proof_verify_queue.snapshot(),
            managed_derive: state.managed_derive_queue.snapshot(),
            managed_verify_base: state.managed_verify_base_queue.snapshot(),
            managed_proof_verify: state.managed_proof_verify_queue.snapshot(),
            managed_bulk_pipelines: state.managed_bulk_pipelines,
            managed_bulk_max_items: state.managed_bulk_max_items,
        },
    })
}

async fn managed_register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ManagedRegisterRequest>,
) -> Result<Json<ManagedRegisterResponse>, ApiError> {
    state
        .managed_vc_store
        .register(req)
        .await
        .map(Json)
        .map_err(ApiError::bad_request)
}

async fn managed_derive(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ManagedDeriveLookupRequest>,
) -> Result<Json<Value>, ApiError> {
    let prepared = state
        .managed_vc_store
        .derive(&req.credential_id, &req.profile_id)
        .map_err(ApiError::bad_request)?;
    let derived = state
        .managed_derive_queue
        .submit(prepared)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(derived))
}

async fn managed_derive_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchRequest<ManagedDeriveLookupRequest>>,
) -> Result<Json<Value>, ApiError> {
    validate_managed_bulk_size(req.items.len(), state.managed_bulk_max_items)?;
    let request_started = Instant::now();
    let prepared = req
        .items
        .into_iter()
        .map(|item| {
            state
                .managed_vc_store
                .derive(&item.credential_id, &item.profile_id)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(ApiError::bad_request)?;
    let lookup_ms = request_started.elapsed().as_secs_f64() * 1000.0;
    let wait_started = Instant::now();
    let slot = Arc::clone(&state.managed_bulk_slots)
        .acquire_owned()
        .await
        .map_err(|_| ApiError::bad_request("managed bulk scheduler is closed"))?;
    let wait_ms = wait_started.elapsed().as_secs_f64() * 1000.0;
    let backend_started = Instant::now();
    let engine = Arc::clone(&state.managed_engine);
    let compact = req.compact;
    let results = tokio::task::spawn_blocking(move || {
        if compact {
            managed::process_derivation_compact(engine.as_ref(), prepared)
        } else {
            managed::process_derivation(engine.as_ref(), prepared)
        }
    })
    .await
    .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let backend_ms = backend_started.elapsed().as_secs_f64() * 1000.0;
    drop(slot);
    let response_started = Instant::now();
    let mut response = if req.compact {
        compact_derive_results(results, req.fail_fast)
    } else {
        serde_json::to_value(api_results_from_values(results, req.fail_fast))
            .expect("serialize managed derive batch")
    };
    if req.compact {
        response["serverTiming"] = json!({
            "lookupMs": lookup_ms,
            "waitMs": wait_ms,
            "backendMs": backend_ms,
            "responseAssemblyMs": response_started.elapsed().as_secs_f64() * 1000.0,
        });
    }
    Ok(Json(response))
}

async fn managed_verify(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ManagedLookupRequest>,
) -> Result<Json<Value>, ApiError> {
    let verified = match state
        .managed_vc_store
        .verification(&req.credential_id)
        .map_err(ApiError::bad_request)?
    {
        ManagedVerification::Base(prepared) => state
            .managed_verify_base_queue
            .submit(prepared)
            .await
            .map_err(ApiError::bad_request)?,
        ManagedVerification::Proof(prepared) => state
            .managed_proof_verify_queue
            .submit(prepared)
            .await
            .map_err(ApiError::bad_request)?,
    };
    Ok(Json(verified))
}

async fn managed_verify_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchRequest<ManagedLookupRequest>>,
) -> Result<Json<Value>, ApiError> {
    validate_managed_bulk_size(req.items.len(), state.managed_bulk_max_items)?;
    let request_started = Instant::now();
    let expected = req.items.len();
    let mut base = Vec::new();
    let mut proof = Vec::new();
    for (index, item) in req.items.into_iter().enumerate() {
        match state
            .managed_vc_store
            .verification(&item.credential_id)
            .map_err(ApiError::bad_request)?
        {
            ManagedVerification::Base(prepared) => base.push((index, prepared)),
            ManagedVerification::Proof(prepared) => proof.push((index, prepared)),
        }
    }
    let lookup_ms = request_started.elapsed().as_secs_f64() * 1000.0;
    let wait_started = Instant::now();
    let slot = Arc::clone(&state.managed_bulk_slots)
        .acquire_owned()
        .await
        .map_err(|_| ApiError::bad_request("managed bulk scheduler is closed"))?;
    let wait_ms = wait_started.elapsed().as_secs_f64() * 1000.0;
    let backend_started = Instant::now();
    let engine = Arc::clone(&state.managed_engine);
    let results = tokio::task::spawn_blocking(move || {
        let mut ordered = (0..expected)
            .map(|_| None)
            .collect::<Vec<Option<Result<Value, String>>>>();
        let (base_indexes, base_items): (Vec<_>, Vec<_>) = base.into_iter().unzip();
        for (index, result) in base_indexes
            .into_iter()
            .zip(managed::process_base_verification(
                engine.as_ref(),
                base_items,
            ))
        {
            ordered[index] = Some(result);
        }
        let (proof_indexes, proof_items): (Vec<_>, Vec<_>) = proof.into_iter().unzip();
        for (index, result) in proof_indexes
            .into_iter()
            .zip(managed::process_proof_verification(
                engine.as_ref(),
                proof_items,
            ))
        {
            ordered[index] = Some(result);
        }
        ordered
            .into_iter()
            .map(|result| result.unwrap_or_else(|| Err("missing managed bulk result".to_owned())))
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let backend_ms = backend_started.elapsed().as_secs_f64() * 1000.0;
    drop(slot);
    let response_started = Instant::now();
    let mut response = if req.compact {
        compact_verify_results(results, req.fail_fast)
    } else {
        serde_json::to_value(api_results_from_values(results, req.fail_fast))
            .expect("serialize managed verify batch")
    };
    if req.compact {
        response["serverTiming"] = json!({
            "lookupMs": lookup_ms,
            "waitMs": wait_ms,
            "backendMs": backend_ms,
            "responseAssemblyMs": response_started.elapsed().as_secs_f64() * 1000.0,
        });
    }
    Ok(Json(response))
}

async fn issue(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    require_json_content_type(&headers)?;
    let item = if let Some(template) = state.issue_template_cache.get(&body) {
        let mut hmac_key = [0; HMAC_KEY_LEN];
        rand::thread_rng().fill_bytes(&mut hmac_key);
        IssueBatchItem::Prepared { template, hmac_key }
    } else {
        let req = serde_json::from_slice::<IssueRequest>(&body)
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let general = prepare_issue_request(&state, req)?;
        match prepare_jsonld_base_issue(&general).await? {
            Some(prepared) => {
                let hmac_key = general
                    .hmac_key
                    .as_slice()
                    .try_into()
                    .expect("prepare_issue_request creates a 32-byte HMAC key");
                let template =
                    PreparedIssueTemplate::new(prepared).map_err(ApiError::bad_request)?;
                let template = state.issue_template_cache.insert(body, template);
                IssueBatchItem::Prepared { template, hmac_key }
            }
            None => IssueBatchItem::General(general),
        }
    };
    let signed = state
        .issue_queue
        .submit(item)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(match signed {
        IssueBatchOutput::Serialized(body) => {
            ([(header::CONTENT_TYPE, "application/json")], body).into_response()
        }
        IssueBatchOutput::Value(value) => Json(value).into_response(),
    })
}

async fn issue_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchRequest<IssueRequest>>,
) -> Result<Json<ApiBatchResult<Value>>, ApiError> {
    let mut pending = Vec::with_capacity(req.items.len());
    for item in req.items {
        let item = prepare_issue_request(&state, item)?;
        pending.push(
            state
                .issue_queue
                .enqueue(IssueBatchItem::General(item))
                .await
                .map_err(ApiError::bad_request)?,
        );
    }
    let mut results = Vec::with_capacity(pending.len());
    for receiver in pending {
        let result = receiver.receive().await.and_then(|output| match output {
            IssueBatchOutput::Value(value) => Ok(value),
            IssueBatchOutput::Serialized(body) => {
                serde_json::from_slice(&body).map_err(|error| error.to_string())
            }
        });
        let failed = result.is_err();
        results.push(result);
        if req.fail_fast && failed {
            break;
        }
    }
    Ok(Json(api_results_from_values(results, req.fail_fast)))
}

fn prepare_issue_request(
    state: &AppState,
    req: IssueRequest,
) -> Result<IssueJsonLdBaseProofRequest, ApiError> {
    if requires_commitment_with_proof(req.options.feature_option.as_deref())
        && req.options.commitment_with_proof.is_none()
    {
        return Err(ApiError::bad_request(
            "featureOption requires commitment_with_proof",
        ));
    }
    let verification_method = req
        .options
        .verification_method
        .unwrap_or_else(|| state.verification_method.clone());
    let proof_purpose = req
        .options
        .proof_purpose
        .unwrap_or_else(|| "assertionMethod".to_owned());
    let proof_options = default_bbs_2023_proof_options(verification_method, proof_purpose);
    let mut hmac_key = vec![0; HMAC_KEY_LEN];
    rand::thread_rng().fill_bytes(&mut hmac_key);
    Ok(IssueJsonLdBaseProofRequest {
        document: req.credential,
        proof_options,
        secret_key: state.secret_key.clone(),
        public_key: state.public_key.clone(),
        mandatory_pointers: req.options.mandatory_pointers,
        hmac_key,
    })
}

fn requires_commitment_with_proof(feature_option: Option<&str>) -> bool {
    matches!(
        feature_option,
        Some("anonymous_holder_binding" | "pseudonym_hidden_pid")
    )
}

async fn verify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    require_json_content_type(&headers)?;
    let prepared = match state.verification_template_cache.get(&body) {
        Some(prepared) => prepared,
        None => {
            let req = serde_json::from_slice::<VerifyRequest>(&body)
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
            let prepared = prepare_verification(req).await?;
            state
                .verification_template_cache
                .insert(body.clone(), prepared)
        }
    };
    verify_prepared(&state, prepared.as_ref()).await?;
    Ok(verified_response())
}

async fn verify_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchRequest<VerifyRequest>>,
) -> Result<Json<ApiBatchResult<Value>>, ApiError> {
    enum Pending {
        Cached,
        Queued(BatchReceiver<bool>, Arc<Value>),
    }

    let mut pending = Vec::with_capacity(req.items.len());
    for item in req.items {
        let prepared = prepare_verification(item).await?;
        let document = prepared.document();
        if state.verification_cache.contains(&document) {
            pending.push(Pending::Cached);
            continue;
        }
        let receiver = enqueue_verification(&state, &prepared).await?;
        pending.push(Pending::Queued(receiver, document));
    }
    let mut results = Vec::with_capacity(pending.len());
    for item in pending {
        let result = match item {
            Pending::Cached => Ok(verification_response_value()),
            Pending::Queued(receiver, document) => receiver.receive().await.map(|_| {
                state.verification_cache.insert(&document);
                verification_response_value()
            }),
        };
        let failed = result.is_err();
        results.push(result);
        if req.fail_fast && failed {
            break;
        }
    }
    Ok(Json(api_results_from_values(results, req.fail_fast)))
}

async fn derive(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeriveRequest>,
) -> Result<Json<Value>, ApiError> {
    let derived = state
        .derive_queue
        .submit(DeriveJsonLdDisclosureProofRequest {
            signed_document: req.verifiable_credential,
            selective_pointers: req.options.selective_pointers,
            presentation_header: req.options.presentation_header,
        })
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(derived))
}

async fn derive_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchRequest<DeriveRequest>>,
) -> Result<Json<ApiBatchResult<Value>>, ApiError> {
    let mut pending = Vec::with_capacity(req.items.len());
    for item in req.items {
        pending.push(
            state
                .derive_queue
                .enqueue(DeriveJsonLdDisclosureProofRequest {
                    signed_document: item.verifiable_credential,
                    selective_pointers: item.options.selective_pointers,
                    presentation_header: item.options.presentation_header,
                })
                .await
                .map_err(ApiError::bad_request)?,
        );
    }
    Ok(Json(
        api_results_from_receivers(pending, req.fail_fast).await,
    ))
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl<E> From<E> for ApiError
where
    E: std::error::Error,
{
    fn from(error: E) -> Self {
        Self::bad_request(error.to_string())
    }
}

#[derive(Clone, Copy)]
enum VerificationKind {
    Signature,
    Proof,
}

impl PreparedVerification {
    fn document(&self) -> Arc<Value> {
        match self {
            Self::Signature { document, .. } | Self::Proof { document, .. } => Arc::clone(document),
        }
    }
}

async fn prepare_verification(req: VerifyRequest) -> Result<PreparedVerification, ApiError> {
    let (document, kind) = validate_verify_request(req)?;
    let cached_document = Arc::new(document.clone());
    match kind {
        VerificationKind::Signature => {
            let prepared = prepare_jsonld_base_verification(document).await?;
            let request = lower_base_verification(prepared)?
                .ok_or_else(|| ApiError::bad_request("proof verification failed"))?;
            Ok(PreparedVerification::Signature {
                document: cached_document,
                request: Arc::new(request),
            })
        }
        VerificationKind::Proof => {
            let prepared = prepare_jsonld_disclosure_verification(document, None).await?;
            let request = lower_disclosure_verification(prepared)?
                .ok_or_else(|| ApiError::bad_request("proof verification failed"))?;
            Ok(PreparedVerification::Proof {
                document: cached_document,
                request: Arc::new(request),
            })
        }
    }
}

async fn enqueue_verification(
    state: &AppState,
    prepared: &PreparedVerification,
) -> Result<BatchReceiver<bool>, ApiError> {
    match prepared {
        PreparedVerification::Signature { request, .. } => state
            .verify_signature_queue
            .enqueue(Arc::clone(request))
            .await
            .map_err(ApiError::bad_request),
        PreparedVerification::Proof { request, .. } => state
            .proof_verify_queue
            .enqueue(Arc::clone(request))
            .await
            .map_err(ApiError::bad_request),
    }
}

async fn verify_prepared(
    state: &AppState,
    prepared: &PreparedVerification,
) -> Result<(), ApiError> {
    let document = if state.verification_cache.enabled() {
        let document = prepared.document();
        if state.verification_cache.contains(&document) {
            return Ok(());
        }
        Some(document)
    } else {
        None
    };
    let verified = match prepared {
        PreparedVerification::Signature { request, .. } => {
            state
                .verify_signature_queue
                .submit(Arc::clone(request))
                .await
        }
        PreparedVerification::Proof { request, .. } => {
            state.proof_verify_queue.submit(Arc::clone(request)).await
        }
    }
    .map_err(ApiError::bad_request)?;
    if !verified {
        return Err(ApiError::bad_request("proof verification failed"));
    }
    if let Some(document) = document {
        state.verification_cache.insert(&document);
    }
    Ok(())
}

fn require_json_content_type(headers: &HeaderMap) -> Result<(), ApiError> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .unwrap_or_default();
    if content_type.eq_ignore_ascii_case("application/json")
        || (content_type
            .get(..12)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("application/"))
            && content_type.to_ascii_lowercase().ends_with("+json"))
    {
        Ok(())
    } else {
        Err(ApiError::bad_request(
            "Content-Type must be application/json or application/*+json",
        ))
    }
}

fn verification_response_value() -> Value {
    json!({"verified": true, "checks": ["proof"]})
}

fn verified_response() -> Response {
    (
        [(header::CONTENT_TYPE, "application/json")],
        Bytes::from_static(br#"{"verified":true,"checks":["proof"]}"#),
    )
        .into_response()
}

fn validate_verify_request(req: VerifyRequest) -> Result<(Value, VerificationKind), ApiError> {
    let proof = req
        .verifiable_credential
        .get("proof")
        .ok_or_else(|| ApiError::bad_request("missing proof"))?;
    let expected_proof_purpose = req
        .options
        .expected_proof_purpose
        .as_deref()
        .unwrap_or("assertionMethod");
    if proof.get("proofPurpose").and_then(Value::as_str) != Some(expected_proof_purpose) {
        return Err(ApiError::bad_request("proof.proofPurpose does not match"));
    }
    if let Some(expected_domain) = req.options.domain.as_deref() {
        if proof.get("domain").and_then(Value::as_str) != Some(expected_domain) {
            return Err(ApiError::bad_request("proof.domain does not match"));
        }
    }
    if let Some(expected_challenge) = req.options.challenge.as_deref() {
        if proof.get("challenge").and_then(Value::as_str) != Some(expected_challenge) {
            return Err(ApiError::bad_request("proof.challenge does not match"));
        }
    }

    let proof_value = proof
        .get("proofValue")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("missing proof.proofValue"))?;
    let kind = if proof_value.starts_with("u2V0C") {
        VerificationKind::Signature
    } else if proof_value.starts_with("u2V0D") {
        VerificationKind::Proof
    } else {
        return Err(ApiError::bad_request(
            "unsupported bbs-2023 proofValue multibase/CBOR prefix",
        ));
    };
    Ok((req.verifiable_credential, kind))
}

fn compact_verify_results(results: Vec<Result<Value, String>>, fail_fast: bool) -> Value {
    let mut verified = Vec::with_capacity(results.len());
    let mut errors = Vec::new();
    for (index, result) in results.into_iter().enumerate() {
        match result {
            Ok(value) => {
                verified.push(value.get("verified").and_then(Value::as_bool) == Some(true))
            }
            Err(error) => {
                verified.push(false);
                errors.push(json!({"index": index, "error": error}));
                if fail_fast {
                    break;
                }
            }
        }
    }
    let attempted = verified.len();
    let failed = errors.len();
    json!({
        "ok": failed == 0,
        "attempted": attempted,
        "succeeded": attempted - failed,
        "failed": failed,
        "verified": verified,
        "errors": errors,
    })
}

fn validate_managed_bulk_size(item_count: usize, max_items: usize) -> Result<(), ApiError> {
    if item_count == 0 {
        return Err(ApiError::bad_request(
            "managed bulk request must contain at least one item",
        ));
    }
    if item_count > max_items {
        return Err(ApiError::bad_request(format!(
            "managed bulk request contains {item_count} items; maximum is {max_items}"
        )));
    }
    Ok(())
}

fn compact_derive_results(results: Vec<Result<Value, String>>, fail_fast: bool) -> Value {
    let mut proof_values = Vec::with_capacity(results.len());
    let mut errors = Vec::new();
    for (index, result) in results.into_iter().enumerate() {
        match result {
            Ok(value) => match value
                .as_str()
                .or_else(|| value.pointer("/proof/proofValue").and_then(Value::as_str))
            {
                Some(proof_value) => proof_values.push(Some(proof_value.to_owned())),
                None => {
                    proof_values.push(None);
                    errors.push(json!({
                        "index": index,
                        "error": "derived document is missing proof.proofValue"
                    }));
                    if fail_fast {
                        break;
                    }
                }
            },
            Err(error) => {
                proof_values.push(None);
                errors.push(json!({"index": index, "error": error}));
                if fail_fast {
                    break;
                }
            }
        }
    }
    let attempted = proof_values.len();
    let failed = errors.len();
    json!({
        "ok": failed == 0,
        "attempted": attempted,
        "succeeded": attempted - failed,
        "failed": failed,
        "proofValues": proof_values,
        "errors": errors,
    })
}

fn api_results_from_values<T>(
    results: Vec<Result<T, String>>,
    fail_fast: bool,
) -> ApiBatchResult<T> {
    let mut items = Vec::with_capacity(results.len());
    for (index, result) in results.into_iter().enumerate() {
        let failed = result.is_err();
        items.push(match result {
            Ok(value) => ApiBatchItem {
                index,
                ok: true,
                value: Some(value),
                error: None,
            },
            Err(error) => ApiBatchItem {
                index,
                ok: false,
                value: None,
                error: Some(error),
            },
        });
        if fail_fast && failed {
            break;
        }
    }
    let succeeded = items.iter().filter(|item| item.ok).count();
    let failed = items.len() - succeeded;
    ApiBatchResult {
        ok: failed == 0,
        attempted: items.len(),
        succeeded,
        failed,
        results: items,
    }
}

async fn api_results_from_receivers<T>(
    receivers: Vec<BatchReceiver<T>>,
    fail_fast: bool,
) -> ApiBatchResult<T> {
    let mut results = Vec::with_capacity(receivers.len());
    for (index, receiver) in receivers.into_iter().enumerate() {
        let result = receiver.receive().await;
        let failed = result.is_err();
        results.push(match result {
            Ok(value) => ApiBatchItem {
                index,
                ok: true,
                value: Some(value),
                error: None,
            },
            Err(error) => ApiBatchItem {
                index,
                ok: false,
                value: None,
                error: Some(error),
            },
        });
        if fail_fast && failed {
            break;
        }
    }
    let succeeded = results.iter().filter(|item| item.ok).count();
    let failed = results.len() - succeeded;
    ApiBatchResult {
        ok: failed == 0,
        attempted: results.len(),
        succeeded,
        failed,
        results,
    }
}

fn managed_verification_values(results: Vec<Result<bool, String>>) -> Vec<Result<Value, String>> {
    results
        .into_iter()
        .map(|result| result.map(|verified| json!({"verified": verified, "checks": ["proof"]})))
        .collect()
}

async fn process_issue_batch(
    engine: Arc<BbsEngine>,
    items: Vec<IssueBatchItem>,
    verification_cache: Arc<VerificationResultCache>,
) -> Vec<Result<IssueBatchOutput, String>> {
    let expected = items.len();
    let mut ordered = (0..expected)
        .map(|_| None)
        .collect::<Vec<Option<Result<IssueBatchOutput, String>>>>();
    let mut prepared_contexts = Vec::new();
    let mut sign_requests = Vec::new();
    let mut general_indexes = Vec::new();
    let mut general_requests = Vec::new();

    for (index, item) in items.into_iter().enumerate() {
        match item {
            IssueBatchItem::Prepared { template, hmac_key } => {
                sign_requests.push(template.sign_request());
                prepared_contexts.push((index, template, hmac_key));
            }
            IssueBatchItem::General(request) => {
                general_indexes.push(index);
                general_requests.push(request);
            }
        }
    }

    if !sign_requests.is_empty() {
        match tokio::task::spawn_blocking(move || engine.sign_batch_shared(&sign_requests, false))
            .await
        {
            Ok(batch) => {
                for item in batch.results {
                    if let Some((index, template, hmac_key)) = prepared_contexts.get(item.index) {
                        let result = item
                            .result
                            .map_err(|error| error.to_string())
                            .and_then(|signature| template.finish(signature, hmac_key))
                            .map(IssueBatchOutput::Serialized);
                        ordered[*index] = Some(result);
                    }
                }
            }
            Err(error) => {
                for (index, _, _) in &prepared_contexts {
                    ordered[*index] = Some(Err(format!("BBS Sign worker failed: {error}")));
                }
            }
        }
    }

    if !general_requests.is_empty() {
        let batch = issue_jsonld_base_proof_batch(IssueJsonLdBaseProofBatchRequest {
            items: general_requests,
            fail_fast: false,
        })
        .await;
        for item in batch.results {
            if let Some(index) = general_indexes.get(item.index) {
                let result = item.result.map(IssueBatchOutput::Value);
                ordered[*index] = Some(result);
            }
        }
    }

    ordered
        .into_iter()
        .map(|result| {
            let result = result.unwrap_or_else(|| Err("missing W3C Issue batch result".to_owned()));
            if verification_cache.enabled() {
                if let Ok(output) = &result {
                    match output {
                        IssueBatchOutput::Value(document) => verification_cache.insert(document),
                        IssueBatchOutput::Serialized(body) => {
                            if let Ok(document) = serde_json::from_slice(body) {
                                verification_cache.insert(&document);
                            }
                        }
                    }
                }
            }
            result
        })
        .collect()
}

async fn process_derive_batch(
    items: Vec<DeriveJsonLdDisclosureProofRequest>,
    verification_cache: Arc<VerificationResultCache>,
) -> Vec<Result<Value, String>> {
    let expected = items.len();
    let cacheable_outputs = items
        .iter()
        .map(|item| verification_cache.contains(&item.signed_document))
        .collect::<Vec<_>>();
    let results = ordered_batch_results(
        derive_jsonld_disclosure_proof_batch(DeriveJsonLdDisclosureProofBatchRequest {
            items,
            fail_fast: false,
        })
        .await,
        expected,
    );
    for (cacheable, result) in cacheable_outputs.into_iter().zip(&results) {
        if cacheable {
            if let Ok(document) = result {
                verification_cache.insert(document);
            }
        }
    }
    results
}

async fn process_prepared_base_verification(
    engine: Arc<BbsEngine>,
    backend: BbsBackend,
    items: Vec<Arc<BbsVerifyRequest>>,
) -> Vec<Result<bool, String>> {
    let expected = items.len();
    let _admission = match acquire_w3c_verify_crypto_admission_for_backend(backend, expected).await
    {
        Ok(permit) => permit,
        Err(error) => return (0..expected).map(|_| Err(error.clone())).collect(),
    };
    let mut ordered = (0..expected)
        .map(|_| None)
        .collect::<Vec<Option<Result<bool, String>>>>();
    let batch = match tokio::task::spawn_blocking(move || engine.verify_batch_shared(&items, false))
        .await
    {
        Ok(batch) => batch,
        Err(error) => {
            return (0..expected)
                .map(|_| Err(format!("BBS Verify worker failed: {error}")))
                .collect();
        }
    };
    for item in batch.results {
        if item.index < expected {
            ordered[item.index] = Some(match item.result {
                Ok(true) => Ok(true),
                Ok(false) => Err("proof verification failed".to_owned()),
                Err(error) => Err(error.to_string()),
            });
        }
    }
    ordered
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err("missing BBS batch result".to_owned())))
        .collect()
}

async fn process_prepared_proof_verification(
    engine: Arc<BbsEngine>,
    backend: BbsBackend,
    items: Vec<Arc<ProofVerifyRequest>>,
) -> Vec<Result<bool, String>> {
    let expected = items.len();
    let _admission = match acquire_w3c_verify_crypto_admission_for_backend(backend, expected).await
    {
        Ok(permit) => permit,
        Err(error) => return (0..expected).map(|_| Err(error.clone())).collect(),
    };
    let batch = match tokio::task::spawn_blocking(move || {
        let requests = items.iter().map(|item| (**item).clone()).collect();
        engine.proof_verify_batch(requests, false)
    })
    .await
    {
        Ok(batch) => batch,
        Err(error) => {
            return (0..expected)
                .map(|_| Err(format!("BBS ProofVerify worker failed: {error}")))
                .collect();
        }
    };
    let mut ordered = (0..expected)
        .map(|_| None)
        .collect::<Vec<Option<Result<bool, String>>>>();
    for item in batch.results {
        if item.index < expected {
            ordered[item.index] = Some(match item.result {
                Ok(true) => Ok(true),
                Ok(false) => Err("proof verification failed".to_owned()),
                Err(error) => Err(error.to_string()),
            });
        }
    }
    ordered
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err("missing BBS batch result".to_owned())))
        .collect()
}

fn ordered_batch_results<T>(batch: BatchRunResult<T>, expected: usize) -> Vec<Result<T, String>> {
    let mut ordered = (0..expected)
        .map(|_| None)
        .collect::<Vec<Option<Result<T, String>>>>();
    for item in batch.results {
        if item.index < expected {
            ordered[item.index] = Some(item.result);
        }
    }
    ordered
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err("missing W3C batch result".to_owned())))
        .collect()
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_template_hot_entry_remains_exact_and_bounded() {
        let cache = RawTemplateCache {
            capacity: 1,
            state: RwLock::new(RawTemplateCacheState::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        };
        let first = Bytes::from_static(br#"{"subject":"alice"}"#);
        let second = Bytes::from_static(br#"{"subject":"bob"}"#);

        cache.insert(first.clone(), 11u64);
        assert_eq!(*cache.get(&first).unwrap(), 11);
        assert!(cache.get(&second).is_none());

        cache.insert(second.clone(), 22u64);
        assert!(cache.get(&first).is_none());
        assert_eq!(*cache.get(&second).unwrap(), 22);
        assert_eq!(cache.snapshot().entries, 1);
    }

    #[test]
    fn verification_cache_is_content_exact_and_bounded() {
        let cache = VerificationResultCache::with_capacity(2);
        let first = json!({"credentialSubject": {"name": "Alice"}, "proof": "one"});
        let second = json!({"credentialSubject": {"name": "Bob"}, "proof": "two"});
        let third = json!({"credentialSubject": {"name": "Carol"}, "proof": "three"});

        cache.insert(&first);
        assert!(cache.contains(&first));

        let mut tampered = first.clone();
        tampered["credentialSubject"]["name"] = Value::String("Mallory".to_owned());
        assert!(!cache.contains(&tampered));

        cache.insert(&second);
        cache.insert(&third);
        assert!(!cache.contains(&first));
        assert!(cache.contains(&second));
        assert!(cache.contains(&third));
        assert_eq!(cache.snapshot().entries, 2);
    }

    #[test]
    fn compact_managed_results_preserve_item_count_and_errors() {
        let verified = compact_verify_results(
            vec![Ok(json!({"verified": true})), Ok(json!({"verified": true}))],
            false,
        );
        assert_eq!(verified["attempted"], 2);
        assert_eq!(verified["verified"], json!([true, true]));

        let derived = compact_derive_results(
            vec![Ok(json!("proof-1")), Err("bad proof".to_owned())],
            false,
        );
        assert_eq!(derived["attempted"], 2);
        assert_eq!(derived["failed"], 1);
        assert_eq!(derived["proofValues"], json!(["proof-1", null]));
        assert_eq!(derived["errors"][0]["index"], 1);
    }

    #[test]
    fn managed_bulk_size_is_bounded() {
        assert!(validate_managed_bulk_size(1, 4096).is_ok());
        assert!(validate_managed_bulk_size(4096, 4096).is_ok());
        assert!(validate_managed_bulk_size(0, 4096).is_err());
        assert!(validate_managed_bulk_size(4097, 4096).is_err());
    }
}
