use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Instant,
};

use cubbs_bbs::{BbsBackend, SignRequest};
use cubbs_bbs_proofvalue::{parse_base, parse_derived, serialize_base, BaseProofValue};
use futures::{stream, StreamExt};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::suite::{
    attach_proof, detach_single_proof, ensure_data_integrity_context, require_credential_subject,
};
use crate::{
    build_proof_configuration, canonicalize_and_group_jsonld, canonicalize_jsonld,
    derive_disclosure_proof_batch, hash_joined_strings, issue_base_proof_batch,
    json_content_digest, public_key_from_verification_method, relabel_and_sort_canonical_nquads,
    verify_base_proof_batch, verify_disclosure_proof_batch, DeriveJsonLdDisclosureProofRequest,
    DisclosureGroups, IssueBaseProofBatchRequest, IssueBaseProofRequest,
    IssueJsonLdBaseProofRequest, PreparedBaseVerification, PreparedBaseVerificationBatchRequest,
    PreparedDisclosureBatchRequest, PreparedDisclosureRequest, PreparedDisclosureVerification,
    PreparedDisclosureVerificationBatchRequest, SuiteJsonLdError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchItemResult<T> {
    pub index: usize,
    pub result: Result<T, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchRunResult<T> {
    pub results: Vec<BatchItemResult<T>>,
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IssueJsonLdBaseProofBatchRequest {
    pub items: Vec<IssueJsonLdBaseProofRequest>,
    pub fail_fast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeriveJsonLdDisclosureProofBatchRequest {
    pub items: Vec<DeriveJsonLdDisclosureProofRequest>,
    pub fail_fast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyJsonLdBaseProofBatchRequest {
    pub items: Vec<Value>,
    pub fail_fast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyJsonLdDisclosureProofBatchRequest {
    pub items: Vec<(Value, Option<Vec<u8>>)>,
    pub fail_fast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedJsonLdDisclosureProof {
    pub selected_document: Value,
    pub base_proof: Value,
    pub primitive: PreparedDisclosureRequest,
}

/// HMAC-independent issuance material for credentials whose canonical
/// N-Quads contain no blank-node labels. The W3C service can share this
/// immutable template while still generating a fresh HMAC key and executing
/// one BBS Sign operation for every request.
#[derive(Debug, Clone)]
pub struct PreparedJsonLdBaseIssue {
    document: Value,
    proof_options: Value,
    sign_request: Arc<SignRequest>,
    header: Vec<u8>,
    mandatory_pointers: Vec<String>,
}

impl PreparedJsonLdBaseIssue {
    pub fn sign_request(&self) -> &Arc<SignRequest> {
        &self.sign_request
    }

    pub fn proof_value(
        &self,
        bbs_signature: Vec<u8>,
        hmac_key: &[u8],
    ) -> Result<String, SuiteJsonLdError> {
        if hmac_key.len() != crate::HMAC_KEY_LEN {
            return Err(
                crate::JsonLdCanonicalizationError::InvalidHmacKeyLength(hmac_key.len()).into(),
            );
        }
        Ok(serialize_base(&BaseProofValue {
            bbs_signature,
            bbs_header: self.header.clone(),
            public_key: self.sign_request.public_key.clone(),
            hmac_key: hmac_key.to_vec(),
            mandatory_pointers: self.mandatory_pointers.clone(),
        })?)
    }

    pub fn assemble_document(&self, proof_value: String) -> Result<Value, SuiteJsonLdError> {
        let mut proof = self.proof_options.clone();
        proof
            .as_object_mut()
            .ok_or(SuiteJsonLdError::DocumentMustBeObject)?
            .insert("proofValue".to_owned(), Value::String(proof_value));
        attach_proof(self.document.clone(), proof)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct W3cBatchTimingSnapshot {
    pub issue_prepare_ms: f64,
    pub issue_proof_config_ms: f64,
    pub issue_proof_canonicalize_ms: f64,
    pub issue_group_ms: f64,
    pub issue_crypto_ms: f64,
    pub derive_prepare_ms: f64,
    pub derive_parse_ms: f64,
    pub derive_group_ms: f64,
    pub derive_crypto_ms: f64,
    pub verify_base_prepare_ms: f64,
    pub verify_base_crypto_ms: f64,
    pub verify_disclosure_prepare_ms: f64,
    pub verify_disclosure_crypto_ms: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedCacheSnapshot {
    pub capacity_per_operation: usize,
    pub issue_entries: usize,
    pub derive_entries: usize,
    pub verify_base_entries: usize,
    pub verify_disclosure_entries: usize,
    pub issue_hits: u64,
    pub issue_misses: u64,
    pub derive_hits: u64,
    pub derive_misses: u64,
    pub verify_base_hits: u64,
    pub verify_base_misses: u64,
    pub verify_disclosure_hits: u64,
    pub verify_disclosure_misses: u64,
}

#[derive(Debug, Default)]
struct W3cBatchTimings {
    issue_prepare_micros: AtomicU64,
    issue_proof_config_micros: AtomicU64,
    issue_proof_canonicalize_micros: AtomicU64,
    issue_group_micros: AtomicU64,
    issue_crypto_micros: AtomicU64,
    derive_prepare_micros: AtomicU64,
    derive_parse_micros: AtomicU64,
    derive_group_micros: AtomicU64,
    derive_crypto_micros: AtomicU64,
    verify_base_prepare_micros: AtomicU64,
    verify_base_crypto_micros: AtomicU64,
    verify_disclosure_prepare_micros: AtomicU64,
    verify_disclosure_crypto_micros: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PreparedCacheKey([u8; 32]);

#[derive(Debug)]
struct PreparedCacheState<V> {
    entries: HashMap<PreparedCacheKey, Arc<V>>,
    insertion_order: VecDeque<PreparedCacheKey>,
}

impl<V> Default for PreparedCacheState<V> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
        }
    }
}

#[derive(Debug)]
struct PreparedCache<V> {
    capacity: usize,
    state: Mutex<PreparedCacheState<V>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<V: Clone> PreparedCache<V> {
    fn new() -> Self {
        Self {
            capacity: prepared_cache_capacity(),
            state: Mutex::new(PreparedCacheState::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    fn get(&self, key: &PreparedCacheKey) -> Option<V> {
        if self.capacity == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let value = self
            .state
            .lock()
            .expect("prepared cache mutex poisoned")
            .entries
            .get(key)
            .cloned();
        match value {
            Some(value) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some((*value).clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    fn insert(&self, key: PreparedCacheKey, value: V) {
        if self.capacity == 0 {
            return;
        }
        let mut state = self.state.lock().expect("prepared cache mutex poisoned");
        if state.entries.contains_key(&key) {
            return;
        }
        while state.entries.len() >= self.capacity {
            let Some(oldest) = state.insertion_order.pop_front() else {
                break;
            };
            state.entries.remove(&oldest);
        }
        state.insertion_order.push_back(key);
        state.entries.insert(key, Arc::new(value));
    }

    fn entries(&self) -> usize {
        self.state
            .lock()
            .expect("prepared cache mutex poisoned")
            .entries
            .len()
    }
}

type PreparedIssueStatic = (Value, Value, crate::Sha256Digest);
type PreparedDeriveItem = (Value, Value, PreparedDisclosureRequest);

// The GPU verifier allocates sizeable per-batch MSM workspaces.  W3C keeps
// JSON-LD preparation concurrent, but must not let independent prepared
// batches collectively exhaust device memory before their crypto sections
// run.  This is deliberately counted in items rather than batches: a 512-item
// batch costs materially more GPU memory than a singleton.
// A 16 GiB RTX 5060 Ti reaches allocator pressure near an 8,192-item
// admission budget under 20 concurrent 512-item HTTP batches.  Keep the
// process-wide default below that observed boundary; larger devices may opt
// in to a higher value only after an equivalent sustained qualification run.
const DEFAULT_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS: usize = 4_096;

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct W3cVerifyCryptoAdmissionSnapshot {
    /// `false` for CPU/no-backend operation, where Verify crypto is not
    /// throttled by this GPU-memory admission controller.
    pub enabled: bool,
    /// The configured GPU item budget when enabled; `None` when disabled.
    pub item_budget: Option<usize>,
    pub in_flight_items: usize,
    pub max_in_flight_items: usize,
    pub waiting_batches: usize,
    pub waiting_items: usize,
    pub total_admission_wait_ms: f64,
}

struct W3cVerifyCryptoAdmission {
    item_budget: usize,
    permits: Arc<Semaphore>,
    in_flight_items: AtomicUsize,
    max_in_flight_items: AtomicUsize,
    waiting_batches: AtomicUsize,
    waiting_items: AtomicUsize,
    total_admission_wait_micros: AtomicU64,
}

/// Opaque RAII permit covering one W3C Verify crypto batch.
pub struct W3cVerifyCryptoPermit {
    admission: Arc<W3cVerifyCryptoAdmission>,
    item_count: usize,
    permit: Option<OwnedSemaphorePermit>,
}

struct W3cVerifyCryptoWaiter {
    admission: Arc<W3cVerifyCryptoAdmission>,
    item_count: usize,
    active: bool,
}

impl W3cVerifyCryptoAdmission {
    fn from_env() -> Self {
        let item_budget = std::env::var("CUBBS_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS")
            .ok()
            .map(|value| {
                value.parse::<usize>().unwrap_or_else(|_| {
                    panic!("CUBBS_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS must be a positive integer")
                })
            })
            .unwrap_or(DEFAULT_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS);
        Self::new(item_budget)
    }

    fn new(item_budget: usize) -> Self {
        assert!(
            item_budget > 0 && item_budget <= u32::MAX as usize,
            "W3C GPU Verify item budget must be in 1..={}",
            u32::MAX
        );
        Self {
            item_budget,
            permits: Arc::new(Semaphore::new(item_budget)),
            in_flight_items: AtomicUsize::new(0),
            max_in_flight_items: AtomicUsize::new(0),
            waiting_batches: AtomicUsize::new(0),
            waiting_items: AtomicUsize::new(0),
            total_admission_wait_micros: AtomicU64::new(0),
        }
    }

    async fn acquire(self: &Arc<Self>, item_count: usize) -> Result<W3cVerifyCryptoPermit, String> {
        if item_count == 0 {
            return Ok(W3cVerifyCryptoPermit {
                admission: Arc::clone(self),
                item_count,
                permit: None,
            });
        }
        if item_count > self.item_budget {
            return Err(format!(
                "W3C GPU Verify batch has {item_count} prepared items, exceeding the configured admission budget {}",
                self.item_budget
            ));
        }

        let mut waiter = W3cVerifyCryptoWaiter::new(self, item_count);
        let wait_started = Instant::now();
        let permit = self
            .permits
            .clone()
            .acquire_many_owned(item_count as u32)
            .await
            .map_err(|_| "W3C GPU Verify admission semaphore is closed".to_owned());
        waiter.finish();
        self.total_admission_wait_micros
            .fetch_add(elapsed_micros(wait_started), Ordering::Relaxed);
        let permit = permit?;
        let in_flight = self
            .in_flight_items
            .fetch_add(item_count, Ordering::Relaxed)
            + item_count;
        self.max_in_flight_items
            .fetch_max(in_flight, Ordering::Relaxed);
        Ok(W3cVerifyCryptoPermit {
            admission: Arc::clone(self),
            item_count,
            permit: Some(permit),
        })
    }

    fn snapshot(&self) -> W3cVerifyCryptoAdmissionSnapshot {
        W3cVerifyCryptoAdmissionSnapshot {
            enabled: true,
            item_budget: Some(self.item_budget),
            in_flight_items: self.in_flight_items.load(Ordering::Relaxed),
            max_in_flight_items: self.max_in_flight_items.load(Ordering::Relaxed),
            waiting_batches: self.waiting_batches.load(Ordering::Relaxed),
            waiting_items: self.waiting_items.load(Ordering::Relaxed),
            total_admission_wait_ms: self.total_admission_wait_micros.load(Ordering::Relaxed)
                as f64
                / 1000.0,
        }
    }
}

impl W3cVerifyCryptoAdmissionSnapshot {
    fn disabled() -> Self {
        Self {
            enabled: false,
            item_budget: None,
            in_flight_items: 0,
            max_in_flight_items: 0,
            waiting_batches: 0,
            waiting_items: 0,
            total_admission_wait_ms: 0.0,
        }
    }
}

impl W3cVerifyCryptoWaiter {
    fn new(admission: &Arc<W3cVerifyCryptoAdmission>, item_count: usize) -> Self {
        admission.waiting_batches.fetch_add(1, Ordering::Relaxed);
        admission
            .waiting_items
            .fetch_add(item_count, Ordering::Relaxed);
        Self {
            admission: Arc::clone(admission),
            item_count,
            active: true,
        }
    }

    fn finish(&mut self) {
        if self.active {
            self.admission
                .waiting_batches
                .fetch_sub(1, Ordering::Relaxed);
            self.admission
                .waiting_items
                .fetch_sub(self.item_count, Ordering::Relaxed);
            self.active = false;
        }
    }
}

impl Drop for W3cVerifyCryptoWaiter {
    fn drop(&mut self) {
        self.finish();
    }
}

impl Drop for W3cVerifyCryptoPermit {
    fn drop(&mut self) {
        if self.item_count != 0 {
            self.admission
                .in_flight_items
                .fetch_sub(self.item_count, Ordering::Relaxed);
        }
        // Drop the actual semaphore permit after recording the item count so a
        // health snapshot never reports more admitted work than the limit.
        drop(self.permit.take());
    }
}

pub async fn issue_jsonld_base_proof_batch(
    req: IssueJsonLdBaseProofBatchRequest,
) -> BatchRunResult<Value> {
    let mut results = Vec::with_capacity(req.items.len());
    let mut prepared = Vec::new();
    let mut contexts = Vec::new();

    let prepare_started = Instant::now();
    let workers = jsonld_batch_workers();
    let prepared_items = if tokio::runtime::Handle::try_current().is_ok() {
        stream::iter(
            req.items
                .into_iter()
                .map(|item| tokio::spawn(prepare_issue_item(item))),
        )
        .buffered(workers)
        .map(|joined| {
            joined.unwrap_or_else(|error| Err(SuiteJsonLdError::TaskJoin(error.to_string())))
        })
        .collect::<Vec<_>>()
        .await
    } else {
        stream::iter(req.items.into_iter().map(prepare_issue_item))
            .buffered(workers)
            .collect::<Vec<_>>()
            .await
    };
    w3c_batch_timings()
        .issue_prepare_micros
        .fetch_add(elapsed_micros(prepare_started), Ordering::Relaxed);
    for (index, prepared_item) in prepared_items.into_iter().enumerate() {
        match prepared_item {
            Ok((document, proof_options, prepared_item)) => {
                contexts.push((index, document, proof_options));
                prepared.push(prepared_item);
            }
            Err(error) => {
                results.push(BatchItemResult {
                    index,
                    result: Err(error.to_string()),
                });
                if req.fail_fast {
                    return batch_result(results);
                }
            }
        }
    }

    let crypto_started = Instant::now();
    let issued = match issue_base_proof_batch(IssueBaseProofBatchRequest {
        items: prepared,
        fail_fast: req.fail_fast,
    }) {
        Ok(result) => result,
        Err(error) => {
            for (index, _, _) in contexts {
                results.push(BatchItemResult {
                    index,
                    result: Err(error.to_string()),
                });
                if req.fail_fast {
                    break;
                }
            }
            results.sort_by_key(|item| item.index);
            return batch_result(results);
        }
    };
    w3c_batch_timings()
        .issue_crypto_micros
        .fetch_add(elapsed_micros(crypto_started), Ordering::Relaxed);

    for item in issued.results {
        let (index, document, mut proof) = contexts[item.index].clone();
        let result = item
            .result
            .map_err(|error| error.to_string())
            .and_then(|proof_value| {
                proof
                    .as_object_mut()
                    .ok_or_else(|| SuiteJsonLdError::DocumentMustBeObject.to_string())?
                    .insert("proofValue".to_owned(), Value::String(proof_value));
                attach_proof(document, proof).map_err(|error| error.to_string())
            });
        results.push(BatchItemResult { index, result });
    }
    results.sort_by_key(|item| item.index);
    batch_result(results)
}

/// Prepare the immutable portion of a repeated JSON-LD issuance request.
///
/// A fresh HMAC changes blank-node labels, so documents containing blank
/// nodes return `Ok(None)` and must use the general issuance path. For the
/// common credential shape without blank nodes, all primitive messages and
/// the response document are safe to share across independent requests.
pub async fn prepare_jsonld_base_issue(
    req: &IssueJsonLdBaseProofRequest,
) -> Result<Option<PreparedJsonLdBaseIssue>, SuiteJsonLdError> {
    let (document, proof_options, primitive) = prepare_issue_item(req.clone()).await?;
    if !canonicalize_jsonld(&document)
        .await?
        .canonical_id_map
        .is_empty()
    {
        return Ok(None);
    }

    let mandatory_hash = crate::hash_mandatory(&primitive.mandatory);
    let header = crate::bbs_header(primitive.proof_hash, mandatory_hash);
    Ok(Some(PreparedJsonLdBaseIssue {
        document,
        proof_options,
        sign_request: Arc::new(SignRequest {
            secret_key: primitive.secret_key,
            public_key: primitive.public_key,
            header: header.clone(),
            messages: primitive
                .non_mandatory
                .iter()
                .map(|message| message.as_bytes().to_vec())
                .collect(),
        }),
        header,
        mandatory_pointers: primitive.mandatory_pointers,
    }))
}

pub async fn derive_jsonld_disclosure_proof_batch(
    req: DeriveJsonLdDisclosureProofBatchRequest,
) -> BatchRunResult<Value> {
    let mut results = Vec::with_capacity(req.items.len());
    let mut prepared = Vec::new();
    let mut contexts = Vec::new();

    let prepare_started = Instant::now();
    let workers = jsonld_batch_workers();
    let prepared_items = if tokio::runtime::Handle::try_current().is_ok() {
        stream::iter(
            req.items
                .into_iter()
                .map(|item| tokio::spawn(prepare_derive_item(item))),
        )
        .buffered(workers)
        .map(|joined| {
            joined.unwrap_or_else(|error| Err(SuiteJsonLdError::TaskJoin(error.to_string())))
        })
        .collect::<Vec<_>>()
        .await
    } else {
        stream::iter(req.items.into_iter().map(prepare_derive_item))
            .buffered(workers)
            .collect::<Vec<_>>()
            .await
    };
    w3c_batch_timings()
        .derive_prepare_micros
        .fetch_add(elapsed_micros(prepare_started), Ordering::Relaxed);
    for (index, prepared_item) in prepared_items.into_iter().enumerate() {
        match prepared_item {
            Ok((selected_document, base_proof, prepared_item)) => {
                contexts.push((index, selected_document, base_proof));
                prepared.push(prepared_item);
            }
            Err(error) => {
                results.push(BatchItemResult {
                    index,
                    result: Err(error.to_string()),
                });
                if req.fail_fast {
                    return batch_result(results);
                }
            }
        }
    }

    let crypto_started = Instant::now();
    let derived = match derive_disclosure_proof_batch(PreparedDisclosureBatchRequest {
        items: prepared,
        fail_fast: req.fail_fast,
    }) {
        Ok(result) => result,
        Err(error) => {
            for (index, _, _) in contexts {
                results.push(BatchItemResult {
                    index,
                    result: Err(error.to_string()),
                });
                if req.fail_fast {
                    break;
                }
            }
            results.sort_by_key(|item| item.index);
            return batch_result(results);
        }
    };
    w3c_batch_timings()
        .derive_crypto_micros
        .fetch_add(elapsed_micros(crypto_started), Ordering::Relaxed);

    for item in derived.results {
        let (index, selected_document, mut proof) = contexts[item.index].clone();
        let result = item
            .result
            .map_err(|error| error.to_string())
            .and_then(|material| {
                proof
                    .as_object_mut()
                    .ok_or_else(|| SuiteJsonLdError::DocumentMustBeObject.to_string())?
                    .insert("proofValue".to_owned(), Value::String(material.proof_value));
                attach_proof(selected_document, proof).map_err(|error| error.to_string())
            });
        results.push(BatchItemResult { index, result });
    }
    results.sort_by_key(|item| item.index);
    batch_result(results)
}

pub async fn verify_jsonld_base_proof_batch(
    req: VerifyJsonLdBaseProofBatchRequest,
) -> BatchRunResult<bool> {
    let mut results = Vec::with_capacity(req.items.len());
    let mut prepared = Vec::new();
    let mut contexts = Vec::new();

    let prepare_started = Instant::now();
    let workers = jsonld_batch_workers();
    let prepared_items = if tokio::runtime::Handle::try_current().is_ok() {
        stream::iter(
            req.items
                .into_iter()
                .map(|item| tokio::spawn(prepare_verify_base_item(item))),
        )
        .buffered(workers)
        .map(|joined| {
            joined.unwrap_or_else(|error| Err(SuiteJsonLdError::TaskJoin(error.to_string())))
        })
        .collect::<Vec<_>>()
        .await
    } else {
        stream::iter(req.items.into_iter().map(prepare_verify_base_item))
            .buffered(workers)
            .collect::<Vec<_>>()
            .await
    };
    w3c_batch_timings()
        .verify_base_prepare_micros
        .fetch_add(elapsed_micros(prepare_started), Ordering::Relaxed);
    for (index, prepared_item) in prepared_items.into_iter().enumerate() {
        match prepared_item {
            Ok(prepared_item) => {
                contexts.push(index);
                prepared.push(prepared_item);
            }
            Err(error) => {
                results.push(BatchItemResult {
                    index,
                    result: Err(error.to_string()),
                });
                if req.fail_fast {
                    return batch_result(results);
                }
            }
        }
    }

    // Do not take a GPU permit while JSON-LD/RDFC work is still in progress:
    // preparation remains fully concurrent across W3C pipelines.  The permit
    // only spans the device-backed Verify crypto call below.
    let _admission = if let Some(admission) = w3c_verify_crypto_admission() {
        match admission.acquire(prepared.len()).await {
            Ok(permit) => Some(permit),
            Err(error) => {
                for index in contexts {
                    results.push(BatchItemResult {
                        index,
                        result: Err(error.clone()),
                    });
                    if req.fail_fast {
                        break;
                    }
                }
                results.sort_by_key(|item| item.index);
                return batch_result(results);
            }
        }
    } else {
        None
    };
    let crypto_started = Instant::now();
    let verified = match verify_base_proof_batch(PreparedBaseVerificationBatchRequest {
        items: prepared,
        fail_fast: req.fail_fast,
    }) {
        Ok(result) => result,
        Err(error) => {
            for index in contexts {
                results.push(BatchItemResult {
                    index,
                    result: Err(error.to_string()),
                });
                if req.fail_fast {
                    break;
                }
            }
            results.sort_by_key(|item| item.index);
            return batch_result(results);
        }
    };
    w3c_batch_timings()
        .verify_base_crypto_micros
        .fetch_add(elapsed_micros(crypto_started), Ordering::Relaxed);

    for item in verified.results {
        results.push(BatchItemResult {
            index: contexts[item.index],
            result: item.result.map_err(|error| error.to_string()),
        });
    }
    results.sort_by_key(|item| item.index);
    batch_result(results)
}

pub async fn verify_jsonld_disclosure_proof_batch(
    req: VerifyJsonLdDisclosureProofBatchRequest,
) -> BatchRunResult<bool> {
    let mut results = Vec::with_capacity(req.items.len());
    let mut prepared = Vec::new();
    let mut contexts = Vec::new();

    let prepare_started = Instant::now();
    let workers = jsonld_batch_workers();
    let prepared_items =
        if tokio::runtime::Handle::try_current().is_ok() {
            stream::iter(req.items.into_iter().map(|(item, expected)| {
                tokio::spawn(prepare_verify_disclosure_item(item, expected))
            }))
            .buffered(workers)
            .map(|joined| {
                joined.unwrap_or_else(|error| Err(SuiteJsonLdError::TaskJoin(error.to_string())))
            })
            .collect::<Vec<_>>()
            .await
        } else {
            stream::iter(
                req.items
                    .into_iter()
                    .map(|(item, expected)| prepare_verify_disclosure_item(item, expected)),
            )
            .buffered(workers)
            .collect::<Vec<_>>()
            .await
        };
    w3c_batch_timings()
        .verify_disclosure_prepare_micros
        .fetch_add(elapsed_micros(prepare_started), Ordering::Relaxed);
    for (index, prepared_item) in prepared_items.into_iter().enumerate() {
        match prepared_item {
            Ok(prepared_item) => {
                contexts.push(index);
                prepared.push(prepared_item);
            }
            Err(error) => {
                results.push(BatchItemResult {
                    index,
                    result: Err(error.to_string()),
                });
                if req.fail_fast {
                    return batch_result(results);
                }
            }
        }
    }

    // The same weighted admission is shared with base verification.  Derived
    // proof verification also invokes GPU MSM/pairing work and therefore must
    // not bypass the device-memory budget under a mixed W3C workload.
    let _admission = if let Some(admission) = w3c_verify_crypto_admission() {
        match admission.acquire(prepared.len()).await {
            Ok(permit) => Some(permit),
            Err(error) => {
                for index in contexts {
                    results.push(BatchItemResult {
                        index,
                        result: Err(error.clone()),
                    });
                    if req.fail_fast {
                        break;
                    }
                }
                results.sort_by_key(|item| item.index);
                return batch_result(results);
            }
        }
    } else {
        None
    };
    let crypto_started = Instant::now();
    let verified = match verify_disclosure_proof_batch(PreparedDisclosureVerificationBatchRequest {
        items: prepared,
        fail_fast: req.fail_fast,
    }) {
        Ok(result) => result,
        Err(error) => {
            for index in contexts {
                results.push(BatchItemResult {
                    index,
                    result: Err(error.to_string()),
                });
                if req.fail_fast {
                    break;
                }
            }
            results.sort_by_key(|item| item.index);
            return batch_result(results);
        }
    };
    w3c_batch_timings()
        .verify_disclosure_crypto_micros
        .fetch_add(elapsed_micros(crypto_started), Ordering::Relaxed);

    for item in verified.results {
        results.push(BatchItemResult {
            index: contexts[item.index],
            result: item.result.map_err(|error| error.to_string()),
        });
    }
    results.sort_by_key(|item| item.index);
    batch_result(results)
}

/// Prepare a base VC once for managed-ID verification.
pub async fn prepare_jsonld_base_verification(
    document: Value,
) -> Result<PreparedBaseVerification, SuiteJsonLdError> {
    prepare_verify_base_item(document).await
}

/// Prepare a derived VP once for managed-ID proof verification.
pub async fn prepare_jsonld_disclosure_verification(
    document: Value,
    expected_presentation_header: Option<Vec<u8>>,
) -> Result<PreparedDisclosureVerification, SuiteJsonLdError> {
    prepare_verify_disclosure_item(document, expected_presentation_header).await
}

/// Prepare a named disclosure profile once for managed-ID proof generation.
pub async fn prepare_jsonld_disclosure_proof(
    req: DeriveJsonLdDisclosureProofRequest,
) -> Result<PreparedJsonLdDisclosureProof, SuiteJsonLdError> {
    let (selected_document, base_proof, primitive) = prepare_derive_item(req).await?;
    Ok(PreparedJsonLdDisclosureProof {
        selected_document,
        base_proof,
        primitive,
    })
}

/// Assemble the W3C document after a managed ProofGen primitive completes.
pub fn assemble_jsonld_disclosure_proof(
    prepared: &PreparedJsonLdDisclosureProof,
    proof_value: String,
) -> Result<Value, SuiteJsonLdError> {
    let mut proof = prepared.base_proof.clone();
    proof
        .as_object_mut()
        .ok_or(SuiteJsonLdError::DocumentMustBeObject)?
        .insert("proofValue".to_owned(), Value::String(proof_value));
    attach_proof(prepared.selected_document.clone(), proof)
}

fn prepared_cache_capacity() -> usize {
    std::env::var("CUBBS_W3C_PREPARED_CACHE_CAPACITY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4_096)
}

fn issue_prepared_cache() -> &'static PreparedCache<PreparedIssueStatic> {
    static CACHE: OnceLock<PreparedCache<PreparedIssueStatic>> = OnceLock::new();
    CACHE.get_or_init(PreparedCache::new)
}

fn derive_prepared_cache() -> &'static PreparedCache<PreparedDeriveItem> {
    static CACHE: OnceLock<PreparedCache<PreparedDeriveItem>> = OnceLock::new();
    CACHE.get_or_init(PreparedCache::new)
}

fn verify_base_prepared_cache() -> &'static PreparedCache<PreparedBaseVerification> {
    static CACHE: OnceLock<PreparedCache<PreparedBaseVerification>> = OnceLock::new();
    CACHE.get_or_init(PreparedCache::new)
}

fn verify_disclosure_prepared_cache() -> &'static PreparedCache<PreparedDisclosureVerification> {
    static CACHE: OnceLock<PreparedCache<PreparedDisclosureVerification>> = OnceLock::new();
    CACHE.get_or_init(PreparedCache::new)
}

fn prepared_key_hasher(domain: &[u8], document: &Value) -> Option<Sha256> {
    let document_digest = json_content_digest(document).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(b"cuBBS-Cred W3C prepared cache v1");
    update_digest_component(&mut hasher, domain);
    update_digest_component(&mut hasher, &document_digest);
    Some(hasher)
}

fn update_digest_component(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn issue_prepared_key(document: &Value, proof_options: &Value) -> Option<PreparedCacheKey> {
    let mut hasher = prepared_key_hasher(b"issue", document)?;
    update_digest_component(&mut hasher, &json_content_digest(proof_options).ok()?);
    Some(PreparedCacheKey(hasher.finalize().into()))
}

fn derive_prepared_key(req: &DeriveJsonLdDisclosureProofRequest) -> Option<PreparedCacheKey> {
    let mut hasher = prepared_key_hasher(b"derive", &req.signed_document)?;
    for pointer in &req.selective_pointers {
        update_digest_component(&mut hasher, pointer.as_bytes());
    }
    update_digest_component(&mut hasher, &req.presentation_header);
    Some(PreparedCacheKey(hasher.finalize().into()))
}

fn verify_base_prepared_key(document: &Value) -> Option<PreparedCacheKey> {
    let hasher = prepared_key_hasher(b"verify-base", document)?;
    Some(PreparedCacheKey(hasher.finalize().into()))
}

fn verify_disclosure_prepared_key(
    document: &Value,
    expected_presentation_header: Option<&[u8]>,
) -> Option<PreparedCacheKey> {
    let mut hasher = prepared_key_hasher(b"verify-disclosure", document)?;
    match expected_presentation_header {
        Some(header) => {
            update_digest_component(&mut hasher, &[1]);
            update_digest_component(&mut hasher, header);
        }
        None => update_digest_component(&mut hasher, &[0]),
    }
    Some(PreparedCacheKey(hasher.finalize().into()))
}

pub fn prepared_cache_snapshot() -> PreparedCacheSnapshot {
    let issue = issue_prepared_cache();
    let derive = derive_prepared_cache();
    let verify_base = verify_base_prepared_cache();
    let verify_disclosure = verify_disclosure_prepared_cache();
    PreparedCacheSnapshot {
        capacity_per_operation: prepared_cache_capacity(),
        issue_entries: issue.entries(),
        derive_entries: derive.entries(),
        verify_base_entries: verify_base.entries(),
        verify_disclosure_entries: verify_disclosure.entries(),
        issue_hits: issue.hits.load(Ordering::Relaxed),
        issue_misses: issue.misses.load(Ordering::Relaxed),
        derive_hits: derive.hits.load(Ordering::Relaxed),
        derive_misses: derive.misses.load(Ordering::Relaxed),
        verify_base_hits: verify_base.hits.load(Ordering::Relaxed),
        verify_base_misses: verify_base.misses.load(Ordering::Relaxed),
        verify_disclosure_hits: verify_disclosure.hits.load(Ordering::Relaxed),
        verify_disclosure_misses: verify_disclosure.misses.load(Ordering::Relaxed),
    }
}

async fn prepare_issue_item(
    req: IssueJsonLdBaseProofRequest,
) -> Result<(Value, Value, IssueBaseProofRequest), SuiteJsonLdError> {
    let IssueJsonLdBaseProofRequest {
        document: input_document,
        proof_options: input_proof_options,
        secret_key,
        public_key,
        mandatory_pointers,
        hmac_key,
    } = req;
    let cache_key = issue_prepared_key(&input_document, &input_proof_options);
    let cached = cache_key.and_then(|key| issue_prepared_cache().get(&key));
    let (document, proof_options, proof_hash) = if let Some(cached) = cached {
        cached
    } else {
        let proof_config_started = Instant::now();
        let document = ensure_data_integrity_context(input_document)?;
        require_credential_subject(&document)?;
        let proof_config = build_proof_configuration(&document, &input_proof_options)?;
        w3c_batch_timings()
            .issue_proof_config_micros
            .fetch_add(elapsed_micros(proof_config_started), Ordering::Relaxed);
        let proof_canonicalize_started = Instant::now();
        let proof_hash = hash_joined_strings(&canonicalize_jsonld(&proof_config).await?.lines);
        w3c_batch_timings()
            .issue_proof_canonicalize_micros
            .fetch_add(
                elapsed_micros(proof_canonicalize_started),
                Ordering::Relaxed,
            );
        let prepared = (document, input_proof_options, proof_hash);
        if let Some(key) = cache_key {
            issue_prepared_cache().insert(key, prepared.clone());
        }
        prepared
    };

    let groups = BTreeMap::from([("mandatory".to_owned(), mandatory_pointers.clone())]);
    let group_started = Instant::now();
    let grouped = canonicalize_and_group_jsonld(&document, &groups, &hmac_key).await?;
    w3c_batch_timings()
        .issue_group_micros
        .fetch_add(elapsed_micros(group_started), Ordering::Relaxed);
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

    Ok((
        document,
        proof_options,
        IssueBaseProofRequest {
            secret_key,
            public_key,
            proof_hash,
            mandatory_pointers,
            mandatory,
            non_mandatory,
            hmac_key,
        },
    ))
}

async fn prepare_derive_item(
    req: DeriveJsonLdDisclosureProofRequest,
) -> Result<PreparedDeriveItem, SuiteJsonLdError> {
    let cache_key = derive_prepared_key(&req);
    if let Some(cached) = cache_key.and_then(|key| derive_prepared_cache().get(&key)) {
        return Ok(cached);
    }

    let parse_started = Instant::now();
    let (unsigned_document, base_proof) = detach_single_proof(&req.signed_document)?;
    let base_proof_value = base_proof
        .get("proofValue")
        .and_then(Value::as_str)
        .ok_or(SuiteJsonLdError::MissingProof)?
        .to_owned();
    let parsed_base = parse_base(&base_proof_value)?;
    w3c_batch_timings()
        .derive_parse_micros
        .fetch_add(elapsed_micros(parse_started), Ordering::Relaxed);

    let combined_pointers = parsed_base
        .mandatory_pointers
        .iter()
        .cloned()
        .chain(req.selective_pointers.iter().cloned())
        .collect::<Vec<_>>();
    let groups = BTreeMap::from([
        (
            "mandatory".to_owned(),
            parsed_base.mandatory_pointers.clone(),
        ),
        ("selective".to_owned(), req.selective_pointers),
        ("combined".to_owned(), combined_pointers),
    ]);
    let group_started = Instant::now();
    let grouped =
        canonicalize_and_group_jsonld(&unsigned_document, &groups, &parsed_base.hmac_key).await?;
    w3c_batch_timings()
        .derive_group_micros
        .fetch_add(elapsed_micros(group_started), Ordering::Relaxed);

    let prepared = (
        grouped.groups["combined"].selected_document.clone(),
        base_proof,
        PreparedDisclosureRequest {
            base_proof_value,
            groups: DisclosureGroups {
                mandatory: grouped.groups["mandatory"].group.clone(),
                selective: grouped.groups["selective"].group.clone(),
                combined: grouped.groups["combined"].group.clone(),
                verifier_label_map: grouped.label_map,
            },
            presentation_header: req.presentation_header,
        },
    );
    if let Some(key) = cache_key {
        derive_prepared_cache().insert(key, prepared.clone());
    }
    Ok(prepared)
}

async fn prepare_verify_disclosure_item(
    document: Value,
    expected_presentation_header: Option<Vec<u8>>,
) -> Result<PreparedDisclosureVerification, SuiteJsonLdError> {
    let cache_key =
        verify_disclosure_prepared_key(&document, expected_presentation_header.as_deref());
    if let Some(cached) = cache_key.and_then(|key| verify_disclosure_prepared_cache().get(&key)) {
        return Ok(cached);
    }

    let (unsigned_document, proof) = detach_single_proof(&document)?;
    let proof_config = build_proof_configuration(&unsigned_document, &proof)?;
    let proof_hash = hash_joined_strings(&canonicalize_jsonld(&proof_config).await?.lines);

    let proof_value = proof
        .get("proofValue")
        .and_then(Value::as_str)
        .ok_or(SuiteJsonLdError::MissingProof)?;
    let parsed = parse_derived(proof_value)?;

    let canonicalized = canonicalize_jsonld(&unsigned_document).await?;
    let canonical_nquads =
        relabel_and_sort_canonical_nquads(&canonicalized.lines, &parsed.label_map);
    let verification_method = proof
        .get("verificationMethod")
        .and_then(Value::as_str)
        .ok_or(SuiteJsonLdError::MissingVerificationMethod)?;
    let public_key = public_key_from_verification_method(verification_method)?;

    let prepared = PreparedDisclosureVerification {
        public_key,
        proof_hash,
        canonical_nquads,
        proof_value: proof_value.to_owned(),
        expected_presentation_header,
    };
    if let Some(key) = cache_key {
        verify_disclosure_prepared_cache().insert(key, prepared.clone());
    }
    Ok(prepared)
}

async fn prepare_verify_base_item(
    document: Value,
) -> Result<PreparedBaseVerification, SuiteJsonLdError> {
    let cache_key = verify_base_prepared_key(&document);
    if let Some(cached) = cache_key.and_then(|key| verify_base_prepared_cache().get(&key)) {
        return Ok(cached);
    }

    let (unsigned_document, proof) = detach_single_proof(&document)?;
    let proof_config = build_proof_configuration(&unsigned_document, &proof)?;
    let proof_hash = hash_joined_strings(&canonicalize_jsonld(&proof_config).await?.lines);

    let proof_value = proof
        .get("proofValue")
        .and_then(Value::as_str)
        .ok_or(SuiteJsonLdError::MissingProof)?;
    let parsed = parse_base(proof_value)?;
    let verification_method = proof
        .get("verificationMethod")
        .and_then(Value::as_str)
        .ok_or(SuiteJsonLdError::MissingVerificationMethod)?;
    let public_key = public_key_from_verification_method(verification_method)?;

    let groups = BTreeMap::from([("mandatory".to_owned(), parsed.mandatory_pointers.clone())]);
    let grouped =
        canonicalize_and_group_jsonld(&unsigned_document, &groups, &parsed.hmac_key).await?;
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

    let prepared = PreparedBaseVerification {
        public_key,
        proof_hash,
        mandatory,
        non_mandatory,
        proof_value: proof_value.to_owned(),
    };
    if let Some(key) = cache_key {
        verify_base_prepared_cache().insert(key, prepared.clone());
    }
    Ok(prepared)
}

fn batch_result<T>(results: Vec<BatchItemResult<T>>) -> BatchRunResult<T> {
    let attempted = results.len();
    let succeeded = results.iter().filter(|item| item.result.is_ok()).count();
    let failed = attempted - succeeded;
    BatchRunResult {
        attempted,
        results,
        succeeded,
        failed,
    }
}

pub fn jsonld_batch_workers() -> usize {
    let default = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8)
        .saturating_mul(4)
        .min(64);
    std::env::var("CUBBS_W3C_JSONLD_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
        .max(1)
}

pub fn w3c_batch_timing_snapshot() -> W3cBatchTimingSnapshot {
    let timings = w3c_batch_timings();
    W3cBatchTimingSnapshot {
        issue_prepare_ms: millis(&timings.issue_prepare_micros),
        issue_proof_config_ms: millis(&timings.issue_proof_config_micros),
        issue_proof_canonicalize_ms: millis(&timings.issue_proof_canonicalize_micros),
        issue_group_ms: millis(&timings.issue_group_micros),
        issue_crypto_ms: millis(&timings.issue_crypto_micros),
        derive_prepare_ms: millis(&timings.derive_prepare_micros),
        derive_parse_ms: millis(&timings.derive_parse_micros),
        derive_group_ms: millis(&timings.derive_group_micros),
        derive_crypto_ms: millis(&timings.derive_crypto_micros),
        verify_base_prepare_ms: millis(&timings.verify_base_prepare_micros),
        verify_base_crypto_ms: millis(&timings.verify_base_crypto_micros),
        verify_disclosure_prepare_ms: millis(&timings.verify_disclosure_prepare_micros),
        verify_disclosure_crypto_ms: millis(&timings.verify_disclosure_crypto_micros),
    }
}

/// Returns the per-process weighted admission state for W3C Verify crypto.
///
/// The budget is initialized once from
/// `CUBBS_W3C_GPU_VERIFY_MAX_INFLIGHT_ITEMS` (default: 4096).  It is shared
/// by base-signature and derived-proof verification when a GPU backend is
/// requested. CPU/no-backend operation is explicitly reported as disabled and
/// does not acquire permits.
pub fn w3c_verify_crypto_admission_snapshot() -> W3cVerifyCryptoAdmissionSnapshot {
    w3c_verify_crypto_admission_snapshot_for_backend(w3c_bbs_backend_from_env())
}

/// Returns the W3C Verify admission state for a parsed BBS backend.
///
/// W3C services should call this variant with their startup backend so health
/// reporting and startup validation cannot diverge from backend selection.
pub fn w3c_verify_crypto_admission_snapshot_for_backend(
    backend: BbsBackend,
) -> W3cVerifyCryptoAdmissionSnapshot {
    w3c_verify_crypto_admission_for_backend(backend)
        .map(|admission| admission.snapshot())
        .unwrap_or_else(W3cVerifyCryptoAdmissionSnapshot::disabled)
}

/// Acquires the weighted GPU admission budget for a prepared Verify batch.
/// CPU backends return `Ok(None)` and need no permit.
pub async fn acquire_w3c_verify_crypto_admission_for_backend(
    backend: BbsBackend,
    item_count: usize,
) -> Result<Option<W3cVerifyCryptoPermit>, String> {
    match w3c_verify_crypto_admission_for_backend(backend) {
        Some(admission) => admission.acquire(item_count).await.map(Some),
        None => Ok(None),
    }
}

fn w3c_verify_crypto_admission() -> Option<&'static Arc<W3cVerifyCryptoAdmission>> {
    w3c_verify_crypto_admission_for_backend(w3c_bbs_backend_from_env())
}

fn w3c_verify_crypto_admission_for_backend(
    backend: BbsBackend,
) -> Option<&'static Arc<W3cVerifyCryptoAdmission>> {
    if !backend.is_gpu_requested() {
        return None;
    }
    static ADMISSION: OnceLock<Arc<W3cVerifyCryptoAdmission>> = OnceLock::new();
    Some(ADMISSION.get_or_init(|| Arc::new(W3cVerifyCryptoAdmission::from_env())))
}

fn w3c_bbs_backend_from_env() -> BbsBackend {
    std::env::var("CUBBS_BBS_BACKEND")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .and_then(|value| BbsBackend::parse(value.trim()).ok())
        .unwrap_or(BbsBackend::ZkryptiumCpu)
}

fn w3c_batch_timings() -> &'static W3cBatchTimings {
    static TIMINGS: OnceLock<W3cBatchTimings> = OnceLock::new();
    TIMINGS.get_or_init(W3cBatchTimings::default)
}

fn elapsed_micros(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u64::MAX as u128) as u64
}

fn millis(value: &AtomicU64) -> f64 {
    value.load(Ordering::Relaxed) as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        default_bbs_2023_proof_options, verification_method_from_public_key,
        verify_jsonld_base_proof, HMAC_KEY_LEN,
    };
    use cubbs_bbs::ZkryptiumBbsEngine;
    use serde_json::json;

    fn issue_request_with_subject(subject: Value) -> IssueJsonLdBaseProofRequest {
        let engine = ZkryptiumBbsEngine::new();
        let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
        let verification_method = verification_method_from_public_key(&keys.public_key).unwrap();
        IssueJsonLdBaseProofRequest {
            document: json!({
                "@context": {
                    "id": "@id",
                    "type": "@type",
                    "VerifiableCredential": "https://www.w3.org/2018/credentials#VerifiableCredential",
                    "credentialSubject": "https://www.w3.org/2018/credentials#credentialSubject",
                    "name": "http://xmlns.com/foaf/0.1/name"
                },
                "id": "urn:cred:prepared-issue",
                "type": "VerifiableCredential",
                "credentialSubject": subject
            }),
            proof_options: default_bbs_2023_proof_options(verification_method, "assertionMethod"),
            secret_key: keys.secret_key,
            public_key: keys.public_key,
            mandatory_pointers: vec!["/credentialSubject/name".to_owned()],
            hmac_key: vec![7; HMAC_KEY_LEN],
        }
    }

    #[tokio::test]
    async fn prepared_issue_keeps_fresh_hmac_and_valid_independent_signatures() {
        let request = issue_request_with_subject(json!({
            "id": "did:example:alice",
            "name": "Alice"
        }));
        let prepared = prepare_jsonld_base_issue(&request)
            .await
            .unwrap()
            .expect("identified JSON-LD nodes support the prepared path");
        let engine = ZkryptiumBbsEngine::new();

        let first_signature = engine.sign((**prepared.sign_request()).clone()).unwrap();
        let second_signature = engine.sign((**prepared.sign_request()).clone()).unwrap();
        let first_proof = prepared
            .proof_value(first_signature, &[1; HMAC_KEY_LEN])
            .unwrap();
        let second_proof = prepared
            .proof_value(second_signature, &[2; HMAC_KEY_LEN])
            .unwrap();
        assert_ne!(first_proof, second_proof);

        let first = prepared.assemble_document(first_proof).unwrap();
        let second = prepared.assemble_document(second_proof).unwrap();
        assert!(verify_jsonld_base_proof(&first).await.unwrap());
        assert!(verify_jsonld_base_proof(&second).await.unwrap());
    }

    #[tokio::test]
    async fn prepared_issue_rejects_documents_with_blank_nodes() {
        let request = issue_request_with_subject(json!({"name": "Alice"}));
        assert!(prepare_jsonld_base_issue(&request).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn weighted_verify_admission_limits_work_and_releases_cancelled_waiters() {
        let admission = Arc::new(W3cVerifyCryptoAdmission::new(3));
        let first = admission.acquire(2).await.unwrap();
        let second = admission.acquire(1).await.unwrap();
        let saturated = admission.snapshot();
        assert_eq!(saturated.in_flight_items, 3);
        assert_eq!(saturated.max_in_flight_items, 3);

        let queued_admission = Arc::clone(&admission);
        let waiting = tokio::spawn(async move { queued_admission.acquire(1).await.unwrap() });
        for _ in 0..100 {
            if admission.snapshot().waiting_batches == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        let queued = admission.snapshot();
        assert_eq!(queued.waiting_batches, 1);
        assert_eq!(queued.waiting_items, 1);
        assert!(queued.max_in_flight_items <= queued.item_budget.unwrap());

        waiting.abort();
        assert!(matches!(waiting.await, Err(error) if error.is_cancelled()));
        for _ in 0..100 {
            if admission.snapshot().waiting_batches == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(admission.snapshot().waiting_batches, 0);
        assert_eq!(admission.snapshot().waiting_items, 0);

        drop(first);
        drop(second);
        let released = admission.snapshot();
        assert_eq!(released.in_flight_items, 0);
        assert_eq!(released.max_in_flight_items, 3);
    }

    #[test]
    fn verify_admission_is_explicitly_disabled_for_cpu_backend() {
        assert!(w3c_verify_crypto_admission_for_backend(BbsBackend::ZkryptiumCpu).is_none());

        let cpu = w3c_verify_crypto_admission_snapshot_for_backend(BbsBackend::ZkryptiumCpu);
        assert!(!cpu.enabled);
        assert_eq!(cpu.item_budget, None);
        assert_eq!(cpu.in_flight_items, 0);
        assert_eq!(cpu.max_in_flight_items, 0);
        assert_eq!(cpu.waiting_batches, 0);
        assert_eq!(cpu.waiting_items, 0);
        assert_eq!(cpu.total_admission_wait_ms, 0.0);

        let gpu = W3cVerifyCryptoAdmission::new(3).snapshot();
        assert!(gpu.enabled);
        assert_eq!(gpu.item_budget, Some(3));
        assert!(BbsBackend::GpuRequestedCpuFallback.is_gpu_requested());
    }
}
