use std::collections::{BTreeMap, BTreeSet};

use cubbs_bbs::{
    BbsBackend, BbsEngine, ProofGenRequest, ProofVerifyRequest, SignRequest, VerifyRequest,
};
use cubbs_bbs_proofvalue::{
    parse_base, parse_derived, serialize_base, serialize_derived, BaseProofValue, DerivedProofValue,
};
use rand::RngCore;

use crate::hashing::{bbs_header, hash_mandatory, Sha256Digest};
use crate::hmac_label::HMAC_KEY_LEN;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid HMAC key length: expected 32 bytes, got {0}")]
    InvalidHmacKeyLength(usize),

    #[error("group indexes are not a partition of the canonical N-Quads")]
    InvalidGroupPartition,

    #[error("selective indexes must match verifier disclosed messages")]
    InvalidSelectiveDisclosure,

    #[error("BBS primitive error: {0}")]
    Bbs(#[from] cubbs_bbs::Error),

    #[error("proofValue error: {0}")]
    ProofValue(#[from] cubbs_bbs_proofvalue::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedNQuads {
    pub matching: BTreeMap<usize, String>,
    pub non_matching: BTreeMap<usize, String>,
    pub deskolemized_nquads: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisclosureGroups {
    pub mandatory: GroupedNQuads,
    pub selective: GroupedNQuads,
    pub combined: GroupedNQuads,
    /// Map from canonical labels such as `c14n0` to HMAC labels such as `b0`
    /// as seen by the verifier after canonicalizing the revealed document.
    pub verifier_label_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IssueBaseProofRequest {
    pub secret_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub proof_hash: Sha256Digest,
    pub mandatory_pointers: Vec<String>,
    pub mandatory: Vec<String>,
    pub non_mandatory: Vec<String>,
    pub hmac_key: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IssueBaseProofBatchRequest {
    pub items: Vec<IssueBaseProofRequest>,
    pub fail_fast: bool,
}

#[derive(Debug)]
pub struct PreparedBatchItemResult<T> {
    pub index: usize,
    pub result: Result<T>,
}

#[derive(Debug)]
pub struct PreparedBatchRunResult<T> {
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub results: Vec<PreparedBatchItemResult<T>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedBaseVerification {
    pub public_key: Vec<u8>,
    pub proof_hash: Sha256Digest,
    pub mandatory: Vec<String>,
    pub non_mandatory: Vec<String>,
    pub proof_value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDisclosureRequest {
    pub base_proof_value: String,
    pub groups: DisclosureGroups,
    pub presentation_header: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDisclosureBatchRequest {
    pub items: Vec<PreparedDisclosureRequest>,
    pub fail_fast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedProofMaterial {
    pub proof_value: String,
    pub mandatory_indexes: Vec<usize>,
    pub selective_indexes: Vec<usize>,
    pub label_map: BTreeMap<String, String>,
}

/// A disclosure-proof request lowered to the BBS primitive boundary.
///
/// Managed-VC services retain this value after JSON-LD preparation and clone
/// only the compact primitive input for each online proof request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredDisclosureProof {
    pub request: ProofGenRequest,
    mandatory_indexes: Vec<usize>,
    selective_indexes: Vec<usize>,
    label_map: BTreeMap<String, String>,
    presentation_header: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDisclosureVerification {
    pub public_key: Vec<u8>,
    pub proof_hash: Sha256Digest,
    pub canonical_nquads: Vec<String>,
    pub proof_value: String,
    pub expected_presentation_header: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDisclosureVerificationBatchRequest {
    pub items: Vec<PreparedDisclosureVerification>,
    pub fail_fast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedBaseVerificationBatchRequest {
    pub items: Vec<PreparedBaseVerification>,
    pub fail_fast: bool,
}

#[derive(Debug, Default, Clone)]
pub struct Bbs2023Core {
    engine: BbsEngine,
}

impl Bbs2023Core {
    pub fn new() -> Self {
        Self::with_backend(BbsBackend::ZkryptiumCpu)
    }

    pub fn from_env() -> Result<Self> {
        let backend = std::env::var("CUBBS_BBS_BACKEND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| BbsBackend::parse(value.trim()))
            .transpose()?
            .unwrap_or(BbsBackend::ZkryptiumCpu);
        if gpu_required_from_env() {
            backend.ensure_hardware_available()?;
        }
        Ok(Self::with_backend(backend))
    }

    pub fn with_backend(backend: BbsBackend) -> Self {
        Self {
            engine: BbsEngine::new(backend),
        }
    }

    pub fn backend(&self) -> BbsBackend {
        self.engine.backend()
    }

    pub fn backend_name(&self) -> &'static str {
        self.engine.backend_name()
    }

    pub fn issue_base_proof(&self, req: IssueBaseProofRequest) -> Result<String> {
        issue_base_proof_with_engine(&self.engine, req)
    }

    pub fn issue_base_proof_batch(
        &self,
        req: IssueBaseProofBatchRequest,
    ) -> PreparedBatchRunResult<String> {
        issue_base_proof_batch_with_engine(&self.engine, req)
    }

    pub fn verify_base_proof(&self, req: PreparedBaseVerification) -> Result<bool> {
        verify_base_proof_with_engine(&self.engine, req)
    }

    pub fn verify_base_proof_batch(
        &self,
        req: PreparedBaseVerificationBatchRequest,
    ) -> PreparedBatchRunResult<bool> {
        verify_base_proof_batch_with_engine(&self.engine, req)
    }

    pub fn derive_disclosure_proof(
        &self,
        req: PreparedDisclosureRequest,
    ) -> Result<DerivedProofMaterial> {
        derive_disclosure_proof_with_engine(&self.engine, req)
    }

    pub fn derive_disclosure_proof_batch(
        &self,
        req: PreparedDisclosureBatchRequest,
    ) -> PreparedBatchRunResult<DerivedProofMaterial> {
        derive_disclosure_proof_batch_with_engine(&self.engine, req)
    }

    pub fn verify_disclosure_proof(&self, req: PreparedDisclosureVerification) -> Result<bool> {
        verify_disclosure_proof_with_engine(&self.engine, req)
    }

    pub fn verify_disclosure_proof_batch(
        &self,
        req: PreparedDisclosureVerificationBatchRequest,
    ) -> PreparedBatchRunResult<bool> {
        verify_disclosure_proof_batch_with_engine(&self.engine, req)
    }

    pub fn random_hmac_key(&self, rng: &mut impl RngCore) -> Vec<u8> {
        let mut key = vec![0; HMAC_KEY_LEN];
        rng.fill_bytes(&mut key);
        key
    }
}

fn gpu_required_from_env() -> bool {
    std::env::var("CUBBS_BBS_REQUIRE_HARDWARE_GPU")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub fn issue_base_proof(req: IssueBaseProofRequest) -> Result<String> {
    Bbs2023Core::from_env()?.issue_base_proof(req)
}

pub fn issue_base_proof_batch(
    req: IssueBaseProofBatchRequest,
) -> Result<PreparedBatchRunResult<String>> {
    Ok(Bbs2023Core::from_env()?.issue_base_proof_batch(req))
}

pub fn verify_base_proof(req: PreparedBaseVerification) -> Result<bool> {
    Bbs2023Core::from_env()?.verify_base_proof(req)
}

pub fn verify_base_proof_batch(
    req: PreparedBaseVerificationBatchRequest,
) -> Result<PreparedBatchRunResult<bool>> {
    Ok(Bbs2023Core::from_env()?.verify_base_proof_batch(req))
}

pub fn derive_disclosure_proof(req: PreparedDisclosureRequest) -> Result<DerivedProofMaterial> {
    Bbs2023Core::from_env()?.derive_disclosure_proof(req)
}

pub fn derive_disclosure_proof_batch(
    req: PreparedDisclosureBatchRequest,
) -> Result<PreparedBatchRunResult<DerivedProofMaterial>> {
    Ok(Bbs2023Core::from_env()?.derive_disclosure_proof_batch(req))
}

pub fn verify_disclosure_proof(req: PreparedDisclosureVerification) -> Result<bool> {
    Bbs2023Core::from_env()?.verify_disclosure_proof(req)
}

pub fn verify_disclosure_proof_batch(
    req: PreparedDisclosureVerificationBatchRequest,
) -> Result<PreparedBatchRunResult<bool>> {
    Ok(Bbs2023Core::from_env()?.verify_disclosure_proof_batch(req))
}

pub fn mandatory_indexes_in_revealed_document(
    combined_matching: &BTreeMap<usize, String>,
    mandatory_matching: &BTreeMap<usize, String>,
) -> Vec<usize> {
    combined_matching
        .keys()
        .enumerate()
        .filter_map(|(relative_index, absolute_index)| {
            mandatory_matching
                .contains_key(absolute_index)
                .then_some(relative_index)
        })
        .collect()
}

pub fn selective_indexes_in_bbs_messages(
    mandatory_non_matching: &BTreeMap<usize, String>,
    selective_matching: &BTreeMap<usize, String>,
) -> Vec<usize> {
    mandatory_non_matching
        .keys()
        .enumerate()
        .filter_map(|(relative_index, absolute_index)| {
            selective_matching
                .contains_key(absolute_index)
                .then_some(relative_index)
        })
        .collect()
}

pub fn group_canonical_nquads(
    canonical_nquads: &[String],
    selected_nquads: &[String],
    deskolemized_nquads: Vec<String>,
) -> GroupedNQuads {
    let selected: BTreeSet<&String> = selected_nquads.iter().collect();
    let mut matching = BTreeMap::new();
    let mut non_matching = BTreeMap::new();
    for (index, nq) in canonical_nquads.iter().enumerate() {
        if selected.contains(nq) {
            matching.insert(index, nq.clone());
        } else {
            non_matching.insert(index, nq.clone());
        }
    }
    GroupedNQuads {
        matching,
        non_matching,
        deskolemized_nquads,
    }
}

fn issue_base_proof_with_engine(engine: &BbsEngine, req: IssueBaseProofRequest) -> Result<String> {
    ensure_hmac_key(&req.hmac_key)?;
    let mandatory_hash = hash_mandatory(&req.mandatory);
    let header = bbs_header(req.proof_hash, mandatory_hash);
    let messages = strings_to_message_bytes(&req.non_mandatory);
    let bbs_signature = engine.sign(SignRequest {
        secret_key: req.secret_key,
        public_key: req.public_key.clone(),
        header: header.clone(),
        messages,
    })?;

    Ok(serialize_base(&BaseProofValue {
        bbs_signature,
        bbs_header: header,
        public_key: req.public_key,
        hmac_key: req.hmac_key,
        mandatory_pointers: req.mandatory_pointers,
    })?)
}

struct PreparedIssueBatchItem {
    index: usize,
    public_key: Vec<u8>,
    header: Vec<u8>,
    hmac_key: Vec<u8>,
    mandatory_pointers: Vec<String>,
}

fn issue_base_proof_batch_with_engine(
    engine: &BbsEngine,
    req: IssueBaseProofBatchRequest,
) -> PreparedBatchRunResult<String> {
    let mut results = Vec::with_capacity(req.items.len());
    let mut contexts = Vec::new();
    let mut sign_reqs = Vec::new();

    for (index, item) in req.items.into_iter().enumerate() {
        if let Err(error) = ensure_hmac_key(&item.hmac_key) {
            results.push(PreparedBatchItemResult {
                index,
                result: Err(error),
            });
            if req.fail_fast {
                return prepared_batch_result(results);
            }
            continue;
        }

        let mandatory_hash = hash_mandatory(&item.mandatory);
        let header = bbs_header(item.proof_hash, mandatory_hash);
        contexts.push(PreparedIssueBatchItem {
            index,
            public_key: item.public_key.clone(),
            header: header.clone(),
            hmac_key: item.hmac_key,
            mandatory_pointers: item.mandatory_pointers,
        });
        sign_reqs.push(SignRequest {
            secret_key: item.secret_key,
            public_key: item.public_key,
            header,
            messages: strings_to_message_bytes(&item.non_mandatory),
        });
    }

    let signed = engine.sign_batch(sign_reqs, req.fail_fast);
    for item in signed.results {
        let context = &contexts[item.index];
        let result = item.result.map_err(Error::from).and_then(|bbs_signature| {
            serialize_base(&BaseProofValue {
                bbs_signature,
                bbs_header: context.header.clone(),
                public_key: context.public_key.clone(),
                hmac_key: context.hmac_key.clone(),
                mandatory_pointers: context.mandatory_pointers.clone(),
            })
            .map_err(Error::from)
        });
        results.push(PreparedBatchItemResult {
            index: context.index,
            result,
        });
    }
    results.sort_by_key(|item| item.index);
    prepared_batch_result(results)
}

fn verify_base_proof_with_engine(
    engine: &BbsEngine,
    req: PreparedBaseVerification,
) -> Result<bool> {
    let parsed = parse_base(&req.proof_value)?;
    if parsed.public_key != req.public_key {
        return Ok(false);
    }

    let mandatory_hash = hash_mandatory(&req.mandatory);
    let header = bbs_header(req.proof_hash, mandatory_hash);
    if parsed.bbs_header != header {
        return Ok(false);
    }

    engine
        .verify(VerifyRequest {
            public_key: req.public_key,
            signature: parsed.bbs_signature,
            header,
            messages: strings_to_message_bytes(&req.non_mandatory),
        })
        .map_err(Error::from)
}

fn verify_base_proof_batch_with_engine(
    engine: &BbsEngine,
    req: PreparedBaseVerificationBatchRequest,
) -> PreparedBatchRunResult<bool> {
    let mut results = Vec::with_capacity(req.items.len());
    let mut contexts = Vec::new();
    let mut verify_reqs = Vec::new();

    for (index, item) in req.items.into_iter().enumerate() {
        match prepare_base_verify_request(item) {
            Ok(Some(verify_req)) => {
                contexts.push(index);
                verify_reqs.push(verify_req);
            }
            Ok(None) => results.push(PreparedBatchItemResult {
                index,
                result: Ok(false),
            }),
            Err(error) => {
                results.push(PreparedBatchItemResult {
                    index,
                    result: Err(error),
                });
                if req.fail_fast {
                    return prepared_batch_result(results);
                }
            }
        }
    }

    let verified = engine.verify_batch(verify_reqs, req.fail_fast);
    for item in verified.results {
        results.push(PreparedBatchItemResult {
            index: contexts[item.index],
            result: item.result.map_err(Error::from),
        });
    }
    results.sort_by_key(|item| item.index);
    prepared_batch_result(results)
}

fn prepare_base_verify_request(req: PreparedBaseVerification) -> Result<Option<VerifyRequest>> {
    let parsed = parse_base(&req.proof_value)?;
    if parsed.public_key != req.public_key {
        return Ok(None);
    }

    let mandatory_hash = hash_mandatory(&req.mandatory);
    let header = bbs_header(req.proof_hash, mandatory_hash);
    if parsed.bbs_header != header {
        return Ok(None);
    }

    Ok(Some(VerifyRequest {
        public_key: req.public_key,
        signature: parsed.bbs_signature,
        header,
        messages: strings_to_message_bytes(&req.non_mandatory),
    }))
}

/// Lower a prepared W3C base-proof verification to its BBS primitive input.
/// `Ok(None)` means the embedded key or header does not match.
pub fn lower_base_verification(req: PreparedBaseVerification) -> Result<Option<VerifyRequest>> {
    prepare_base_verify_request(req)
}

fn derive_disclosure_proof_with_engine(
    engine: &BbsEngine,
    req: PreparedDisclosureRequest,
) -> Result<DerivedProofMaterial> {
    validate_group_partition(&req.groups.mandatory)?;
    validate_group_partition(&req.groups.selective)?;
    validate_group_partition(&req.groups.combined)?;

    let base = parse_base(&req.base_proof_value)?;
    let mandatory_indexes = mandatory_indexes_in_revealed_document(
        &req.groups.combined.matching,
        &req.groups.mandatory.matching,
    );
    let selective_indexes = selective_indexes_in_bbs_messages(
        &req.groups.mandatory.non_matching,
        &req.groups.selective.matching,
    );
    let messages: Vec<Vec<u8>> = req
        .groups
        .mandatory
        .non_matching
        .values()
        .map(|nq| nq.as_bytes().to_vec())
        .collect();

    let bbs_proof = engine.proof_gen(ProofGenRequest {
        public_key: base.public_key,
        signature: base.bbs_signature,
        header: base.bbs_header,
        presentation_header: req.presentation_header.clone(),
        messages,
        disclosed_indexes: selective_indexes.clone(),
    })?;

    let proof_value = serialize_derived(&DerivedProofValue {
        bbs_proof,
        label_map: req.groups.verifier_label_map.clone(),
        mandatory_indexes: mandatory_indexes.iter().map(|i| *i as u64).collect(),
        selective_indexes: selective_indexes.iter().map(|i| *i as u64).collect(),
        presentation_header: req.presentation_header,
    })?;

    Ok(DerivedProofMaterial {
        proof_value,
        mandatory_indexes,
        selective_indexes,
        label_map: req.groups.verifier_label_map,
    })
}

struct PreparedDeriveBatchItem {
    index: usize,
    mandatory_indexes: Vec<usize>,
    selective_indexes: Vec<usize>,
    label_map: BTreeMap<String, String>,
    presentation_header: Vec<u8>,
}

fn derive_disclosure_proof_batch_with_engine(
    engine: &BbsEngine,
    req: PreparedDisclosureBatchRequest,
) -> PreparedBatchRunResult<DerivedProofMaterial> {
    let mut results = Vec::with_capacity(req.items.len());
    let mut contexts = Vec::new();
    let mut proof_reqs = Vec::new();

    for (index, item) in req.items.into_iter().enumerate() {
        match prepare_derive_batch_item(index, item) {
            Ok((context, proof_req)) => {
                contexts.push(context);
                proof_reqs.push(proof_req);
            }
            Err(error) => {
                results.push(PreparedBatchItemResult {
                    index,
                    result: Err(error),
                });
                if req.fail_fast {
                    return prepared_batch_result(results);
                }
            }
        }
    }

    let proved = engine.proof_gen_batch(proof_reqs, req.fail_fast);
    for item in proved.results {
        let context = &contexts[item.index];
        let result = item.result.map_err(Error::from).and_then(|bbs_proof| {
            let proof_value = serialize_derived(&DerivedProofValue {
                bbs_proof,
                label_map: context.label_map.clone(),
                mandatory_indexes: context
                    .mandatory_indexes
                    .iter()
                    .map(|i| *i as u64)
                    .collect(),
                selective_indexes: context
                    .selective_indexes
                    .iter()
                    .map(|i| *i as u64)
                    .collect(),
                presentation_header: context.presentation_header.clone(),
            })?;
            Ok(DerivedProofMaterial {
                proof_value,
                mandatory_indexes: context.mandatory_indexes.clone(),
                selective_indexes: context.selective_indexes.clone(),
                label_map: context.label_map.clone(),
            })
        });
        results.push(PreparedBatchItemResult {
            index: context.index,
            result,
        });
    }
    results.sort_by_key(|item| item.index);
    prepared_batch_result(results)
}

fn prepare_derive_batch_item(
    index: usize,
    req: PreparedDisclosureRequest,
) -> Result<(PreparedDeriveBatchItem, ProofGenRequest)> {
    validate_group_partition(&req.groups.mandatory)?;
    validate_group_partition(&req.groups.selective)?;
    validate_group_partition(&req.groups.combined)?;

    let base = parse_base(&req.base_proof_value)?;
    let mandatory_indexes = mandatory_indexes_in_revealed_document(
        &req.groups.combined.matching,
        &req.groups.mandatory.matching,
    );
    let selective_indexes = selective_indexes_in_bbs_messages(
        &req.groups.mandatory.non_matching,
        &req.groups.selective.matching,
    );
    let messages = req
        .groups
        .mandatory
        .non_matching
        .values()
        .map(|nq| nq.as_bytes().to_vec())
        .collect();
    Ok((
        PreparedDeriveBatchItem {
            index,
            mandatory_indexes,
            selective_indexes: selective_indexes.clone(),
            label_map: req.groups.verifier_label_map,
            presentation_header: req.presentation_header.clone(),
        },
        ProofGenRequest {
            public_key: base.public_key,
            signature: base.bbs_signature,
            header: base.bbs_header,
            presentation_header: req.presentation_header,
            messages,
            disclosed_indexes: selective_indexes,
        },
    ))
}

/// Lower a prepared selective-disclosure request to a reusable BBS request.
pub fn lower_disclosure_proof(req: PreparedDisclosureRequest) -> Result<LoweredDisclosureProof> {
    let (context, request) = prepare_derive_batch_item(0, req)?;
    Ok(LoweredDisclosureProof {
        request,
        mandatory_indexes: context.mandatory_indexes,
        selective_indexes: context.selective_indexes,
        label_map: context.label_map,
        presentation_header: context.presentation_header,
    })
}

/// Serialize a W3C derived proof after the lowered ProofGen request completes.
pub fn complete_disclosure_proof(
    lowered: &LoweredDisclosureProof,
    bbs_proof: Vec<u8>,
) -> Result<DerivedProofMaterial> {
    let proof_value = serialize_derived(&DerivedProofValue {
        bbs_proof,
        label_map: lowered.label_map.clone(),
        mandatory_indexes: lowered
            .mandatory_indexes
            .iter()
            .map(|i| *i as u64)
            .collect(),
        selective_indexes: lowered
            .selective_indexes
            .iter()
            .map(|i| *i as u64)
            .collect(),
        presentation_header: lowered.presentation_header.clone(),
    })?;
    Ok(DerivedProofMaterial {
        proof_value,
        mandatory_indexes: lowered.mandatory_indexes.clone(),
        selective_indexes: lowered.selective_indexes.clone(),
        label_map: lowered.label_map.clone(),
    })
}

fn verify_disclosure_proof_with_engine(
    engine: &BbsEngine,
    req: PreparedDisclosureVerification,
) -> Result<bool> {
    match prepare_verify_batch_item(0, req)? {
        PreparedVerifyBatchAction::Verify(_, request) => {
            engine.proof_verify(request).map_err(Error::from)
        }
        PreparedVerifyBatchAction::Resolved(value) => Ok(value),
    }
}

struct PreparedVerifyBatchItem {
    index: usize,
}

fn verify_disclosure_proof_batch_with_engine(
    engine: &BbsEngine,
    req: PreparedDisclosureVerificationBatchRequest,
) -> PreparedBatchRunResult<bool> {
    let mut results = Vec::with_capacity(req.items.len());
    let mut contexts = Vec::new();
    let mut verify_reqs = Vec::new();

    for (index, item) in req.items.into_iter().enumerate() {
        match prepare_verify_batch_item(index, item) {
            Ok(PreparedVerifyBatchAction::Verify(context, verify_req)) => {
                contexts.push(context);
                verify_reqs.push(verify_req);
            }
            Ok(PreparedVerifyBatchAction::Resolved(value)) => {
                results.push(PreparedBatchItemResult {
                    index,
                    result: Ok(value),
                });
            }
            Err(error) => {
                results.push(PreparedBatchItemResult {
                    index,
                    result: Err(error),
                });
                if req.fail_fast {
                    return prepared_batch_result(results);
                }
            }
        }
    }

    let verified = engine.proof_verify_batch(verify_reqs, req.fail_fast);
    for item in verified.results {
        let context = &contexts[item.index];
        results.push(PreparedBatchItemResult {
            index: context.index,
            result: item.result.map_err(Error::from),
        });
    }
    results.sort_by_key(|item| item.index);
    prepared_batch_result(results)
}

enum PreparedVerifyBatchAction {
    Verify(PreparedVerifyBatchItem, ProofVerifyRequest),
    Resolved(bool),
}

fn prepare_verify_batch_item(
    index: usize,
    req: PreparedDisclosureVerification,
) -> Result<PreparedVerifyBatchAction> {
    let parsed = parse_derived(&req.proof_value)?;
    if let Some(expected) = req.expected_presentation_header {
        if parsed.presentation_header != expected {
            return Ok(PreparedVerifyBatchAction::Resolved(false));
        }
    }

    let mandatory_index_set: BTreeSet<usize> = parsed
        .mandatory_indexes
        .iter()
        .map(|i| *i as usize)
        .collect();
    let mut mandatory = Vec::new();
    let mut disclosed_messages = Vec::new();
    for (index, nq) in req.canonical_nquads.iter().enumerate() {
        if mandatory_index_set.contains(&index) {
            mandatory.push(nq.clone());
        } else {
            disclosed_messages.push(nq.as_bytes().to_vec());
        }
    }

    let selective_indexes = parsed
        .selective_indexes
        .iter()
        .map(|i| *i as usize)
        .collect::<Vec<_>>();
    if disclosed_messages.len() != selective_indexes.len() {
        return Err(Error::InvalidSelectiveDisclosure);
    }
    let total_message_count =
        total_message_count_from_proof(&parsed.bbs_proof, &selective_indexes)?;

    let mandatory_hash = hash_mandatory(&mandatory);
    let header = bbs_header(req.proof_hash, mandatory_hash);
    Ok(PreparedVerifyBatchAction::Verify(
        PreparedVerifyBatchItem { index },
        ProofVerifyRequest {
            public_key: req.public_key,
            proof: parsed.bbs_proof,
            header,
            presentation_header: parsed.presentation_header,
            disclosed_messages,
            disclosed_indexes: selective_indexes,
            total_message_count,
        },
    ))
}

/// Lower a prepared disclosure verification to its BBS primitive input.
/// `Ok(None)` means the expected presentation header does not match.
pub fn lower_disclosure_verification(
    req: PreparedDisclosureVerification,
) -> Result<Option<ProofVerifyRequest>> {
    match prepare_verify_batch_item(0, req)? {
        PreparedVerifyBatchAction::Verify(_, request) => Ok(Some(request)),
        PreparedVerifyBatchAction::Resolved(false) => Ok(None),
        PreparedVerifyBatchAction::Resolved(true) => {
            unreachable!("disclosure verification lowering never resolves true")
        }
    }
}

fn ensure_hmac_key(key: &[u8]) -> Result<()> {
    if key.len() != HMAC_KEY_LEN {
        return Err(Error::InvalidHmacKeyLength(key.len()));
    }
    Ok(())
}

fn prepared_batch_result<T>(results: Vec<PreparedBatchItemResult<T>>) -> PreparedBatchRunResult<T> {
    let attempted = results.len();
    let succeeded = results.iter().filter(|item| item.result.is_ok()).count();
    let failed = attempted - succeeded;
    PreparedBatchRunResult {
        attempted,
        succeeded,
        failed,
        results,
    }
}

fn validate_group_partition(group: &GroupedNQuads) -> Result<()> {
    let total = group.matching.len() + group.non_matching.len();
    for expected in 0..total {
        if !(group.matching.contains_key(&expected) || group.non_matching.contains_key(&expected)) {
            return Err(Error::InvalidGroupPartition);
        }
    }
    Ok(())
}

fn strings_to_message_bytes(messages: &[String]) -> Vec<Vec<u8>> {
    messages
        .iter()
        .map(|message| message.as_bytes().to_vec())
        .collect()
}

fn total_message_count_from_proof(proof: &[u8], selective_indexes: &[usize]) -> Result<usize> {
    // BBS proof: 3 compressed G1 points, 4 fixed scalars (including c),
    // and one scalar response per hidden message. The last disclosed index
    // cannot reveal how many hidden messages follow it.
    let hidden_bytes = proof
        .len()
        .checked_sub(3 * 48 + 4 * 32)
        .filter(|len| len % 32 == 0)
        .ok_or(cubbs_bbs::Error::InvalidProof)?;
    let total = (hidden_bytes / 32)
        .checked_add(selective_indexes.len())
        .ok_or(cubbs_bbs::Error::InvalidProof)?;
    if selective_indexes.windows(2).any(|pair| pair[0] >= pair[1])
        || selective_indexes
            .last()
            .is_some_and(|index| *index >= total)
    {
        return Err(Error::InvalidSelectiveDisclosure);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubbs_bbs::ZkryptiumBbsEngine;

    fn sample_nquads() -> Vec<String> {
        vec![
            "<urn:cred> <issuer> <did:example:issuer> .\n".to_owned(),
            "<urn:cred> <type> \"VerifiableCredential\" .\n".to_owned(),
            "<urn:subject> <name> \"Alice\" .\n".to_owned(),
            "<urn:subject> <age> \"42\" .\n".to_owned(),
            "<urn:subject> <license> \"A123\" .\n".to_owned(),
        ]
    }

    fn group_from_indexes(nquads: &[String], indexes: &[usize]) -> GroupedNQuads {
        let index_set: BTreeSet<usize> = indexes.iter().copied().collect();
        let mut matching = BTreeMap::new();
        let mut non_matching = BTreeMap::new();
        for (index, nq) in nquads.iter().enumerate() {
            if index_set.contains(&index) {
                matching.insert(index, nq.clone());
            } else {
                non_matching.insert(index, nq.clone());
            }
        }
        GroupedNQuads {
            matching,
            non_matching,
            deskolemized_nquads: nquads.to_vec(),
        }
    }

    #[test]
    fn computes_w3c_relative_indexes() {
        let nquads = sample_nquads();
        let mandatory = group_from_indexes(&nquads, &[0, 1]);
        let selective = group_from_indexes(&nquads, &[3, 4]);
        let combined = group_from_indexes(&nquads, &[0, 1, 3, 4]);

        assert_eq!(
            mandatory_indexes_in_revealed_document(&combined.matching, &mandatory.matching),
            vec![0, 1]
        );
        assert_eq!(
            selective_indexes_in_bbs_messages(&mandatory.non_matching, &selective.matching),
            vec![1, 2]
        );
    }

    #[test]
    fn groups_canonical_nquads_by_absolute_index() {
        let nquads = sample_nquads();
        let selected = vec![nquads[1].clone(), nquads[3].clone()];
        let grouped = group_canonical_nquads(&nquads, &selected, Vec::new());
        assert_eq!(
            grouped.matching.keys().copied().collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(
            grouped.non_matching.keys().copied().collect::<Vec<_>>(),
            vec![0, 2, 4]
        );
    }

    #[test]
    fn prepared_issue_derive_verify_round_trip() {
        prepared_hidden_tail_round_trip(Bbs2023Core::new());
    }

    #[test]
    #[ignore = "requires a real CUDA device and CUBBS_BBS_FORCE_GPU=1"]
    fn loaded_gpu_prepared_hidden_tail_round_trip() {
        let backend = BbsBackend::GpuRequestedCpuFallback;
        backend.ensure_hardware_available().unwrap();
        prepared_hidden_tail_round_trip(Bbs2023Core::with_backend(backend));
    }

    #[test]
    fn proof_message_count_includes_hidden_tail_and_checks_encoding() {
        assert_eq!(
            total_message_count_from_proof(&[0; 368], &[0, 1, 2, 3, 5]).unwrap(),
            8
        );
        assert_eq!(total_message_count_from_proof(&[0; 368], &[]).unwrap(), 3);
        assert_eq!(
            total_message_count_from_proof(&[0; 272], &[0, 1, 2]).unwrap(),
            3
        );
        assert_eq!(total_message_count_from_proof(&[0; 272], &[]).unwrap(), 0);
        for len in [0, 271, 273, 303] {
            assert!(total_message_count_from_proof(&vec![0; len], &[]).is_err());
        }
        for indexes in [vec![0, 0], vec![1, 0], vec![3], vec![usize::MAX]] {
            assert!(total_message_count_from_proof(&[0; 272], &indexes).is_err());
        }
    }

    fn prepared_hidden_tail_round_trip(core: Bbs2023Core) {
        let engine = ZkryptiumBbsEngine::new();
        let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
        let nquads = sample_nquads();
        let proof_hash = [7; 32];
        let mandatory = group_from_indexes(&nquads, &[0, 1]);
        let selective = group_from_indexes(&nquads, &[2, 3]);
        let combined = group_from_indexes(&nquads, &[0, 1, 2, 3]);
        let mandatory_values: Vec<String> = mandatory.matching.values().cloned().collect();
        let non_mandatory_values: Vec<String> = mandatory.non_matching.values().cloned().collect();

        let base_proof = core
            .issue_base_proof(IssueBaseProofRequest {
                secret_key: keys.secret_key,
                public_key: keys.public_key.clone(),
                proof_hash,
                mandatory_pointers: vec!["/issuer".to_owned()],
                mandatory: mandatory_values.clone(),
                non_mandatory: non_mandatory_values.clone(),
                hmac_key: vec![1; HMAC_KEY_LEN],
            })
            .unwrap();
        assert!(core
            .verify_base_proof(PreparedBaseVerification {
                public_key: keys.public_key.clone(),
                proof_hash,
                mandatory: mandatory_values,
                non_mandatory: non_mandatory_values,
                proof_value: base_proof.clone(),
            })
            .unwrap());

        let derived = core
            .derive_disclosure_proof(PreparedDisclosureRequest {
                base_proof_value: base_proof,
                groups: DisclosureGroups {
                    mandatory,
                    selective,
                    combined,
                    verifier_label_map: BTreeMap::new(),
                },
                presentation_header: b"domain-bound".to_vec(),
            })
            .unwrap();
        let revealed_nquads = vec![
            nquads[0].clone(),
            nquads[1].clone(),
            nquads[2].clone(),
            nquads[3].clone(),
        ];
        let request = PreparedDisclosureVerification {
            public_key: keys.public_key,
            proof_hash,
            canonical_nquads: revealed_nquads,
            proof_value: derived.proof_value,
            expected_presentation_header: Some(b"domain-bound".to_vec()),
        };
        let lowered = lower_disclosure_verification(request.clone())
            .unwrap()
            .unwrap();
        assert_eq!(lowered.disclosed_indexes, [0, 1]);
        assert_eq!(lowered.total_message_count, 3);
        assert!(core.verify_disclosure_proof(request.clone()).unwrap());
        let batch =
            core.verify_disclosure_proof_batch(PreparedDisclosureVerificationBatchRequest {
                items: vec![request.clone(), request],
                fail_fast: false,
            });
        assert_eq!(batch.succeeded, 2);
        assert!(batch.results.into_iter().all(|item| item.result.unwrap()));
    }
}
