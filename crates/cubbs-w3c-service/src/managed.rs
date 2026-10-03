use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
};

use cubbs_bbs::{BbsEngine, ProofVerifyRequest, VerifyRequest};
use cubbs_vc_di_bbs::{
    assemble_jsonld_disclosure_proof, complete_disclosure_proof, lower_base_verification,
    lower_disclosure_proof, lower_disclosure_verification, prepare_jsonld_base_verification,
    prepare_jsonld_disclosure_proof, prepare_jsonld_disclosure_verification,
    DeriveJsonLdDisclosureProofRequest, LoweredDisclosureProof, PreparedJsonLdDisclosureProof,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManagedRegisterRequest {
    pub credential_id: String,
    pub verifiable_credential: Value,
    #[serde(default)]
    pub derive_profiles: Vec<ManagedDeriveProfile>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManagedDeriveProfile {
    pub profile_id: String,
    #[serde(default)]
    pub selective_pointers: Vec<String>,
    #[serde(default)]
    pub presentation_header: Vec<u8>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManagedLookupRequest {
    pub credential_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManagedDeriveLookupRequest {
    pub credential_id: String,
    pub profile_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ManagedRegisterResponse {
    pub credential_id: String,
    pub replaced: bool,
    pub verification_kind: &'static str,
    pub derive_profiles: usize,
    pub message_count: usize,
    pub disclosed_message_count: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ManagedVcSnapshot {
    pub enabled: bool,
    pub capacity: usize,
    pub entries: usize,
    pub derive_profiles: usize,
    pub inserts: u64,
    pub replacements: u64,
    pub evictions: u64,
    pub lookup_hits: u64,
    pub lookup_misses: u64,
}

#[derive(Debug, Clone)]
pub enum ManagedVerification {
    Base(Arc<VerifyRequest>),
    Proof(Arc<ProofVerifyRequest>),
}

#[derive(Debug, Clone)]
pub struct ManagedDeriveEntry {
    prepared: Arc<PreparedJsonLdDisclosureProof>,
    lowered: Arc<LoweredDisclosureProof>,
}

#[derive(Debug)]
struct ManagedVcEntry {
    verification: ManagedVerification,
    derive_profiles: HashMap<String, Arc<ManagedDeriveEntry>>,
}

#[derive(Debug, Default)]
struct ManagedVcState {
    entries: HashMap<String, Arc<ManagedVcEntry>>,
    insertion_order: VecDeque<String>,
    derive_profiles: usize,
}

#[derive(Debug)]
pub struct ManagedVcStore {
    enabled: bool,
    capacity: usize,
    state: RwLock<ManagedVcState>,
    inserts: AtomicU64,
    replacements: AtomicU64,
    evictions: AtomicU64,
    lookup_hits: AtomicU64,
    lookup_misses: AtomicU64,
}

impl ManagedVcStore {
    pub fn from_env() -> Self {
        let enabled = env_bool("CUBBS_W3C_MANAGED_VC_MODE", false);
        let capacity = if enabled {
            env_usize("CUBBS_W3C_MANAGED_VC_CAPACITY", 50_000).max(1)
        } else {
            0
        };
        Self {
            enabled,
            capacity,
            state: RwLock::new(ManagedVcState::default()),
            inserts: AtomicU64::new(0),
            replacements: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            lookup_hits: AtomicU64::new(0),
            lookup_misses: AtomicU64::new(0),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub async fn register(
        &self,
        req: ManagedRegisterRequest,
    ) -> Result<ManagedRegisterResponse, String> {
        self.require_enabled()?;
        validate_id("credentialId", &req.credential_id)?;
        let proof_value = req
            .verifiable_credential
            .pointer("/proof/proofValue")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing proof.proofValue".to_owned())?;

        let (
            verification,
            verification_kind,
            derive_profiles,
            message_count,
            disclosed_message_count,
        ) = if proof_value.starts_with("u2V0C") {
            let prepared = prepare_jsonld_base_verification(req.verifiable_credential.clone())
                .await
                .map_err(|error| error.to_string())?;
            let verification = lower_base_verification(prepared)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "base VC public key or BBS header mismatch".to_owned())?;
            let mut profiles = HashMap::with_capacity(req.derive_profiles.len());
            let mut seen = HashSet::with_capacity(req.derive_profiles.len());
            for profile in req.derive_profiles {
                validate_id("profileId", &profile.profile_id)?;
                if !seen.insert(profile.profile_id.clone()) {
                    return Err(format!("duplicate profileId: {}", profile.profile_id));
                }
                let prepared =
                    prepare_jsonld_disclosure_proof(DeriveJsonLdDisclosureProofRequest {
                        signed_document: req.verifiable_credential.clone(),
                        selective_pointers: profile.selective_pointers,
                        presentation_header: profile.presentation_header,
                    })
                    .await
                    .map_err(|error| error.to_string())?;
                let lowered = lower_disclosure_proof(prepared.primitive.clone())
                    .map_err(|error| error.to_string())?;
                profiles.insert(
                    profile.profile_id,
                    Arc::new(ManagedDeriveEntry {
                        prepared: Arc::new(prepared),
                        lowered: Arc::new(lowered),
                    }),
                );
            }
            let message_count = verification.messages.len();
            (
                ManagedVerification::Base(Arc::new(verification)),
                "base",
                profiles,
                message_count,
                None,
            )
        } else if proof_value.starts_with("u2V0D") {
            if !req.derive_profiles.is_empty() {
                return Err("derived VPs cannot define deriveProfiles".to_owned());
            }
            let prepared = prepare_jsonld_disclosure_verification(req.verifiable_credential, None)
                .await
                .map_err(|error| error.to_string())?;
            let verification = lower_disclosure_verification(prepared)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "derived VP presentation header mismatch".to_owned())?;
            let message_count = verification.total_message_count;
            let disclosed_message_count = verification.disclosed_messages.len();
            (
                ManagedVerification::Proof(Arc::new(verification)),
                "proof",
                HashMap::new(),
                message_count,
                Some(disclosed_message_count),
            )
        } else {
            return Err("unsupported bbs-2023 proofValue prefix".to_owned());
        };

        let profile_count = derive_profiles.len();
        let replaced = self.insert(
            req.credential_id.clone(),
            ManagedVcEntry {
                verification,
                derive_profiles,
            },
        );
        Ok(ManagedRegisterResponse {
            credential_id: req.credential_id,
            replaced,
            verification_kind,
            derive_profiles: profile_count,
            message_count,
            disclosed_message_count,
        })
    }

    pub fn verification(&self, credential_id: &str) -> Result<ManagedVerification, String> {
        self.require_enabled()?;
        validate_id("credentialId", credential_id)?;
        Ok(self.lookup(credential_id)?.verification.clone())
    }

    pub fn derive(
        &self,
        credential_id: &str,
        profile_id: &str,
    ) -> Result<Arc<ManagedDeriveEntry>, String> {
        self.require_enabled()?;
        validate_id("credentialId", credential_id)?;
        validate_id("profileId", profile_id)?;
        let entry = self.lookup(credential_id)?;
        entry
            .derive_profiles
            .get(profile_id)
            .cloned()
            .ok_or_else(|| {
                format!("managed derive profile not found: {credential_id}/{profile_id}")
            })
    }

    pub fn snapshot(&self) -> ManagedVcSnapshot {
        let state = self.state.read().expect("managed VC lock poisoned");
        ManagedVcSnapshot {
            enabled: self.enabled,
            capacity: self.capacity,
            entries: state.entries.len(),
            derive_profiles: state.derive_profiles,
            inserts: self.inserts.load(Ordering::Relaxed),
            replacements: self.replacements.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            lookup_hits: self.lookup_hits.load(Ordering::Relaxed),
            lookup_misses: self.lookup_misses.load(Ordering::Relaxed),
        }
    }

    fn require_enabled(&self) -> Result<(), String> {
        self.enabled
            .then_some(())
            .ok_or_else(|| "managed VC mode is disabled".to_owned())
    }

    fn lookup(&self, credential_id: &str) -> Result<Arc<ManagedVcEntry>, String> {
        let entry = self
            .state
            .read()
            .expect("managed VC lock poisoned")
            .entries
            .get(credential_id)
            .cloned();
        match entry {
            Some(entry) => {
                self.lookup_hits.fetch_add(1, Ordering::Relaxed);
                Ok(entry)
            }
            None => {
                self.lookup_misses.fetch_add(1, Ordering::Relaxed);
                Err(format!("managed credential not found: {credential_id}"))
            }
        }
    }

    fn insert(&self, credential_id: String, entry: ManagedVcEntry) -> bool {
        let profile_count = entry.derive_profiles.len();
        let mut state = self.state.write().expect("managed VC lock poisoned");
        let replaced = state.entries.contains_key(&credential_id);
        if let Some(previous) = state.entries.remove(&credential_id) {
            state.derive_profiles = state
                .derive_profiles
                .saturating_sub(previous.derive_profiles.len());
            state.insertion_order.retain(|id| id != &credential_id);
            self.replacements.fetch_add(1, Ordering::Relaxed);
        }
        while state.entries.len() >= self.capacity {
            let Some(oldest) = state.insertion_order.pop_front() else {
                break;
            };
            if let Some(evicted) = state.entries.remove(&oldest) {
                state.derive_profiles = state
                    .derive_profiles
                    .saturating_sub(evicted.derive_profiles.len());
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        state.derive_profiles += profile_count;
        state.insertion_order.push_back(credential_id.clone());
        state.entries.insert(credential_id, Arc::new(entry));
        self.inserts.fetch_add(1, Ordering::Relaxed);
        replaced
    }
}

pub fn process_base_verification(
    engine: &BbsEngine,
    items: Vec<Arc<VerifyRequest>>,
) -> Vec<Result<Value, String>> {
    let requests = items.iter().map(|item| (**item).clone()).collect();
    bool_results(engine.verify_batch(requests, false))
}

pub fn process_proof_verification(
    engine: &BbsEngine,
    items: Vec<Arc<ProofVerifyRequest>>,
) -> Vec<Result<Value, String>> {
    let requests = items.iter().map(|item| (**item).clone()).collect();
    bool_results(engine.proof_verify_batch(requests, false))
}

pub fn process_derivation(
    engine: &BbsEngine,
    items: Vec<Arc<ManagedDeriveEntry>>,
) -> Vec<Result<Value, String>> {
    process_derivation_with_output(engine, items, true)
}

pub fn process_derivation_compact(
    engine: &BbsEngine,
    items: Vec<Arc<ManagedDeriveEntry>>,
) -> Vec<Result<Value, String>> {
    process_derivation_with_output(engine, items, false)
}

fn process_derivation_with_output(
    engine: &BbsEngine,
    items: Vec<Arc<ManagedDeriveEntry>>,
    assemble_document: bool,
) -> Vec<Result<Value, String>> {
    let requests = items
        .iter()
        .map(|item| item.lowered.request.clone())
        .collect();
    let batch = engine.proof_gen_batch(requests, false);
    let mut ordered = (0..items.len())
        .map(|_| None)
        .collect::<Vec<Option<Result<Value, String>>>>();
    for result in batch.results {
        let index = result.index;
        let value = result
            .result
            .map_err(|error| error.to_string())
            .and_then(|bbs_proof| {
                let material = complete_disclosure_proof(&items[index].lowered, bbs_proof)
                    .map_err(|error| error.to_string())?;
                if assemble_document {
                    assemble_jsonld_disclosure_proof(&items[index].prepared, material.proof_value)
                        .map_err(|error| error.to_string())
                } else {
                    Ok(Value::String(material.proof_value))
                }
            });
        ordered[index] = Some(value);
    }
    finish_ordered(ordered)
}

fn bool_results(batch: cubbs_bbs::BbsBatchRunResult<bool>) -> Vec<Result<Value, String>> {
    let mut ordered = (0..batch.attempted)
        .map(|_| None)
        .collect::<Vec<Option<Result<Value, String>>>>();
    for result in batch.results {
        ordered[result.index] = Some(match result.result {
            Ok(true) => Ok(json!({"verified": true, "checks": ["proof"]})),
            Ok(false) => Err("proof verification failed".to_owned()),
            Err(error) => Err(error.to_string()),
        });
    }
    finish_ordered(ordered)
}

fn finish_ordered(ordered: Vec<Option<Result<Value, String>>>) -> Vec<Result<Value, String>> {
    ordered
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err("missing managed batch result".to_owned())))
        .collect()
}

fn validate_id(name: &str, value: &str) -> Result<(), String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    if trimmed.len() > 512 {
        return Err(format!("{name} exceeds 512 bytes"));
    }
    Ok(())
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_bounded() {
        assert!(validate_id("credentialId", "vc-1").is_ok());
        assert!(validate_id("credentialId", "").is_err());
        assert!(validate_id("credentialId", &"x".repeat(513)).is_err());
    }

    fn test_store(capacity: usize) -> ManagedVcStore {
        ManagedVcStore {
            enabled: true,
            capacity,
            state: RwLock::new(ManagedVcState::default()),
            inserts: AtomicU64::new(0),
            replacements: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            lookup_hits: AtomicU64::new(0),
            lookup_misses: AtomicU64::new(0),
        }
    }

    fn entry(tag: u8) -> ManagedVcEntry {
        ManagedVcEntry {
            verification: ManagedVerification::Base(Arc::new(VerifyRequest {
                public_key: vec![tag],
                signature: vec![tag],
                header: vec![tag],
                messages: vec![vec![tag]],
            })),
            derive_profiles: HashMap::new(),
        }
    }

    #[test]
    fn replacement_and_capacity_are_exact() {
        let store = test_store(2);
        assert!(!store.insert("vc-1".to_owned(), entry(1)));
        assert!(!store.insert("vc-2".to_owned(), entry(2)));
        assert!(store.insert("vc-2".to_owned(), entry(3)));
        assert!(!store.insert("vc-3".to_owned(), entry(4)));

        assert!(store.verification("vc-1").is_err());
        assert!(store.verification("vc-2").is_ok());
        assert!(store.verification("vc-3").is_ok());
        let snapshot = store.snapshot();
        assert_eq!(snapshot.entries, 2);
        assert_eq!(snapshot.replacements, 1);
        assert_eq!(snapshot.evictions, 1);
    }
}
