use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    io::Write,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Instant,
};

use json_ld::{
    expansion::{Action, Policy},
    rdf::RdfDirection,
    syntax::{Parse, Value as JsonLdValue},
    ChainLoader, JsonLdProcessor, Loader, Options, RemoteDocument, ReqwestLoader,
};
use rdf_types::{Id, RdfDisplay, Term};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::Notify;

use crate::{
    canonical_label_map_from_input_map, canonicalize_nquads, group_canonical_nquads,
    relabel_and_sort_canonical_nquads, select_json, shuffled_bnode_label_map,
    CanonicalizationError, CanonicalizedNQuads, GroupedNQuads, HMAC_KEY_LEN,
};

#[derive(Debug, thiserror::Error)]
pub enum JsonLdCanonicalizationError {
    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("JSON-LD parse failed: {0}")]
    Parse(String),

    #[error("JSON-LD context cache failed: {0}")]
    ContextCache(String),

    #[error("JSON-LD to RDF failed: {0}")]
    ToRdf(String),

    #[error("JSON-LD data loss detected: {0}")]
    DataLoss(String),

    #[error(transparent)]
    RdfCanon(#[from] CanonicalizationError),

    #[error(transparent)]
    Selection(#[from] crate::SelectionError),

    #[error("invalid HMAC key length: expected 32 bytes, got {0}")]
    InvalidHmacKeyLength(usize),

    #[error("coalesced JSON-LD canonicalization failed: {0}")]
    Coalesced(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelReplacementCanonicalizedJsonLd {
    pub nquads: Vec<String>,
    pub canonical_id_map: BTreeMap<String, String>,
    pub label_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizedJsonLdGroup {
    pub group: GroupedNQuads,
    pub selected_document: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizedJsonLdGroups {
    pub nquads: Vec<String>,
    pub label_map: BTreeMap<String, String>,
    pub groups: BTreeMap<String, CanonicalizedJsonLdGroup>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalizationCacheSnapshot {
    pub capacity: usize,
    pub entries: usize,
    pub label_capacity: usize,
    pub label_entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub label_hits: u64,
    pub label_misses: u64,
    pub in_flight: usize,
    pub coalesced_waiters: u64,
    pub canonicalization_leaders: u64,
    pub canonicalize_total_ms: f64,
    pub cache_key_ms: f64,
    pub cache_lookup_ms: f64,
    pub jsonld_to_nquads_total_ms: f64,
    pub jsonld_pre_to_rdf_ms: f64,
    pub jsonld_to_rdf_ms: f64,
    pub nquads_serialize_ms: f64,
    pub rdf_canonicalize_ms: f64,
    pub label_replacement_ms: f64,
    pub select_ms: f64,
    pub group_ms: f64,
}

#[derive(Debug)]
struct CanonicalizationCacheState {
    entries: HashMap<DocumentCacheKey, Arc<CanonicalizedNQuads>>,
    insertion_order: VecDeque<DocumentCacheKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DocumentCacheKey([u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LabelCacheKey {
    document: DocumentCacheKey,
    hmac_key: [u8; HMAC_KEY_LEN],
}

#[derive(Debug)]
struct LabelReplacementCacheState {
    entries: HashMap<LabelCacheKey, Arc<LabelReplacementCanonicalizedJsonLd>>,
    insertion_order: VecDeque<LabelCacheKey>,
}

#[derive(Debug)]
struct CanonicalizationCache {
    capacity: usize,
    state: Mutex<CanonicalizationCacheState>,
    label_capacity: usize,
    label_state: Mutex<LabelReplacementCacheState>,
    in_flight: Mutex<HashMap<DocumentCacheKey, Arc<CanonicalizationFlight>>>,
    hits: AtomicU64,
    misses: AtomicU64,
    label_hits: AtomicU64,
    label_misses: AtomicU64,
    coalesced_waiters: AtomicU64,
    canonicalization_leaders: AtomicU64,
    canonicalize_total_micros: AtomicU64,
    cache_key_micros: AtomicU64,
    cache_lookup_micros: AtomicU64,
    jsonld_to_nquads_total_micros: AtomicU64,
    jsonld_pre_to_rdf_micros: AtomicU64,
    jsonld_to_rdf_micros: AtomicU64,
    nquads_serialize_micros: AtomicU64,
    rdf_canonicalize_micros: AtomicU64,
    label_replacement_micros: AtomicU64,
    select_micros: AtomicU64,
    group_micros: AtomicU64,
}

#[derive(Debug)]
struct CanonicalizationFlight {
    result: Mutex<Option<Result<CanonicalizedNQuads, String>>>,
    ready: Notify,
}

struct DocumentHashWriter(Sha256);

impl Write for DocumentHashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn json_content_digest(document: &Value) -> Result<[u8; 32], serde_json::Error> {
    let mut writer = DocumentHashWriter(Sha256::new());
    serde_json::to_writer(&mut writer, document)?;
    Ok(writer.0.finalize().into())
}

fn document_cache_key(document: &Value) -> Result<DocumentCacheKey, serde_json::Error> {
    json_content_digest(document).map(DocumentCacheKey)
}

pub async fn canonicalize_jsonld(
    document: &Value,
) -> Result<CanonicalizedNQuads, JsonLdCanonicalizationError> {
    let total_started = Instant::now();
    let cache = canonicalization_cache();
    if cache.capacity == 0 {
        cache.misses.fetch_add(1, Ordering::Relaxed);
        let nquads = jsonld_to_nquads(document).await?;
        let rdf_canon_started = Instant::now();
        let canonicalized = canonicalize_nquads(&nquads)?;
        cache
            .rdf_canonicalize_micros
            .fetch_add(elapsed_micros(rdf_canon_started), Ordering::Relaxed);
        cache
            .canonicalize_total_micros
            .fetch_add(elapsed_micros(total_started), Ordering::Relaxed);
        return Ok(canonicalized);
    }
    let cache_key_started = Instant::now();
    let cache_key = document_cache_key(document)?;
    cache
        .cache_key_micros
        .fetch_add(elapsed_micros(cache_key_started), Ordering::Relaxed);
    canonicalize_jsonld_with_key(document, cache_key, total_started).await
}

async fn canonicalize_jsonld_with_key(
    document: &Value,
    cache_key: DocumentCacheKey,
    total_started: Instant,
) -> Result<CanonicalizedNQuads, JsonLdCanonicalizationError> {
    let cache_lookup_started = Instant::now();
    if let Some(canonicalized) = canonicalization_cache_get(&cache_key) {
        let cache = canonicalization_cache();
        cache
            .cache_lookup_micros
            .fetch_add(elapsed_micros(cache_lookup_started), Ordering::Relaxed);
        cache
            .canonicalize_total_micros
            .fetch_add(elapsed_micros(total_started), Ordering::Relaxed);
        return Ok(canonicalized);
    }
    canonicalization_cache()
        .cache_lookup_micros
        .fetch_add(elapsed_micros(cache_lookup_started), Ordering::Relaxed);

    let (flight, leader) = canonicalization_flight(&cache_key);
    if !leader {
        let result = wait_for_canonicalization(&flight).await;
        canonicalization_cache()
            .canonicalize_total_micros
            .fetch_add(elapsed_micros(total_started), Ordering::Relaxed);
        return result.map_err(JsonLdCanonicalizationError::Coalesced);
    }

    let flight_guard = CanonicalizationFlightGuard {
        key: cache_key,
        flight: Arc::clone(&flight),
        completed: false,
    };
    let computed = async {
        let nquads = jsonld_to_nquads(document).await?;
        let rdf_canon_started = Instant::now();
        let canonicalized = canonicalize_nquads(&nquads)?;
        canonicalization_cache()
            .rdf_canonicalize_micros
            .fetch_add(elapsed_micros(rdf_canon_started), Ordering::Relaxed);
        Ok::<_, JsonLdCanonicalizationError>(canonicalized)
    }
    .await;
    let canonicalized = match computed {
        Ok(canonicalized) => {
            canonicalization_cache_insert(cache_key, canonicalized.clone());
            flight_guard.complete(Ok(canonicalized.clone()));
            canonicalized
        }
        Err(error) => {
            flight_guard.complete(Err(error.to_string()));
            canonicalization_cache()
                .canonicalize_total_micros
                .fetch_add(elapsed_micros(total_started), Ordering::Relaxed);
            return Err(error);
        }
    };
    canonicalization_cache()
        .canonicalize_total_micros
        .fetch_add(elapsed_micros(total_started), Ordering::Relaxed);
    Ok(canonicalized)
}

pub async fn label_replacement_canonicalize_jsonld(
    document: &Value,
    hmac_key: &[u8],
) -> Result<LabelReplacementCanonicalizedJsonLd, JsonLdCanonicalizationError> {
    let started = Instant::now();
    let hmac_key: [u8; HMAC_KEY_LEN] = hmac_key
        .try_into()
        .map_err(|_| JsonLdCanonicalizationError::InvalidHmacKeyLength(hmac_key.len()))?;
    let cache = canonicalization_cache();
    if cache.capacity == 0 && cache.label_capacity == 0 {
        let canonicalized = canonicalize_jsonld(document).await?;
        if canonicalized.canonical_id_map.is_empty() {
            cache
                .label_replacement_micros
                .fetch_add(elapsed_micros(started), Ordering::Relaxed);
            return Ok(LabelReplacementCanonicalizedJsonLd {
                nquads: canonicalized.lines,
                canonical_id_map: BTreeMap::new(),
                label_map: BTreeMap::new(),
            });
        }
        let input_to_replacement =
            shuffled_bnode_label_map(&canonicalized.canonical_id_map, &hmac_key);
        let label_map = canonical_label_map_from_input_map(
            &canonicalized.canonical_id_map,
            &input_to_replacement,
        );
        let nquads = relabel_and_sort_canonical_nquads(&canonicalized.lines, &label_map);
        cache
            .label_replacement_micros
            .fetch_add(elapsed_micros(started), Ordering::Relaxed);
        return Ok(LabelReplacementCanonicalizedJsonLd {
            nquads,
            canonical_id_map: canonicalized.canonical_id_map,
            label_map,
        });
    }
    let cache_key_started = Instant::now();
    let document_key = document_cache_key(document)?;
    cache
        .cache_key_micros
        .fetch_add(elapsed_micros(cache_key_started), Ordering::Relaxed);
    let cache_lookup_started = Instant::now();
    let canonicalized = match canonicalization_cache_peek(&document_key) {
        Some(canonicalized) => {
            let cache = canonicalization_cache();
            cache.hits.fetch_add(1, Ordering::Relaxed);
            cache
                .cache_lookup_micros
                .fetch_add(elapsed_micros(cache_lookup_started), Ordering::Relaxed);
            cache
                .canonicalize_total_micros
                .fetch_add(elapsed_micros(started), Ordering::Relaxed);
            canonicalized
        }
        None => Arc::new(canonicalize_jsonld_with_key(document, document_key, started).await?),
    };

    // No blank-node labels means the HMAC cannot affect the canonical N-Quads.
    // Issuance deliberately generates a fresh HMAC key for every proof, so
    // caching these values by (document, HMAC) would create only misses and
    // continuous eviction while returning the same structural result.
    if canonicalized.canonical_id_map.is_empty() {
        canonicalization_cache()
            .label_replacement_micros
            .fetch_add(elapsed_micros(started), Ordering::Relaxed);
        return Ok(LabelReplacementCanonicalizedJsonLd {
            nquads: canonicalized.lines.clone(),
            canonical_id_map: BTreeMap::new(),
            label_map: BTreeMap::new(),
        });
    }

    let label_key = LabelCacheKey {
        document: document_key,
        hmac_key,
    };
    if let Some(cached) = label_replacement_cache_get(&label_key) {
        canonicalization_cache()
            .label_replacement_micros
            .fetch_add(elapsed_micros(started), Ordering::Relaxed);
        return Ok(cached);
    }

    let input_to_replacement = shuffled_bnode_label_map(&canonicalized.canonical_id_map, &hmac_key);
    let label_map =
        canonical_label_map_from_input_map(&canonicalized.canonical_id_map, &input_to_replacement);
    let nquads = relabel_and_sort_canonical_nquads(&canonicalized.lines, &label_map);

    let result = LabelReplacementCanonicalizedJsonLd {
        nquads,
        canonical_id_map: canonicalized.canonical_id_map.clone(),
        label_map,
    };
    label_replacement_cache_insert(label_key, result.clone());
    canonicalization_cache()
        .label_replacement_micros
        .fetch_add(elapsed_micros(started), Ordering::Relaxed);
    Ok(result)
}

pub async fn canonicalize_and_group_jsonld(
    document: &Value,
    groups: &BTreeMap<String, Vec<String>>,
    hmac_key: &[u8],
) -> Result<CanonicalizedJsonLdGroups, JsonLdCanonicalizationError> {
    let canonicalized = label_replacement_canonicalize_jsonld(document, hmac_key).await?;
    let mut grouped = BTreeMap::new();

    for (name, pointers) in groups {
        let select_started = Instant::now();
        let selected_document = select_json(document, pointers)?;
        canonicalization_cache()
            .select_micros
            .fetch_add(elapsed_micros(select_started), Ordering::Relaxed);
        let selected_nquads = if selected_document.is_null() {
            Vec::new()
        } else {
            label_replacement_canonicalize_jsonld(&selected_document, hmac_key)
                .await?
                .nquads
        };
        let group_started = Instant::now();
        grouped.insert(
            name.clone(),
            CanonicalizedJsonLdGroup {
                group: group_canonical_nquads(&canonicalized.nquads, &selected_nquads, Vec::new()),
                selected_document,
            },
        );
        canonicalization_cache()
            .group_micros
            .fetch_add(elapsed_micros(group_started), Ordering::Relaxed);
    }

    Ok(CanonicalizedJsonLdGroups {
        nquads: canonicalized.nquads,
        label_map: canonicalized.label_map,
        groups: grouped,
    })
}

pub async fn jsonld_to_nquads(document: &Value) -> Result<String, JsonLdCanonicalizationError> {
    let started = Instant::now();
    if let Some(fast_nquads) = try_fast_w3c_v2_jsonld_to_nquads(document) {
        canonicalization_cache()
            .jsonld_to_nquads_total_micros
            .fetch_add(elapsed_micros(started), Ordering::Relaxed);
        return Ok(fast_nquads);
    }
    let result = if shared_jsonld_loader_enabled() {
        let loader = ChainLoader::new(w3c_context_cache()?, reqwest_loader());
        jsonld_to_nquads_using(document, &loader).await
    } else {
        let fallback_loader = ReqwestLoader::default();
        let loader = ChainLoader::new(w3c_context_cache()?, fallback_loader);
        jsonld_to_nquads_using(document, &loader).await
    };
    canonicalization_cache()
        .jsonld_to_nquads_total_micros
        .fetch_add(elapsed_micros(started), Ordering::Relaxed);
    result
}

const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const XSD_BOOLEAN_IRI: &str = "http://www.w3.org/2001/XMLSchema#boolean";
const XSD_INTEGER_IRI: &str = "http://www.w3.org/2001/XMLSchema#integer";
const XSD_DATETIME_IRI: &str = "http://www.w3.org/2001/XMLSchema#dateTime";

#[derive(Clone, Copy)]
enum FastTermKind {
    PlainString,
    Iri,
    Typed(&'static str),
    ProofPurposeVocab,
    CredentialSubject,
}

#[derive(Clone, Copy)]
struct FastTermDef {
    iri: &'static str,
    kind: FastTermKind,
}

fn is_fast_abs_iri(s: &str) -> bool {
    if s.is_empty() || s.starts_with("_:") {
        return false;
    }
    let Some(colon) = s.find(':') else {
        return false;
    };
    if colon == 0 {
        return false;
    }
    let scheme = &s.as_bytes()[..colon];
    if !scheme[0].is_ascii_alphabetic()
        || !scheme
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
    {
        return false;
    }
    s.bytes().all(|b| {
        b > 0x20
            && b != 0x7f
            && !matches!(b, b'<' | b'>' | b'"' | b'{' | b'}' | b'|' | b'^' | b'`' | b'\\')
    })
}

fn is_protected_w3c_v2_term(key: &str) -> bool {
    matches!(
        key,
        "id" | "type"
            | "description"
            | "digestMultibase"
            | "digestSRI"
            | "mediaType"
            | "name"
            | "VerifiableCredential"
            | "EnvelopedVerifiableCredential"
            | "VerifiablePresentation"
            | "EnvelopedVerifiablePresentation"
            | "JsonSchemaCredential"
            | "JsonSchema"
            | "BitstringStatusListCredential"
            | "BitstringStatusList"
            | "BitstringStatusListEntry"
            | "DataIntegrityProof"
            | "proof"
            | "credentialSubject"
            | "issuer"
            | "validFrom"
            | "validUntil"
            | "cryptosuite"
            | "proofPurpose"
            | "proofValue"
            | "verificationMethod"
            | "created"
            | "expires"
            | "domain"
            | "challenge"
            | "nonce"
            | "previousProof"
    )
}

fn lookup_root_term<'a>(
    key: &str,
    has_vc_v2: bool,
    inline: &'a HashMap<&'a str, &'a str>,
) -> Option<(&'a str, FastTermKind)> {
    if has_vc_v2 {
        match key {
            "description" => return Some(("https://schema.org/description", FastTermKind::PlainString)),
            "name" => return Some(("https://schema.org/name", FastTermKind::PlainString)),
            "mediaType" => return Some(("https://schema.org/encodingFormat", FastTermKind::PlainString)),
            "digestMultibase" => {
                return Some((
                    "https://w3id.org/security#digestMultibase",
                    FastTermKind::Typed("https://w3id.org/security#multibase"),
                ));
            }
            "digestSRI" => {
                return Some((
                    "https://www.w3.org/2018/credentials#digestSRI",
                    FastTermKind::Typed("https://www.w3.org/2018/credentials#sriString"),
                ));
            }
            _ => {}
        }
    }
    inline.get(key).map(|iri| (*iri, FastTermKind::PlainString))
}

fn lookup_vc_scoped_term(key: &str) -> Option<FastTermDef> {
    match key {
        "credentialSubject" => Some(FastTermDef {
            iri: "https://www.w3.org/2018/credentials#credentialSubject",
            kind: FastTermKind::CredentialSubject,
        }),
        "issuer" => Some(FastTermDef {
            iri: "https://www.w3.org/2018/credentials#issuer",
            kind: FastTermKind::Iri,
        }),
        "validFrom" => Some(FastTermDef {
            iri: "https://www.w3.org/2018/credentials#validFrom",
            kind: FastTermKind::Typed(XSD_DATETIME_IRI),
        }),
        "validUntil" => Some(FastTermDef {
            iri: "https://www.w3.org/2018/credentials#validUntil",
            kind: FastTermKind::Typed(XSD_DATETIME_IRI),
        }),
        "description" => Some(FastTermDef {
            iri: "https://schema.org/description",
            kind: FastTermKind::PlainString,
        }),
        "name" => Some(FastTermDef {
            iri: "https://schema.org/name",
            kind: FastTermKind::PlainString,
        }),
        _ => None,
    }
}

fn lookup_di_scoped_term(key: &str) -> Option<FastTermDef> {
    match key {
        "cryptosuite" => Some(FastTermDef {
            iri: "https://w3id.org/security#cryptosuite",
            kind: FastTermKind::Typed("https://w3id.org/security#cryptosuiteString"),
        }),
        "verificationMethod" => Some(FastTermDef {
            iri: "https://w3id.org/security#verificationMethod",
            kind: FastTermKind::Iri,
        }),
        "proofPurpose" => Some(FastTermDef {
            iri: "https://w3id.org/security#proofPurpose",
            kind: FastTermKind::ProofPurposeVocab,
        }),
        "created" => Some(FastTermDef {
            iri: "http://purl.org/dc/terms/created",
            kind: FastTermKind::Typed(XSD_DATETIME_IRI),
        }),
        "expires" => Some(FastTermDef {
            iri: "https://w3id.org/security#expiration",
            kind: FastTermKind::Typed(XSD_DATETIME_IRI),
        }),
        "domain" => Some(FastTermDef {
            iri: "https://w3id.org/security#domain",
            kind: FastTermKind::PlainString,
        }),
        "challenge" => Some(FastTermDef {
            iri: "https://w3id.org/security#challenge",
            kind: FastTermKind::PlainString,
        }),
        "nonce" => Some(FastTermDef {
            iri: "https://w3id.org/security#nonce",
            kind: FastTermKind::PlainString,
        }),
        "proofValue" => Some(FastTermDef {
            iri: "https://w3id.org/security#proofValue",
            kind: FastTermKind::Typed("https://w3id.org/security#multibase"),
        }),
        "previousProof" => Some(FastTermDef {
            iri: "https://w3id.org/security#previousProof",
            kind: FastTermKind::Iri,
        }),
        _ => None,
    }
}

fn push_escaped_nquad_literal(out: &mut String, s: &str) -> bool {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || (c as u32) == 0x7f => return false,
            c => out.push(c),
        }
    }
    out.push('"');
    true
}

fn push_nquad_triple_iri(out: &mut String, subj: &str, pred: &str, obj_iri: &str) {
    out.push_str(subj);
    out.push_str(" <");
    out.push_str(pred);
    out.push_str("> <");
    out.push_str(obj_iri);
    out.push_str("> .\n");
}

fn emit_fast_value(
    out: &mut String,
    subj: &str,
    pred: &str,
    kind: FastTermKind,
    value: &Value,
    has_vc_v2: bool,
    inline: &HashMap<&str, &str>,
    bnode_counter: &mut usize,
) -> Option<()> {
    match value {
        Value::Null => Some(()),
        Value::Array(items) => {
            for item in items {
                emit_fast_value(out, subj, pred, kind, item, has_vc_v2, inline, bnode_counter)?;
            }
            Some(())
        }
        Value::String(s) => match kind {
            FastTermKind::PlainString => {
                out.push_str(subj);
                out.push_str(" <");
                out.push_str(pred);
                out.push_str("> ");
                if !push_escaped_nquad_literal(out, s) {
                    return None;
                }
                out.push_str(" .\n");
                Some(())
            }
            FastTermKind::Typed(dt) => {
                out.push_str(subj);
                out.push_str(" <");
                out.push_str(pred);
                out.push_str("> ");
                if !push_escaped_nquad_literal(out, s) {
                    return None;
                }
                out.push_str("^^<");
                out.push_str(dt);
                out.push_str("> .\n");
                Some(())
            }
            FastTermKind::Iri => {
                if s.is_empty() {
                    // Without a document base IRI, json-ld drops empty relative IRIs.
                    return Some(());
                }
                if !is_fast_abs_iri(s) {
                    return None;
                }
                push_nquad_triple_iri(out, subj, pred, s);
                Some(())
            }
            FastTermKind::ProofPurposeVocab => {
                let iri = match s.as_str() {
                    "assertionMethod" => "https://w3id.org/security#assertionMethod",
                    "authentication" => "https://w3id.org/security#authenticationMethod",
                    "capabilityDelegation" => "https://w3id.org/security#capabilityDelegationMethod",
                    "capabilityInvocation" => "https://w3id.org/security#capabilityInvocationMethod",
                    "keyAgreement" => "https://w3id.org/security#keyAgreementMethod",
                    other if is_fast_abs_iri(other) => other,
                    _ => return None,
                };
                push_nquad_triple_iri(out, subj, pred, iri);
                Some(())
            }
            FastTermKind::CredentialSubject => {
                if !is_fast_abs_iri(s) {
                    return None;
                }
                push_nquad_triple_iri(out, subj, pred, s);
                Some(())
            }
        },
        Value::Bool(b) => match kind {
            FastTermKind::PlainString => {
                out.push_str(subj);
                out.push_str(" <");
                out.push_str(pred);
                out.push_str(if *b { "> \"true\"^^<" } else { "> \"false\"^^<" });
                out.push_str(XSD_BOOLEAN_IRI);
                out.push_str("> .\n");
                Some(())
            }
            _ => None,
        },
        Value::Number(n) => match kind {
            FastTermKind::PlainString => {
                let int_str = if let Some(i) = n.as_i64() {
                    i.to_string()
                } else if let Some(u) = n.as_u64() {
                    u.to_string()
                } else {
                    return None;
                };
                out.push_str(subj);
                out.push_str(" <");
                out.push_str(pred);
                out.push_str("> \"");
                out.push_str(&int_str);
                out.push_str("\"^^<");
                out.push_str(XSD_INTEGER_IRI);
                out.push_str("> .\n");
                Some(())
            }
            _ => None,
        },
        Value::Object(nested) => match kind {
            FastTermKind::CredentialSubject => {
                // Type-scoped VC context does not propagate into credentialSubject.
                let nested_subj = resolve_node_subject(nested, bnode_counter)?;
                out.push_str(subj);
                out.push_str(" <");
                out.push_str(pred);
                out.push_str("> ");
                out.push_str(&nested_subj);
                out.push_str(" .\n");
                emit_flat_node(out, &nested_subj, nested, has_vc_v2, false, false, inline, bnode_counter)
            }
            _ => None,
        },
    }
}

fn resolve_node_subject(
    obj: &serde_json::Map<String, Value>,
    bnode_counter: &mut usize,
) -> Option<String> {
    if obj.contains_key("@id") {
        return None;
    }
    match obj.get("id") {
        Some(Value::String(id)) => {
            if !is_fast_abs_iri(id) {
                return None;
            }
            Some(format!("<{id}>"))
        }
        Some(_) => None,
        None => {
            let n = *bnode_counter;
            *bnode_counter += 1;
            Some(format!("_:{n}"))
        }
    }
}

fn emit_flat_node(
    out: &mut String,
    subj: &str,
    obj: &serde_json::Map<String, Value>,
    has_vc_v2: bool,
    is_vc: bool,
    is_di: bool,
    inline: &HashMap<&str, &str>,
    bnode_counter: &mut usize,
) -> Option<()> {
    for (key, value) in obj {
        if key == "@context" || key == "id" {
            continue;
        }
        if key.starts_with('@') {
            return None;
        }
        if key == "type" {
            emit_fast_types(out, subj, value, has_vc_v2, inline)?;
            continue;
        }
        if let Some(def) = if is_vc {
            lookup_vc_scoped_term(key)
        } else if is_di {
            lookup_di_scoped_term(key)
        } else {
            None
        } {
            emit_fast_value(
                out,
                subj,
                def.iri,
                def.kind,
                value,
                has_vc_v2,
                inline,
                bnode_counter,
            )?;
            continue;
        }
        let (iri, kind) = lookup_root_term(key, has_vc_v2, inline)?;
        emit_fast_value(out, subj, iri, kind, value, has_vc_v2, inline, bnode_counter)?;
    }
    Some(())
}

fn emit_fast_types(
    out: &mut String,
    subj: &str,
    value: &Value,
    has_vc_v2: bool,
    inline: &HashMap<&str, &str>,
) -> Option<()> {
    let mut emit_one = |s: &str| -> Option<()> {
        let iri = match s {
            "VerifiableCredential" if has_vc_v2 => {
                "https://www.w3.org/2018/credentials#VerifiableCredential"
            }
            "DataIntegrityProof" => "https://w3id.org/security#DataIntegrityProof",
            other => {
                if let Some(mapped) = inline.get(other) {
                    *mapped
                } else if is_fast_abs_iri(other) {
                    other
                } else {
                    return None;
                }
            }
        };
        push_nquad_triple_iri(out, subj, RDF_TYPE_IRI, iri);
        Some(())
    };
    match value {
        Value::String(s) => emit_one(s),
        Value::Array(arr) => {
            for item in arr {
                let Value::String(s) = item else {
                    return None;
                };
                emit_one(s)?;
            }
            Some(())
        }
        _ => None,
    }
}

fn parse_fast_ctx_entry<'a>(
    entry: &'a Value,
    has_vc_v2: &mut bool,
    has_di_v2: &mut bool,
    inline: &mut HashMap<&'a str, &'a str>,
) -> Option<()> {
    match entry {
        Value::String(url) if url == "https://www.w3.org/ns/credentials/v2" => {
            *has_vc_v2 = true;
            *has_di_v2 = true;
            Some(())
        }
        Value::String(url) if url == "https://w3id.org/security/data-integrity/v2" => {
            *has_di_v2 = true;
            Some(())
        }
        Value::Object(map) => {
            for (k, v) in map {
                if k.starts_with('@') || is_protected_w3c_v2_term(k) {
                    return None;
                }
                let Value::String(iri) = v else {
                    return None;
                };
                if !is_fast_abs_iri(iri) {
                    return None;
                }
                inline.insert(k.as_str(), iri.as_str());
            }
            Some(())
        }
        _ => None,
    }
}

fn try_fast_w3c_v2_jsonld_to_nquads(document: &Value) -> Option<String> {
    let obj = document.as_object()?;
    let ctx = obj.get("@context")?;
    let mut has_vc_v2 = false;
    let mut has_di_v2 = false;
    let mut inline: HashMap<&str, &str> = HashMap::new();

    match ctx {
        Value::String(_) => parse_fast_ctx_entry(ctx, &mut has_vc_v2, &mut has_di_v2, &mut inline)?,
        Value::Array(arr) if !arr.is_empty() => {
            for entry in arr {
                parse_fast_ctx_entry(entry, &mut has_vc_v2, &mut has_di_v2, &mut inline)?;
            }
        }
        _ => return None,
    }
    if !has_vc_v2 && !has_di_v2 {
        return None;
    }

    // Determine whether the root node activates VerifiableCredential or DataIntegrityProof scope.
    let mut is_vc = false;
    let mut is_di = false;
    if let Some(ty) = obj.get("type") {
        match ty {
            Value::String(s) => {
                if s == "VerifiableCredential" && has_vc_v2 {
                    is_vc = true;
                } else if s == "DataIntegrityProof" && has_di_v2 {
                    is_di = true;
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    if let Value::String(s) = item {
                        if s == "VerifiableCredential" && has_vc_v2 {
                            is_vc = true;
                        } else if s == "DataIntegrityProof" && has_di_v2 {
                            is_di = true;
                        }
                    }
                }
            }
            _ => return None,
        }
    }
    if is_vc && is_di {
        return None;
    }

    let mut bnode_counter = 0usize;
    let root_subj = resolve_node_subject(obj, &mut bnode_counter)?;
    let mut nquads = String::with_capacity(4096);
    emit_flat_node(
        &mut nquads,
        &root_subj,
        obj,
        has_vc_v2,
        is_vc,
        is_di,
        &inline,
        &mut bnode_counter,
    )?;
    if ensure_type_values_were_not_dropped(document, &nquads).is_err() {
        return None;
    }
    Some(nquads)
}

pub async fn jsonld_to_nquads_using(
    document: &Value,
    loader: &impl Loader,
) -> Result<String, JsonLdCanonicalizationError> {
    let pre_to_rdf_started = Instant::now();
    let parsed = JsonLdValue::from_serde_json(document.clone());
    let input = RemoteDocument::new(None::<iref::IriBuf>, None, parsed);

    let mut generator = rdf_types::generator::Blank::new();
    let mut options = Options::default();
    options.rdf_direction = Some(RdfDirection::I18nDatatype);
    options.produce_generalized_rdf = false;
    options.expansion_policy = Policy {
        invalid: Action::Reject,
        vocab: Action::Reject,
        allow_undefined: false,
    };
    canonicalization_cache()
        .jsonld_pre_to_rdf_micros
        .fetch_add(elapsed_micros(pre_to_rdf_started), Ordering::Relaxed);
    let to_rdf_started = Instant::now();
    let mut rdf = input
        .to_rdf_using(&mut generator, loader, options)
        .await
        .map_err(|e| JsonLdCanonicalizationError::ToRdf(format!("{e:?}")))?;
    canonicalization_cache()
        .jsonld_to_rdf_micros
        .fetch_add(elapsed_micros(to_rdf_started), Ordering::Relaxed);

    let serialize_started = Instant::now();
    let mut nquads = String::new();
    for rdf_types::Quad(subject, predicate, object, graph) in rdf.quads() {
        nquads.push_str(&serialize_id(subject.as_ref()));
        nquads.push(' ');
        nquads.push_str(&serialize_id(predicate.as_ref()));
        nquads.push(' ');
        nquads.push_str(&serialize_term(&object));
        if let Some(graph) = graph {
            nquads.push(' ');
            nquads.push_str(&serialize_id(graph));
        }
        nquads.push_str(" .\n");
    }
    ensure_type_values_were_not_dropped(document, &nquads)?;
    canonicalization_cache()
        .nquads_serialize_micros
        .fetch_add(elapsed_micros(serialize_started), Ordering::Relaxed);
    Ok(nquads)
}

fn reqwest_loader() -> &'static ReqwestLoader {
    static LOADER: OnceLock<ReqwestLoader> = OnceLock::new();
    LOADER.get_or_init(ReqwestLoader::default)
}

pub fn shared_jsonld_loader_enabled() -> bool {
    std::env::var("CUBBS_W3C_SHARED_JSONLD_LOADER")
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        .unwrap_or(true)
}

fn w3c_context_cache(
) -> Result<&'static BTreeMap<iref::IriBuf, RemoteDocument>, JsonLdCanonicalizationError> {
    static CACHE: OnceLock<Result<BTreeMap<iref::IriBuf, RemoteDocument>, String>> =
        OnceLock::new();
    match CACHE.get_or_init(|| build_w3c_context_cache().map_err(|error| error.to_string())) {
        Ok(contexts) => Ok(contexts),
        Err(error) => Err(JsonLdCanonicalizationError::ContextCache(error.clone())),
    }
}

fn canonicalization_cache() -> &'static CanonicalizationCache {
    static CACHE: OnceLock<CanonicalizationCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        let capacity = std::env::var("CUBBS_W3C_CANON_CACHE_CAPACITY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(32_768);
        let label_capacity = std::env::var("CUBBS_W3C_LABEL_CACHE_CAPACITY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| capacity.min(16_384));
        CanonicalizationCache {
            capacity,
            state: Mutex::new(CanonicalizationCacheState {
                entries: HashMap::new(),
                insertion_order: VecDeque::new(),
            }),
            label_capacity,
            label_state: Mutex::new(LabelReplacementCacheState {
                entries: HashMap::new(),
                insertion_order: VecDeque::new(),
            }),
            in_flight: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            label_hits: AtomicU64::new(0),
            label_misses: AtomicU64::new(0),
            coalesced_waiters: AtomicU64::new(0),
            canonicalization_leaders: AtomicU64::new(0),
            canonicalize_total_micros: AtomicU64::new(0),
            cache_key_micros: AtomicU64::new(0),
            cache_lookup_micros: AtomicU64::new(0),
            jsonld_to_nquads_total_micros: AtomicU64::new(0),
            jsonld_pre_to_rdf_micros: AtomicU64::new(0),
            jsonld_to_rdf_micros: AtomicU64::new(0),
            nquads_serialize_micros: AtomicU64::new(0),
            rdf_canonicalize_micros: AtomicU64::new(0),
            label_replacement_micros: AtomicU64::new(0),
            select_micros: AtomicU64::new(0),
            group_micros: AtomicU64::new(0),
        }
    })
}

// Ownership of an in-flight entry includes cancellation, not just success/error.
struct CanonicalizationFlightGuard {
    key: DocumentCacheKey,
    flight: Arc<CanonicalizationFlight>,
    completed: bool,
}

impl CanonicalizationFlightGuard {
    fn complete(mut self, result: Result<CanonicalizedNQuads, String>) {
        complete_canonicalization_flight(&self.key, &self.flight, result);
        self.completed = true;
    }
}

impl Drop for CanonicalizationFlightGuard {
    fn drop(&mut self) {
        if !self.completed {
            complete_canonicalization_flight(
                &self.key,
                &self.flight,
                Err("canonicalization leader cancelled".to_owned()),
            );
        }
    }
}

fn canonicalization_flight(key: &DocumentCacheKey) -> (Arc<CanonicalizationFlight>, bool) {
    let cache = canonicalization_cache();
    let mut in_flight = cache
        .in_flight
        .lock()
        .expect("canonicalization in-flight mutex poisoned");
    if let Some(flight) = in_flight.get(key) {
        cache.coalesced_waiters.fetch_add(1, Ordering::Relaxed);
        return (Arc::clone(flight), false);
    }
    let flight = Arc::new(CanonicalizationFlight {
        result: Mutex::new(None),
        ready: Notify::new(),
    });
    in_flight.insert(*key, Arc::clone(&flight));
    cache
        .canonicalization_leaders
        .fetch_add(1, Ordering::Relaxed);
    (flight, true)
}

async fn wait_for_canonicalization(
    flight: &CanonicalizationFlight,
) -> Result<CanonicalizedNQuads, String> {
    loop {
        let notified = flight.ready.notified();
        if let Some(result) = flight
            .result
            .lock()
            .expect("canonicalization flight result mutex poisoned")
            .clone()
        {
            return result;
        }
        notified.await;
    }
}

fn complete_canonicalization_flight(
    key: &DocumentCacheKey,
    flight: &Arc<CanonicalizationFlight>,
    result: Result<CanonicalizedNQuads, String>,
) {
    *flight
        .result
        .lock()
        .expect("canonicalization flight result mutex poisoned") = Some(result);
    let cache = canonicalization_cache();
    let mut in_flight = cache
        .in_flight
        .lock()
        .expect("canonicalization in-flight mutex poisoned");
    if in_flight
        .get(key)
        .is_some_and(|current| Arc::ptr_eq(current, flight))
    {
        in_flight.remove(key);
    }
    drop(in_flight);
    flight.ready.notify_waiters();
}

fn canonicalization_cache_get(key: &DocumentCacheKey) -> Option<CanonicalizedNQuads> {
    canonicalization_cache_get_arc(key).map(|cached| (*cached).clone())
}

fn canonicalization_cache_get_arc(key: &DocumentCacheKey) -> Option<Arc<CanonicalizedNQuads>> {
    let cache = canonicalization_cache();
    if cache.capacity == 0 {
        cache.misses.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let cached = canonicalization_cache_peek(key);
    if let Some(cached) = cached {
        cache.hits.fetch_add(1, Ordering::Relaxed);
        Some(cached)
    } else {
        cache.misses.fetch_add(1, Ordering::Relaxed);
        None
    }
}

fn canonicalization_cache_peek(key: &DocumentCacheKey) -> Option<Arc<CanonicalizedNQuads>> {
    let cache = canonicalization_cache();
    (cache.capacity != 0)
        .then(|| {
            cache
                .state
                .lock()
                .expect("canonicalization cache mutex poisoned")
                .entries
                .get(key)
                .cloned()
        })
        .flatten()
}

fn canonicalization_cache_insert(key: DocumentCacheKey, value: CanonicalizedNQuads) {
    let cache = canonicalization_cache();
    if cache.capacity == 0 {
        return;
    }
    let mut state = cache
        .state
        .lock()
        .expect("canonicalization cache mutex poisoned");
    if state.entries.contains_key(&key) {
        return;
    }
    while state.entries.len() >= cache.capacity {
        let Some(oldest) = state.insertion_order.pop_front() else {
            break;
        };
        state.entries.remove(&oldest);
    }
    state.insertion_order.push_back(key);
    state.entries.insert(key, Arc::new(value));
}

fn label_replacement_cache_get(key: &LabelCacheKey) -> Option<LabelReplacementCanonicalizedJsonLd> {
    let cache = canonicalization_cache();
    if cache.label_capacity == 0 {
        cache.label_misses.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let cached = cache
        .label_state
        .lock()
        .expect("label replacement cache mutex poisoned")
        .entries
        .get(key)
        .cloned();
    if let Some(cached) = cached {
        cache.label_hits.fetch_add(1, Ordering::Relaxed);
        Some((*cached).clone())
    } else {
        cache.label_misses.fetch_add(1, Ordering::Relaxed);
        None
    }
}

fn label_replacement_cache_insert(key: LabelCacheKey, value: LabelReplacementCanonicalizedJsonLd) {
    let cache = canonicalization_cache();
    if cache.label_capacity == 0 {
        return;
    }
    let mut state = cache
        .label_state
        .lock()
        .expect("label replacement cache mutex poisoned");
    if state.entries.contains_key(&key) {
        return;
    }
    while state.entries.len() >= cache.label_capacity {
        let Some(oldest) = state.insertion_order.pop_front() else {
            break;
        };
        state.entries.remove(&oldest);
    }
    state.insertion_order.push_back(key);
    state.entries.insert(key, Arc::new(value));
}

pub fn canonicalization_cache_snapshot() -> CanonicalizationCacheSnapshot {
    let cache = canonicalization_cache();
    CanonicalizationCacheSnapshot {
        capacity: cache.capacity,
        entries: cache
            .state
            .lock()
            .expect("canonicalization cache mutex poisoned")
            .entries
            .len(),
        label_capacity: cache.label_capacity,
        label_entries: cache
            .label_state
            .lock()
            .expect("label replacement cache mutex poisoned")
            .entries
            .len(),
        hits: cache.hits.load(Ordering::Relaxed),
        misses: cache.misses.load(Ordering::Relaxed),
        label_hits: cache.label_hits.load(Ordering::Relaxed),
        label_misses: cache.label_misses.load(Ordering::Relaxed),
        in_flight: cache
            .in_flight
            .lock()
            .expect("canonicalization in-flight mutex poisoned")
            .len(),
        coalesced_waiters: cache.coalesced_waiters.load(Ordering::Relaxed),
        canonicalization_leaders: cache.canonicalization_leaders.load(Ordering::Relaxed),
        canonicalize_total_ms: millis(&cache.canonicalize_total_micros),
        cache_key_ms: millis(&cache.cache_key_micros),
        cache_lookup_ms: millis(&cache.cache_lookup_micros),
        jsonld_to_nquads_total_ms: millis(&cache.jsonld_to_nquads_total_micros),
        jsonld_pre_to_rdf_ms: millis(&cache.jsonld_pre_to_rdf_micros),
        jsonld_to_rdf_ms: cache.jsonld_to_rdf_micros.load(Ordering::Relaxed) as f64 / 1000.0,
        nquads_serialize_ms: cache.nquads_serialize_micros.load(Ordering::Relaxed) as f64 / 1000.0,
        rdf_canonicalize_ms: cache.rdf_canonicalize_micros.load(Ordering::Relaxed) as f64 / 1000.0,
        label_replacement_ms: millis(&cache.label_replacement_micros),
        select_ms: millis(&cache.select_micros),
        group_ms: millis(&cache.group_micros),
    }
}

fn elapsed_micros(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u64::MAX as u128) as u64
}

fn millis(value: &AtomicU64) -> f64 {
    value.load(Ordering::Relaxed) as f64 / 1000.0
}

fn build_w3c_context_cache(
) -> Result<BTreeMap<iref::IriBuf, RemoteDocument>, JsonLdCanonicalizationError> {
    let mut contexts = BTreeMap::new();
    insert_cached_context(
        &mut contexts,
        "https://www.w3.org/2018/credentials/v1",
        include_str!("contexts/credentials-v1.jsonld"),
    )?;
    insert_cached_context(
        &mut contexts,
        "https://www.w3.org/ns/credentials/v2",
        include_str!("contexts/credentials-v2.jsonld"),
    )?;
    insert_cached_context(
        &mut contexts,
        "https://w3id.org/security/data-integrity/v2",
        include_str!("contexts/data-integrity-v2.jsonld"),
    )?;
    insert_cached_context(
        &mut contexts,
        "https://w3id.org/vdl/v1",
        include_str!("contexts/vdl-v1.jsonld"),
    )?;
    Ok(contexts)
}

fn insert_cached_context(
    contexts: &mut BTreeMap<iref::IriBuf, RemoteDocument>,
    url: &str,
    json: &str,
) -> Result<(), JsonLdCanonicalizationError> {
    let iri = url
        .parse::<iref::IriBuf>()
        .map_err(|e| JsonLdCanonicalizationError::ContextCache(format!("{url}: {e:?}")))?;
    let parsed = JsonLdValue::parse_str(json)
        .map_err(|e| JsonLdCanonicalizationError::ContextCache(format!("{url}: {e:?}")))?;
    contexts.insert(iri.clone(), RemoteDocument::new(Some(iri), None, parsed.0));
    Ok(())
}

fn ensure_type_values_were_not_dropped(
    document: &Value,
    nquads: &str,
) -> Result<(), JsonLdCanonicalizationError> {
    let expected_type_values = count_json_type_values(document);
    if expected_type_values == 0 {
        return Ok(());
    }

    let rdf_type_quads = nquads
        .lines()
        .filter(|line| line.contains("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"))
        .count();
    if rdf_type_quads < expected_type_values {
        return Err(JsonLdCanonicalizationError::DataLoss(format!(
            "JSON-LD processor dropped one or more type values: input had {expected_type_values}, RDF output had {rdf_type_quads}"
        )));
    }
    Ok(())
}

fn count_json_type_values(value: &Value) -> usize {
    match value {
        Value::Object(map) => map
            .iter()
            .filter(|(key, _)| key.as_str() != "@context")
            .map(|(key, value)| {
                if key == "type" || key == "@type" {
                    count_type_value_strings(value)
                } else {
                    count_json_type_values(value)
                }
            })
            .sum(),
        Value::Array(values) => values.iter().map(count_json_type_values).sum(),
        _ => 0,
    }
}

fn count_type_value_strings(value: &Value) -> usize {
    match value {
        Value::String(_) => 1,
        Value::Array(values) => values.iter().filter(|value| value.is_string()).count(),
        _ => 0,
    }
}

fn serialize_id<I, B>(id: &Id<I, B>) -> String
where
    I: RdfDisplay,
    B: core::fmt::Display,
{
    match id {
        Id::Iri(iri) => format!("{}", iri.rdf_display()),
        Id::Blank(blank) => {
            let blank = blank.to_string();
            if blank.starts_with("_:") {
                blank
            } else {
                format!("_:{blank}")
            }
        }
    }
}

fn serialize_term(term: &Term) -> String {
    match term {
        Term::Id(id) => serialize_id(id),
        Term::Literal(literal) => format!("{}", literal.rdf_display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[async_std::test]
    async fn cancelled_leader_releases_waiters_and_allows_retry() {
        let key = document_cache_key(&json!({"test": "cancelled-flight"})).unwrap();
        let (flight, leader) = canonicalization_flight(&key);
        assert!(leader);
        let guard = CanonicalizationFlightGuard {
            key,
            flight: Arc::clone(&flight),
            completed: false,
        };
        let (waiter, leader) = canonicalization_flight(&key);
        assert!(!leader);
        drop(guard); // Dropping a cancelled async future drops its owned guard.
        let result = async_std::future::timeout(
            std::time::Duration::from_secs(1),
            wait_for_canonicalization(&waiter),
        )
        .await
        .expect("cancelled leader must not leave a waiter blocked");
        assert!(result.unwrap_err().contains("cancelled"));
        let (retry, leader) = canonicalization_flight(&key);
        assert!(leader);
        assert!(!Arc::ptr_eq(&retry, &flight));
        CanonicalizationFlightGuard {
            key,
            flight: retry,
            completed: false,
        }
        .complete(Err("test cleanup".to_owned()));
    }

    #[async_std::test]
    async fn converts_inline_jsonld_to_nquads() {
        let document = json!({
            "@context": {
                "name": "http://xmlns.com/foaf/0.1/name"
            },
            "@id": "did:example:alice",
            "name": "Alice"
        });
        let nquads = jsonld_to_nquads(&document).await.unwrap();
        assert!(nquads.contains("<did:example:alice>"));
        assert!(nquads.contains("<http://xmlns.com/foaf/0.1/name>"));
        assert!(nquads.contains("\"Alice\""));
    }

    #[async_std::test]
    async fn canonicalizes_inline_jsonld() {
        let document = json!({
            "@context": {
                "name": "http://xmlns.com/foaf/0.1/name"
            },
            "@id": "_:alice",
            "name": "Alice"
        });
        let canonical = canonicalize_jsonld(&document).await.unwrap();
        assert!(canonical.serialized.contains("_:c14n"));
        assert_eq!(canonical.lines.len(), 1);
    }

    #[async_std::test]
    async fn canonicalizes_and_groups_inline_jsonld() {
        let document = json!({
            "@context": {
                "name": "http://xmlns.com/foaf/0.1/name",
                "age": "http://xmlns.com/foaf/0.1/age"
            },
            "@id": "did:example:alice",
            "name": "Alice",
            "age": 42
        });
        let groups = BTreeMap::from([("mandatory".to_owned(), vec!["/name".to_owned()])]);
        let grouped = canonicalize_and_group_jsonld(&document, &groups, &[1; HMAC_KEY_LEN])
            .await
            .unwrap();
        let mandatory = &grouped.groups["mandatory"].group;
        assert_eq!(mandatory.matching.len(), 1);
        assert_eq!(mandatory.non_matching.len(), 1);
        assert!(mandatory
            .matching
            .values()
            .next()
            .unwrap()
            .contains("foaf/0.1/name"));
    }

    #[async_std::test]
    async fn hmac_does_not_change_documents_without_blank_nodes() {
        let document = json!({
            "@context": {"name": "http://xmlns.com/foaf/0.1/name"},
            "@id": "did:example:no-blank-node",
            "name": "Alice"
        });
        let one = label_replacement_canonicalize_jsonld(&document, &[1; HMAC_KEY_LEN])
            .await
            .unwrap();
        let two = label_replacement_canonicalize_jsonld(&document, &[2; HMAC_KEY_LEN])
            .await
            .unwrap();
        assert_eq!(one, two);
        assert!(one.canonical_id_map.is_empty());
        assert!(one.label_map.is_empty());
    }

    #[test]
    fn detects_dropped_type_values() {
        let document = json!({
            "@context": {
                "type": "@type",
                "Known": "https://example.test/Known"
            },
            "type": ["Known", "UndefinedType"]
        });
        let nquads =
            "<urn:test> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://example.test/Known> .\n";
        assert!(matches!(
            ensure_type_values_were_not_dropped(&document, nquads),
            Err(JsonLdCanonicalizationError::DataLoss(_))
        ));
    }

    #[async_std::test]
    async fn fast_w3c_v2_matches_generic_jsonld_bit_for_bit() {
        let loader = ChainLoader::new(w3c_context_cache().unwrap(), reqwest_loader());
        let docs = [
            json!({
                "@context": [
                    "https://www.w3.org/ns/credentials/v2",
                    {
                        "ExampleCredential": "https://example.com/ExampleCredential",
                        "claim0": "https://example.com/claim/0",
                        "claim1": "https://example.com/claim/1",
                        "age": "https://example.com/age",
                        "active": "https://example.com/active"
                    }
                ],
                "id": "urn:uuid:omni-w3c-64",
                "type": ["VerifiableCredential", "ExampleCredential"],
                "issuer": "",
                "credentialSubject": {
                    "id": "did:example:subject-64",
                    "claim0": "value-0",
                    "claim1": "value-\"1\"\nline2",
                    "age": 42,
                    "active": true
                }
            }),
            json!({
                "@context": [
                    "https://www.w3.org/ns/credentials/v2",
                    {
                        "ExampleCredential": "https://example.com/ExampleCredential"
                    }
                ],
                "id": "urn:uuid:omni-w3c-2",
                "type": ["VerifiableCredential", "ExampleCredential"],
                "issuer": "did:key:zUC79tMoWT1WjLn6RnjkfKSGcFtFhKcnDqji5mXX82GFMQLjFg8qnz8aSGtqnJc7paBmE3zpKSy3zzUQCAwM9DBJxDZohsaUqmdmTq4sCRWF4nioJ2YqKAmNQM296RNsJ2YKdZG",
                "validFrom": "2026-01-01T00:00:00Z",
                "credentialSubject": {
                    "id": "did:example:alice",
                    "name": "Alice"
                }
            }),
            json!({
                "@context": [
                    "https://www.w3.org/ns/credentials/v2",
                    { "ExampleCredential": "https://example.com/ExampleCredential" }
                ],
                "type": "DataIntegrityProof",
                "cryptosuite": "bbs-2023",
                "verificationMethod": "did:key:zUC79#zUC79",
                "proofPurpose": "assertionMethod",
                "created": "2026-01-01T00:00:00Z"
            }),
        ];
        for doc in &docs {
            let fast_raw = try_fast_w3c_v2_jsonld_to_nquads(doc)
                .expect("fast path should handle standard W3C VC v2 / DI v2 doc");
            let slow_raw = jsonld_to_nquads_using(doc, &loader).await.unwrap();
            let fast_canon = canonicalize_nquads(&fast_raw).unwrap();
            let slow_canon = canonicalize_nquads(&slow_raw).unwrap();
            assert_eq!(fast_canon, slow_canon, "mismatch on doc {doc}");
        }
    }
}
