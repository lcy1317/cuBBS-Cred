use std::collections::BTreeMap;

use cubbs_bbs_proofvalue::{parse_base, parse_derived};
use serde_json::Value;

use crate::{
    build_proof_configuration, canonicalize_and_group_jsonld, canonicalize_jsonld,
    derive_disclosure_proof, hash_joined_strings, issue_base_proof,
    public_key_from_verification_method, relabel_and_sort_canonical_nquads, verify_base_proof,
    verify_disclosure_proof, DisclosureGroups, IssueBaseProofRequest, JsonLdCanonicalizationError,
    PreparedBaseVerification, PreparedDisclosureRequest, PreparedDisclosureVerification,
    ProofConfigError,
};

pub const DATA_INTEGRITY_CONTEXT_V2: &str = "https://w3id.org/security/data-integrity/v2";
pub const VC_CONTEXT_V1: &str = "https://www.w3.org/2018/credentials/v1";
pub const VC_CONTEXT_V2: &str = "https://www.w3.org/ns/credentials/v2";

#[derive(Debug, thiserror::Error)]
pub enum SuiteJsonLdError {
    #[error("document must be a JSON object")]
    DocumentMustBeObject,

    #[error("document does not contain a proof")]
    MissingProof,

    #[error("proof must contain verificationMethod")]
    MissingVerificationMethod,

    #[error("credential must contain credentialSubject")]
    MissingCredentialSubject,

    #[error("verificationMethod public key does not match proofValue public key")]
    VerificationMethodMismatch,

    #[error("W3C preparation task failed: {0}")]
    TaskJoin(String),

    #[error(transparent)]
    JsonLd(#[from] JsonLdCanonicalizationError),

    #[error(transparent)]
    ProofConfig(#[from] ProofConfigError),

    #[error(transparent)]
    Core(#[from] crate::Error),

    #[error(transparent)]
    ProofValue(#[from] cubbs_bbs_proofvalue::Error),

    #[error(transparent)]
    Key(#[from] crate::KeyEncodingError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IssueJsonLdBaseProofRequest {
    pub document: Value,
    pub proof_options: Value,
    pub secret_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub mandatory_pointers: Vec<String>,
    pub hmac_key: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeriveJsonLdDisclosureProofRequest {
    pub signed_document: Value,
    pub selective_pointers: Vec<String>,
    pub presentation_header: Vec<u8>,
}

pub async fn issue_jsonld_base_proof(
    req: IssueJsonLdBaseProofRequest,
) -> Result<Value, SuiteJsonLdError> {
    let document = ensure_data_integrity_context(req.document)?;
    require_credential_subject(&document)?;
    let proof_config = build_proof_configuration(&document, &req.proof_options)?;
    let proof_hash = hash_joined_strings(&canonicalize_jsonld(&proof_config).await?.lines);
    let groups = BTreeMap::from([("mandatory".to_owned(), req.mandatory_pointers.clone())]);
    let grouped = canonicalize_and_group_jsonld(&document, &groups, &req.hmac_key).await?;
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
    let proof_value = issue_base_proof(IssueBaseProofRequest {
        secret_key: req.secret_key,
        public_key: req.public_key,
        proof_hash,
        mandatory_pointers: req.mandatory_pointers,
        mandatory,
        non_mandatory,
        hmac_key: req.hmac_key,
    })?;

    let mut proof = req.proof_options;
    let proof_object = proof
        .as_object_mut()
        .ok_or(SuiteJsonLdError::DocumentMustBeObject)?;
    proof_object.insert("proofValue".to_owned(), Value::String(proof_value));

    attach_proof(document, proof)
}

pub async fn verify_jsonld_base_proof(document: &Value) -> Result<bool, SuiteJsonLdError> {
    let (unsigned_document, proof) = detach_single_proof(document)?;
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
    if public_key != parsed.public_key {
        return Ok(false);
    }

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

    Ok(verify_base_proof(PreparedBaseVerification {
        public_key,
        proof_hash,
        mandatory,
        non_mandatory,
        proof_value: proof_value.to_owned(),
    })?)
}

pub async fn derive_jsonld_disclosure_proof(
    req: DeriveJsonLdDisclosureProofRequest,
) -> Result<Value, SuiteJsonLdError> {
    let (unsigned_document, base_proof) = detach_single_proof(&req.signed_document)?;
    let base_proof_value = base_proof
        .get("proofValue")
        .and_then(Value::as_str)
        .ok_or(SuiteJsonLdError::MissingProof)?;
    let parsed_base = parse_base(base_proof_value)?;

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
    let grouped =
        canonicalize_and_group_jsonld(&unsigned_document, &groups, &parsed_base.hmac_key).await?;

    let derived = derive_disclosure_proof(PreparedDisclosureRequest {
        base_proof_value: base_proof_value.to_owned(),
        groups: DisclosureGroups {
            mandatory: grouped.groups["mandatory"].group.clone(),
            selective: grouped.groups["selective"].group.clone(),
            combined: grouped.groups["combined"].group.clone(),
            verifier_label_map: grouped.label_map,
        },
        presentation_header: req.presentation_header,
    })?;

    let mut derived_proof = base_proof;
    derived_proof
        .as_object_mut()
        .ok_or(SuiteJsonLdError::DocumentMustBeObject)?
        .insert("proofValue".to_owned(), Value::String(derived.proof_value));
    attach_proof(
        grouped.groups["combined"].selected_document.clone(),
        derived_proof,
    )
}

pub async fn verify_jsonld_disclosure_proof(
    document: &Value,
    expected_presentation_header: Option<Vec<u8>>,
) -> Result<bool, SuiteJsonLdError> {
    let (unsigned_document, proof) = detach_single_proof(document)?;
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

    Ok(verify_disclosure_proof(PreparedDisclosureVerification {
        public_key,
        proof_hash,
        canonical_nquads,
        proof_value: proof_value.to_owned(),
        expected_presentation_header,
    })?)
}

pub(crate) fn attach_proof(mut document: Value, proof: Value) -> Result<Value, SuiteJsonLdError> {
    let document = document
        .as_object_mut()
        .ok_or(SuiteJsonLdError::DocumentMustBeObject)?;
    document.insert("proof".to_owned(), proof);
    Ok(Value::Object(document.clone()))
}

pub(crate) fn ensure_data_integrity_context(
    mut document: Value,
) -> Result<Value, SuiteJsonLdError> {
    let object = document
        .as_object_mut()
        .ok_or(SuiteJsonLdError::DocumentMustBeObject)?;
    if object
        .get("@context")
        .is_some_and(context_defines_data_integrity_terms)
    {
        return Ok(document);
    }
    match object.get_mut("@context") {
        Some(Value::Array(contexts)) => {
            let has_context = contexts
                .iter()
                .any(|context| context.as_str() == Some(DATA_INTEGRITY_CONTEXT_V2));
            if !has_context {
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
                Value::Array(vec![
                    Value::String(VC_CONTEXT_V1.to_owned()),
                    Value::String(DATA_INTEGRITY_CONTEXT_V2.to_owned()),
                ]),
            );
        }
    }
    Ok(document)
}

pub(crate) fn require_credential_subject(document: &Value) -> Result<(), SuiteJsonLdError> {
    if document.get("credentialSubject").is_none() {
        return Err(SuiteJsonLdError::MissingCredentialSubject);
    }
    Ok(())
}

fn context_defines_data_integrity_terms(context: &Value) -> bool {
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

pub(crate) fn detach_single_proof(document: &Value) -> Result<(Value, Value), SuiteJsonLdError> {
    let mut unsigned = document
        .as_object()
        .ok_or(SuiteJsonLdError::DocumentMustBeObject)?
        .clone();
    let proof = unsigned
        .remove("proof")
        .ok_or(SuiteJsonLdError::MissingProof)?;
    Ok((Value::Object(unsigned), proof))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        default_bbs_2023_proof_options, verification_method_from_public_key, BBS_2023_CRYPTOSUITE,
        DATA_INTEGRITY_PROOF_TYPE, HMAC_KEY_LEN,
    };
    use cubbs_bbs::ZkryptiumBbsEngine;
    use serde_json::json;

    fn context() -> Value {
        json!({
            "id": "@id",
            "type": "@type",
            "VerifiableCredential": "https://www.w3.org/2018/credentials#VerifiableCredential",
            "credentialSubject": "https://www.w3.org/2018/credentials#credentialSubject",
            "name": "http://xmlns.com/foaf/0.1/name",
            "age": "http://xmlns.com/foaf/0.1/age",
            "DataIntegrityProof": "https://w3id.org/security#DataIntegrityProof",
            "cryptosuite": "https://w3id.org/security#cryptosuite",
            "verificationMethod": {
                "@id": "https://w3id.org/security#verificationMethod",
                "@type": "@id"
            },
            "proofPurpose": {
                "@id": "https://w3id.org/security#proofPurpose",
                "@type": "@vocab"
            },
            "assertionMethod": "https://w3id.org/security#assertionMethod"
        })
    }

    #[async_std::test]
    async fn issues_and_verifies_jsonld_base_proof() {
        let engine = ZkryptiumBbsEngine::new();
        let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
        let verification_method = verification_method_from_public_key(&keys.public_key).unwrap();
        let document = json!({
            "@context": context(),
            "id": "urn:cred:1",
            "type": "VerifiableCredential",
            "credentialSubject": {"id": "did:example:alice"},
            "name": "Alice",
            "age": 42
        });
        let proof_options = default_bbs_2023_proof_options(verification_method, "assertionMethod");

        let signed = issue_jsonld_base_proof(IssueJsonLdBaseProofRequest {
            document,
            proof_options,
            secret_key: keys.secret_key,
            public_key: keys.public_key,
            mandatory_pointers: vec!["/name".to_owned()],
            hmac_key: vec![3; HMAC_KEY_LEN],
        })
        .await
        .unwrap();

        assert_eq!(
            signed["proof"]["type"].as_str().unwrap(),
            DATA_INTEGRITY_PROOF_TYPE
        );
        assert_eq!(
            signed["proof"]["cryptosuite"].as_str().unwrap(),
            BBS_2023_CRYPTOSUITE
        );
        assert!(signed["proof"]["proofValue"]
            .as_str()
            .unwrap()
            .starts_with("u2V0C"));
        assert!(verify_jsonld_base_proof(&signed).await.unwrap());
    }

    #[async_std::test]
    async fn derives_and_verifies_jsonld_disclosure_proof() {
        let engine = ZkryptiumBbsEngine::new();
        let keys = engine.generate_keypair(&mut rand::thread_rng()).unwrap();
        let verification_method = verification_method_from_public_key(&keys.public_key).unwrap();
        let document = json!({
            "@context": context(),
            "id": "urn:cred:1",
            "type": "VerifiableCredential",
            "credentialSubject": {"id": "did:example:alice"},
            "name": "Alice",
            "age": 42
        });
        let proof_options = default_bbs_2023_proof_options(verification_method, "assertionMethod");
        let signed = issue_jsonld_base_proof(IssueJsonLdBaseProofRequest {
            document,
            proof_options,
            secret_key: keys.secret_key,
            public_key: keys.public_key,
            mandatory_pointers: vec!["/name".to_owned()],
            hmac_key: vec![4; HMAC_KEY_LEN],
        })
        .await
        .unwrap();

        let derived = derive_jsonld_disclosure_proof(DeriveJsonLdDisclosureProofRequest {
            signed_document: signed,
            selective_pointers: vec!["/age".to_owned()],
            presentation_header: b"example-domain".to_vec(),
        })
        .await
        .unwrap();

        assert_eq!(derived["name"].as_str().unwrap(), "Alice");
        assert_eq!(derived["age"].as_i64().unwrap(), 42);
        assert!(derived["proof"]["proofValue"]
            .as_str()
            .unwrap()
            .starts_with("u2V0D"));
        assert!(
            verify_jsonld_disclosure_proof(&derived, Some(b"example-domain".to_vec()))
                .await
                .unwrap()
        );
    }
}
