use serde_json::{Map, Value};

pub const DATA_INTEGRITY_PROOF_TYPE: &str = "DataIntegrityProof";
pub const BBS_2023_CRYPTOSUITE: &str = "bbs-2023";

#[derive(Debug, thiserror::Error)]
pub enum ProofConfigError {
    #[error("proof must be a JSON object")]
    ProofMustBeObject,

    #[error("proof.type must be DataIntegrityProof")]
    InvalidProofType,

    #[error("proof.cryptosuite must be bbs-2023")]
    InvalidCryptosuite,

    #[error("proof.verificationMethod must be present")]
    MissingVerificationMethod,

    #[error("proof.proofPurpose must be present")]
    MissingProofPurpose,

    #[error("proof.created must be a valid XMLSchema dateTimeStamp")]
    InvalidCreated,

    #[error("proof.expires must be a valid XMLSchema dateTimeStamp")]
    InvalidExpires,
}

pub fn default_bbs_2023_proof_options(
    verification_method: impl Into<String>,
    proof_purpose: impl Into<String>,
) -> Value {
    let mut proof = Map::new();
    proof.insert(
        "type".to_owned(),
        Value::String(DATA_INTEGRITY_PROOF_TYPE.to_owned()),
    );
    proof.insert(
        "cryptosuite".to_owned(),
        Value::String(BBS_2023_CRYPTOSUITE.to_owned()),
    );
    proof.insert(
        "verificationMethod".to_owned(),
        Value::String(verification_method.into()),
    );
    proof.insert(
        "proofPurpose".to_owned(),
        Value::String(proof_purpose.into()),
    );
    Value::Object(proof)
}

pub fn normalize_bbs_2023_proof_options(proof: &Value) -> Result<Value, ProofConfigError> {
    validate_bbs_2023_proof(proof)?;
    let mut proof = proof
        .as_object()
        .ok_or(ProofConfigError::ProofMustBeObject)?
        .clone();
    proof.remove("proofValue");
    Ok(Value::Object(proof))
}

pub fn build_proof_configuration(
    document: &Value,
    proof: &Value,
) -> Result<Value, ProofConfigError> {
    let normalized = normalize_bbs_2023_proof_options(proof)?;
    let mut config = Map::new();
    if let Some(context) = document.get("@context") {
        config.insert("@context".to_owned(), context.clone());
    }
    let Value::Object(proof) = normalized else {
        unreachable!("normalize returns object")
    };
    for (key, value) in proof {
        config.insert(key, value);
    }
    Ok(Value::Object(config))
}

fn validate_bbs_2023_proof(proof: &Value) -> Result<(), ProofConfigError> {
    let proof = proof
        .as_object()
        .ok_or(ProofConfigError::ProofMustBeObject)?;
    match proof.get("type").and_then(Value::as_str) {
        Some(DATA_INTEGRITY_PROOF_TYPE) => {}
        _ => return Err(ProofConfigError::InvalidProofType),
    }
    match proof.get("cryptosuite").and_then(Value::as_str) {
        Some(BBS_2023_CRYPTOSUITE) => {}
        _ => return Err(ProofConfigError::InvalidCryptosuite),
    }
    if proof
        .get("verificationMethod")
        .and_then(Value::as_str)
        .is_none()
    {
        return Err(ProofConfigError::MissingVerificationMethod);
    }
    if proof.get("proofPurpose").and_then(Value::as_str).is_none() {
        return Err(ProofConfigError::MissingProofPurpose);
    }
    if let Some(created) = proof.get("created") {
        let Some(created) = created.as_str() else {
            return Err(ProofConfigError::InvalidCreated);
        };
        validate_datetime(created).map_err(|_| ProofConfigError::InvalidCreated)?;
    }
    if let Some(expires) = proof.get("expires") {
        let Some(expires) = expires.as_str() else {
            return Err(ProofConfigError::InvalidExpires);
        };
        validate_datetime(expires).map_err(|_| ProofConfigError::InvalidExpires)?;
    }
    Ok(())
}

fn validate_datetime(value: &str) -> Result<(), ()> {
    if chrono::DateTime::parse_from_rfc3339(value).is_ok() {
        return Ok(());
    }
    chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S")
        .map(|_| ())
        .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builds_proof_configuration_with_document_context_and_without_signature_fields() {
        let document = json!({
            "@context": ["https://www.w3.org/ns/credentials/v2"],
            "id": "urn:cred:1"
        });
        let proof = json!({
            "type": "DataIntegrityProof",
            "cryptosuite": "bbs-2023",
            "verificationMethod": "did:key:zIssuer#zKey",
            "proofPurpose": "assertionMethod",
            "created": "2026-01-01T00:00:00Z",
            "proofValue": "u2V0C..."
        });
        assert_eq!(
            build_proof_configuration(&document, &proof).unwrap(),
            json!({
                "@context": ["https://www.w3.org/ns/credentials/v2"],
                "type": "DataIntegrityProof",
                "cryptosuite": "bbs-2023",
                "verificationMethod": "did:key:zIssuer#zKey",
                "proofPurpose": "assertionMethod",
                "created": "2026-01-01T00:00:00Z"
            })
        );
    }

    #[test]
    fn rejects_wrong_cryptosuite() {
        let proof = json!({
            "type": "DataIntegrityProof",
            "cryptosuite": "bbs-2024"
        });
        assert!(matches!(
            normalize_bbs_2023_proof_options(&proof),
            Err(ProofConfigError::InvalidCryptosuite)
        ));
    }

    #[test]
    fn rejects_invalid_created() {
        let proof = json!({
            "type": "DataIntegrityProof",
            "cryptosuite": "bbs-2023",
            "verificationMethod": "did:key:zIssuer#zKey",
            "proofPurpose": "assertionMethod",
            "created": "2026-06-24 09:30:00"
        });
        assert!(matches!(
            normalize_bbs_2023_proof_options(&proof),
            Err(ProofConfigError::InvalidCreated)
        ));
    }

    #[test]
    fn rejects_invalid_expires() {
        let proof = json!({
            "type": "DataIntegrityProof",
            "cryptosuite": "bbs-2023",
            "verificationMethod": "did:key:zIssuer#zKey",
            "proofPurpose": "assertionMethod",
            "expires": "invalidDate"
        });
        assert!(matches!(
            normalize_bbs_2023_proof_options(&proof),
            Err(ProofConfigError::InvalidExpires)
        ));
    }

    #[test]
    fn accepts_no_offset_timestamps_for_compatibility() {
        let proof = json!({
            "type": "DataIntegrityProof",
            "cryptosuite": "bbs-2023",
            "verificationMethod": "did:key:zIssuer#zKey",
            "proofPurpose": "assertionMethod",
            "created": "2026-06-24T09:30:00",
            "expires": "2026-06-24T09:30:00"
        });
        assert!(normalize_bbs_2023_proof_options(&proof).is_ok());
    }
}
