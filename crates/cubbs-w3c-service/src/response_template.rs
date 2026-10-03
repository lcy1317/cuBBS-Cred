//! Serialize immutable credential structure once; never search user bytes for a marker.
use axum::body::Bytes;
use serde_json::Value;

#[derive(Debug)]
pub(super) struct ProofResponseTemplate {
    prefix: Bytes,
}

impl ProofResponseTemplate {
    pub(super) fn new(mut document: Value) -> Result<Self, String> {
        let object = document
            .as_object_mut()
            .ok_or("credential must be an object")?;
        let mut proof = object
            .remove("proof")
            .ok_or("credential is missing proof")?;
        let proof_object = proof.as_object_mut().ok_or("proof must be an object")?;
        proof_object.remove("proofValue");
        let mut prefix = serde_json::to_vec(&document).map_err(|e| e.to_string())?;
        prefix.pop(); // Root object's closing brace.
        if prefix.len() > 1 {
            prefix.push(b',');
        }
        prefix.extend_from_slice(b"\"proof\":");
        let mut encoded_proof = serde_json::to_vec(&proof).map_err(|e| e.to_string())?;
        encoded_proof.pop();
        let has_proof_fields = encoded_proof.len() > 1;
        prefix.extend_from_slice(&encoded_proof);
        if has_proof_fields {
            prefix.push(b',');
        }
        prefix.extend_from_slice(b"\"proofValue\":\"");
        Ok(Self {
            prefix: Bytes::from(prefix),
        })
    }

    pub(super) fn finish(&self, proof_value: &str) -> Result<Bytes, String> {
        // The proof-value encoder emits multibase/base64url, never JSON syntax.
        if !proof_value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err("proofValue is not a multibase/base64url string".to_owned());
        }
        let mut response = Vec::with_capacity(self.prefix.len() + proof_value.len() + 3);
        response.extend_from_slice(&self.prefix);
        response.extend_from_slice(proof_value.as_bytes());
        response.extend_from_slice(b"\"}}");
        Ok(Bytes::from(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn user_strings_are_never_proof_slots() {
        let document = json!({"credentialSubject": {"name": "__CUBBS_CRED_FRESH_PROOF_VALUE_SLOT__"}, "proof": {"type": "DataIntegrityProof", "proofValue": "old"}});
        let template = ProofResponseTemplate::new(document.clone()).unwrap();
        let result: Value =
            serde_json::from_slice(&template.finish("uFreshProof-1").unwrap()).unwrap();
        assert_eq!(result["credentialSubject"], document["credentialSubject"]);
        assert_eq!(result["proof"]["proofValue"], "uFreshProof-1");
        assert_eq!(result["proof"]["type"], "DataIntegrityProof");
        assert!(template.finish("\"}").is_err());
    }

    #[test]
    fn empty_objects_serialize_correctly() {
        let template = ProofResponseTemplate::new(json!({"proof": {"proofValue": ""}})).unwrap();
        let result: Value = serde_json::from_slice(&template.finish("uAA").unwrap()).unwrap();
        assert_eq!(result, json!({"proof": {"proofValue": "uAA"}}));
    }
}
