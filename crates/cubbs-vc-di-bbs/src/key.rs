use cubbs_bbs::PUBLIC_KEY_LEN;
use serde_json::{json, Value};

pub const MULTIBASE_BASE58BTC_PREFIX: char = 'z';
pub const MULTICODEC_G2_PUBLIC_KEY_HEADER: [u8; 2] = [0xeb, 0x01];

#[derive(Debug, thiserror::Error)]
pub enum KeyEncodingError {
    #[error("public key must be 96 bytes, got {0}")]
    InvalidPublicKeyLength(usize),

    #[error("multibase value must start with base58btc prefix 'z'")]
    InvalidMultibasePrefix,

    #[error("multicodec header must be BLS12-381 G2 public key 0xeb01")]
    InvalidMulticodecHeader,

    #[error("base58btc decode failed: {0}")]
    Base58(#[from] bs58::decode::Error),

    #[error("verification method must be a did:key or contain a did:key fragment")]
    InvalidVerificationMethod,
}

pub fn public_key_to_multibase(public_key: &[u8]) -> Result<String, KeyEncodingError> {
    if public_key.len() != PUBLIC_KEY_LEN {
        return Err(KeyEncodingError::InvalidPublicKeyLength(public_key.len()));
    }
    let mut multicodec = Vec::with_capacity(MULTICODEC_G2_PUBLIC_KEY_HEADER.len() + PUBLIC_KEY_LEN);
    multicodec.extend_from_slice(&MULTICODEC_G2_PUBLIC_KEY_HEADER);
    multicodec.extend_from_slice(public_key);
    Ok(format!(
        "{MULTIBASE_BASE58BTC_PREFIX}{}",
        bs58::encode(multicodec).into_string()
    ))
}

pub fn public_key_from_multibase(multibase: &str) -> Result<Vec<u8>, KeyEncodingError> {
    let encoded = multibase
        .strip_prefix(MULTIBASE_BASE58BTC_PREFIX)
        .ok_or(KeyEncodingError::InvalidMultibasePrefix)?;
    let decoded = bs58::decode(encoded).into_vec()?;
    if !decoded.starts_with(&MULTICODEC_G2_PUBLIC_KEY_HEADER) {
        return Err(KeyEncodingError::InvalidMulticodecHeader);
    }
    let public_key = decoded[MULTICODEC_G2_PUBLIC_KEY_HEADER.len()..].to_vec();
    if public_key.len() != PUBLIC_KEY_LEN {
        return Err(KeyEncodingError::InvalidPublicKeyLength(public_key.len()));
    }
    Ok(public_key)
}

pub fn did_key_from_public_key(public_key: &[u8]) -> Result<String, KeyEncodingError> {
    Ok(format!("did:key:{}", public_key_to_multibase(public_key)?))
}

pub fn public_key_from_did_key(did_key: &str) -> Result<Vec<u8>, KeyEncodingError> {
    let multibase = did_key
        .strip_prefix("did:key:")
        .ok_or(KeyEncodingError::InvalidVerificationMethod)?;
    public_key_from_multibase(multibase)
}

pub fn verification_method_from_public_key(public_key: &[u8]) -> Result<String, KeyEncodingError> {
    let public_key_multibase = public_key_to_multibase(public_key)?;
    Ok(format!(
        "did:key:{public_key_multibase}#{public_key_multibase}"
    ))
}

pub fn public_key_from_verification_method(
    verification_method: &str,
) -> Result<Vec<u8>, KeyEncodingError> {
    if let Some(fragment) = verification_method
        .split_once('#')
        .map(|(_, fragment)| fragment)
    {
        return public_key_from_multibase(fragment);
    }
    public_key_from_did_key(verification_method)
}

pub fn verification_method_document(public_key: &[u8]) -> Result<Value, KeyEncodingError> {
    let public_key_multibase = public_key_to_multibase(public_key)?;
    let did = format!("did:key:{public_key_multibase}");
    let verification_method = format!("{did}#{public_key_multibase}");
    Ok(json!({
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/multikey/v1"
        ],
        "id": did,
        "assertionMethod": [verification_method],
        "verificationMethod": [{
            "id": verification_method,
            "type": "Multikey",
            "controller": did,
            "publicKeyMultibase": public_key_multibase
        }]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubbs_bbs::ZkryptiumBbsEngine;

    #[test]
    fn multibase_round_trip_for_generated_bbs_key() {
        let keys = ZkryptiumBbsEngine::new()
            .generate_keypair(&mut rand::thread_rng())
            .unwrap();
        let multibase = public_key_to_multibase(&keys.public_key).unwrap();
        assert!(multibase.starts_with('z'));
        assert_eq!(
            public_key_from_multibase(&multibase).unwrap(),
            keys.public_key
        );
    }

    #[test]
    fn parses_digital_bazaar_bbs_multikey_vector() {
        let multibase = "zUC7DerdEmfZ8f4pFajXgGwJoMkV1ofMTmEG5UoNvnWiPiLuGKNeqgRpLH2TV4Xe5mJ2cXV76gRN7LFQwapF1VFu6x2yrr5ci1mXqC1WNUrnHnLgvfZfMH7h6xP6qsf9EKRQrPQ";
        let public_key = public_key_from_multibase(multibase).unwrap();
        assert_eq!(public_key.len(), PUBLIC_KEY_LEN);
        assert_eq!(public_key_to_multibase(&public_key).unwrap(), multibase);
    }

    #[test]
    fn builds_did_key_verification_method() {
        let keys = ZkryptiumBbsEngine::new()
            .generate_keypair(&mut rand::thread_rng())
            .unwrap();
        let vm = verification_method_from_public_key(&keys.public_key).unwrap();
        assert!(vm.starts_with("did:key:z"));
        assert!(vm.contains("#z"));
        assert_eq!(
            public_key_from_verification_method(&vm).unwrap(),
            keys.public_key
        );
    }
}
