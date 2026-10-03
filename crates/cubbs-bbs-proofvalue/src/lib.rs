//! W3C `bbs-2023` proofValue encoding.
//!
//! This crate implements only the proofValue wire format. It does not depend
//! on SpruceID's Data Integrity layer and does not perform JSON-LD processing.

use std::collections::BTreeMap;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};

const CBOR_PREFIX_BASE: [u8; 3] = [0xd9, 0x5d, 0x02];
const CBOR_PREFIX_DERIVED: [u8; 3] = [0xd9, 0x5d, 0x03];

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("proofValue must use multibase base64url-no-pad prefix 'u'")]
    InvalidMultibasePrefix,

    #[error("proofValue does not have the expected bbs-2023 CBOR prefix")]
    InvalidCborPrefix,

    #[error("invalid base proof value")]
    InvalidBaseProof,

    #[error("invalid derived proof value")]
    InvalidDerivedProof,

    #[error("base64url decode failed: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("CBOR error: {0}")]
    Cbor(#[from] serde_cbor::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseProofValue {
    pub bbs_signature: Vec<u8>,
    pub bbs_header: Vec<u8>,
    pub public_key: Vec<u8>,
    pub hmac_key: Vec<u8>,
    pub mandatory_pointers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedProofValue {
    pub bbs_proof: Vec<u8>,
    /// Map from canonical labels such as `c14n0` to HMAC labels such as `b0`.
    pub label_map: BTreeMap<String, String>,
    pub mandatory_indexes: Vec<u64>,
    pub selective_indexes: Vec<u64>,
    pub presentation_header: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BasePayload(
    #[serde(with = "serde_bytes")] Vec<u8>,
    #[serde(with = "serde_bytes")] Vec<u8>,
    #[serde(with = "serde_bytes")] Vec<u8>,
    #[serde(with = "serde_bytes")] Vec<u8>,
    Vec<String>,
);

#[derive(Debug, Serialize, Deserialize)]
struct DerivedPayload(
    #[serde(with = "serde_bytes")] Vec<u8>,
    BTreeMap<u64, u64>,
    Vec<u64>,
    Vec<u64>,
    #[serde(with = "serde_bytes")] Vec<u8>,
);

pub fn serialize_base(value: &BaseProofValue) -> Result<String> {
    validate_base(value)?;
    let payload = BasePayload(
        value.bbs_signature.clone(),
        value.bbs_header.clone(),
        value.public_key.clone(),
        value.hmac_key.clone(),
        value.mandatory_pointers.clone(),
    );
    encode_with_prefix(CBOR_PREFIX_BASE, &payload)
}

pub fn parse_base(proof_value: &str) -> Result<BaseProofValue> {
    let payload = decode_with_prefix::<BasePayload>(proof_value, CBOR_PREFIX_BASE)?;
    let value = BaseProofValue {
        bbs_signature: payload.0,
        bbs_header: payload.1,
        public_key: payload.2,
        hmac_key: payload.3,
        mandatory_pointers: payload.4,
    };
    validate_base(&value)?;
    Ok(value)
}

pub fn serialize_derived(value: &DerivedProofValue) -> Result<String> {
    validate_derived(value)?;
    let payload = DerivedPayload(
        value.bbs_proof.clone(),
        compress_label_map(&value.label_map)?,
        value.mandatory_indexes.clone(),
        value.selective_indexes.clone(),
        value.presentation_header.clone(),
    );
    encode_with_prefix(CBOR_PREFIX_DERIVED, &payload)
}

pub fn parse_derived(proof_value: &str) -> Result<DerivedProofValue> {
    let payload = decode_with_prefix::<DerivedPayload>(proof_value, CBOR_PREFIX_DERIVED)?;
    let value = DerivedProofValue {
        bbs_proof: payload.0,
        label_map: decompress_label_map(&payload.1),
        mandatory_indexes: payload.2,
        selective_indexes: payload.3,
        presentation_header: payload.4,
    };
    validate_derived(&value)?;
    Ok(value)
}

fn encode_with_prefix<T: Serialize>(prefix: [u8; 3], payload: &T) -> Result<String> {
    let mut bytes = prefix.to_vec();
    bytes.extend_from_slice(&serde_cbor::to_vec(payload)?);
    Ok(format!("u{}", URL_SAFE_NO_PAD.encode(bytes)))
}

fn decode_with_prefix<T>(proof_value: &str, prefix: [u8; 3]) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let encoded = proof_value
        .strip_prefix('u')
        .ok_or(Error::InvalidMultibasePrefix)?;
    let bytes = URL_SAFE_NO_PAD.decode(encoded)?;
    if !bytes.starts_with(&prefix) {
        return Err(Error::InvalidCborPrefix);
    }
    Ok(serde_cbor::from_slice(&bytes[prefix.len()..])?)
}

fn validate_base(value: &BaseProofValue) -> Result<()> {
    if value.bbs_signature.len() != 80
        || value.bbs_header.len() != 64
        || value.public_key.len() != 96
        || value.hmac_key.len() != 32
    {
        return Err(Error::InvalidBaseProof);
    }
    Ok(())
}

fn validate_derived(value: &DerivedProofValue) -> Result<()> {
    if value.bbs_proof.is_empty() {
        return Err(Error::InvalidDerivedProof);
    }
    Ok(())
}

fn compress_label_map(map: &BTreeMap<String, String>) -> Result<BTreeMap<u64, u64>> {
    let mut compressed = BTreeMap::new();
    for (k, v) in map {
        let k = k
            .strip_prefix("c14n")
            .ok_or(Error::InvalidDerivedProof)?
            .parse()
            .map_err(|_| Error::InvalidDerivedProof)?;
        let v = v
            .strip_prefix('b')
            .ok_or(Error::InvalidDerivedProof)?
            .parse()
            .map_err(|_| Error::InvalidDerivedProof)?;
        compressed.insert(k, v);
    }
    Ok(compressed)
}

fn decompress_label_map(map: &BTreeMap<u64, u64>) -> BTreeMap<String, String> {
    map.iter()
        .map(|(k, v)| (format!("c14n{k}"), format!("b{v}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_proof_value_round_trip() {
        let value = BaseProofValue {
            bbs_signature: vec![1; 80],
            bbs_header: vec![2; 64],
            public_key: vec![3; 96],
            hmac_key: vec![4; 32],
            mandatory_pointers: vec!["/issuer".to_owned()],
        };
        let proof_value = serialize_base(&value).unwrap();
        assert!(proof_value.starts_with("u2V0C"));
        assert_eq!(parse_base(&proof_value).unwrap(), value);
    }

    #[test]
    fn derived_proof_value_round_trip() {
        let value = DerivedProofValue {
            bbs_proof: vec![1; 272],
            label_map: BTreeMap::from([
                ("c14n0".to_owned(), "b2".to_owned()),
                ("c14n1".to_owned(), "b0".to_owned()),
            ]),
            mandatory_indexes: vec![0, 1, 4],
            selective_indexes: vec![2, 3],
            presentation_header: b"ph".to_vec(),
        };
        let proof_value = serialize_derived(&value).unwrap();
        assert!(proof_value.starts_with("u2V0D"));
        assert_eq!(parse_derived(&proof_value).unwrap(), value);
    }
}
