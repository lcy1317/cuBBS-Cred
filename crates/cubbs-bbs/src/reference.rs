//! Versioned CPU reference and conformance backends.
//!
//! Production GPU lowering must not depend on these types.  They are kept
//! behind this adapter for explicit CPU execution, conformance checks, and
//! recoverable GPU fallback.

use crate::{
    Error, KeyPair, ProofGenRequest, ProofVerifyRequest, Result, SignRequest, VerifyRequest,
    PUBLIC_KEY_LEN, SECRET_KEY_LEN, SIGNATURE_LEN,
};
use zkryptium::{
    bbsplus::{
        ciphersuites::Bls12381Sha256 as Bls12381Sha256_022,
        keys::{BBSplusPublicKey as BBSplusPublicKey022, BBSplusSecretKey as BBSplusSecretKey022},
    },
    keys::pair::KeyPair as ZkKeyPair022,
    schemes::{
        algorithms::BBSplus as BBSplus022,
        generics::{PoKSignature as PoKSignature022, Signature as Signature022},
    },
};
use zkryptium_061::{
    bbsplus::keys::{BBSplusPublicKey, BBSplusSecretKey},
    keys::pair::KeyPair as ZkKeyPair061,
    schemes::{
        algorithms::BbsBls12381Sha256,
        generics::{PoKSignature, Signature},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZkryptiumReferenceVersion {
    V0_2_2,
    V0_6_1,
}

impl ZkryptiumReferenceVersion {
    pub const fn name(self) -> &'static str {
        match self {
            Self::V0_2_2 => "zkryptium-0.2.2",
            Self::V0_6_1 => "zkryptium-0.6.1",
        }
    }
}

pub trait CpuReferenceBackend: Send + Sync {
    fn version(&self) -> ZkryptiumReferenceVersion;
    fn generate_keypair(&self) -> Result<KeyPair>;
    fn sign(&self, request: &SignRequest) -> Result<Vec<u8>>;
    fn verify(&self, request: &VerifyRequest) -> Result<bool>;
    fn proof_gen(&self, request: &ProofGenRequest) -> Result<Vec<u8>>;
    fn proof_verify(&self, request: &ProofVerifyRequest) -> Result<bool>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Zkryptium022Reference;

impl Zkryptium022Reference {
    pub const fn new() -> Self {
        Self
    }
}

impl CpuReferenceBackend for Zkryptium022Reference {
    fn version(&self) -> ZkryptiumReferenceVersion {
        ZkryptiumReferenceVersion::V0_2_2
    }

    fn generate_keypair(&self) -> Result<KeyPair> {
        let pair = ZkKeyPair022::<BBSplus022<Bls12381Sha256_022>>::random()
            .map_err(reference_error_022)?;
        let (secret_key, public_key) = pair.into_parts();
        Ok(KeyPair {
            secret_key: secret_key.to_bytes().to_vec(),
            public_key: public_key.to_bytes().to_vec(),
        })
    }

    fn sign(&self, request: &SignRequest) -> Result<Vec<u8>> {
        let secret_key = parse_secret_key_022(&request.secret_key)?;
        let public_key = parse_public_key_022(&request.public_key)?;
        Signature022::<BBSplus022<Bls12381Sha256_022>>::sign(
            Some(&request.messages),
            &secret_key,
            &public_key,
            Some(&request.header),
        )
        .map(|signature| signature.to_bytes().to_vec())
        .map_err(reference_error_022)
    }

    fn verify(&self, request: &VerifyRequest) -> Result<bool> {
        let public_key = parse_public_key_022(&request.public_key)?;
        let signature_bytes: &[u8; SIGNATURE_LEN] = request
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| Error::InvalidSignature)?;
        let signature = Signature022::<BBSplus022<Bls12381Sha256_022>>::from_bytes(signature_bytes)
            .map_err(|_| Error::InvalidSignature)?;
        Ok(signature
            .verify(&public_key, Some(&request.messages), Some(&request.header))
            .is_ok())
    }

    fn proof_gen(&self, request: &ProofGenRequest) -> Result<Vec<u8>> {
        let public_key = parse_public_key_022(&request.public_key)?;
        PoKSignature022::<BBSplus022<Bls12381Sha256_022>>::proof_gen(
            &public_key,
            &request.signature,
            Some(&request.header),
            Some(&request.presentation_header),
            Some(&request.messages),
            Some(&request.disclosed_indexes),
        )
        .map(|proof| proof.to_bytes())
        .map_err(reference_error_022)
    }

    fn proof_verify(&self, request: &ProofVerifyRequest) -> Result<bool> {
        validate_disclosed_indexes(request)?;
        let public_key = parse_public_key_022(&request.public_key)?;
        let proof = PoKSignature022::<BBSplus022<Bls12381Sha256_022>>::from_bytes(&request.proof)
            .map_err(|_| Error::InvalidProof)?;
        Ok(proof
            .proof_verify(
                &public_key,
                Some(&request.disclosed_messages),
                Some(&request.disclosed_indexes),
                Some(&request.header),
                Some(&request.presentation_header),
            )
            .is_ok())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Zkryptium061Reference;

impl Zkryptium061Reference {
    pub const fn new() -> Self {
        Self
    }
}

impl CpuReferenceBackend for Zkryptium061Reference {
    fn version(&self) -> ZkryptiumReferenceVersion {
        ZkryptiumReferenceVersion::V0_6_1
    }

    fn generate_keypair(&self) -> Result<KeyPair> {
        let pair = ZkKeyPair061::<BbsBls12381Sha256>::random().map_err(reference_error)?;
        let (secret_key, public_key) = pair.into_parts();
        Ok(KeyPair {
            secret_key: secret_key.to_bytes().to_vec(),
            public_key: public_key.to_bytes().to_vec(),
        })
    }

    fn sign(&self, request: &SignRequest) -> Result<Vec<u8>> {
        let secret_key = parse_secret_key(&request.secret_key)?;
        let public_key = parse_public_key(&request.public_key)?;
        Signature::<BbsBls12381Sha256>::sign(
            Some(&request.messages),
            &secret_key,
            &public_key,
            Some(&request.header),
        )
        .map(|signature| signature.to_bytes().to_vec())
        .map_err(reference_error)
    }

    fn verify(&self, request: &VerifyRequest) -> Result<bool> {
        let public_key = parse_public_key(&request.public_key)?;
        let signature_bytes: &[u8; SIGNATURE_LEN] = request
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| Error::InvalidSignature)?;
        let signature = Signature::<BbsBls12381Sha256>::from_bytes(signature_bytes)
            .map_err(|_| Error::InvalidSignature)?;
        Ok(signature
            .verify(&public_key, Some(&request.messages), Some(&request.header))
            .is_ok())
    }

    fn proof_gen(&self, request: &ProofGenRequest) -> Result<Vec<u8>> {
        let public_key = parse_public_key(&request.public_key)?;
        PoKSignature::<BbsBls12381Sha256>::proof_gen(
            &public_key,
            &request.signature,
            Some(&request.header),
            Some(&request.presentation_header),
            Some(&request.messages),
            Some(&request.disclosed_indexes),
        )
        .map(|proof| proof.to_bytes())
        .map_err(reference_error)
    }

    fn proof_verify(&self, request: &ProofVerifyRequest) -> Result<bool> {
        validate_disclosed_indexes(request)?;
        let public_key = parse_public_key(&request.public_key)?;
        let proof = PoKSignature::<BbsBls12381Sha256>::from_bytes(&request.proof)
            .map_err(|_| Error::InvalidProof)?;
        Ok(proof
            .proof_verify(
                &public_key,
                Some(&request.disclosed_messages),
                Some(&request.disclosed_indexes),
                Some(&request.header),
                Some(&request.presentation_header),
            )
            .is_ok())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ZkryptiumReference {
    V0_2_2(Zkryptium022Reference),
    V0_6_1(Zkryptium061Reference),
}

impl Default for ZkryptiumReference {
    fn default() -> Self {
        Self::V0_6_1(Zkryptium061Reference::new())
    }
}

impl ZkryptiumReference {
    pub fn from_env() -> Result<Self> {
        match std::env::var("CUBBS_BBS_ZKRYPTIUM_REFERENCE")
            .unwrap_or_else(|_| "0.6.1".to_owned())
            .as_str()
        {
            "0.2.2" | "022" | "legacy" => Ok(Self::V0_2_2(Zkryptium022Reference::new())),
            "0.6.1" | "061" | "current" => Ok(Self::V0_6_1(Zkryptium061Reference::new())),
            value => Err(Error::Zkryptium(format!(
                "unsupported reference version: {value}"
            ))),
        }
    }
}

impl CpuReferenceBackend for ZkryptiumReference {
    fn version(&self) -> ZkryptiumReferenceVersion {
        match self {
            Self::V0_2_2(reference) => reference.version(),
            Self::V0_6_1(reference) => reference.version(),
        }
    }

    fn generate_keypair(&self) -> Result<KeyPair> {
        match self {
            Self::V0_2_2(reference) => reference.generate_keypair(),
            Self::V0_6_1(reference) => reference.generate_keypair(),
        }
    }

    fn sign(&self, request: &SignRequest) -> Result<Vec<u8>> {
        match self {
            Self::V0_2_2(reference) => reference.sign(request),
            Self::V0_6_1(reference) => reference.sign(request),
        }
    }

    fn verify(&self, request: &VerifyRequest) -> Result<bool> {
        match self {
            Self::V0_2_2(reference) => reference.verify(request),
            Self::V0_6_1(reference) => reference.verify(request),
        }
    }

    fn proof_gen(&self, request: &ProofGenRequest) -> Result<Vec<u8>> {
        match self {
            Self::V0_2_2(reference) => reference.proof_gen(request),
            Self::V0_6_1(reference) => reference.proof_gen(request),
        }
    }

    fn proof_verify(&self, request: &ProofVerifyRequest) -> Result<bool> {
        match self {
            Self::V0_2_2(reference) => reference.proof_verify(request),
            Self::V0_6_1(reference) => reference.proof_verify(request),
        }
    }
}

fn parse_secret_key_022(bytes: &[u8]) -> Result<BBSplusSecretKey022> {
    if bytes.len() != SECRET_KEY_LEN {
        return Err(Error::InvalidSecretKey);
    }
    BBSplusSecretKey022::from_bytes(bytes).map_err(|_| Error::InvalidSecretKey)
}

fn parse_public_key_022(bytes: &[u8]) -> Result<BBSplusPublicKey022> {
    if bytes.len() != PUBLIC_KEY_LEN {
        return Err(Error::InvalidPublicKey);
    }
    BBSplusPublicKey022::from_bytes(bytes).map_err(|_| Error::InvalidPublicKey)
}

fn reference_error_022(error: zkryptium::errors::Error) -> Error {
    Error::Zkryptium(format!("zkryptium-0.2.2: {error}"))
}

fn parse_secret_key(bytes: &[u8]) -> Result<BBSplusSecretKey> {
    if bytes.len() != SECRET_KEY_LEN {
        return Err(Error::InvalidSecretKey);
    }
    BBSplusSecretKey::from_bytes(bytes).map_err(|_| Error::InvalidSecretKey)
}

fn parse_public_key(bytes: &[u8]) -> Result<BBSplusPublicKey> {
    if bytes.len() != PUBLIC_KEY_LEN {
        return Err(Error::InvalidPublicKey);
    }
    BBSplusPublicKey::from_bytes(bytes).map_err(|_| Error::InvalidPublicKey)
}

fn validate_disclosed_indexes(request: &ProofVerifyRequest) -> Result<()> {
    if request.disclosed_messages.len() != request.disclosed_indexes.len() {
        return Err(Error::InvalidDisclosedIndexes);
    }
    let mut previous = None;
    for &index in &request.disclosed_indexes {
        if index >= request.total_message_count || previous.is_some_and(|value| index <= value) {
            return Err(Error::InvalidDisclosedIndexes);
        }
        previous = Some(index);
    }
    Ok(())
}

fn reference_error(error: zkryptium_061::errors::Error) -> Error {
    Error::Zkryptium(format!("zkryptium-0.6.1: {error}"))
}
#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn default_reference_is_zkryptium_0_6_1() {
        assert_eq!(
            ZkryptiumReference::default().version(),
            ZkryptiumReferenceVersion::V0_6_1
        );
    }

    fn fixture() -> (Zkryptium022Reference, SignRequest) {
        let keygen = crate::ZkryptiumBbsEngine::new();
        let keypair = keygen.generate_keypair(&mut thread_rng()).unwrap();
        let legacy = Zkryptium022Reference::new();
        (
            legacy,
            SignRequest {
                secret_key: keypair.secret_key,
                public_key: keypair.public_key,
                header: vec![0x5a; 64],
                messages: (0..8)
                    .map(|index| format!("message-{index}").into_bytes())
                    .collect(),
            },
        )
    }

    #[test]
    fn versions_accept_each_others_valid_signatures() {
        let (legacy, request) = fixture();
        let current = Zkryptium061Reference::new();

        let legacy_signature = legacy.sign(&request).unwrap();
        let current_signature = current.sign(&request).unwrap();

        for signature in [legacy_signature, current_signature] {
            let verify_request = VerifyRequest {
                public_key: request.public_key.clone(),
                signature,
                header: request.header.clone(),
                messages: request.messages.clone(),
            };
            assert!(legacy.verify(&verify_request).unwrap());
            assert!(current.verify(&verify_request).unwrap());
        }
    }

    #[test]
    fn proof_transcripts_require_matching_version_adapter() {
        let (legacy, request) = fixture();
        let current = Zkryptium061Reference::new();
        let disclosed_indexes = vec![0, 3, 7];
        let presentation_header = b"omni-reference-cross-check".to_vec();

        let signatures = [
            legacy.sign(&request).unwrap(),
            current.sign(&request).unwrap(),
        ];
        for signature in signatures {
            let proof_request = ProofGenRequest {
                public_key: request.public_key.clone(),
                signature,
                header: request.header.clone(),
                presentation_header: presentation_header.clone(),
                messages: request.messages.clone(),
                disclosed_indexes: disclosed_indexes.clone(),
            };
            let proofs = [
                legacy.proof_gen(&proof_request).unwrap(),
                current.proof_gen(&proof_request).unwrap(),
            ];
            for (proof_version, proof) in proofs.into_iter().enumerate() {
                let verify_request = ProofVerifyRequest {
                    public_key: request.public_key.clone(),
                    proof,
                    header: request.header.clone(),
                    presentation_header: presentation_header.clone(),
                    disclosed_messages: disclosed_indexes
                        .iter()
                        .map(|&index| request.messages[index].clone())
                        .collect(),
                    disclosed_indexes: disclosed_indexes.clone(),
                    total_message_count: request.messages.len(),
                };
                let legacy_valid = legacy.proof_verify(&verify_request).unwrap();
                let current_valid = current.proof_verify(&verify_request).unwrap();
                assert_eq!(legacy_valid, proof_version == 0);
                assert_eq!(current_valid, proof_version == 1);
            }
        }
    }
    #[test]
    fn cubbs_protocol_proof_is_accepted_by_current_reference() {
        let (_, request) = fixture();
        let protocol = crate::CuBbsProtocolEngine::new();
        let current = Zkryptium061Reference::new();
        let disclosed_indexes = vec![0, 3, 7];
        let presentation_header = b"omni-current-reference".to_vec();
        let signature = protocol.sign(request.clone()).unwrap();
        let proof = protocol
            .proof_gen(ProofGenRequest {
                public_key: request.public_key.clone(),
                signature,
                header: request.header.clone(),
                presentation_header: presentation_header.clone(),
                messages: request.messages.clone(),
                disclosed_indexes: disclosed_indexes.clone(),
            })
            .unwrap();
        let verify_request = ProofVerifyRequest {
            public_key: request.public_key,
            proof,
            header: request.header,
            presentation_header,
            disclosed_messages: disclosed_indexes
                .iter()
                .map(|&index| request.messages[index].clone())
                .collect(),
            disclosed_indexes,
            total_message_count: request.messages.len(),
        };
        assert!(current.proof_verify(&verify_request).unwrap());
    }
}
