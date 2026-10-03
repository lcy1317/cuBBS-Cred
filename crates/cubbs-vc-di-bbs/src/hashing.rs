use sha2::{Digest, Sha256};

pub const BBS2023_HEADER_LEN: usize = 64;

pub type Sha256Digest = [u8; 32];

pub fn hash_joined_strings(lines: &[String]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    for line in lines {
        hasher.update(line.as_bytes());
    }
    hasher.finalize().into()
}

pub fn hash_mandatory(mandatory: &[String]) -> Sha256Digest {
    hash_joined_strings(mandatory)
}

pub fn hash_canonical_proof(canonical_proof: &str) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(canonical_proof.as_bytes());
    hasher.finalize().into()
}

pub fn bbs_header(proof_hash: Sha256Digest, mandatory_hash: Sha256Digest) -> Vec<u8> {
    let mut header = Vec::with_capacity(BBS2023_HEADER_LEN);
    header.extend_from_slice(&proof_hash);
    header.extend_from_slice(&mandatory_hash);
    header
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mandatory_hash_concatenates_without_separators() {
        assert_eq!(
            hash_mandatory(&["ab".to_owned(), "c".to_owned()]),
            hash_mandatory(&["a".to_owned(), "bc".to_owned()])
        );
    }

    #[test]
    fn bbs_header_is_proof_hash_then_mandatory_hash() {
        let header = bbs_header([1; 32], [2; 32]);
        assert_eq!(header.len(), BBS2023_HEADER_LEN);
        assert_eq!(&header[..32], &[1; 32]);
        assert_eq!(&header[32..], &[2; 32]);
    }
}
