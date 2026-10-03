use std::collections::BTreeMap;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, Mac};
use sha2::Sha256;

pub const HMAC_KEY_LEN: usize = 32;

type HmacSha256 = Hmac<Sha256>;

pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut hmac = HmacSha256::new_from_slice(key)
        .expect("HMAC-SHA256 accepts keys of any size, including empty keys");
    hmac.update(data);
    hmac.finalize().into_bytes().into()
}

/// Build the shuffled BBS blank-node label map used by `bbs-2023`.
///
/// The input mirrors the RDFC canonical id map after `_:` prefixes have been
/// removed: input blank-node label -> canonical label (`c14n0`, `c14n1`, ...).
/// The output maps input blank-node labels to `bN` labels. Sort order follows
/// the W3C/Digital Bazaar reference algorithm: HMAC each canonical label,
/// encode as multibase base64url with a leading `u`, sort the HMAC identifiers,
/// then replace each blank-node label with the rank of its HMAC identifier.
pub fn shuffled_bnode_label_map(
    canonical_id_map: &BTreeMap<String, String>,
    hmac_key: &[u8],
) -> BTreeMap<String, String> {
    let hmac_ids_by_input: BTreeMap<String, String> = canonical_id_map
        .iter()
        .map(|(input, c14n_label)| {
            let digest = hmac_sha256(hmac_key, c14n_label.as_bytes());
            let hmac_id = format!("u{}", URL_SAFE_NO_PAD.encode(digest));
            (input.clone(), hmac_id)
        })
        .collect();

    let mut sorted_hmac_ids: Vec<String> = hmac_ids_by_input.values().cloned().collect();
    sorted_hmac_ids.sort();

    hmac_ids_by_input
        .into_iter()
        .map(|(input, hmac_id)| {
            let index = sorted_hmac_ids
                .binary_search(&hmac_id)
                .expect("HMAC identifier exists in sorted list");
            (input, format!("b{index}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffled_map_is_deterministic_and_uses_b_labels() {
        let canonical_id_map = BTreeMap::from([
            ("alice".to_owned(), "c14n0".to_owned()),
            ("bob".to_owned(), "c14n1".to_owned()),
            ("carol".to_owned(), "c14n2".to_owned()),
        ]);
        let one = shuffled_bnode_label_map(&canonical_id_map, &[9; HMAC_KEY_LEN]);
        let two = shuffled_bnode_label_map(&canonical_id_map, &[9; HMAC_KEY_LEN]);
        assert_eq!(one, two);
        assert_eq!(one.len(), canonical_id_map.len());
        assert!(one.values().all(|v| v.starts_with('b')));
    }
}
