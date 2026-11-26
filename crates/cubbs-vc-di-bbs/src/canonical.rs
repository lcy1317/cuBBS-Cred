use std::{collections::BTreeMap, io::Cursor};

use oxrdf::Dataset;
use oxttl::NQuadsParser;

pub type CanonicalizationResult<T> = core::result::Result<T, CanonicalizationError>;

#[derive(Debug, thiserror::Error)]
pub enum CanonicalizationError {
    #[error("N-Quads parse error: {0}")]
    NQuads(String),

    #[error("RDFC-1.0 canonicalization error: {0}")]
    RdfCanon(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizedNQuads {
    pub serialized: String,
    pub lines: Vec<String>,
    /// Map from input blank-node labels to canonical labels (`c14nN`).
    pub canonical_id_map: BTreeMap<String, String>,
}

pub fn canonicalize_nquads(input: &str) -> CanonicalizationResult<CanonicalizedNQuads> {
    if let Some(fast) = try_fast_canonicalize_nquads_no_bnodes(input) {
        return Ok(fast);
    }
    let dataset = parse_nquads(input)?;
    let issued_identifiers =
        rdf_canon::issue(&dataset).map_err(|e| CanonicalizationError::RdfCanon(e.to_string()))?;
    let relabeled = rdf_canon::relabel(&dataset, &issued_identifiers)
        .map_err(|e| CanonicalizationError::RdfCanon(e.to_string()))?;
    let serialized = rdf_canon::serialize(&relabeled);
    let canonical_id_map = issued_identifiers.into_iter().collect();
    let lines = split_canonical_nquads(&serialized);

    Ok(CanonicalizedNQuads {
        serialized,
        lines,
        canonical_id_map,
    })
}

fn try_fast_canonicalize_nquads_no_bnodes(input: &str) -> Option<CanonicalizedNQuads> {
    let has_bnode = input.contains("_:");
    if has_bnode {
        return try_fast_canonicalize_nquads_single_bnode0(input);
    }
    let mut lines: Vec<String> = Vec::new();
    for line in input.lines() {
        if line.is_empty() {
            continue;
        }
        if !is_canonical_default_graph_nquad_line(line) {
            return None;
        }
        let mut owned = String::with_capacity(line.len() + 1);
        owned.push_str(line);
        owned.push('\n');
        lines.push(owned);
    }
    lines.sort_unstable();
    lines.dedup();
    let total_len: usize = lines.iter().map(String::len).sum();
    let mut serialized = String::with_capacity(total_len);
    for line in &lines {
        serialized.push_str(line);
    }
    Some(CanonicalizedNQuads {
        serialized,
        lines,
        canonical_id_map: BTreeMap::new(),
    })
}

/// Fast path for datasets whose only blank node is `_:0` in subject position
/// (the exact shape produced by W3C Data Integrity proof configurations).
/// Under RDFC-1.0, a dataset with a single blank node `_:0` deterministically
/// assigns `0 -> c14n0`.
fn try_fast_canonicalize_nquads_single_bnode0(input: &str) -> Option<CanonicalizedNQuads> {
    let mut lines: Vec<String> = Vec::new();
    let mut saw_bnode0 = false;
    for line in input.lines() {
        if line.is_empty() {
            continue;
        }
        let Some(rest) = line.strip_prefix("_:0 ") else {
            return None;
        };
        // Ensure no other blank nodes appear in predicate/object/graph positions.
        if rest.contains("_:") {
            return None;
        }
        // Validate predicate + object + " ." using a synthetic IRI subject.
        if !is_canonical_default_graph_nquad_rest(rest) {
            return None;
        }
        saw_bnode0 = true;
        let mut owned = String::with_capacity(rest.len() + 9);
        owned.push_str("_:c14n0 ");
        owned.push_str(rest);
        owned.push('\n');
        lines.push(owned);
    }
    if !saw_bnode0 {
        return None;
    }
    lines.sort_unstable();
    lines.dedup();
    let total_len: usize = lines.iter().map(String::len).sum();
    let mut serialized = String::with_capacity(total_len);
    for line in &lines {
        serialized.push_str(line);
    }
    Some(CanonicalizedNQuads {
        serialized,
        lines,
        canonical_id_map: BTreeMap::from([("0".to_owned(), "c14n0".to_owned())]),
    })
}

fn is_simple_iri(iri: &str) -> bool {
    !iri.is_empty()
        && iri.bytes().all(|b| {
            b > 0x20
                && b != 0x7f
                && !matches!(b, b'<' | b'>' | b'"' | b'{' | b'}' | b'|' | b'^' | b'`' | b'\\')
        })
}

fn is_canonical_default_graph_nquad_line(line: &str) -> bool {
    let Some(rest) = line.strip_prefix('<') else {
        return false;
    };
    let Some(subj_end) = rest.find('>') else {
        return false;
    };
    if !is_simple_iri(&rest[..subj_end]) {
        return false;
    }
    let Some(after_subj) = rest[subj_end + 1..].strip_prefix(' ') else {
        return false;
    };
    is_canonical_default_graph_nquad_rest(after_subj)
}

fn is_canonical_default_graph_nquad_rest(after_subj: &str) -> bool {
    let Some(body) = after_subj.strip_suffix(" .") else {
        return false;
    };
    let Some(rest) = body.strip_prefix('<') else {
        return false;
    };
    let Some(pred_end) = rest.find('>') else {
        return false;
    };
    if !is_simple_iri(&rest[..pred_end]) {
        return false;
    }
    let Some(obj) = rest[pred_end + 1..].strip_prefix(' ') else {
        return false;
    };
    if let Some(iri) = obj.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        return is_simple_iri(iri);
    }
    let Some(lit) = obj.strip_prefix('"') else {
        return false;
    };
    let bytes = lit.as_bytes();
    let mut i = 0;
    let mut close_quote = None;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x20 || b == 0x7f {
            return false;
        }
        if b == b'\\' {
            i += 1;
            if i >= bytes.len() {
                return false;
            }
            if !matches!(bytes[i], b'"' | b'\\' | b'n' | b'r' | b't' | b'b' | b'f') {
                return false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            close_quote = Some(i);
            break;
        }
        i += 1;
    }
    let Some(end_idx) = close_quote else {
        return false;
    };
    let suffix = &lit[end_idx + 1..];
    if suffix.is_empty() {
        return true;
    }
    if let Some(dt) = suffix.strip_prefix("^^<").and_then(|s| s.strip_suffix('>')) {
        return is_simple_iri(dt) && dt != "http://www.w3.org/2001/XMLSchema#string";
    }
    if let Some(lang) = suffix.strip_prefix('@') {
        return !lang.is_empty()
            && lang
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    }
    false
}

pub fn split_canonical_nquads(serialized: &str) -> Vec<String> {
    serialized
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| format!("{line}\n"))
        .collect()
}

pub fn canonical_label_map_from_input_map(
    input_to_canonical: &BTreeMap<String, String>,
    input_to_replacement: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    input_to_canonical
        .iter()
        .filter_map(|(input, canonical)| {
            input_to_replacement
                .get(input)
                .map(|replacement| (canonical.clone(), replacement.clone()))
        })
        .collect()
}

pub fn relabel_and_sort_canonical_nquads(
    lines: &[String],
    canonical_to_replacement: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut relabeled = replace_blank_node_labels(lines, canonical_to_replacement);
    relabeled.sort();
    relabeled
}

pub fn replace_blank_node_labels(
    lines: &[String],
    label_map: &BTreeMap<String, String>,
) -> Vec<String> {
    lines
        .iter()
        .map(|line| replace_blank_node_labels_in_line(line, label_map))
        .collect()
}

fn parse_nquads(input: &str) -> CanonicalizationResult<Dataset> {
    let mut quads = Vec::new();
    for parsed in NQuadsParser::new().for_reader(Cursor::new(input.as_bytes())) {
        quads.push(parsed.map_err(|e| CanonicalizationError::NQuads(e.to_string()))?);
    }
    Ok(Dataset::from_iter(quads))
}

fn replace_blank_node_labels_in_line(line: &str, label_map: &BTreeMap<String, String>) -> String {
    let mut output = String::with_capacity(line.len());
    let mut remaining = line;

    while let Some(pos) = remaining.find("_:") {
        let (before, after_prefix) = remaining.split_at(pos);
        output.push_str(before);
        output.push_str("_:");

        let label_start = 2;
        let label_end = after_prefix[label_start..]
            .find(char::is_whitespace)
            .map_or(after_prefix.len(), |offset| label_start + offset);
        let label = &after_prefix[label_start..label_end];
        output.push_str(label_map.get(label).map_or(label, String::as_str));
        remaining = &after_prefix[label_end..];
    }

    output.push_str(remaining);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_nquads_and_returns_identifier_map() {
        let input = r#"_:e0 <http://example.org/vocab#next> _:e1 _:g .
_:e1 <http://example.org/vocab#next> _:e0 _:g .
"#;
        let canonicalized = canonicalize_nquads(input).unwrap();
        assert!(canonicalized.serialized.contains("_:c14n"));
        assert_eq!(canonicalized.lines.len(), 2);
        assert!(canonicalized.canonical_id_map.contains_key("e0"));
        assert!(canonicalized.canonical_id_map.contains_key("e1"));
        assert!(canonicalized.canonical_id_map.contains_key("g"));
    }

    #[test]
    fn relabels_blank_nodes_and_sorts_lines() {
        let lines = vec![
            "_:c14n1 <urn:p> _:c14n0 .\n".to_owned(),
            "_:c14n0 <urn:p> \"x\" .\n".to_owned(),
        ];
        let label_map = BTreeMap::from([
            ("c14n0".to_owned(), "b1".to_owned()),
            ("c14n1".to_owned(), "b0".to_owned()),
        ]);
        let relabeled = relabel_and_sort_canonical_nquads(&lines, &label_map);
        assert_eq!(
            relabeled,
            vec![
                "_:b0 <urn:p> _:b1 .\n".to_owned(),
                "_:b1 <urn:p> \"x\" .\n".to_owned(),
            ]
        );
    }

    #[test]
    fn converts_input_label_map_to_canonical_label_map() {
        let input_to_canonical = BTreeMap::from([
            ("alice".to_owned(), "c14n1".to_owned()),
            ("bob".to_owned(), "c14n0".to_owned()),
        ]);
        let input_to_replacement = BTreeMap::from([
            ("alice".to_owned(), "b0".to_owned()),
            ("bob".to_owned(), "b1".to_owned()),
        ]);
        assert_eq!(
            canonical_label_map_from_input_map(&input_to_canonical, &input_to_replacement),
            BTreeMap::from([
                ("c14n0".to_owned(), "b1".to_owned()),
                ("c14n1".to_owned(), "b0".to_owned()),
            ])
        );
    }
}
