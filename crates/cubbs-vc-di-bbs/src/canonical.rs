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
