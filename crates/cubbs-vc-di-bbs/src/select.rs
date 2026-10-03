use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonPointerToken {
    Key(String),
    Index(usize),
}

#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("JSON pointer must be empty or start with '/'")]
    InvalidPointer,

    #[error("invalid JSON pointer escape sequence")]
    InvalidEscape,

    #[error("JSON pointer does not match document: {0}")]
    PointerNotFound(String),

    #[error("array index is out of bounds in JSON pointer: {0}")]
    ArrayIndexOutOfBounds(usize),

    #[error("cannot traverse non-container value in JSON pointer")]
    NonContainer,
}

pub fn parse_json_pointer(pointer: &str) -> Result<Vec<JsonPointerToken>, SelectionError> {
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    if !pointer.starts_with('/') {
        return Err(SelectionError::InvalidPointer);
    }

    pointer
        .split('/')
        .skip(1)
        .map(|part| {
            let unescaped = unescape_pointer_part(part)?;
            Ok(match unescaped.parse::<usize>() {
                Ok(index) => JsonPointerToken::Index(index),
                Err(_) => JsonPointerToken::Key(unescaped),
            })
        })
        .collect()
}

pub fn select_json(document: &Value, pointers: &[String]) -> Result<Value, SelectionError> {
    if pointers.is_empty() {
        return Ok(Value::Null);
    }

    let mut selection = Value::Object(Map::new());
    if let Some(context) = document.get("@context") {
        selection["@context"] = context.clone();
    }
    init_selection(&mut selection, document);

    for pointer in pointers {
        let paths = parse_json_pointer(pointer)?;
        if paths.is_empty() {
            return Ok(document.clone());
        }
        select_path(document, &paths, &mut selection, pointer)?;
    }

    compact_arrays(&mut selection);
    Ok(selection)
}

fn unescape_pointer_part(part: &str) -> Result<String, SelectionError> {
    let mut output = String::with_capacity(part.len());
    let mut chars = part.chars();
    while let Some(ch) = chars.next() {
        if ch != '~' {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some('0') => output.push('~'),
            Some('1') => output.push('/'),
            _ => return Err(SelectionError::InvalidEscape),
        }
    }
    Ok(output)
}

fn select_path(
    document: &Value,
    paths: &[JsonPointerToken],
    selection: &mut Value,
    pointer: &str,
) -> Result<(), SelectionError> {
    let mut source = document;
    for token in paths {
        source = get_child(source, token)
            .ok_or_else(|| SelectionError::PointerNotFound(pointer.to_owned()))?;
    }

    let mut selected_parent = selection;
    let mut source_cursor = document;
    for token in &paths[..paths.len() - 1] {
        let next_source = get_child(source_cursor, token)
            .ok_or_else(|| SelectionError::PointerNotFound(pointer.to_owned()))?;
        selected_parent = ensure_child(selected_parent, token, next_source)?;
        source_cursor = next_source;
    }

    let last = paths.last().expect("non-empty path");
    set_child(selected_parent, last, selected_value(source))?;
    Ok(())
}

fn selected_value(source: &Value) -> Value {
    match source {
        Value::Object(_) => {
            let mut value = Value::Object(Map::new());
            init_selection(&mut value, source);
            merge_objects(&mut value, source.clone());
            value
        }
        _ => source.clone(),
    }
}

fn get_child<'a>(source: &'a Value, token: &JsonPointerToken) -> Option<&'a Value> {
    match (source, token) {
        (Value::Object(map), JsonPointerToken::Key(key)) => map.get(key),
        (Value::Array(values), JsonPointerToken::Index(index)) => values.get(*index),
        _ => None,
    }
}

fn ensure_child<'a>(
    selected_parent: &'a mut Value,
    token: &JsonPointerToken,
    source_child: &Value,
) -> Result<&'a mut Value, SelectionError> {
    match (selected_parent, token) {
        (Value::Object(map), JsonPointerToken::Key(key)) => {
            if !map.contains_key(key) {
                map.insert(key.clone(), empty_selection_for(source_child));
            }
            Ok(map.get_mut(key).expect("inserted key exists"))
        }
        (Value::Array(values), JsonPointerToken::Index(index)) => {
            while values.len() <= *index {
                values.push(Value::Null);
            }
            if values[*index].is_null() {
                values[*index] = empty_selection_for(source_child);
            }
            Ok(&mut values[*index])
        }
        (Value::Array(_), JsonPointerToken::Key(_)) => Err(SelectionError::NonContainer),
        (Value::Object(_), JsonPointerToken::Index(index)) => {
            Err(SelectionError::ArrayIndexOutOfBounds(*index))
        }
        _ => Err(SelectionError::NonContainer),
    }
}

fn set_child(
    selected_parent: &mut Value,
    token: &JsonPointerToken,
    value: Value,
) -> Result<(), SelectionError> {
    match (selected_parent, token) {
        (Value::Object(map), JsonPointerToken::Key(key)) => {
            map.insert(key.clone(), value);
            Ok(())
        }
        (Value::Array(values), JsonPointerToken::Index(index)) => {
            while values.len() <= *index {
                values.push(Value::Null);
            }
            values[*index] = value;
            Ok(())
        }
        (Value::Array(_), JsonPointerToken::Key(_)) => Err(SelectionError::NonContainer),
        (Value::Object(_), JsonPointerToken::Index(index)) => {
            Err(SelectionError::ArrayIndexOutOfBounds(*index))
        }
        _ => Err(SelectionError::NonContainer),
    }
}

fn empty_selection_for(source: &Value) -> Value {
    match source {
        Value::Array(_) => Value::Array(Vec::new()),
        Value::Object(_) => {
            let mut selection = Value::Object(Map::new());
            init_selection(&mut selection, source);
            selection
        }
        _ => Value::Null,
    }
}

fn init_selection(selection: &mut Value, source: &Value) {
    let (Some(selection), Some(source)) = (selection.as_object_mut(), source.as_object()) else {
        return;
    };

    if let Some(id) = source.get("id").and_then(Value::as_str) {
        if !id.starts_with("_:") {
            selection.insert("id".to_owned(), Value::String(id.to_owned()));
        }
    }
    if let Some(id) = source.get("@id").and_then(Value::as_str) {
        if !id.starts_with("_:") {
            selection.insert("@id".to_owned(), Value::String(id.to_owned()));
        }
    }
    if let Some(value_type) = source.get("type") {
        selection.insert("type".to_owned(), value_type.clone());
    }
    if let Some(value_type) = source.get("@type") {
        selection.insert("@type".to_owned(), value_type.clone());
    }
}

fn merge_objects(target: &mut Value, source: Value) {
    let (Some(target), Value::Object(source)) = (target.as_object_mut(), source) else {
        return;
    };
    for (key, value) in source {
        target.insert(key, value);
    }
}

fn compact_arrays(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values.iter_mut() {
                compact_arrays(value);
            }
            values.retain(|value| !value.is_null());
        }
        Value::Object(map) => {
            for value in map.values_mut() {
                compact_arrays(value);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_json_pointer_escapes() {
        assert_eq!(
            parse_json_pointer("/credentialSubject/a~1b/c~0d").unwrap(),
            vec![
                JsonPointerToken::Key("credentialSubject".to_owned()),
                JsonPointerToken::Key("a/b".to_owned()),
                JsonPointerToken::Key("c~d".to_owned()),
            ]
        );
    }

    #[test]
    fn selects_literal_and_keeps_context_id_type() {
        let document = json!({
            "@context": ["https://www.w3.org/ns/credentials/v2"],
            "id": "urn:cred:1",
            "type": ["VerifiableCredential"],
            "issuer": "did:example:issuer",
            "credentialSubject": {
                "id": "did:example:alice",
                "type": "Person",
                "name": "Alice",
                "age": 42
            }
        });

        let selected = select_json(
            &document,
            &["/credentialSubject/name".to_owned(), "/issuer".to_owned()],
        )
        .unwrap();
        assert_eq!(
            selected,
            json!({
                "@context": ["https://www.w3.org/ns/credentials/v2"],
                "id": "urn:cred:1",
                "type": ["VerifiableCredential"],
                "issuer": "did:example:issuer",
                "credentialSubject": {
                    "id": "did:example:alice",
                    "type": "Person",
                    "name": "Alice"
                }
            })
        );
    }

    #[test]
    fn selecting_array_index_makes_dense_array() {
        let document = json!({
            "@context": {},
            "credentialSubject": {
                "achievements": [
                    {"name": "A"},
                    {"name": "B"},
                    {"name": "C"}
                ]
            }
        });
        let selected = select_json(
            &document,
            &["/credentialSubject/achievements/2/name".to_owned()],
        )
        .unwrap();
        assert_eq!(
            selected,
            json!({
                "@context": {},
                "credentialSubject": {
                    "achievements": [
                        {"name": "C"}
                    ]
                }
            })
        );
    }
}
