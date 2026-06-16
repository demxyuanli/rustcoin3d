//! ISO 10303-28 (XML STEP) support.
//!
//! XML STEP is the XML encoding of EXPRESS/STEP data.
//! This module implements a zero-dependency XML writer (EntityIndex → XML)
//! and a lightweight regex-based XML reader (XML → EntityIndex).

use rc3d_core::math::Real;
use std::collections::HashMap;
use super::parser::EntityIndex;
use super::value::StepValue;

/// Write an EntityIndex as ISO 10303-28 XML.
pub fn write_xml_step(entities: &EntityIndex, schema: &str) -> String {
    let mut out = String::with_capacity(entities.len() * 512);
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<iso_10303_28 version=\"2.0\">\n  <express schema=\"{}\"/>\n",
        xml_escape(schema)
    ));
    out.push_str("  <Data>\n");

    // Sort by entity ID for deterministic output
    let mut ids: Vec<u64> = entities.keys().copied().collect();
    ids.sort();

    for id in ids {
        let record = &entities[&id];
        out.push_str(&format!(
            "    <{} id=\"{}\">\n",
            record.name, id
        ));
        write_params_xml(&record.params, &mut out, 3);
        out.push_str(&format!("    </{}>\n", record.name));
    }

    out.push_str("  </Data>\n</iso_10303_28>\n");
    out
}

fn write_params_xml(value: &StepValue, out: &mut String, depth: usize) {
    let indent = "  ".repeat(depth);
    match value {
        StepValue::Omitted => {
            out.push_str(&format!("{}<Omitted/>\n", indent));
        }
        StepValue::Integer(v) => {
            out.push_str(&format!("{}<Integer>{}</Integer>\n", indent, v));
        }
        StepValue::Real(v) => {
            out.push_str(&format!("{}<Real>{:.6}</Real>\n", indent, v));
        }
        StepValue::String(s) => {
            if !s.is_empty() {
                out.push_str(&format!("{}<String>{}</String>\n", indent, xml_escape(s)));
            }
        }
        StepValue::Enum(s) => {
            out.push_str(&format!("{}<Enum>{}</Enum>\n", indent, s.trim_matches('.')));
        }
        StepValue::Ref(id) => {
            out.push_str(&format!("{}<Ref id=\"{}\"/>\n", indent, id));
        }
        StepValue::Typed(tag, inner) => {
            out.push_str(&format!("{}<Typed name=\"{}\">\n", indent, xml_escape(tag)));
            write_params_xml(inner, out, depth + 1);
            out.push_str(&format!("{}</Typed>\n", indent));
        }
        StepValue::List(items) => {
            out.push_str(&format!("{}<List>\n", indent));
            for item in items.iter() {
                write_params_xml(item, out, depth + 1);
            }
            out.push_str(&format!("{}</List>\n", indent));
        }
    }
}

/// Parse XML STEP text into an EntityIndex.
/// Uses simple regex-based extraction — handles the standard XML STEP format.
pub fn parse_xml_step(input: &str) -> Result<EntityIndex, String> {
    let mut entities = HashMap::new();
    let mut entity_count = 0usize;

    // Find all entity elements with id attributes
    let mut pos = 0;
    let bytes = input.as_bytes();

    while pos < bytes.len() {
        // Find next opening tag with id attribute
        if let Some(tag_start) = find_str(bytes, pos, "<") {
            let tag_start = tag_start + 1;
            if tag_start >= bytes.len() || bytes[tag_start] == b'/' || bytes[tag_start] == b'?' {
                pos = tag_start + 1;
                continue;
            }

            // Read tag name
            let name_end = find_whitespace_or_close(bytes, tag_start);
            if name_end <= tag_start { pos = tag_start + 1; continue; }
            let name = std::str::from_utf8(&bytes[tag_start..name_end])
                .map_err(|_| "invalid UTF-8 in tag name".to_string())?
                .to_string();

            // Check if this is an entity element (has id attribute)
            let after_name = &bytes[name_end..];
            if let Some(id_pos) = find_str(after_name, 0, "id=\"") {
                let id_start = name_end + id_pos + 4;
                let id_end = find_char(bytes, id_start, b'"');
                if id_end <= id_start { pos = name_end; continue; }
                let id_str = std::str::from_utf8(&bytes[id_start..id_end])
                    .map_err(|_| "invalid UTF-8 in id".to_string())?;
                let id: u64 = id_str.parse().map_err(|e| format!("invalid entity id '{}': {}", id_str, e))?;

                // Find closing tag
                let closing_tag = format!("</{}>", name);
                if let Some(close_pos) = find_str(bytes, id_end, &closing_tag) {
                    let params_start = find_char(bytes, id_end, b'>') + 1;
                    if params_start < close_pos {
                        let params_text = std::str::from_utf8(&bytes[params_start..close_pos])
                            .map_err(|_| "invalid UTF-8 in params".to_string())?;
                        let params = parse_xml_params(params_text)?;
                        let etype = super::entity_types::EntityType::from_name(&name);
                        entities.insert(id, super::parser::EntityRecord {
                            name,
                            params,
                            entity_type: etype,
                        });
                        entity_count += 1;
                    }
                }
            }

            pos = name_end + 1;
        } else {
            break;
        }
    }

    if entity_count == 0 {
        return Err("no entities found in XML input".into());
    }

    Ok(entities)
}

fn parse_xml_params(text: &str) -> Result<StepValue, String> {
    let mut items = Vec::new();
    let bytes = text.as_bytes();
    let mut pos = 0;

    while pos < bytes.len() {
        // Skip whitespace
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() { break; }

        if bytes[pos] == b'<' {
            let tag_start = pos + 1;
            let tag_name_end = find_whitespace_or_close(bytes, tag_start);
            if tag_name_end <= tag_start { break; }
            let tag_name = std::str::from_utf8(&bytes[tag_start..tag_name_end])
                .map_err(|_| "invalid tag".to_string())?;

            // Find tag content end
            let close_tag = format!("</{}>", tag_name);
            let self_close = bytes[tag_name_end] == b'/';

            match tag_name {
                "Omitted" => {
                    items.push(StepValue::Omitted);
                }
                "Integer" => {
                    let val_start = find_char(bytes, tag_name_end, b'>') + 1;
                    let val_end = find_str(bytes, val_start, &close_tag).unwrap_or(val_start);
                    let s = std::str::from_utf8(&bytes[val_start..val_end]).unwrap_or("0").trim();
                    items.push(StepValue::Integer(s.parse().unwrap_or(0)));
                    pos = val_end + close_tag.len();
                    continue;
                }
                "Real" => {
                    let val_start = find_char(bytes, tag_name_end, b'>') + 1;
                    let val_end = find_str(bytes, val_start, &close_tag).unwrap_or(val_start);
                    let s = std::str::from_utf8(&bytes[val_start..val_end]).unwrap_or("0").trim();
                    items.push(StepValue::Real(s.parse().unwrap_or(0.0)));
                    pos = val_end + close_tag.len();
                    continue;
                }
                "String" => {
                    let val_start = find_char(bytes, tag_name_end, b'>') + 1;
                    let val_end = find_str(bytes, val_start, &close_tag).unwrap_or(val_start);
                    let s = std::str::from_utf8(&bytes[val_start..val_end]).unwrap_or("").trim();
                    items.push(StepValue::String(s.to_string()));
                    pos = val_end + close_tag.len();
                    continue;
                }
                "Enum" => {
                    let val_start = find_char(bytes, tag_name_end, b'>') + 1;
                    let val_end = find_str(bytes, val_start, &close_tag).unwrap_or(val_start);
                    let s = std::str::from_utf8(&bytes[val_start..val_end]).unwrap_or("").trim();
                    let dot_val = format!(".{}.", s);
                    items.push(StepValue::Enum(dot_val));
                    pos = val_end + close_tag.len();
                    continue;
                }
                "Ref" => {
                    // Find id="NNN" attribute
                    if let Some(a_pos) = find_str(bytes, tag_name_end, "id=\"") {
                        let id_start = a_pos + 4;
                        let id_end = find_char(bytes, id_start, b'"');
                        let id_s = std::str::from_utf8(&bytes[id_start..id_end]).unwrap_or("0");
                        items.push(StepValue::Ref(id_s.parse().unwrap_or(0)));
                        pos = find_char(bytes, id_end, b'>') + 1;
                        continue;
                    }
                }
                "List" => {
                    let content_start = find_char(bytes, tag_name_end, b'>') + 1;
                    let content_end = find_str(bytes, content_start, &close_tag).unwrap_or(content_start);
                    let inner = std::str::from_utf8(&bytes[content_start..content_end])
                        .map_err(|_| "invalid list content".to_string())?;
                    let list_items = parse_xml_params(inner)?;
                    items.push(list_items);
                    pos = content_end + close_tag.len();
                    continue;
                }
                "Typed" => {
                    // Find name attribute
                    let typed_name = if let Some(n_pos) = find_str(bytes, tag_name_end, "name=\"") {
                        let n_start = n_pos + 6;
                        let n_end = find_char(bytes, n_start, b'"');
                        std::str::from_utf8(&bytes[n_start..n_end]).unwrap_or("").to_string()
                    } else {
                        String::new()
                    };
                    let content_start = find_char(bytes, tag_name_end, b'>') + 1;
                    let content_end = find_str(bytes, content_start, &close_tag).unwrap_or(content_start);
                    let inner = std::str::from_utf8(&bytes[content_start..content_end])
                        .map_err(|_| "invalid typed content".to_string())?;
                    let inner_val = parse_xml_params(inner)?;
                    items.push(StepValue::Typed(typed_name, Box::new(inner_val)));
                    pos = content_end + close_tag.len();
                    continue;
                }
                _ => {
                    // Unknown tag — skip
                    let close = format!("</{}>", tag_name);
                    if let Some(cp) = find_str(bytes, pos, &close) {
                        pos = cp + close.len();
                    } else {
                        pos += 1;
                    }
                    continue;
                }
            }

            // For self-closing tags, skip past />
            if self_close {
                pos = find_char(bytes, tag_name_end, b'>') + 1;
            } else {
                pos = find_str(bytes, pos, &close_tag).unwrap_or(pos + 1) + close_tag.len();
            }
        } else {
            pos += 1;
        }
    }

    Ok(StepValue::List(items))
}

// ── Byte-level search helpers ──

fn find_str(haystack: &[u8], start: usize, needle: &str) -> Option<usize> {
    let n = needle.as_bytes();
    if start + n.len() > haystack.len() { return None; }
    for i in start..=haystack.len() - n.len() {
        if &haystack[i..i + n.len()] == n { return Some(i); }
    }
    None
}

fn find_char(haystack: &[u8], start: usize, ch: u8) -> usize {
    for i in start..haystack.len() {
        if haystack[i] == ch { return i; }
    }
    haystack.len()
}

fn find_whitespace_or_close(bytes: &[u8], start: usize) -> usize {
    for i in start..bytes.len() {
        if bytes[i].is_ascii_whitespace() || bytes[i] == b'>' || bytes[i] == b'/' {
            return i;
        }
    }
    bytes.len()
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_xml_basic() {
        let mut entities = HashMap::new();
        entities.insert(1, super::super::parser::EntityRecord {
            name: "CARTESIAN_POINT".into(),
            params: StepValue::List(vec![
                StepValue::String("".into()),
                StepValue::List(vec![
                    StepValue::Real(0.0),
                    StepValue::Real(1.0),
                    StepValue::Real(2.0),
                ]),
            ]),
            entity_type: super::super::entity_types::EntityType::CartesianPoint,
        });

        let xml = write_xml_step(&entities, "AP203");
        assert!(xml.contains("<?xml"));
        assert!(xml.contains("CARTESIAN_POINT"));
        assert!(xml.contains("iso_10303_28"));
    }

    #[test]
    fn test_xml_roundtrip() {
        let mut entities = HashMap::new();
        entities.insert(1, super::super::parser::EntityRecord {
            name: "CARTESIAN_POINT".into(),
            params: StepValue::List(vec![
                StepValue::String("".into()),
                StepValue::List(vec![
                    StepValue::Real(0.0),
                    StepValue::Real(1.0),
                    StepValue::Real(2.0),
                ]),
            ]),
            entity_type: super::super::entity_types::EntityType::CartesianPoint,
        });
        entities.insert(10, super::super::parser::EntityRecord {
            name: "LINE".into(),
            params: StepValue::List(vec![
                StepValue::String("".into()),
                StepValue::Ref(1),
                StepValue::Ref(3),
            ]),
            entity_type: super::super::entity_types::EntityType::Line,
        });

        let xml = write_xml_step(&entities, "AP203");
        let parsed = parse_xml_step(&xml).expect("XML round-trip parse");

        assert_eq!(parsed.len(), 2);
        assert!(parsed.contains_key(&1));
        assert!(parsed.contains_key(&10));
        assert_eq!(parsed[&10].name, "LINE");
    }
}
