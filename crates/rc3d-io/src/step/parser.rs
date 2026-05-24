use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::path::Path;
use super::entity_types::EntityType;
use super::value::StepValue;

#[derive(Debug, Clone)]
pub struct EntityRecord {
    pub name: String,
    pub params: StepValue,
    pub entity_type: EntityType,
}

pub type EntityIndex = HashMap<u64, EntityRecord>;

#[derive(Debug)]
pub struct Exchange {
    pub header: Option<super::header::HeaderInfo>,
    pub entities: EntityIndex,
}

/// Parse ISO 10303-21 ASCII exchange structure text.
pub fn parse_exchange(input: &str) -> Result<Exchange, String> {
    let input = input.trim();
    if !input.starts_with("ISO-10303-21;") {
        return Err("missing ISO-10303-21 header".into());
    }
    let rest = &input["ISO-10303-21;".len()..];

    // Parse HEADER section (between ISO-10303-21 and DATA)
    let data_pos = rest.find("DATA;");
    let header = if let Some(dp) = data_pos {
        let header_region = &rest[..dp];
        super::header::parse_header(header_region)
            .map(|(h, _)| h)
    } else {
        None
    };
    let rest = skip_until(rest, "DATA;").ok_or("DATA section not found")?;
    let rest = &rest["DATA;".len()..];

    // Parse DATA section entities
    let rest = rest.trim_start();
    let (mut entities, _) = parse_entities(rest)?;

    // Tag each entity with its EntityType
    for record in entities.values_mut() {
        record.entity_type = EntityType::from_name(&record.name);
    }

    Ok(Exchange { header, entities })
}

fn skip_until<'a>(input: &'a str, marker: &str) -> Option<&'a str> {
    let pos = input.find(marker)?;
    Some(&input[pos..])
}

fn parse_entities(input: &str) -> Result<(EntityIndex, &str), String> {
    let mut entities = EntityIndex::new();
    let mut rest = input;
    loop {
        rest = rest.trim_start();
        if rest.is_empty() || rest.starts_with("ENDSEC;") || rest.starts_with("END-ISO") {
            break;
        }
        // Skip comments
        if rest.starts_with("/*") {
            if let Some(end) = rest.find("*/") {
                rest = &rest[end + 2..];
                continue;
            }
        }
        match parse_entity(rest) {
            Ok((eid, ename, eparams, new_rest)) => {
                entities.insert(eid, EntityRecord { name: ename, params: eparams, entity_type: EntityType::Unknown });
                rest = new_rest;
            }
            Err(e) => {
                log::warn!("STEP parse error, skipping entity: {}", e);
                // Attempt to recover by finding the next entity start or statement end
                if let Some(next) = rest[1..].find('#') {
                    rest = &rest[next + 1..];
                } else if let Some(semi) = rest.find(';') {
                    rest = &rest[semi + 1..];
                } else {
                    break;
                }
            }
        }
    }
    Ok((entities, rest))
}

fn parse_entity(input: &str) -> Result<(u64, String, StepValue, &str), String> {
    let rest = input.trim_start();
    if !rest.starts_with('#') {
        return Err(format!("expected entity ID, got: {}", &rest[..rest.len().min(20)]));
    }
    let rest = &rest[1..];
    let (id, rest) = parse_u64(rest).ok_or("invalid entity ID")?;
    let rest = rest.trim_start();
    if !rest.starts_with('=') {
        return Err(format!("expected '=' after entity ID, got: {}", &rest[..rest.len().min(20)]));
    }
    let rest = &rest[1..].trim_start();

    // Handle complex entity (subsuper).
    // Format 1: #ID = (SUPER1() SUPER2())ENTITY_NAME(...)
    // Format 2: #ID = (SUPER1() SUPER2()ENTITY_NAME(...))
    if rest.starts_with('(') {
        let (close_pos, inner) = find_outer_paren(rest);
        let after = rest[close_pos..].trim_start();
        let after_starts_keyword = after.chars().next()
            .map_or(false, |c| c.is_ascii_alphanumeric() || c == '_');
        if !after_starts_keyword {
            // Format 2: entity name + params are inside the outer parens.
            return extract_subsuper_entity(id, inner, after);
        }
        // Format 1: entity name follows the outer parens.
        return finish_entity(id, after);
    }

    finish_entity(id, rest)
}

/// Find the matching close-paren for a string starting with '('.
/// Returns (close_pos, inner_content).
fn find_outer_paren<'a>(input: &'a str) -> (usize, &'a str) {
    let mut depth = 1i32;
    let mut close_pos = 1usize;
    for (i, c) in input.char_indices().skip(1) {
        match c {
            '(' => depth += 1,
            ')' => { depth -= 1; if depth == 0 { close_pos = i + c.len_utf8(); break; } }
            _ => {}
        }
    }
    (close_pos, &input[1..close_pos - 1])
}

/// Format 2: extract the actual entity name+params from multi-line subsuper chain.
/// AP242 writes each supertype on its own line: KEYWORD(params)\nKEYWORD2(params2)\n...
/// We need the most specific geometry type, not REPRESENTATION_ITEM.
/// For AP242 multi-line format, params from ancestor supertypes are merged so that
/// e.g. B_SPLINE_CURVE_WITH_KNOTS gets degree from B_SPLINE_CURVE plus knots from itself.
fn extract_subsuper_entity<'a>(
    id: u64, inner: &'a str, after: &'a str,
) -> Result<(u64, String, StepValue, &'a str), String> {
    // Priority-ordered type list: prefer geometry types over meta wrappers.
    const PRIORITY_TYPES: &[&str] = &[
        "B_SPLINE_CURVE_WITH_KNOTS", "B_SPLINE_CURVE", "RATIONAL_B_SPLINE_CURVE",
        "B_SPLINE_SURFACE_WITH_KNOTS", "B_SPLINE_SURFACE", "RATIONAL_B_SPLINE_SURFACE",
        "LINE", "CIRCLE", "ELLIPSE", "HYPERBOLA", "PARABOLA", "POLYLINE",
        "TRIMMED_CURVE", "COMPOSITE_CURVE", "SEAM_CURVE", "INTERSECTION_CURVE",
        "OFFSET_CURVE_3D",
        "PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE",
        "SPHERICAL_SURFACE", "TOROIDAL_SURFACE",
        "SURFACE_OF_LINEAR_EXTRUSION", "SURFACE_OF_REVOLUTION",
        "RECTANGULAR_TRIMMED_SURFACE",
        "CURVE_BOUNDED_SURFACE",
        "FACE_SURFACE", "ADVANCED_FACE", "FACE_OUTER_BOUND", "FACE_BOUND",
        "CLOSED_SHELL", "OPEN_SHELL", "SHELL",
        "EDGE_CURVE", "ORIENTED_EDGE", "EDGE_LOOP",
        "VERTEX_POINT", "CARTESIAN_POINT",
        "DIRECTION", "VECTOR", "AXIS2_PLACEMENT_3D", "AXIS2_PLACEMENT_2D",
        "CURVE", "SURFACE",
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "PRODUCT_DEFINITION_SHAPE",
        "SHAPE_DEFINITION_REPRESENTATION", "ITEM_DEFINED_TRANSFORMATION",
        "MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS",
    ];

    // Parse all KEYWORD(params) pairs from inner text.
    let pairs = extract_keyword_param_pairs(inner)?;
    if pairs.is_empty() {
        return Err("no keyword+params pairs found in subsuper entity".into());
    }

    // Find the highest-priority type, or fall back to last pair with non-empty params.
    let mut best_idx = pairs.len() - 1;
    let mut best_priority = usize::MAX;
    for (idx, (kw, _pd)) in pairs.iter().enumerate() {
        if let Some(prio) = PRIORITY_TYPES.iter().position(|t| t == kw) {
            if prio < best_priority {
                best_priority = prio;
                best_idx = idx;
            }
        }
    }
    // If no priority type matched, use the last pair with non-empty, non-trivial params.
    if best_priority == usize::MAX {
        best_idx = pairs.iter().rposition(|(_, pd)| {
            !pd.is_empty() && pd != "''" && pd != "*"
        }).unwrap_or(pairs.len() - 1);
    }

    let (name, _) = &pairs[best_idx];

    // Merge params: combine all ancestor pairs' params + selected pair's params.
    let merged_params = merge_subsuper_params(&pairs, best_idx);
    let (params, _) = parse_parameter_list(&merged_params)?;

    let rest = if after.starts_with(';') { &after[1..] } else { after };
    Ok((id, name.clone(), params, rest))
}

/// Merge parameter texts from pairs[0..=best_idx] into a single comma-separated list.
/// Skips empty/trivial params (e.g. `()`, `''`, `*`, `$`).
fn merge_subsuper_params(pairs: &[(String, String)], _best_idx: usize) -> String {
    let mut parts = Vec::new();
    for i in 0..pairs.len() {
        let pd = &pairs[i].1;
        let trimmed = pd.trim();
        if !trimmed.is_empty() && trimmed != "''" && trimmed != "*" && trimmed != "$" {
            parts.push(trimmed.to_string());
        }
    }
    parts.join(",")
}

/// Extract all KEYWORD(...) pairs from subsuper inner text.
fn extract_keyword_param_pairs(inner: &str) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
    let mut rest = inner.trim();
    while !rest.is_empty() {
        let end = rest.find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        if end == 0 { break; }
        let keyword = rest[..end].to_string();
        rest = rest[end..].trim_start();
        if !rest.starts_with('(') { break; }
        let inner_start = 1usize;
        let close = find_matching_close(&rest[inner_start..])
            .map(|p| p + inner_start)
            .unwrap_or(rest.len());
        let params = rest[inner_start..close].to_string();
        pairs.push((keyword, params));
        rest = rest[close + 1..].trim_start();
    }
    Ok(pairs)
}

fn find_matching_close(input: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in input.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                if depth == 0 { return Some(i); }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// Finish parsing entity: name + (params) + ;
fn finish_entity<'a>(
    id: u64, rest: &'a str,
) -> Result<(u64, String, StepValue, &'a str), String> {
    let (name, rest) = parse_keyword(rest)?;

    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return Err(format!("expected '(' after entity name '{}', got: {}", name, &rest[..rest.len().min(20)]));
    }
    let rest = &rest[1..];

    let (params, rest) = parse_parameter_list(rest)?;

    let rest = rest.trim_start();
    if !rest.starts_with(')') {
        return Err(format!("expected ')' closing params, got: {}", &rest[..rest.len().min(20)]));
    }
    let rest = &rest[1..];
    let rest = rest.trim_start();
    let rest = if rest.starts_with(';') { &rest[1..] } else { rest };

    Ok((id, name, params, rest))
}

fn parse_parameter_list(input: &str) -> Result<(StepValue, &str), String> {
    let mut params = Vec::new();
    let mut rest = input;
    loop {
        rest = rest.trim_start();
        if rest.is_empty() || rest.starts_with(')') {
            break;
        }
        if !params.is_empty() {
            if rest.starts_with(',') {
                rest = &rest[1..];
            } else {
                break;
            }
        }
        let (param, new_rest) = parse_parameter(rest)?;
        params.push(param);
        rest = new_rest;
    }
    Ok((StepValue::List(params), rest))
}

fn parse_parameter(input: &str) -> Result<(StepValue, &str), String> {
    let rest = input.trim_start();
    if rest.is_empty() {
        return Err("unexpected end of input".into());
    }

    // Reference: #123
    if rest.starts_with('#') {
        let rest = &rest[1..];
        if let Some((id, rest)) = parse_u64(rest) {
            return Ok((StepValue::Ref(id), rest));
        }
    }

    // Omitted: $ or *
    if rest.starts_with('$') {
        return Ok((StepValue::Omitted, &rest[1..]));
    }
    if rest.starts_with('*') {
        return Ok((StepValue::Omitted, &rest[1..]));
    }

    // Enum: starts with .
    if rest.starts_with('.') {
        let (enum_val, rest) = parse_enum(rest)?;
        return Ok((StepValue::Enum(enum_val), rest));
    }

    // Nested list: (...)
    if rest.starts_with('(') {
        let (val, rest) = parse_parameter_list(&rest[1..])?;
        let rest = rest.trim_start();
        let rest = if rest.starts_with(')') { &rest[1..] } else { rest };
        return Ok((val, rest));
    }

    // String: '...' (with '' escape for embedded quote)
    if rest.starts_with('\'') {
        let (s, rest) = parse_string(&rest[1..])?;
        return Ok((StepValue::String(s), rest));
    }

    // Integer, real, or typed parameter
    parse_number_or_typed(rest)
}

fn parse_number_or_typed(input: &str) -> Result<(StepValue, &str), String> {
    let end = input.find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.' && c != '-' && c != '+')
        .unwrap_or(input.len());
    let token = &input[..end];
    let rest = &input[end..];

    if token.is_empty() {
        return Err("unexpected empty token".into());
    }

    let rest_trimmed = rest.trim_start();
    // Typed parameter: NAME(...)
    if rest_trimmed.starts_with('(') && token.chars().all(|c| c.is_ascii_uppercase() || c == '_') {
        let inner = &rest_trimmed[1..];
        let (inner_params, inner_rest) = parse_parameter_list(inner)?;
        let inner_rest = inner_rest.trim_start();
        if !inner_rest.starts_with(')') {
            return Err(format!("unclosed typed parameter '{}'", token));
        }
        return Ok((StepValue::Typed(token.to_string(), Box::new(inner_params)), &inner_rest[1..]));
    }

    // Integer
    if let Ok(v) = token.parse::<i64>() {
        return Ok((StepValue::Integer(v), rest));
    }
    // Real
    if let Ok(v) = token.parse::<f64>() {
        return Ok((StepValue::Real(v), rest));
    }

    Err(format!("unrecognized token: '{}'", token))
}

fn parse_u64(input: &str) -> Option<(u64, &str)> {
    let end = input.find(|c: char| !c.is_ascii_digit()).unwrap_or(input.len());
    if end == 0 { return None; }
    let n = input[..end].parse().ok()?;
    Some((n, &input[end..]))
}

fn parse_keyword(input: &str) -> Result<(String, &str), String> {
    let rest = input.trim_start();
    let end = rest.find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(rest.len());
    if end == 0 {
        return Err("expected keyword".into());
    }
    Ok((rest[..end].to_string(), &rest[end..]))
}

/// Streaming parser: parse STEP file incrementally to reduce memory usage.
/// Note: does not parse the HEADER section; `Exchange.header` will be `None`.
/// Reads the file line by line, skipping HEADER, then parses DATA section entities.
/// Properly handles multi-line entities by buffering until complete.
pub fn parse_exchange_streaming<R: BufRead>(
    reader: &mut R,
) -> Result<Exchange, String> {
    let mut entities = EntityIndex::new();
    let mut line_buf = String::new();
    let mut entity_buf = String::new();
    let mut in_data = false;
    let mut paren_depth: i32 = 0;
    let mut line_num: u64 = 0;
    let mut current_entity_id: u64 = 0;

    loop {
        line_buf.clear();
        let bytes_read = reader.read_line(&mut line_buf)
            .map_err(|e| format!("read error at line {}: {}", line_num, e))?;
        line_num += 1;

        // EOF
        if bytes_read == 0 {
            // If we have an incomplete entity buffered, try to parse it
            if paren_depth > 0 && !entity_buf.is_empty() {
                log::warn!(
                    "EOF reached with incomplete entity (depth={}), skipping entity #{}",
                    paren_depth, current_entity_id
                );
            }
            break;
        }

        let line = line_buf.trim();

        // Look for DATA section start
        if !in_data {
            if line.contains("DATA;") {
                in_data = true;
            }
            continue;
        }

        // End of DATA section
        if line.starts_with("ENDSEC;") || line.starts_with("END-ISO") {
            // Flush any buffered entity
            if paren_depth > 0 && !entity_buf.is_empty() {
                match parse_entity(&entity_buf) {
                    Ok((eid, ename, eparams, _)) => {
                        entities.insert(eid, EntityRecord {
                            name: ename,
                            params: eparams,
                            entity_type: EntityType::Unknown,
                        });
                    }
                    Err(e) => {
                        log::warn!("STEP stream: failed to parse buffered entity: {}", e);
                    }
                }
            }
            break;
        }

        // Skip comments
        if line.starts_with("/*") {
            // Multi-line comment support
            if !line.contains("*/") {
                // Skip until comment ends
                loop {
                    line_buf.clear();
                    match reader.read_line(&mut line_buf) {
                        Ok(0) => break, // EOF
                        Ok(_) => {}
                        Err(e) => return Err(format!("read error: {}", e)),
                    }
                    line_num += 1;
                    if line_buf.contains("*/") {
                        break;
                    }
                }
            }
            continue;
        }

        // Track parenthesis depth
        for c in line.chars() {
            match c {
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                _ => {}
            }
        }

        // Append to entity buffer (preserve newlines for subsuper multi-line format)
        if !entity_buf.is_empty() {
            entity_buf.push('\n');
        }
        entity_buf.push_str(line);

        // Check if entity is complete (depth back to 0 and ends with semicolon)
        if paren_depth == 0 && line.ends_with(';') {
            // Try to parse the complete entity
            match parse_entity(&entity_buf) {
                Ok((eid, ename, eparams, _)) => {
                    current_entity_id = eid;
                    log::debug!("Parsed entity #{} = {}", eid, ename);
                    entities.insert(eid, EntityRecord {
                        name: ename,
                        params: eparams,
                        entity_type: EntityType::Unknown,
                    });
                }
                Err(e) => {
                    log::warn!("STEP stream parse error at line {}: {}", line_num, e);
                    // Try to recover by finding the next entity start
                    if let Some(next_hash) = entity_buf[1..].find("#") {
                        let remaining = &entity_buf[next_hash..];
                        entity_buf = remaining.to_string();
                        paren_depth = 0;
                        // Attempt to continue parsing from remaining content
                        match parse_entity(&entity_buf) {
                            Ok((eid, ename, eparams, _)) => {
                                entities.insert(eid, EntityRecord {
                                    name: ename,
                                    params: eparams,
                                    entity_type: EntityType::Unknown,
                                });
                                entity_buf.clear();
                            }
                            Err(_) => {}
                        }
                    }
                }
            }
            entity_buf.clear();
        } else if paren_depth < 0 {
            // Unmatched closing paren - try to recover
            log::warn!("STEP stream: unmatched ')' at line {}, recovering", line_num);
            paren_depth = 0;
            entity_buf.clear();
        }
    }

    // Tag each entity with its EntityType
    for record in entities.values_mut() {
        record.entity_type = EntityType::from_name(&record.name);
    }

    log::info!("STEP streaming parse complete: {} entities", entities.len());
    Ok(Exchange { header: None, entities })
}

/// Parse a STEP file from disk using streaming I/O (reduces memory vs. read_to_string).
/// For very large files (>100MB), prefer this over `parse_step`.
pub fn parse_step_from_file(path: &Path) -> Result<Exchange, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {}", path.display(), e))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file); // 1MB read buffer
    parse_exchange_streaming(&mut reader)
}

// ---------------------------------------------------------------------------
// Original functions below (unchanged)
// ---------------------------------------------------------------------------

fn parse_string(input: &str) -> Result<(String, &str), String> {
    let mut s = String::new();
    let mut chars = input.char_indices();
    loop {
        match chars.next() {
            Some((_, '\'')) => {
                // Check for escaped quote ''
                if let Some((_, '\'')) = chars.clone().next() {
                    chars.next();
                    s.push('\'');
                } else {
                    let pos = chars.clone().next().map(|(i, _)| i).unwrap_or(input.len());
                    return Ok((s, &input[pos..]));
                }
            }
            Some((_, c)) => s.push(c),
            None => return Err("unterminated string".into()),
        }
    }
}

fn parse_enum(input: &str) -> Result<(String, &str), String> {
    let rest = &input[1..]; // skip leading .
    let end = rest.find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.')
        .unwrap_or(rest.len());
    if end == 0 {
        return Err("expected enum value".into());
    }
    Ok((format!(".{}", &rest[..end]), &rest[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_entity() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = CARTESIAN_POINT('', (0.0, 1.0, 2.0));\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        assert_eq!(ex.entities.len(), 1);
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "CARTESIAN_POINT");
    }

    #[test]
    fn test_parse_nested_list() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = ENTITY(#2, (3.0, 4.0));\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        let e = ex.entities.get(&1).unwrap();
        if let StepValue::List(params) = &e.params {
            assert!(matches!(params[0], StepValue::Ref(2)));
            assert!(matches!(params[1], StepValue::List(_)));
        } else { panic!("expected List"); }
    }

    #[test]
    fn test_parse_typed_param() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = ENTITY(LENGTH_MEASURE(0.001));\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        let e = ex.entities.get(&1).unwrap();
        if let StepValue::List(params) = &e.params {
            assert!(matches!(&params[0], StepValue::Typed(name, _) if name == "LENGTH_MEASURE"));
        } else { panic!("expected List"); }
    }

    #[test]
    fn test_parse_omitted() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = ENTITY($, *);\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        let e = ex.entities.get(&1).unwrap();
        if let StepValue::List(params) = &e.params {
            assert!(matches!(params[0], StepValue::Omitted));
            assert!(matches!(params[1], StepValue::Omitted));
        } else { panic!(); }
    }

    #[test]
    fn test_parse_empty_data() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        assert!(ex.entities.is_empty());
    }

    #[test]
    fn test_parse_escaped_string() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = ENTITY('it''s');\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        let e = ex.entities.get(&1).unwrap();
        if let StepValue::List(params) = &e.params {
            assert!(matches!(&params[0], StepValue::String(s) if s == "it's"));
        }
    }

    #[test]
    fn test_parse_subsuper_inside() {
        // Format 2: entity name + params inside the outer parens
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = (LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.MILLI.,.METRE.));\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "SI_UNIT");
    }

    #[test]
    fn test_parse_subsuper_outside() {
        // Format 1: entity name follows the outer parens
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = (SUPER1()SUPER2())ENTITY(1.0, 2.0);\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "ENTITY");
    }

    #[test]
    fn test_parse_subsuper_with_params() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.MILLI.,.METRE.));\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "SI_UNIT");
        if let StepValue::List(params) = &e.params {
            assert_eq!(params.len(), 3);
            assert!(matches!(params[0], StepValue::Omitted));
        } else { panic!("expected List params"); }
    }

    #[test]
    fn test_parse_subsuper_multiline_ap242() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = (\nBOUNDED_CURVE()\nB_SPLINE_CURVE(2,(#10,#11),.UNSPECIFIED.,.F.,.F.)\nB_SPLINE_CURVE_WITH_KNOTS((3,2,3),(0.625,0.667,0.75),.UNSPECIFIED.)\nCURVE()\nGEOMETRIC_REPRESENTATION_ITEM()\nRATIONAL_B_SPLINE_CURVE((0.933,0.933,1.))\nREPRESENTATION_ITEM('')\n);\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        let e = ex.entities.get(&1).unwrap();
        // Should pick B_SPLINE_CURVE_WITH_KNOTS as the entity name
        assert_eq!(e.name, "B_SPLINE_CURVE_WITH_KNOTS");
        // Params should include degree=2 from B_SPLINE_CURVE merged with knot data
        if let StepValue::List(params) = &e.params {
            // First param should be degree=2 (from B_SPLINE_CURVE)
            assert!(matches!(params.first(), Some(StepValue::Integer(2))));
        } else { panic!("expected List params"); }
    }

    #[test]
    fn test_parse_error_recovery_skips_bad_entity() {
        let ex = parse_exchange(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = CARTESIAN_POINT('', (0.0, 1.0, 2.0));\n#2 = BAD_ENTITY(no closing paren\n#3 = CARTESIAN_POINT('', (3.0, 4.0, 5.0));\nENDSEC;\nEND-ISO-10303-21;\n",
        ).unwrap();
        // Should recover and parse #1 and #3, skipping #2
        assert!(ex.entities.contains_key(&1), "should parse #1");
        assert!(!ex.entities.contains_key(&2), "should skip broken #2");
        assert!(ex.entities.contains_key(&3), "should parse #3");
    }
}
