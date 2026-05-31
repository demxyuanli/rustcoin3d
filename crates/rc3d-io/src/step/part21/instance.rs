//! Parse `#id = KEYWORD(params);` and complex (subsuper) instances.

use crate::step::model::{ComplexMapping, Record, StepInstance};

use super::error::{ParseError, ParseResult};
use super::params::parse_param_list_at;
use super::token::Span;

/// Parse one entity instance; returns remaining text after `;` if present.
pub fn parse_instance<'a>(input: &'a str) -> ParseResult<(StepInstance, &'a str)> {
    parse_instance_at(input, Span { line: 1, col: 1 })
}

/// Parse one entity with errors anchored at `at` (file-absolute line/col).
pub fn parse_instance_at<'a>(input: &'a str, at: Span) -> ParseResult<(StepInstance, &'a str)> {
    let rest = input.trim_start();
    if !rest.starts_with('#') {
        let snippet: String = rest.chars().take(20).collect();
        return Err(ParseError::new(
            format!("expected '#', got '{}'", snippet),
            at,
        ));
    }
    let rest = &rest[1..];
    let (id, rest) = parse_u64(rest).ok_or_else(|| ParseError::new("invalid entity id", at))?;
    let rest = rest.trim_start();
    if !rest.starts_with('=') {
        return Err(ParseError::new("expected '=' after entity id", at));
    }
    let rest = rest[1..].trim_start();

    if rest.starts_with('(') {
        let (close_pos, inner) = find_outer_paren(rest);
        let after = rest[close_pos..].trim_start();
        let after_starts_keyword = after
            .chars()
            .next()
            .map_or(false, |c| c.is_ascii_alphanumeric() || c == '_');
        if !after_starts_keyword {
            return parse_subsuper_internal(id, inner, after, at);
        }
        let (inst, rest) = parse_subsuper_external_prefix(id, inner, after, at)?;
        return Ok((inst, rest));
    }

    parse_simple_instance(id, rest, at)
}

fn parse_simple_instance<'a>(
    id: u64,
    rest: &'a str,
    at: Span,
) -> ParseResult<(StepInstance, &'a str)> {
    let (keyword, rest) = parse_keyword_at(rest, at)?;
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return Err(ParseError::new(
            format!("expected '(' after '{}'", keyword),
            at,
        ));
    }
    let (params, rest) = parse_param_list_at(&rest[1..], at)?;
    let rest = rest.trim_start();
    if !rest.starts_with(')') {
        return Err(ParseError::new("expected ')' after parameters", at));
    }
    let mut rest = &rest[1..];
    rest = rest.trim_start();
    if rest.starts_with(';') {
        rest = &rest[1..];
    }
    Ok((
        StepInstance {
            id,
            records: vec![Record { keyword, params }],
            mapping: ComplexMapping::Simple,
        },
        rest,
    ))
}

fn parse_subsuper_external_prefix<'a>(
    id: u64,
    inner: &str,
    after: &'a str,
    at: Span,
) -> ParseResult<(StepInstance, &'a str)> {
    let mut records = extract_keyword_param_pairs(inner, at)?;
    let (keyword, rest) = parse_keyword_at(after, at)?;
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return Err(ParseError::new(
            format!("expected '(' after '{}'", keyword),
            at,
        ));
    }
    let (params, rest) = parse_param_list_at(&rest[1..], at)?;
    let rest = rest.trim_start();
    if !rest.starts_with(')') {
        return Err(ParseError::new("expected ')' after entity parameters", at));
    }
    records.push(Record { keyword, params });
    let mut rest = &rest[1..];
    rest = rest.trim_start();
    if rest.starts_with(';') {
        rest = &rest[1..];
    }
    // Keep original STEP order — the last record is the most-derived subtype.
    // Alphabetical sorting (removed) causes supertypes like "SURFACE" to be
    // selected as the primary record instead of concrete types like
    // "B_SPLINE_SURFACE_WITH_KNOTS".
    let leaf_index = records.len().saturating_sub(1);
    Ok((
        StepInstance {
            id,
            records,
            mapping: ComplexMapping::Internal { leaf_index },
        },
        rest,
    ))
}

fn parse_subsuper_internal<'a>(
    id: u64,
    inner: &str,
    after: &'a str,
    at: Span,
) -> ParseResult<(StepInstance, &'a str)> {
    let records = extract_keyword_param_pairs(inner, at)?;
    if records.is_empty() {
        return Err(ParseError::new(
            "no keyword+params pairs in complex entity",
            at,
        ));
    }
    let leaf_index = records.len().saturating_sub(1);
    let mut rest = after;
    if rest.starts_with(';') {
        rest = &rest[1..];
    }
    Ok((
        StepInstance {
            id,
            records,
            mapping: ComplexMapping::Internal { leaf_index },
        },
        rest,
    ))
}

fn extract_keyword_param_pairs(inner: &str, at: Span) -> ParseResult<Vec<Record>> {
    let mut pairs = Vec::new();
    let mut rest = inner.trim();
    while !rest.is_empty() {
        let (keyword, new_rest) = parse_keyword_at(rest, at)?;
        rest = new_rest.trim_start();
        if !rest.starts_with('(') {
            break;
        }
        let (params, new_rest) = parse_param_list_at(&rest[1..], at)?;
        rest = new_rest.trim_start();
        if !rest.starts_with(')') {
            return Err(ParseError::new(
                format!("expected ')' after '{}'", keyword),
                at,
            ));
        }
        pairs.push(Record { keyword, params });
        rest = rest[1..].trim_start();
    }
    Ok(pairs)
}

fn find_outer_paren(input: &str) -> (usize, &str) {
    let mut depth = 1i32;
    let mut close_pos = 1usize;
    for (i, c) in input.char_indices().skip(1) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close_pos = i + c.len_utf8();
                    break;
                }
            }
            _ => {}
        }
    }
    (close_pos, &input[1..close_pos.saturating_sub(1)])
}

fn parse_u64(input: &str) -> Option<(u64, &str)> {
    let end = input
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(input.len());
    if end == 0 {
        return None;
    }
    let n = input[..end].parse().ok()?;
    Some((n, &input[end..]))
}

fn parse_keyword_at(input: &str, at: Span) -> ParseResult<(String, &str)> {
    let rest = input.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(rest.len());
    if end == 0 {
        return Err(ParseError::new("expected keyword", at));
    }
    Ok((rest[..end].to_string(), &rest[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_cartesian_point() {
        let (inst, rest) =
            parse_instance("#1 = CARTESIAN_POINT('', (0.0, 1.0, 2.0));").unwrap();
        assert_eq!(inst.id, 1);
        assert_eq!(inst.records.len(), 1);
        assert_eq!(inst.records[0].keyword, "CARTESIAN_POINT");
        assert!(rest.trim().is_empty());
        assert_eq!(inst.mapping, ComplexMapping::Simple);
    }

    #[test]
    fn parse_instance_error_has_line_col() {
        let at = Span { line: 5, col: 1 };
        let err =
            parse_instance_at("#1 = CARTESIAN_POINT('', (0.0, 1.0, 2.0", at).unwrap_err();
        assert_eq!(err.span.line, 5);
    }

    #[test]
    fn parse_ap242_multiline_subsuper() {
        let input = "#1 = (\n\
BOUNDED_CURVE()\n\
B_SPLINE_CURVE(2,(#10,#11),.UNSPECIFIED.,.F.,.F.)\n\
B_SPLINE_CURVE_WITH_KNOTS((3,2,3),(0.625,0.667,0.75),.UNSPECIFIED.)\n\
CURVE()\n\
GEOMETRIC_REPRESENTATION_ITEM()\n\
RATIONAL_B_SPLINE_CURVE((0.933,0.933,1.))\n\
REPRESENTATION_ITEM('')\n\
);\n";
        let (inst, _) = parse_instance(input).unwrap();
        assert!(inst.records.len() >= 5);
        assert_eq!(inst.mapping, ComplexMapping::Internal { leaf_index: 6 });
        let keywords: Vec<_> = inst.records.iter().map(|r| r.keyword.as_str()).collect();
        assert!(keywords.contains(&"B_SPLINE_CURVE_WITH_KNOTS"));
        assert!(inst.records.iter().any(|r| r.keyword == "B_SPLINE_CURVE_WITH_KNOTS"
            && r.params.as_list().map(|l| l.len()).unwrap_or(0) >= 2));
    }

    #[test]
    fn subsuper_internal_three_level() {
        let input = "#1 = (CHILD(1.0) PARENT());";
        let (inst, _) = parse_instance(input).unwrap();
        assert_eq!(inst.id, 1);
        assert!(inst.records.len() >= 2);
        let keywords: Vec<&str> = inst.records.iter().map(|r| r.keyword.as_str()).collect();
        assert!(keywords.contains(&"CHILD"));
        assert!(keywords.contains(&"PARENT"));
    }

    #[test]
    fn subsuper_internal_multiline() {
        let input = "#1 = (\nBOUNDED_SURFACE()\nB_SPLINE_SURFACE(2,3,(#10))\n);";
        let (inst, _) = parse_instance(input).unwrap();
        assert_eq!(inst.id, 1);
        assert!(inst.records.len() >= 2);
        let keywords: Vec<&str> = inst.records.iter().map(|r| r.keyword.as_str()).collect();
        assert!(keywords.contains(&"B_SPLINE_SURFACE"));
        assert!(keywords.contains(&"BOUNDED_SURFACE"));
    }

    #[test]
    fn subsuper_external_with_keyword_after_paren() {
        let input = "#1 = (WRAPPER())REAL_ENTITY(1.0, 2.0);";
        let (inst, _) = parse_instance(input).unwrap();
        assert_eq!(inst.id, 1);
        let keywords: Vec<&str> = inst.records.iter().map(|r| r.keyword.as_str()).collect();
        assert!(keywords.contains(&"REAL_ENTITY"));
    }
}
