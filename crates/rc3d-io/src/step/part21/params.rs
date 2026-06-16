//! Parse STEP parameter lists into `StepValue` (strict comma rules).

use rc3d_core::math::Real;
use crate::step::value::StepValue;

use super::error::{ParseError, ParseResult};
use super::token::Span;

/// Parse a comma-separated parameter list; input must not include outer `(` `)`.
pub fn parse_param_list(input: &str) -> ParseResult<(StepValue, &str)> {
    parse_param_list_at(input, Span { line: 1, col: 1 })
}

/// Parse a parameter list with errors anchored at `base`.
pub fn parse_param_list_at(input: &str, base: Span) -> ParseResult<(StepValue, &str)> {
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
                rest = rest.trim_start();
            } else if rest.starts_with(')') {
                break;
            } else {
                let snippet: String = rest.chars().take(24).collect();
                return Err(ParseError::new(
                    format!("expected ',' between parameters, got '{}'", snippet),
                    base,
                ));
            }
        }
        let (param, new_rest) = parse_param_at(rest, base)?;
        params.push(param);
        rest = new_rest;
    }
    Ok((StepValue::List(params), rest))
}

/// Parse one parameter value.
pub fn parse_param(input: &str) -> ParseResult<(StepValue, &str)> {
    parse_param_at(input, Span { line: 1, col: 1 })
}

pub fn parse_param_at(input: &str, base: Span) -> ParseResult<(StepValue, &str)> {
    let rest = input.trim_start();
    if rest.is_empty() {
        return Err(ParseError::new("unexpected end of input", base));
    }

    if rest.starts_with('#') {
        let rest = &rest[1..];
        let (id, rest) = parse_u64(rest)
            .ok_or_else(|| ParseError::new("invalid entity reference", base))?;
        return Ok((StepValue::Ref(id), rest));
    }

    if rest.starts_with('$') || rest.starts_with('*') {
        return Ok((StepValue::Omitted, &rest[1..]));
    }

    if rest.starts_with('.') {
        let (enum_val, rest) = parse_enum_at(rest, base)?;
        return Ok((StepValue::Enum(enum_val), rest));
    }

    if rest.starts_with('(') {
        let (val, rest) = parse_param_list_at(&rest[1..], base)?;
        let rest = rest.trim_start();
        if !rest.starts_with(')') {
            let snippet: String = rest.chars().take(24).collect();
            return Err(ParseError::new(
                format!("expected ')' closing nested list, got '{}'", snippet),
                base,
            ));
        }
        return Ok((val, &rest[1..]));
    }

    if rest.starts_with('\'') {
        let (s, rest) = parse_string_at(&rest[1..], base)?;
        return Ok((StepValue::String(s), rest));
    }

    parse_number_or_typed_at(rest, base)
}

fn parse_number_or_typed_at(input: &str, base: Span) -> ParseResult<(StepValue, &str)> {
    let end = input
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.' && c != '-' && c != '+')
        .unwrap_or(input.len());
    let token = &input[..end];
    let rest = &input[end..];

    if token.is_empty() {
        return Err(ParseError::new("unexpected empty token", base));
    }

    let rest_trimmed = rest.trim_start();
    if rest_trimmed.starts_with('(')
        && token.chars().all(|c| c.is_ascii_uppercase() || c == '_')
    {
        let inner = &rest_trimmed[1..];
        let (inner_params, inner_rest) = parse_param_list_at(inner, base)?;
        let inner_rest = inner_rest.trim_start();
        if !inner_rest.starts_with(')') {
            return Err(ParseError::new(
                format!("unclosed typed parameter '{}'", token),
                base,
            ));
        }
        return Ok((
            StepValue::Typed(token.to_string(), Box::new(inner_params)),
            &inner_rest[1..],
        ));
    }

    if let Ok(v) = token.parse::<i64>() {
        return Ok((StepValue::Integer(v), rest));
    }
    if let Ok(v) = token.parse::<f64>() {
        return Ok((StepValue::Real(v), rest));
    }

    Err(ParseError::new(
        format!("unrecognized token: '{}'", token),
        base,
    ))
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

fn parse_enum_at(input: &str, base: Span) -> ParseResult<(String, &str)> {
    let rest = &input[1..];
    let end = rest
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.')
        .unwrap_or(rest.len());
    if end == 0 {
        return Err(ParseError::new("expected enum value", base));
    }
    Ok((format!(".{}", &rest[..end]), &rest[end..]))
}

fn parse_string_at(input: &str, base: Span) -> ParseResult<(String, &str)> {
    let mut s = String::new();
    let mut chars = input.char_indices();
    loop {
        match chars.next() {
            Some((_, '\'')) => {
                if chars.clone().next().map(|(_, c)| c) == Some('\'') {
                    chars.next();
                    s.push('\'');
                } else {
                    let pos = chars.next().map(|(i, _)| i).unwrap_or(input.len());
                    return Ok((s, &input[pos..]));
                }
            }
            Some((_, c)) => s.push(c),
            None => return Err(ParseError::new("unterminated string", base)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_params_nested() {
        let (val, rest) =
            parse_param_list("#2, (3.0, 4.0), LENGTH_MEASURE(0.001), $").unwrap();
        assert!(rest.trim_start().is_empty());
        let list = val.as_list().unwrap();
        assert_eq!(list.len(), 4);
        assert!(matches!(list[3], StepValue::Omitted));
    }

    #[test]
    fn missing_comma_is_error() {
        let err = parse_param_list("(1.0 2.0)").unwrap_err();
        assert!(err.message.contains("expected ','"));
    }

    #[test]
    fn parse_deeply_nested_param_list() {
        // 5 levels of nesting
        let input = "#1 = NEST((((((42))))));";
        let (inst, _) = crate::step::part21::instance::parse_instance(input).unwrap();
        assert_eq!(inst.id, 1);
        assert_eq!(inst.records[0].keyword, "NEST");
    }

    #[test]
    fn parse_typed_param_nested_value() {
        let (val, rest) = parse_param("LENGTH_MEASURE(0.001)").unwrap();
        assert!(rest.trim().is_empty());
        assert!(matches!(val, StepValue::Typed(name, _) if name == "LENGTH_MEASURE"));
    }

    #[test]
    fn parse_mixed_types_in_list() {
        let (val, rest) = parse_param_list("#1, 2.0, 'str', .T., $").unwrap();
        assert!(rest.trim().is_empty());
        let list = val.as_list().unwrap();
        assert_eq!(list.len(), 5);
        assert!(matches!(list[0], StepValue::Ref(1)));
        assert!(matches!(list[1], StepValue::Real(v) if (v - 2.0).abs() < 1e-10));
        assert!(matches!(&list[2], StepValue::String(s) if s == "str"));
        assert!(matches!(&list[3], StepValue::Enum(e) if e == ".T."));
        assert!(matches!(list[4], StepValue::Omitted));
    }

    #[test]
    fn parse_enum_variants() {
        for (input, expected) in &[
            (".T.", ".T."),
            (".F.", ".F."),
            (".UNSPECIFIED.", ".UNSPECIFIED."),
            (".MILLI.", ".MILLI."),
            (".METRE.", ".METRE."),
        ] {
            let (val, rest) = parse_param(input).unwrap();
            assert!(rest.trim().is_empty(), "rest not empty for {}", input);
            assert!(matches!(&val, StepValue::Enum(e) if e == expected),
                "expected {} got {:?}", expected, val);
        }
    }
}
