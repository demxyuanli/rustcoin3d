//! Read a full ISO 10303-21 exchange file into `model::Exchange`.

use std::io::{BufRead, BufReader};

use crate::step::header;
use crate::step::model::conformance::from_header;
use crate::step::model::{DataSection, Exchange};

use super::error::{ParseError, ParseResult};
use super::instance::parse_instance_at;
use super::skipped::SkippedEntity;
use super::span_util::line_col_at_offset;
use super::token::Span;

const IO_SPAN: Span = Span { line: 1, col: 1 };

/// Parse an in-memory STEP exchange file (preview: skip malformed instances).
pub fn read_exchange(input: &str) -> ParseResult<Exchange> {
    read_exchange_with_recovery(input, true).map(|(ex, _)| ex)
}

/// Parse exchange; when `recover_skipped` is false, fail on first bad instance.
pub fn read_exchange_with_recovery(
    input: &str,
    recover_skipped: bool,
) -> ParseResult<(Exchange, Vec<SkippedEntity>)> {
    let mut skipped = Vec::new();
    let exchange = read_exchange_inner(input, recover_skipped, &mut skipped)?;
    Ok((exchange, skipped))
}

fn span_at(file: &str, suffix: &str) -> Span {
    let offset = file.len().saturating_sub(suffix.len());
    line_col_at_offset(file, offset, 1)
}

pub(crate) fn read_exchange_inner(
    input: &str,
    recover_skipped: bool,
    skipped: &mut Vec<SkippedEntity>,
) -> ParseResult<Exchange> {
    let file = input.trim();
    if !file.starts_with("ISO-10303-21;") {
        return Err(ParseError::new("missing ISO-10303-21 header", IO_SPAN));
    }
    let mut rest = &file["ISO-10303-21;".len()..];
    rest = rest.trim_start();

    let (header, mut rest) = match header::parse_header(rest) {
        Some(h) => h,
        None => (header::HeaderInfo::default(), rest),
    };

    let implementation_level = from_header(&header);
    let mut exchange = Exchange {
        header: Some(header),
        implementation_level,
        ..Default::default()
    };

    loop {
        rest = rest.trim_start();
        if rest.is_empty() || rest.starts_with("END-ISO-10303-21;") {
            break;
        }
        if rest.starts_with("ANCHOR;") {
            let (body, after) = take_section(file, rest, "ANCHOR;")?;
            exchange.anchor_raw = Some(body.to_string());
            rest = after;
            continue;
        }
        if rest.starts_with("REFERENCE;") {
            let (body, after) = take_section(file, rest, "REFERENCE;")?;
            exchange.reference_raw = Some(body.to_string());
            rest = after;
            continue;
        }
        if rest.starts_with("DATA;") {
            let (section, after) = parse_data_section(file, rest, recover_skipped, skipped)?;
            exchange.data_sections.push(section);
            rest = after;
            continue;
        }
        let snippet: String = rest.chars().take(32).collect();
        return Err(ParseError::new(
            format!("unexpected section marker: '{}'", snippet),
            span_at(file, rest),
        ));
    }

    rest = rest.trim_start();
    if rest.starts_with("END-ISO-10303-21;") {
        rest = &rest["END-ISO-10303-21;".len()..];
    }
    rest = rest.trim_start();
    while rest.starts_with("SIGNATURE") {
        let (sig, after) = take_signature_section(file, rest)?;
        exchange.signatures.push(sig);
        rest = after;
    }

    Ok(exchange)
}

/// Read exchange from a buffered file (loads into memory; same parse path as `read_exchange`).
pub fn read_exchange_buffered<R: BufRead>(reader: &mut R) -> ParseResult<Exchange> {
    read_exchange_buffered_with_recovery(reader, true).map(|(ex, _)| ex)
}

pub fn read_exchange_buffered_with_recovery<R: BufRead>(
    reader: &mut R,
    recover_skipped: bool,
) -> ParseResult<(Exchange, Vec<SkippedEntity>)> {
    super::stream::read_exchange_stream(reader, recover_skipped)
}

/// Open path and parse via buffered reader.
pub fn read_exchange_file(path: &std::path::Path) -> ParseResult<Exchange> {
    read_exchange_file_with_recovery(path, true).map(|(ex, _)| ex)
}

pub fn read_exchange_file_with_recovery(
    path: &std::path::Path,
    recover_skipped: bool,
) -> ParseResult<(Exchange, Vec<SkippedEntity>)> {
    let file = std::fs::File::open(path)
        .map_err(|e| ParseError::new(format!("open {}: {e}", path.display()), IO_SPAN))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    read_exchange_buffered_with_recovery(&mut reader, recover_skipped)
}

fn take_section<'a>(
    file: &str,
    input: &'a str,
    marker: &str,
) -> ParseResult<(&'a str, &'a str)> {
    let body_start = marker.len();
    let rest = &input[body_start..];
    let end = rest
        .find("ENDSEC;")
        .ok_or_else(|| ParseError::new("missing ENDSEC;", span_at(file, input)))?;
    Ok((&rest[..end], &rest[end + "ENDSEC;".len()..]))
}

fn take_signature_section<'a>(file: &str, input: &'a str) -> ParseResult<(String, &'a str)> {
    let rest = input
        .strip_prefix("SIGNATURE")
        .ok_or_else(|| ParseError::new("expected SIGNATURE", span_at(file, input)))?;
    let end = rest
        .find("ENDSEC;")
        .ok_or_else(|| ParseError::new("missing SIGNATURE ENDSEC;", span_at(file, input)))?;
    Ok((rest[..end].trim().to_string(), &rest[end + "ENDSEC;".len()..]))
}

pub(crate) fn peek_entity_id(chunk: &str) -> Option<u64> {
    let s = chunk.trim_start();
    if !s.starts_with('#') {
        return None;
    }
    let digits = &s[1..];
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    if end == 0 {
        return None;
    }
    digits[..end].parse().ok()
}

pub(crate) fn entity_snippet(chunk: &str) -> String {
    chunk.chars().take(80).collect()
}

fn parse_data_section<'a>(
    file: &str,
    input: &'a str,
    recover_skipped: bool,
    skipped: &mut Vec<SkippedEntity>,
) -> ParseResult<(DataSection, &'a str)> {
    let rest = &input["DATA;".len()..];
    let end = rest
        .find("ENDSEC;")
        .ok_or_else(|| ParseError::new("DATA section missing ENDSEC;", span_at(file, input)))?;
    let body = &rest[..end];
    let after = &rest[end + "ENDSEC;".len()..];

    let mut section = DataSection::default();
    let mut chunk = body;
    loop {
        chunk = chunk.trim_start();
        if chunk.is_empty() {
            break;
        }
        if chunk.starts_with("/*") {
            if let Some(end_comment) = chunk.find("*/") {
                chunk = &chunk[end_comment + 2..];
                continue;
            }
            break;
        }
        let body_start = file.len().saturating_sub(input.len()) + "DATA;".len();
        let entity_offset = body_start + body.len() - chunk.len();
        let at = line_col_at_offset(file, entity_offset, 1);
        match parse_instance_at(chunk, at) {
            Ok((inst, new_chunk)) => {
                section.instances.push(inst);
                chunk = new_chunk;
            }
            Err(e) => {
                if !recover_skipped {
                    return Err(e);
                }
                if let Some(next) = chunk[1..].find('#') {
                    skipped.push(SkippedEntity {
                        message: e.message.clone(),
                        span: e.span,
                        id_hint: peek_entity_id(chunk),
                        snippet: entity_snippet(chunk),
                    });
                    log::warn!("STEP part21: skip entity: {}", e);
                    chunk = &chunk[next + 1..];
                    continue;
                }
                return Err(e);
            }
        }
    }
    Ok((section, after))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_two_data_sections() {
        let input = r#"ISO-10303-21;
HEADER;
FILE_SCHEMA(('AP203'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
ENDSEC;
DATA;
#2=CARTESIAN_POINT('',(1.,0.,0.));
ENDSEC;
END-ISO-10303-21;"#;
        let ex = read_exchange(input).unwrap();
        assert_eq!(ex.data_sections.len(), 2);
        assert_eq!(ex.instance_count(), 2);
    }

    #[test]
    fn parse_error_reports_non_default_line() {
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 1.0, 2.0
ENDSEC;
END-ISO-10303-21;"#;
        let err = read_exchange_with_recovery(input, false).unwrap_err();
        assert!(
            err.span.line >= 4,
            "expected DATA line, got line {} col {}",
            err.span.line,
            err.span.col
        );
    }

    #[test]
    fn recovery_records_skipped_id() {
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (0.0, 1.0, 2.0
#3 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
ENDSEC;
END-ISO-10303-21;"#;
        let (ex, skipped) = read_exchange_with_recovery(input, true).unwrap();
        assert_eq!(ex.instance_count(), 2);
        assert!(!skipped.is_empty());
        assert!(skipped.iter().any(|s| s.id_hint == Some(2)));
    }
}
