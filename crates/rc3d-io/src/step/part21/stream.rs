//! Incremental ISO 10303-21 exchange reader (no full-file `String`).

use std::io::BufRead;

use crate::step::header;
use crate::step::model::conformance::from_header;
use crate::step::model::{DataSection, Exchange};

use super::error::{ParseError, ParseResult};
use super::instance::parse_instance_at;
use super::read::{entity_snippet, peek_entity_id};
use super::skipped::SkippedEntity;
use super::span_util::line_col_at_offset;
use super::token::Span;

const READ_CHUNK: usize = 64 * 1024;
const IO_SPAN: Span = Span { line: 1, col: 1 };

/// Stream a STEP exchange from any `BufRead` source without loading the whole file into one `String`.
pub struct StreamExchangeReader<R> {
    reader: R,
    buf: String,
    eof: bool,
    /// Bytes consumed from the stream before `buf` (for span mapping).
    consumed: usize,
}

impl<R: BufRead> StreamExchangeReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: String::new(),
            eof: false,
            consumed: 0,
        }
    }

    pub fn read_exchange(&mut self) -> ParseResult<Exchange> {
        self.read_exchange_with_recovery(true).map(|(ex, _)| ex)
    }

    pub fn read_exchange_with_recovery(
        &mut self,
        recover_skipped: bool,
    ) -> ParseResult<(Exchange, Vec<SkippedEntity>)> {
        let mut skipped = Vec::new();
        let exchange = self.parse_exchange(recover_skipped, &mut skipped)?;
        Ok((exchange, skipped))
    }

    fn parse_exchange(
        &mut self,
        recover_skipped: bool,
        skipped: &mut Vec<SkippedEntity>,
    ) -> ParseResult<Exchange> {
        self.fill_until(14)?;
        if !self.buf.starts_with("ISO-10303-21;") {
            return Err(ParseError::new("missing ISO-10303-21 header", IO_SPAN));
        }
        self.drain(14)?;
        self.buf = self.buf.trim_start().to_string();

        self.fill_until(8)?;
        let header_end = self
            .buf
            .find("ENDSEC;")
            .ok_or_else(|| ParseError::new("HEADER missing ENDSEC;", IO_SPAN))?
            + "ENDSEC;".len();
        let header_slice = self.buf[..header_end].to_string();
        self.drain(header_end)?;

        let (header, _) = header::parse_header(&header_slice)
            .unwrap_or((header::HeaderInfo::default(), ""));
        let implementation_level = from_header(&header);
        let mut exchange = Exchange {
            header: Some(header),
            implementation_level,
            ..Default::default()
        };

        loop {
            self.fill_until(6)?;
            self.buf = self.buf.trim_start().to_string();
            if self.buf.is_empty() && self.eof {
                break;
            }
            if self.buf.starts_with("END-ISO-10303-21;") {
                self.drain("END-ISO-10303-21;".len())?;
                break;
            }
            if self.buf.starts_with("ANCHOR;") {
                let body = self.take_section_body("ANCHOR;")?;
                exchange.anchor_raw = Some(body);
                continue;
            }
            if self.buf.starts_with("REFERENCE;") {
                let body = self.take_section_body("REFERENCE;")?;
                exchange.reference_raw = Some(body);
                continue;
            }
            if self.buf.starts_with("DATA;") {
                let section = self.read_data_section(recover_skipped, skipped)?;
                exchange.data_sections.push(section);
                continue;
            }
            if self.buf.starts_with("SIGNATURE") {
                let sig = self.take_signature_body()?;
                exchange.signatures.push(sig);
                continue;
            }
            if self.eof {
                break;
            }
            let snippet: String = self.buf.chars().take(32).collect();
            return Err(ParseError::new(
                format!("unexpected section marker: '{}'", snippet),
                self.span_at_buf_start(),
            ));
        }

        self.buf = self.buf.trim_start().to_string();
        while self.buf.starts_with("SIGNATURE") {
            let sig = self.take_signature_body()?;
            exchange.signatures.push(sig);
            self.buf = self.buf.trim_start().to_string();
        }

        Ok(exchange)
    }

    fn take_section_body(&mut self, marker: &str) -> ParseResult<String> {
        self.drain(marker.len())?;
        loop {
            if let Some(end) = self.buf.find("ENDSEC;") {
                let body = self.buf[..end].to_string();
                self.drain(end + "ENDSEC;".len())?;
                return Ok(body);
            }
            if self.eof {
                return Err(ParseError::new("missing ENDSEC;", IO_SPAN));
            }
            self.fill_more()?;
        }
    }

    fn take_signature_body(&mut self) -> ParseResult<String> {
        self.buf = self.buf.trim_start().to_string();
        if !self.buf.starts_with("SIGNATURE") {
            return Err(ParseError::new("expected SIGNATURE", IO_SPAN));
        }
        self.drain("SIGNATURE".len())?;
        loop {
            if let Some(end) = self.buf.find("ENDSEC;") {
                let body = self.buf[..end].trim().to_string();
                self.drain(end + "ENDSEC;".len())?;
                return Ok(body);
            }
            if self.eof {
                return Err(ParseError::new("missing SIGNATURE ENDSEC;", IO_SPAN));
            }
            self.fill_more()?;
        }
    }

    fn read_data_section(
        &mut self,
        recover_skipped: bool,
        skipped: &mut Vec<SkippedEntity>,
    ) -> ParseResult<DataSection> {
        self.drain("DATA;".len())?;
        let data_body_start = self.consumed;
        let mut section = DataSection::default();

        loop {
            self.fill_until(2)?;
            self.buf = self.buf.trim_start().to_string();
            if self.buf.starts_with("ENDSEC;") {
                self.drain("ENDSEC;".len())?;
                break;
            }
            if self.buf.starts_with("/*") {
                if let Some(end) = self.buf.find("*/") {
                    self.drain(end + 2)?;
                    continue;
                }
                if !self.eof {
                    self.fill_more()?;
                    continue;
                }
                break;
            }
            let Some(end) = find_entity_terminator(&self.buf) else {
                if self.eof {
                    break;
                }
                self.fill_more()?;
                continue;
            };
            let entity = self.buf[..end].trim().to_string();
            self.drain(end)?;
            if !entity.starts_with('#') {
                continue;
            }
            let entity_offset = data_body_start + self.consumed.saturating_sub(data_body_start);
            let at = line_col_at_offset(&entity, 0, 1);
            let _ = entity_offset;
            let file_span = Span {
                line: at.line,
                col: at.col,
            };
            match parse_instance_at(&entity, file_span) {
                Ok((inst, _)) => section.instances.push(inst),
                Err(e) => {
                    if !recover_skipped {
                        return Err(e);
                    }
                    if entity[1..].contains('#') {
                        skipped.push(SkippedEntity {
                            message: e.message.clone(),
                            span: e.span,
                            id_hint: peek_entity_id(&entity),
                            snippet: entity_snippet(&entity),
                        });
                        log::warn!("STEP part21: skip entity: {}", e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        Ok(section)
    }

    fn span_at_buf_start(&self) -> Span {
        line_col_at_offset(&self.buf, 0, 1)
    }

    fn fill_until(&mut self, min: usize) -> ParseResult<()> {
        while self.buf.len() < min && !self.eof {
            self.fill_more()?;
        }
        Ok(())
    }

    fn fill_more(&mut self) -> ParseResult<()> {
        let mut chunk = [0u8; READ_CHUNK];
        let n = self
            .reader
            .read(&mut chunk)
            .map_err(|e| ParseError::new(format!("read error: {e}"), IO_SPAN))?;
        if n == 0 {
            self.eof = true;
            return Ok(());
        }
        self.consumed += n;
        self.buf.push_str(&String::from_utf8_lossy(&chunk[..n]));
        Ok(())
    }

    fn drain(&mut self, n: usize) -> ParseResult<()> {
        if self.buf.len() >= n {
            self.buf.drain(..n);
            Ok(())
        } else {
            Err(ParseError::new("unexpected end of stream", IO_SPAN))
        }
    }
}

fn find_entity_terminator(s: &str) -> Option<usize> {
    let start = s.find('#')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let bytes = s.as_bytes();
    let mut i = start;
    while i < bytes.len() {
        let c = s[i..].chars().next()?;
        let len = c.len_utf8();
        match c {
            '\'' if !in_string => in_string = true,
            '\'' if in_string => {
                if s[i + len..].starts_with('\'') {
                    i += len * 2;
                    continue;
                }
                in_string = false;
            }
            '(' if !in_string => depth += 1,
            ')' if !in_string => depth -= 1,
            ';' if !in_string && depth == 0 => return Some(i + len),
            _ => {}
        }
        i += len;
    }
    None
}

/// Parse exchange from `BufRead` without allocating one `String` for the entire file.
pub fn read_exchange_stream<R: BufRead>(
    reader: &mut R,
    recover_skipped: bool,
) -> ParseResult<(Exchange, Vec<SkippedEntity>)> {
    StreamExchangeReader::new(reader).read_exchange_with_recovery(recover_skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::part21::read::read_exchange;

    #[test]
    fn stream_matches_memory_two_data_sections() {
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
        let mem = read_exchange(input).unwrap();
        let mut cursor = std::io::Cursor::new(input.as_bytes());
        let (streamed, _) = read_exchange_stream(&mut cursor, true).unwrap();
        assert_eq!(mem.instance_count(), streamed.instance_count());
        assert_eq!(mem.data_sections.len(), streamed.data_sections.len());
    }
}
