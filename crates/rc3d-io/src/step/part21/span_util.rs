//! Map substring offsets to file-absolute line/column.

use super::token::Span;

/// Line/column at `offset` within `text`, with `base_line` as the line of `text[0]`.
pub fn line_col_at_offset(text: &str, offset: usize, base_line: u32) -> Span {
    let offset = offset.min(text.len());
    let prefix = &text[..offset];
    let extra_lines = prefix.chars().filter(|&c| c == '\n').count() as u32;
    let line = base_line.saturating_add(extra_lines);
    let col = prefix
        .rfind('\n')
        .map(|i| prefix[i + 1..].chars().count() as u32 + 1)
        .unwrap_or_else(|| prefix.chars().count() as u32 + 1);
    Span {
        line,
        col: col.max(1),
    }
}

pub fn count_newlines(s: &str) -> u32 {
    s.chars().filter(|&c| c == '\n').count() as u32
}
