//! Recoverable parse skip records.

use super::token::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedEntity {
    pub message: String,
    pub span: Span,
    pub id_hint: Option<u64>,
    pub snippet: String,
}

impl SkippedEntity {
    pub fn format_short(&self) -> String {
        let id = self
            .id_hint
            .map(|i| format!(" #{i}"))
            .unwrap_or_default();
        format!(
            "{}@{}:{}{id} — {}",
            self.message, self.span.line, self.span.col, self.snippet
        )
    }
}
