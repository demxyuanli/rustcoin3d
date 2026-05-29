//! ISO 10303-21 lexical tokens.

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ref(u64),
    Keyword(String),
    String(String),
    Enum(String),
    Integer(i64),
    Real(f64),
    Omitted,
    LParen,
    RParen,
    Comma,
    Semi,
    Eq,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}
