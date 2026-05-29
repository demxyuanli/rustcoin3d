//! Lexer for ISO 10303-21 clear-text exchange structures.

use super::token::{Span, Token};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

pub type LexResult<T> = Result<T, LexError>;

/// Tokenize the entire input; returns `(token, span_at_start)` pairs.
pub fn lex(input: &str) -> LexResult<Vec<(Token, Span)>> {
    let mut lexer = Lexer::new(input);
    let mut out = Vec::new();
    loop {
        lexer.skip_space_and_comments()?;
        if lexer.at_end() {
            break;
        }
        let span = lexer.span();
        let tok = lexer.next_token()?;
        out.push((tok, span));
    }
    Ok(out)
}

struct Lexer<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
            line: 1,
            col: 1,
        }
    }

    fn at_end(&mut self) -> bool {
        self.chars.peek().is_none()
    }

    fn span(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
        }
    }

    fn bump(&mut self) -> Option<char> {
        let (_, c) = self.chars.next()?;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, c)| *c)
    }

    fn err(&self, msg: impl Into<String>) -> LexError {
        LexError {
            message: msg.into(),
            span: self.span(),
        }
    }

    fn skip_space_and_comments(&mut self) -> LexResult<()> {
        loop {
            while matches!(self.peek(), Some(c) if c.is_ascii_whitespace()) {
                self.bump();
            }
            if self.peek() == Some('/') && self.input[self.byte_pos()..].starts_with("/*") {
                self.bump();
                self.bump();
                loop {
                    let c = self.bump().ok_or_else(|| self.err("unterminated block comment"))?;
                    if c == '*' && self.peek() == Some('/') {
                        self.bump();
                        break;
                    }
                }
                continue;
            }
            break;
        }
        Ok(())
    }

    fn byte_pos(&mut self) -> usize {
        self.chars.peek().map(|(i, _)| *i).unwrap_or(self.input.len())
    }

    fn next_token(&mut self) -> LexResult<Token> {
        let c = self.peek().ok_or_else(|| self.err("unexpected end of input"))?;
        match c {
            '(' => {
                self.bump();
                Ok(Token::LParen)
            }
            ')' => {
                self.bump();
                Ok(Token::RParen)
            }
            ',' => {
                self.bump();
                Ok(Token::Comma)
            }
            ';' => {
                self.bump();
                Ok(Token::Semi)
            }
            '=' => {
                self.bump();
                Ok(Token::Eq)
            }
            '#' => {
                self.bump();
                self.read_ref()
            }
            '\'' => {
                self.bump();
                self.read_string()
            }
            '.' => self.read_enum(),
            '$' | '*' => {
                self.bump();
                Ok(Token::Omitted)
            }
            '+' | '-' => self.read_number(),
            '0'..='9' => self.read_number(),
            _ if c.is_ascii_alphabetic() || c == '_' => self.read_keyword(),
            _ => Err(self.err(format!("unexpected character {:?}", c))),
        }
    }

    fn read_ref(&mut self) -> LexResult<Token> {
        let start = self.byte_pos();
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        let end = self.byte_pos();
        if end == start {
            return Err(self.err("expected digits after '#'"));
        }
        let n: u64 = self.input[start..end]
            .parse()
            .map_err(|_| self.err("invalid entity reference"))?;
        Ok(Token::Ref(n))
    }

    fn read_string(&mut self) -> LexResult<Token> {
        let mut s = String::new();
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated string")),
                Some('\'') => {
                    if self.peek() == Some('\'') {
                        self.bump();
                        s.push('\'');
                    } else {
                        return Ok(Token::String(s));
                    }
                }
                Some(ch) => s.push(ch),
            }
        }
    }

    fn read_enum(&mut self) -> LexResult<Token> {
        self.bump(); // leading '.'
        let mut buf = String::from('.');
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '_') {
            buf.push(self.bump().unwrap());
        }
        if buf.len() < 2 {
            return Err(self.err("expected enum value after '.'"));
        }
        if self.peek() == Some('.') {
            self.bump();
            buf.push('.');
        } else {
            return Err(self.err("enum must end with '.'"));
        }
        Ok(Token::Enum(buf))
    }

    fn read_number(&mut self) -> LexResult<Token> {
        let start = self.byte_pos();
        if matches!(self.peek(), Some('+' | '-')) {
            self.bump();
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if self.peek() == Some('.') {
            self.bump();
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.bump();
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.err("invalid exponent in real"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }
        let end = self.byte_pos();
        let text = &self.input[start..end];
        if text.contains('.') || text.contains('e') || text.contains('E') {
            let v: f64 = text
                .parse()
                .map_err(|_| self.err(format!("invalid real literal '{}'", text)))?;
            Ok(Token::Real(v))
        } else {
            let v: i64 = text
                .parse()
                .map_err(|_| self.err(format!("invalid integer literal '{}'", text)))?;
            Ok(Token::Integer(v))
        }
    }

    fn read_keyword(&mut self) -> LexResult<Token> {
        let start = self.byte_pos();
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '_') {
            self.bump();
        }
        let end = self.byte_pos();
        Ok(Token::Keyword(self.input[start..end].to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_cartesian_point_line() {
        let tokens = lex("#12 = CARTESIAN_POINT('', (0.0, 1.0, 2.0));").unwrap();
        assert!(matches!(tokens[0].0, Token::Ref(12)));
        assert!(matches!(tokens[1].0, Token::Eq));
        assert!(matches!(tokens[2].0, Token::Keyword(ref k) if k == "CARTESIAN_POINT"));
    }
}
