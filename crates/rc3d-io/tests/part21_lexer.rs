//! Integration tests for ISO 10303-21 lexer.

use rc3d_io::step::part21::lexer::lex;
use rc3d_io::step::part21::token::Token;

#[test]
fn lex_simple_entity_header() {
    let input = "#12 = CARTESIAN_POINT('', (0.0, 1.0, 2.0));";
    let tokens = lex(input).expect("lex");
    assert!(matches!(tokens[0].0, Token::Ref(12)));
    assert!(matches!(tokens[1].0, Token::Eq));
    assert!(matches!(tokens[2].0, Token::Keyword(ref k) if k == "CARTESIAN_POINT"));
}

#[test]
fn lex_enum_and_omitted() {
    let tokens = lex("ENTITY($, .UNSPECIFIED., *);").expect("lex");
    assert!(matches!(tokens[1].0, Token::LParen));
    assert!(matches!(tokens[2].0, Token::Omitted));
    assert!(matches!(tokens[4].0, Token::Enum(ref e) if e == ".UNSPECIFIED."));
}

#[test]
fn lex_block_comment() {
    let tokens = lex("/* hello */ #1 = X();").expect("lex");
    assert!(matches!(tokens[0].0, Token::Ref(1)));
}
