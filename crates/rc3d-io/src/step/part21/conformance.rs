//! HEADER conformance detection (delegates to `model::conformance`).

pub use crate::step::model::conformance::{
    from_header, parse_implementation_level, ImplementationLevel,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_level_2_1() {
        let l = parse_implementation_level("'2;1'").unwrap();
        assert_eq!(l.version, 2);
        assert_eq!(l.conformance, 1);
        assert!(!l.prefers_external_complex());
    }

    #[test]
    fn parse_level_2_2() {
        let l = parse_implementation_level("2;2").unwrap();
        assert_eq!(l.conformance, 2);
        assert!(l.prefers_external_complex());
    }
}
