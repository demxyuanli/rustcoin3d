//! STEP exchange facade: Part21 read + adapter → legacy `EntityIndex` for `build_brep`.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use super::adapter::{exchange_from_model, AdapterOptions};
use super::entity_types::EntityType;
use super::import_options::StepImportOptions;
use super::model::Exchange as Part21Exchange;
use super::part21::SkippedEntity;
use super::value::StepValue;

const MAX_STORED_SKIPPED: usize = 32;

#[derive(Debug, Clone)]
pub struct EntityRecord {
    pub name: String,
    pub params: StepValue,
    pub entity_type: EntityType,
}

pub type EntityIndex = HashMap<u64, EntityRecord>;

#[derive(Debug, Default, Clone)]
pub struct ParseDiagnostics {
    pub skipped_entities: Vec<SkippedEntity>,
    pub unknown_entity_count: usize,
}

#[derive(Debug, Default, Clone)]
pub struct Part21ImportStats {
    pub data_section_count: usize,
    pub complex_external_count: usize,
}

#[derive(Debug)]
pub struct Exchange {
    pub header: Option<super::header::HeaderInfo>,
    pub entities: EntityIndex,
    pub diagnostics: ParseDiagnostics,
    pub part21_stats: Option<Part21ImportStats>,
    /// Pre-transfer AP242 WR violations (always computed on Part21 path).
    pub schema_violations: Vec<super::validate::SchemaViolation>,
}

fn part21_to_exchange(
    model: Part21Exchange,
    skipped: Vec<SkippedEntity>,
    adapter: &AdapterOptions,
) -> Result<Exchange, String> {
    let mut ex = exchange_from_model(model, adapter)?;
    ex.diagnostics.skipped_entities = skipped.into_iter().take(MAX_STORED_SKIPPED).collect();
    Ok(ex)
}

fn check_strict_skipped(recover: bool, skipped: &[SkippedEntity]) -> Result<(), String> {
    if !recover && !skipped.is_empty() {
        let detail: Vec<_> = skipped.iter().map(|s| s.format_short()).collect();
        return Err(format!(
            "STEP parse: {} malformed entit(y/ies) in strict mode: {}",
            skipped.len(),
            detail.join("; ")
        ));
    }
    Ok(())
}

/// Parse ISO 10303-21 ASCII exchange structure text.
pub fn parse_exchange(input: &str) -> Result<Exchange, String> {
    parse_exchange_with_options(input, &StepImportOptions::default())
}

pub fn parse_exchange_with_options(
    input: &str,
    options: &StepImportOptions,
) -> Result<Exchange, String> {
    let recover = options.recover_skipped_entities();
    let (model, skipped) = super::part21::read::read_exchange_with_recovery(input.trim(), recover)
        .map_err(|e| e.to_string())?;
    check_strict_skipped(recover, &skipped)?;
    part21_to_exchange(model, skipped, &options.adapter_options())
}

/// Streaming entry: buffered read then Part21 parse (same result as `parse_exchange`).
pub fn parse_exchange_streaming<R: BufRead>(
    reader: &mut R,
) -> Result<Exchange, String> {
    parse_exchange_streaming_with_options(reader, &StepImportOptions::default())
}

pub fn parse_exchange_streaming_with_options<R: BufRead>(
    reader: &mut R,
    options: &StepImportOptions,
) -> Result<Exchange, String> {
    let recover = options.recover_skipped_entities();
    let (model, skipped) =
        super::part21::read::read_exchange_buffered_with_recovery(reader, recover)
            .map_err(|e| e.to_string())?;
    check_strict_skipped(recover, &skipped)?;
    part21_to_exchange(model, skipped, &options.adapter_options())
}

/// Parse a STEP file from disk using buffered I/O.
pub fn parse_step_from_file(path: &Path) -> Result<Exchange, String> {
    parse_step_from_file_with_options(path, &StepImportOptions::default())
}

pub fn parse_step_from_file_with_options(
    path: &Path,
    options: &StepImportOptions,
) -> Result<Exchange, String> {
    let recover = options.recover_skipped_entities();
    let (model, skipped) =
        super::part21::read::read_exchange_file_with_recovery(path, recover)
            .map_err(|e| e.to_string())?;
    check_strict_skipped(recover, &skipped)?;
    part21_to_exchange(model, skipped, &options.adapter_options())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n";

    #[test]
    fn test_parse_simple_entity() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1 = CARTESIAN_POINT('', (0.0, 1.0, 2.0));\nENDSEC;\nEND-ISO-10303-21;\n"
        ))
        .unwrap();
        assert_eq!(ex.entities.len(), 1);
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "CARTESIAN_POINT");
    }

    #[test]
    fn test_parse_nested_list() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1 = ENTITY(#2, (3.0, 4.0));\nENDSEC;\nEND-ISO-10303-21;\n"
        ))
        .unwrap();
        let e = ex.entities.get(&1).unwrap();
        if let StepValue::List(params) = &e.params {
            assert!(matches!(params[0], StepValue::Ref(2)));
            assert!(matches!(params[1], StepValue::List(_)));
        } else {
            panic!("expected List");
        }
    }

    #[test]
    fn test_parse_typed_param() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1 = ENTITY(LENGTH_MEASURE(0.001));\nENDSEC;\nEND-ISO-10303-21;\n"
        ))
        .unwrap();
        let e = ex.entities.get(&1).unwrap();
        if let StepValue::List(params) = &e.params {
            assert!(matches!(&params[0], StepValue::Typed(name, _) if name == "LENGTH_MEASURE"));
        } else {
            panic!("expected List");
        }
    }

    #[test]
    fn test_parse_omitted() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1 = ENTITY($, *);\nENDSEC;\nEND-ISO-10303-21;\n"
        ))
        .unwrap();
        let e = ex.entities.get(&1).unwrap();
        if let StepValue::List(params) = &e.params {
            assert!(matches!(params[0], StepValue::Omitted));
            assert!(matches!(params[1], StepValue::Omitted));
        } else {
            panic!();
        }
    }

    #[test]
    fn test_parse_empty_data() {
        let ex = parse_exchange(&format!("{MINIMAL}ENDSEC;\nEND-ISO-10303-21;\n")).unwrap();
        assert!(ex.entities.is_empty());
    }

    #[test]
    fn test_parse_escaped_string() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1 = ENTITY('it''s');\nENDSEC;\nEND-ISO-10303-21;\n"
        ))
        .unwrap();
        let e = ex.entities.get(&1).unwrap();
        if let StepValue::List(params) = &e.params {
            assert!(matches!(&params[0], StepValue::String(s) if s == "it's"));
        }
    }

    #[test]
    fn test_parse_subsuper_inside() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1 = (LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.MILLI.,.METRE.));\nENDSEC;\nEND-ISO-10303-21;\n"
        ))
        .unwrap();
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "SI_UNIT");
    }

    #[test]
    fn test_parse_subsuper_outside() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1 = (SUPER1()SUPER2())ENTITY(1.0, 2.0);\nENDSEC;\nEND-ISO-10303-21;\n"
        ))
        .unwrap();
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "ENTITY");
    }

    #[test]
    fn test_parse_subsuper_with_params() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.MILLI.,.METRE.));\nENDSEC;\nEND-ISO-10303-21;\n"
        ))
        .unwrap();
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "SI_UNIT");
        if let StepValue::List(params) = &e.params {
            assert_eq!(params.len(), 3);
            assert!(matches!(params[0], StepValue::Omitted));
        } else {
            panic!("expected List params");
        }
    }

    #[test]
    fn test_parse_subsuper_multiline_ap242() {
        let ex = parse_exchange(&format!(
            "{MINIMAL}#1 = (
BOUNDED_CURVE()
B_SPLINE_CURVE(2,(#10,#11),.UNSPECIFIED.,.F.,.F.)
B_SPLINE_CURVE_WITH_KNOTS((3,2,3),(0.625,0.667,0.75),.UNSPECIFIED.)
CURVE()
GEOMETRIC_REPRESENTATION_ITEM()
RATIONAL_B_SPLINE_CURVE((0.933,0.933,1.))
REPRESENTATION_ITEM('')
);
ENDSEC;
END-ISO-10303-21;\n"
        ))
        .unwrap();
        let e = ex.entities.get(&1).unwrap();
        assert_eq!(e.name, "B_SPLINE_CURVE_WITH_KNOTS");
        if let StepValue::List(params) = &e.params {
            assert!(matches!(params.first(), Some(StepValue::Integer(2))));
        } else {
            panic!("expected List params");
        }
    }

    #[test]
    fn test_parse_error_recovery_skips_bad_entity() {
        let ex = parse_exchange_with_options(
            &format!(
                "{MINIMAL}#1 = CARTESIAN_POINT('', (0.0, 1.0, 2.0));\n#2 = BAD_ENTITY(no closing paren\n#3 = CARTESIAN_POINT('', (3.0, 4.0, 5.0));\nENDSEC;\nEND-ISO-10303-21;\n"
            ),
            &StepImportOptions::preview(),
        )
        .unwrap();
        assert!(ex.entities.contains_key(&1), "should parse #1");
        assert!(!ex.entities.contains_key(&2), "should skip broken #2");
        assert!(ex.entities.contains_key(&3), "should parse #3");
        assert!(
            ex.diagnostics
                .skipped_entities
                .iter()
                .any(|s| s.id_hint == Some(2)),
            "skipped record should identify #2"
        );
    }
}
