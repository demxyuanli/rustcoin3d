//! Transfer layer: `model::Exchange` → legacy `EntityIndex` for `build_brep`.

use crate::step::entity_types::EntityType;
use crate::step::model::{ComplexMapping, Exchange};
use crate::step::parser::{EntityIndex, EntityRecord, ParseDiagnostics, Part21ImportStats};

mod subsuper;

pub use subsuper::instance_to_name_and_params;

/// How to flatten complex entity records into one `EntityRecord`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdapterMode {
    /// Legacy merge: all supertype parameter texts (matches old `parser.rs`).
    #[default]
    CompatMerge,
    /// Primary record parameters only (no ancestor merge).
    StrictFidelity,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AdapterOptions {
    pub mode: AdapterMode,
}

impl AdapterOptions {
    pub fn compat_merge() -> Self {
        Self {
            mode: AdapterMode::CompatMerge,
        }
    }
}

/// Build `EntityIndex` from a fidelity exchange (all DATA sections).
pub fn to_entity_index(exchange: &Exchange, options: &AdapterOptions) -> Result<EntityIndex, String> {
    let mut entities = EntityIndex::new();
    for inst in exchange.instances() {
        let (name, params) = subsuper::instance_to_name_and_params(inst, options.mode)?;
        entities.insert(
            inst.id,
            EntityRecord {
                name,
                params,
                entity_type: EntityType::Unknown,
            },
        );
    }
    Ok(entities)
}

pub fn finalize_entity_types(entities: &mut EntityIndex, diagnostics: &mut ParseDiagnostics) {
    diagnostics.unknown_entity_count = 0;
    for record in entities.values_mut() {
        record.entity_type = EntityType::from_name(&record.name);
        if record.entity_type == EntityType::Unknown {
            diagnostics.unknown_entity_count += 1;
        }
    }
}

/// Build `parser::Exchange` from a Part21 fidelity model.
pub fn exchange_from_model(
    model: Exchange,
    options: &AdapterOptions,
) -> Result<crate::step::parser::Exchange, String> {
    let schema_violations = crate::step::schema::validate_exchange_with_mode(&model, options.mode);
    let mut entities = to_entity_index(&model, options)?;
    let mut diagnostics = ParseDiagnostics::default();
    finalize_entity_types(&mut entities, &mut diagnostics);
    let complex_external_count = model
        .instances()
        .filter(|i| i.mapping == ComplexMapping::External)
        .count();
    Ok(crate::step::parser::Exchange {
        header: model.header,
        entities,
        diagnostics,
        part21_stats: Some(Part21ImportStats {
            data_section_count: model.data_sections.len(),
            complex_external_count,
        }),
        schema_violations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::part21::read::read_exchange;

    #[test]
    fn strict_fidelity_same_keyword_different_params_on_complex() {
        let input = "ISO-10303-21;\nHEADER;ENDSEC;\nDATA;\n#1 = (\n\
BOUNDED_CURVE()\nB_SPLINE_CURVE(2,(#10,#11),.UNSPECIFIED.,.F.,.F.)\n\
B_SPLINE_CURVE_WITH_KNOTS((3,2,3),(0.625,0.667,0.75),.UNSPECIFIED.)\n\
CURVE()\nGEOMETRIC_REPRESENTATION_ITEM()\nRATIONAL_B_SPLINE_CURVE((0.933,0.933,1.))\n\
REPRESENTATION_ITEM('')
);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = read_exchange(input).expect("part21 read");
        let compat = to_entity_index(&exchange, &AdapterOptions::compat_merge()).expect("compat");
        let strict = to_entity_index(
            &exchange,
            &AdapterOptions {
                mode: AdapterMode::StrictFidelity,
            },
        )
        .expect("strict");
        assert_eq!(compat.get(&1).unwrap().name, strict.get(&1).unwrap().name);
        assert_ne!(compat.get(&1).unwrap().params, strict.get(&1).unwrap().params);
    }

    #[test]
    fn adapter_picks_bspline_with_knots() {
        let input = "ISO-10303-21;\nHEADER;ENDSEC;\nDATA;\n#1 = (\n\
BOUNDED_CURVE()\nB_SPLINE_CURVE(2,(#10,#11),.UNSPECIFIED.,.F.,.F.)\n\
B_SPLINE_CURVE_WITH_KNOTS((3,2,3),(0.625,0.667,0.75),.UNSPECIFIED.)\n\
CURVE()\nGEOMETRIC_REPRESENTATION_ITEM()\nRATIONAL_B_SPLINE_CURVE((0.933,0.933,1.))\n\
REPRESENTATION_ITEM('')
);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = read_exchange(input).expect("part21 read");
        let idx = to_entity_index(&exchange, &AdapterOptions::compat_merge()).expect("adapter");
        let e = idx.get(&1).expect("#1");
        // Internal mapping: leaf_index = 6 (last record) = REPRESENTATION_ITEM('')
        assert_eq!(e.name, "REPRESENTATION_ITEM");
        // CompatMerge merges all records' non-Omitted params; B_SPLINE_CURVE
        // contributes Integer(2) as first meaningful param.
        if let crate::step::value::StepValue::List(params) = &e.params {
            assert!(matches!(params.first(), Some(crate::step::value::StepValue::Integer(2))));
        } else {
            panic!("expected List params");
        }
    }
}
