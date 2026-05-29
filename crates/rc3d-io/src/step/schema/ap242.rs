//! AP242 (and shared AP203/214) WR checks on `StepInstance` before adapter merge.

use std::collections::HashMap;

use crate::step::adapter::{instance_to_name_and_params, AdapterMode};
use crate::step::model::StepInstance;
use crate::step::validate::SchemaViolation;
use crate::step::value::StepValue;

/// Resolved EXPRESS keyword for validation (uses same adapter mode as transfer).
pub fn instance_keyword(inst: &StepInstance, mode: AdapterMode) -> Option<String> {
    instance_to_name_and_params(inst, mode)
        .ok()
        .map(|(name, _)| name)
}

pub fn validate_instance(
    inst: &StepInstance,
    mode: AdapterMode,
    keywords: &HashMap<u64, String>,
) -> Vec<SchemaViolation> {
    let Ok((name, params)) = instance_to_name_and_params(inst, mode) else {
        return Vec::new();
    };
    let id = inst.id;
    match name.as_str() {
        "CLOSED_SHELL" => closed_shell_wr1(id, &params).into_iter().collect(),
        "MANIFOLD_SOLID_BREP" => manifold_solid_wr1(id, &params, keywords).into_iter().collect(),
        "ADVANCED_FACE" => advanced_face_wr1(id, &params).into_iter().collect(),
        _ => Vec::new(),
    }
}

fn closed_shell_wr1(id: u64, params: &StepValue) -> Option<SchemaViolation> {
    let face_list = params.nth_param(1).and_then(|v| v.as_list())?;
    if face_list.is_empty() {
        Some(SchemaViolation {
            entity_id: id,
            entity_name: "CLOSED_SHELL".into(),
            constraint: "WR1".into(),
            description: "Closed shell must contain at least one face".into(),
        })
    } else {
        None
    }
}

fn manifold_solid_wr1(
    id: u64,
    params: &StepValue,
    keywords: &HashMap<u64, String>,
) -> Option<SchemaViolation> {
    let shell_ref = params.nth_param(1).and_then(|v| v.as_ref_id())?;
    let kw = keywords.get(&shell_ref)?;
    if kw != "CLOSED_SHELL" {
        Some(SchemaViolation {
            entity_id: id,
            entity_name: "MANIFOLD_SOLID_BREP".into(),
            constraint: "WR1".into(),
            description: format!("Outer shell must be CLOSED_SHELL, found {kw}"),
        })
    } else {
        None
    }
}

fn advanced_face_wr1(id: u64, params: &StepValue) -> Option<SchemaViolation> {
    let bounds = params.nth_param(1).and_then(|v| v.as_list())?;
    if bounds.is_empty() {
        Some(SchemaViolation {
            entity_id: id,
            entity_name: "ADVANCED_FACE".into(),
            constraint: "WR1".into(),
            description: "Face must have at least one bound (outer boundary)".into(),
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::model::{ComplexMapping, StepInstance};
    use crate::step::model::record::Record;
    use crate::step::value::StepValue;

    fn closed_shell_no_faces() -> StepInstance {
        StepInstance {
            id: 99,
            records: vec![Record {
                keyword: "CLOSED_SHELL".into(),
                params: StepValue::List(vec![
                    StepValue::String(String::new()),
                    StepValue::List(vec![]),
                ]),
            }],
            mapping: ComplexMapping::Simple,
        }
    }

    #[test]
    fn schema_closed_shell_empty_faces() {
        let inst = closed_shell_no_faces();
        let violations = validate_instance(&inst, AdapterMode::CompatMerge, &HashMap::new());
        assert!(violations.iter().any(|v| v.constraint == "WR1"));
    }
}
