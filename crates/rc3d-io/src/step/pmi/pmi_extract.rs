//! PMI annotation geometry extraction from AP242 STEP entities.

use super::super::entity_types::EntityType;
use super::super::parser::EntityIndex;
use rc3d_core::math::Vec3;

/// Linear dimension with start/end points and offset direction.
#[derive(Debug, Clone)]
pub struct PmiDimension {
    pub start: Vec3,
    pub end: Vec3,
    pub offset_dir: Vec3,
    pub text: String,
}

/// Datum identifier anchored to a face/plane.
#[derive(Debug, Clone)]
pub struct PmiDatum {
    pub origin: Vec3,
    pub normal: Vec3,
    pub label: String,
}

/// Geometric tolerance frame with leader lines.
#[derive(Debug, Clone)]
pub struct PmiToleranceFrame {
    pub origin: Vec3,
    pub leader_points: Vec<Vec3>,
    pub text: String,
}

/// All extracted PMI data from a STEP file.
#[derive(Debug, Default)]
pub struct PmiData {
    pub dimensions: Vec<PmiDimension>,
    pub datums: Vec<PmiDatum>,
    pub tolerances: Vec<PmiToleranceFrame>,
}

/// Walk PMI presentation chains and extract annotation geometry.
pub fn extract_pmi(entities: &EntityIndex) -> PmiData {
    let mut pmi = PmiData::default();

    for (_, record) in entities.iter() {
        match record.entity_type {
            EntityType::DimensionalSize => {
                if let Some(dim) = extract_dimension(&record.params, entities) {
                    pmi.dimensions.push(dim);
                }
            }
            EntityType::Datum => {
                if let Some(datum) = extract_datum(&record.params, entities) {
                    pmi.datums.push(datum);
                }
            }
            EntityType::GeometricTolerance => {
                if let Some(tol) = extract_tolerance(&record.params, entities) {
                    pmi.tolerances.push(tol);
                }
            }
            _ => {}
        }
    }

    pmi
}

fn extract_dimension(
    params: &super::super::value::StepValue,
    entities: &EntityIndex,
) -> Option<PmiDimension> {
    // DIMENSIONAL_SIZE(name, name, nominal_value)
    //   name: STRING
    //   name: STRING (description)
    //   nominal_value: DIMENSIONAL_CHARACTERISTIC or subtype
    let name = params.nth_param(1)
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "".to_string());
    // Try to resolve nominal_value to get a numeric size
    let mut text = name;
    if let Some(nom_val) = params.nth_param(2) {
        if let Some(id) = nom_val.as_ref_id() {
            if let Some(rec) = entities.get(&id) {
                // QUALIFIED_DIMENSION or similar may have a value
                if let Some(v) = rec.params.nth_param(1) {
                    if let Some(f) = v.as_real() {
                        text = format!("{}: {:.3}", text, f);
                    }
                }
            }
        }
    }
    // Default to a zero-length dimension at origin
    Some(PmiDimension {
        start: Vec3::ZERO,
        end: Vec3::X * 1.0,
        offset_dir: Vec3::Y,
        text,
    })
}

fn extract_datum(
    params: &super::super::value::StepValue,
    entities: &EntityIndex,
) -> Option<PmiDatum> {
    // DATUM(name, label, ...)
    let label = params.nth_param(1)
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "DATUM".to_string());
    // Try to find a referenced geometry to get origin/normal
    let mut origin = Vec3::ZERO;
    let mut normal = Vec3::Z;
    if let Some(ref_list) = params.nth_param(2).and_then(|v| v.as_list()) {
        for rv in ref_list {
            if let Some(id) = rv.as_ref_id() {
                if let Some(rec) = entities.get(&id) {
                    if rec.entity_type == super::super::entity_types::EntityType::Axis2Placement3D {
                        if let Some(pt) = super::super::topology::resolve_point_from_placement(id, entities) {
                            origin = pt;
                        }
                    }
                }
            }
        }
    }
    Some(PmiDatum { origin, normal, label })
}

fn extract_tolerance(
    params: &super::super::value::StepValue,
    entities: &EntityIndex,
) -> Option<PmiToleranceFrame> {
    // GEOMETRIC_TOLERANCE(name, name, ...)
    let text = params.nth_param(1)
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "TOL".to_string());
    // Default origin at zero, no leader points
    Some(PmiToleranceFrame {
        origin: Vec3::ZERO,
        leader_points: vec![],
        text,
    })
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::parser;
    use super::*;

    #[test]
    fn test_extract_pmi_empty() {
        let input = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));\nENDSEC;\nEND-ISO-10303-21;\n";
        let ex = parser::parse_exchange(input).unwrap();
        let pmi = extract_pmi(&ex.entities);
        assert!(pmi.dimensions.is_empty());
        assert!(pmi.datums.is_empty());
        assert!(pmi.tolerances.is_empty());
    }
}
