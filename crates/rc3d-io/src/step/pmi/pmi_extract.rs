//! PMI annotation geometry extraction from AP242 STEP entities.

use super::super::entity_types::EntityType;
use super::super::parser::EntityIndex;
use super::pmi_types::{FinishSymbol, PmiDataSet, PmiSurfaceFinish};
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::{GdtMaterialCondition, GdtSymbol};

/// Linear dimension with start/end points and offset direction.
#[derive(Debug, Clone)]
pub struct PmiDimension {
    pub entity_id: u64,
    pub start: Vec3,
    pub end: Vec3,
    pub offset_dir: Vec3,
    pub text: String,
}

/// Datum identifier anchored to a face/plane.
#[derive(Debug, Clone)]
pub struct PmiDatum {
    pub entity_id: u64,
    pub origin: Vec3,
    pub normal: Vec3,
    pub label: String,
}

/// Geometric tolerance frame with GD&T typed fields.
#[derive(Debug, Clone)]
pub struct PmiToleranceFrame {
    pub origin: Vec3,
    pub leader_points: Vec<Vec3>,
    pub text: String,
    pub symbol: Option<GdtSymbol>,
    pub value: f32,
    pub diameter: bool,
    pub datum_primary: Option<String>,
    pub datum_secondary: Option<String>,
    pub material_condition: Option<GdtMaterialCondition>,
}

/// All extracted PMI data from a STEP file.
#[derive(Debug, Default)]
pub struct PmiData {
    pub dimensions: Vec<PmiDimension>,
    pub datums: Vec<PmiDatum>,
    pub tolerances: Vec<PmiToleranceFrame>,
}

/// Extract full PMI data set including surface finishes.
pub fn extract_pmi_full(entities: &EntityIndex) -> PmiDataSet {
    let basic = extract_pmi(entities);
    let surface_finishes = extract_surface_finishes(entities);
    PmiDataSet {
        dimensions: basic.dimensions,
        datums: basic.datums,
        tolerances: basic.tolerances,
        surface_finishes,
    }
}

/// Extract surface finish annotations from STEP entities.
///
/// Looks for entities whose name matches common surface finish entity types:
/// `SURFACE_FINISH`, `SURFACE_TEXTURE`, or presentation items referencing them.
fn extract_surface_finishes(entities: &EntityIndex) -> Vec<PmiSurfaceFinish> {
    let mut finishes = Vec::new();
    for (&_eid, record) in entities.iter() {
        let name_upper = record.name.to_uppercase();
        let is_finish = name_upper.contains("SURFACE_FINISH")
            || name_upper.contains("SURFACE_TEXTURE")
            || name_upper.contains("ROUGHNESS");
        if !is_finish {
            continue;
        }
        // Try to extract Ra/Rz values from params
        let mut ra_value: Option<f32> = None;
        let mut rz_value: Option<f32> = None;
        let mut anchor = Vec3::ZERO;
        let mut note: Option<String> = None;

        // Param 0: name / description
        if let Some(n) = record.params.nth_param(0).and_then(|v| v.as_string()) {
            if !n.is_empty() {
                note = Some(n.to_string());
            }
        }
        // Param 1: Ra value (or reference to a value entity)
        if let Some(v) = record.params.nth_param(1) {
            if let Some(f) = v.as_real() {
                ra_value = Some(f as f32);
            } else if let Some(id) = v.as_ref_id() {
                if let Some(rec) = entities.get(&id) {
                    ra_value = rec.params.nth_param(0).and_then(|v| v.as_real()).map(|f| f as f32);
                }
            }
        }
        // Param 2: Rz value
        if let Some(v) = record.params.nth_param(2) {
            if let Some(f) = v.as_real() {
                rz_value = Some(f as f32);
            }
        }

        // Resolve anchor from ANNOTATION_OCCURRENCE chain
        for (_, anno_rec) in entities.iter() {
            if anno_rec.entity_type != EntityType::AnnotationOccurrence {
                continue;
            }
            if let Some(item_id) = anno_rec.params.nth_param(1).and_then(|v| v.as_ref_id()) {
                let actual_id = unwrap_styled_item(item_id, entities);
                if actual_id == _eid {
                    if let Some(ref_pts) = anno_rec.params.nth_param(3).and_then(|v| v.as_list()) {
                        let ref_ids: Vec<u64> = ref_pts.iter().filter_map(|v| v.as_ref_id()).collect();
                        let pts = resolve_pmi_points(&ref_ids, entities);
                        if let Some(&first) = pts.first() {
                            anchor = first;
                        }
                    }
                    break;
                }
            }
        }

        finishes.push(PmiSurfaceFinish {
            ra_value,
            rz_value,
            symbol: FinishSymbol::Basic,
            anchor_point: anchor,
            note,
        });
    }
    finishes
}

/// Walk PMI presentation chains and extract annotation geometry.
pub fn extract_pmi(entities: &EntityIndex) -> PmiData {
    let mut pmi = PmiData::default();

    for (&eid, record) in entities.iter() {
        match record.entity_type {
            EntityType::DimensionalSize => {
                if let Some(dim) = extract_dimension(eid, &record.params, entities) {
                    pmi.dimensions.push(dim);
                }
            }
            EntityType::Datum => {
                if let Some(datum) = extract_datum(eid, &record.params, entities) {
                    pmi.datums.push(datum);
                }
            }
            EntityType::GeometricTolerance
            | EntityType::FlatnessTolerance
            | EntityType::PositionTolerance
            | EntityType::ProfileTolerance
            | EntityType::ParallelismTolerance
            | EntityType::PerpendicularityTolerance
            | EntityType::RunoutTolerance
            | EntityType::StraightnessTolerance
            | EntityType::RoundnessTolerance
            | EntityType::CylindricityTolerance
            | EntityType::AngularityTolerance
            | EntityType::ConcentricityTolerance
            | EntityType::SymmetryTolerance
            | EntityType::CircularRunoutTolerance
            | EntityType::TotalRunoutTolerance
            | EntityType::SurfaceProfileTolerance
            | EntityType::LineProfileTolerance
            | EntityType::PlusMinusTolerance
            | EntityType::LimitsAndFits
            | EntityType::ModifiedGeometricTolerance
            | EntityType::UnequallyDisposedTolerance => {
                if let Some(tol) = extract_tolerance(eid, record.entity_type, &record.name, &record.params, entities) {
                    pmi.tolerances.push(tol);
                }
            }
            EntityType::DimensionalLocation
            | EntityType::DimensionalSizeWithDatum
            | EntityType::AngularSize => {
                if let Some(dim) = extract_dimension(eid, &record.params, entities) {
                    pmi.dimensions.push(dim);
                }
            }
            EntityType::DatumReferenceElement => {
                // Resolved via DATUM_SYSTEM chain during tolerance extraction.
                // Registered here to avoid Unknown classification.
            }
            _ => {}
        }
    }

    pmi
}

fn extract_dimension(
    entity_id: u64,
    params: &super::super::value::StepValue,
    entities: &EntityIndex,
) -> Option<PmiDimension> {
    // DIMENSIONAL_SIZE(name, name, nominal_value)
    //   name: STRING
    //   name: STRING (description)
    //   nominal_value: DIMENSIONAL_CHARACTERISTIC or subtype
    let name = params.nth_param(0)
        .and_then(|v| v.as_string())
        .map(|s| s.to_string())
        .unwrap_or_default();
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
    // Resolve ANNOTATION_OCCURRENCE to get reference points
    let mut start = Vec3::ZERO;
    let mut end = Vec3::new(1.0, 0.0, 0.0);

    for (_, anno_rec) in entities.iter() {
        if anno_rec.entity_type != EntityType::AnnotationOccurrence {
            continue;
        }
        // ANNOTATION_OCCURRENCE(name, item, styled_item, ref_points?)
        if let Some(item_id) = anno_rec.params.nth_param(1).and_then(|v| v.as_ref_id()) {
            let actual_id = unwrap_styled_item(item_id, entities);
            if actual_id == entity_id {
                if let Some(ref_pts) = anno_rec.params.nth_param(3).and_then(|v| v.as_list()) {
                    let ref_ids: Vec<u64> = ref_pts.iter().filter_map(|v| v.as_ref_id()).collect();
                    let pts = resolve_pmi_points(&ref_ids, entities);
                    if pts.len() >= 2 {
                        start = pts[0];
                        end = pts[1];
                    } else if pts.len() == 1 {
                        start = pts[0];
                    }
                }
                break;
            }
        }
    }

    let offset_dir = Vec3::Y;
    Some(PmiDimension {
        entity_id,
        start,
        end,
        offset_dir,
        text,
    })
}

fn extract_datum(
    entity_id: u64,
    params: &super::super::value::StepValue,
    entities: &EntityIndex,
) -> Option<PmiDatum> {
    // DATUM(name, label, reference_list)
    let label = params.nth_param(1)
        .and_then(|v| v.as_string())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "DATUM".to_string());

    let mut origin = Vec3::ZERO;
    let mut normal = Vec3::Z;

    // Walk: DATUM → DATUM_FEATURE → AXIS2_PLACEMENT_3D
    // or: DATUM → AXIS2_PLACEMENT_3D directly
    if let Some(ref_list) = params.nth_param(2).and_then(|v| v.as_list()) {
        for rv in ref_list {
            if let Some(ref_id) = rv.as_ref_id() {
                if let Some(rec) = entities.get(&ref_id) {
                    match rec.entity_type {
                        EntityType::DatumFeature => {
                            // DATUM_FEATURE(name, label, geometry)
                            if let Some(geom_id) = rec.params.nth_param(2).and_then(|v| v.as_ref_id()) {
                                if let Some(geom_rec) = entities.get(&geom_id) {
                                    if geom_rec.entity_type == EntityType::Axis2Placement3D {
                                        if let Some((o, _, axis)) = super::super::topology::resolve_placement(
                                            geom_id, entities
                                        ) {
                                            origin = o;
                                            normal = axis;
                                        }
                                    }
                                }
                            }
                        }
                        EntityType::Axis2Placement3D => {
                            if let Some((o, _, axis)) = super::super::topology::resolve_placement(
                                ref_id, entities
                            ) {
                                origin = o;
                                normal = axis;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Some(PmiDatum {
        entity_id,
        origin,
        normal,
        label,
    })
}

fn extract_tolerance(
    entity_id: u64,
    entity_type: EntityType,
    entity_name: &str,
    params: &super::super::value::StepValue,
    entities: &EntityIndex,
) -> Option<PmiToleranceFrame> {
    let text = params.nth_param(1)
        .and_then(|v| v.as_string())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "TOL".to_string());

    let mut origin = Vec3::ZERO;
    let mut leader_points = vec![];

    // Resolve tolerance value from DIMENSIONAL_CHARACTERISTIC_REPRESENTATION
    let mut value = 0.0f32;
    if let Some(nom_val) = params.nth_param(2) {
        if let Some(id) = nom_val.as_ref_id() {
            if let Some(rec) = entities.get(&id) {
                if let Some(v) = rec.params.nth_param(1) {
                    value = v.as_real().unwrap_or(0.0) as f32;
                }
            }
        }
    }

    // Resolve datum references via DATUM_SYSTEM chain
    let datum_labels = resolve_datum_labels(params, entities);
    let datum_primary = datum_labels.first().cloned();
    let datum_secondary = datum_labels.get(1).cloned();

    // Detect diameter modifier: check tolerance_shape or magnitude entity
    let diameter = resolve_diameter_modifier(params, entities);

    // Resolve position: search ANNOTATION_OCCURRENCE for anchor points
    for (_, anno_rec) in entities.iter() {
        if anno_rec.entity_type != EntityType::AnnotationOccurrence {
            continue;
        }
        // ANNOTATION_OCCURRENCE(name, item, styled_item, ref_points?)
        if let Some(item_id) = anno_rec.params.nth_param(1).and_then(|v| v.as_ref_id()) {
            // Handle STYLED_ITEM indirection
            let actual_id = unwrap_styled_item(item_id, entities);
            if actual_id != entity_id {
                continue;
            }
            if let Some(ref_pts) = anno_rec.params.nth_param(3).and_then(|v| v.as_list()) {
                let ref_ids: Vec<u64> = ref_pts.iter().filter_map(|v| v.as_ref_id()).collect();
                let pts = resolve_pmi_points(&ref_ids, entities);
                if let Some(&first) = pts.first() {
                    origin = first;
                }
                leader_points = pts;
            }
            break;
        }
    }

    Some(PmiToleranceFrame {
        origin,
        leader_points,
        text,
        symbol: gdt_symbol_for_entity(entity_type, entity_name),
        value,
        diameter,
        datum_primary,
        datum_secondary,
        material_condition: None,
    })
}

/// If `id` points to a STYLED_ITEM, return the inner item ref. Otherwise return `id`.
fn unwrap_styled_item(id: u64, entities: &EntityIndex) -> u64 {
    if let Some(rec) = entities.get(&id) {
        if rec.entity_type == EntityType::StyledItem {
            // STYLED_ITEM(name, styles, item)
            if let Some(inner) = rec.params.nth_param(2).and_then(|v| v.as_ref_id()) {
                return inner;
            }
        }
    }
    id
}

/// Walk DATUM_SYSTEM → DATUM_REFERENCE_COMPARTMENT → DATUM_REFERENCE_ELEMENT chain
/// to extract datum labels for a tolerance.
fn resolve_datum_labels(params: &super::super::value::StepValue, entities: &EntityIndex) -> Vec<String> {
    let datum_system_id = match params.nth_param(3).and_then(|v| v.as_ref_id()) {
        Some(id) => id,
        None => return vec![],
    };
    let ds_rec = match entities.get(&datum_system_id) {
        Some(r) => r,
        None => return vec![],
    };
    let compartments = match ds_rec.params.nth_param(1).and_then(|v| v.as_list()) {
        Some(l) => l,
        None => return vec![],
    };
    let mut labels = Vec::new();
    for comp_val in compartments {
        let comp_id = match comp_val.as_ref_id() {
            Some(id) => id,
            None => continue,
        };
        let comp_rec = match entities.get(&comp_id) {
            Some(r) => r,
            None => continue,
        };
        let dre_list = match comp_rec.params.nth_param(1).and_then(|v| v.as_list()) {
            Some(l) => l,
            None => continue,
        };
        for dre_val in dre_list {
            let dre_id = match dre_val.as_ref_id() {
                Some(id) => id,
                None => continue,
            };
            let dre_rec = match entities.get(&dre_id) {
                Some(r) => r,
                None => continue,
            };
            if let Some(label) = dre_rec.params.nth_param(1).and_then(|v| v.as_string()) {
                labels.push(label.to_string());
            }
        }
    }
    labels
}

/// Check if the tolerance has a diameter modifier (e.g. ⊘ 0.05).
fn resolve_diameter_modifier(params: &super::super::value::StepValue, entities: &EntityIndex) -> bool {
    // Check tolerance_shape reference at params[3] or params[4]
    for i in 3..=4 {
        if let Some(id) = params.nth_param(i).and_then(|v| v.as_ref_id()) {
            if let Some(rec) = entities.get(&id) {
                // TOLERANCE_SHAPE or similar may have diameter attribute
                if let Some(dia_param) = rec.params.nth_param(0) {
                    if dia_param.as_ref_id().is_some() {
                        // If first param is a ref (not a name string), it might indicate diameter
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Map an EntityType to the corresponding GD&T symbol.
/// Returns None for unrecognized types — callers must handle fallback explicitly.
fn gdt_symbol_for_entity(entity_type: EntityType, entity_name: &str) -> Option<GdtSymbol> {
    match (entity_type, entity_name) {
        // Form tolerances
        (EntityType::FlatnessTolerance, _) => Some(GdtSymbol::Flatness),
        (EntityType::StraightnessTolerance, _) => Some(GdtSymbol::Straightness),
        (EntityType::RoundnessTolerance, _) => Some(GdtSymbol::Circularity),
        (EntityType::CylindricityTolerance, _) => Some(GdtSymbol::Cylindricity),
        // Profile tolerances
        (EntityType::ProfileTolerance, "LINE_PROFILE_TOLERANCE") | (EntityType::LineProfileTolerance, _) => Some(GdtSymbol::ProfileOfLine),
        (EntityType::ProfileTolerance, _) | (EntityType::SurfaceProfileTolerance, _) => Some(GdtSymbol::ProfileOfSurface),
        // Orientation tolerances
        (EntityType::ParallelismTolerance, _) => Some(GdtSymbol::Parallelism),
        (EntityType::PerpendicularityTolerance, _) => Some(GdtSymbol::Perpendicularity),
        (EntityType::AngularityTolerance, _) => Some(GdtSymbol::Angularity),
        // Location tolerances
        (EntityType::PositionTolerance, _) => Some(GdtSymbol::Position),
        (EntityType::ConcentricityTolerance, _) => Some(GdtSymbol::Concentricity),
        (EntityType::SymmetryTolerance, _) => Some(GdtSymbol::Symmetry),
        // Runout tolerances
        (EntityType::RunoutTolerance, "TOTAL_RUNOUT_TOLERANCE") | (EntityType::TotalRunoutTolerance, _) => Some(GdtSymbol::TotalRunout),
        (EntityType::RunoutTolerance, _) | (EntityType::CircularRunoutTolerance, _) => Some(GdtSymbol::CircularRunout),
        _ => None,
    }
}

/// Resolve a 3D point from a STEP entity reference (CARTESIAN_POINT or AXIS2_PLACEMENT_3D.origin).
fn resolve_pmi_point(ref_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let rec = entities.get(&ref_id)?;
    match rec.entity_type {
        EntityType::CartesianPoint => {
            let coords = rec.params.nth_param(1)?;
            let list = coords.as_list()?;
            let x = list.first().and_then(|v| v.as_real())? as f32;
            let y = list.get(1).and_then(|v| v.as_real())? as f32;
            let z = list.get(2).and_then(|v| v.as_real())? as f32;
            Some(Vec3::new(x, y, z))
        }
        _ => None,
    }
}

/// Resolve first CARTESIAN_POINT from a list of entity references.
fn resolve_pmi_points(ref_ids: &[u64], entities: &EntityIndex) -> Vec<Vec3> {
    ref_ids.iter().filter_map(|&id| resolve_pmi_point(id, entities)).collect()
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

    #[test]
    fn test_extract_pmi_dimension_with_points() {
        let input = "\
ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('pt1', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('pt2', (10.0, 0.0, 0.0));
#3 = DIMENSIONAL_CHARACTERISTIC_REPRESENTATION('', 10.0);
#4 = DIMENSIONAL_SIZE('dist', '', #3);
#5 = ANNOTATION_OCCURRENCE('', #4, $, (#1, #2));
ENDSEC;
END-ISO-10303-21;
";
        let ex = parser::parse_exchange(input).unwrap();
        let pmi = extract_pmi(&ex.entities);
        assert_eq!(pmi.dimensions.len(), 1);
        let dim = &pmi.dimensions[0];
        assert!((dim.start.x - 0.0).abs() < 1e-6);
        assert!((dim.end.x - 10.0).abs() < 1e-6);
        // Name from DIMENSIONAL_SIZE param[0] should be in the text
        assert!(dim.text.contains("dist"), "expected text '{}' to contain 'dist'", dim.text);
    }
}
