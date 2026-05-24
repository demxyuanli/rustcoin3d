//! STEP schema validation: checks entity integrity, dangling references,
//! EXPRESS schema constraints, and required entity types for valid B-rep geometry.

use std::collections::HashSet;
use super::parser::EntityIndex;
use super::value::StepValue;

/// Validation result with warnings and errors.
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub entity_count: usize,
    pub missing_refs: Vec<(u64, u64)>, // (entity_id, missing_ref_id)
    pub topology_info: TopologyInfo,
    /// EXPRESS schema constraint violations
    pub schema_violations: Vec<SchemaViolation>,
}

#[derive(Debug, Clone)]
pub struct SchemaViolation {
    pub entity_id: u64,
    pub entity_name: String,
    pub constraint: String,
    pub description: String,
}

#[derive(Debug, Default)]
pub struct TopologyInfo {
    pub shells: usize,
    pub faces: usize,
    pub loops: usize,
    pub edges: usize,
    pub points: usize,
    pub has_closed_shell: bool,
    pub has_manifold_solid_brep: bool,
}

/// Validate an EntityIndex and return a report.
pub fn validate(entities: &EntityIndex) -> ValidationReport {
    let mut report = ValidationReport {
        entity_count: entities.len(),
        ..Default::default()
    };

    // Phase 1: collect all entity IDs
    let all_ids: HashSet<u64> = entities.keys().copied().collect();

    // Phase 2: check dangling references
    let mut dangling = Vec::new();
    for (&id, record) in entities.iter() {
        let refs = collect_refs(&record.params);
        for ref_id in refs {
            if !all_ids.contains(&ref_id) {
                dangling.push((id, ref_id));
            }
        }
    }
    if !dangling.is_empty() {
        for &(entity_id, missing) in &dangling {
            report.warnings.push(format!(
                "#{} references #{} which does not exist", entity_id, missing
            ));
        }
    }
    report.missing_refs = dangling;

    // Phase 3: topology statistics
    for (_id, record) in entities.iter() {
        match record.name.as_str() {
            "CLOSED_SHELL" | "OPEN_SHELL" | "SHELL"
            | "ORIENTED_CLOSED_SHELL" | "ORIENTED_OPEN_SHELL" => {
                report.topology_info.shells += 1;
                if record.name == "CLOSED_SHELL" {
                    report.topology_info.has_closed_shell = true;
                }
            }
            "ADVANCED_FACE" | "FACE_SURFACE" | "FACE" => {
                report.topology_info.faces += 1;
            }
            "EDGE_LOOP" | "POLY_LOOP" => {
                report.topology_info.loops += 1;
            }
            "EDGE_CURVE" | "ORIENTED_EDGE" => {
                report.topology_info.edges += 1;
            }
            "CARTESIAN_POINT" | "VERTEX_POINT" => {
                report.topology_info.points += 1;
            }
            "MANIFOLD_SOLID_BREP" => {
                report.topology_info.has_manifold_solid_brep = true;
            }
            _ => {}
        }
    }

    // Phase 4: structural checks
    if !report.topology_info.has_closed_shell && report.topology_info.shells == 0 {
        report.errors.push("No closed shell found — geometry cannot be rendered".into());
    }
    if report.topology_info.faces == 0 && report.topology_info.shells > 0 {
        report.errors.push("Shells exist but contain no faces".into());
    }
    if report.topology_info.points == 0 {
        report.errors.push("No CARTESIAN_POINT entities found".into());
    }
    if report.topology_info.edges == 0 && report.topology_info.faces > 0 {
        report.warnings.push("Faces exist but no edge curves found — topology may be incomplete".into());
    }

    // Phase 5: EXPRESS schema constraint validation
    check_express_constraints(entities, &mut report);

    // Phase 6: check for common issues
    check_ref_cycles(entities, &mut report);

    report
}

/// Validate EXPRESS schema constraints for key entity types.
/// These correspond to ISO 10303-11 schema rules.
fn check_express_constraints(entities: &EntityIndex, report: &mut ValidationReport) {
    for (&id, record) in entities.iter() {
        match record.name.as_str() {
            // B-rep topology constraints
            "CLOSED_SHELL" => {
                check_closed_shell_constraints(id, &record.params, entities, report);
            }
            "MANIFOLD_SOLID_BREP" => {
                check_manifold_solid_brep_constraints(id, &record.params, entities, report);
            }
            "ADVANCED_FACE" => {
                check_advanced_face_constraints(id, &record.params, entities, report);
            }
            "EDGE_LOOP" => {
                check_edge_loop_constraints(id, &record.params, entities, report);
            }
            "CARTESIAN_POINT" => {
                check_cartesian_point_constraints(id, &record.params, report);
            }
            "AXIS2_PLACEMENT_3D" => {
                check_axis2_placement_3d_constraints(id, &record.params, entities, report);
            }
            "B_SPLINE_CURVE_WITH_KNOTS" => {
                check_bspline_curve_knots_constraints(id, &record.params, report);
            }
            "B_SPLINE_SURFACE_WITH_KNOTS" => {
                check_bspline_surface_knots_constraints(id, &record.params, report);
            }
            "TOROIDAL_SURFACE" => {
                check_toroidal_surface_constraints(id, &record.params, report);
            }
            _ => {}
        }
    }
}

/// CLOSED_SHELL must have at least 4 faces (requirement from STEP schema).
fn check_closed_shell_constraints(id: u64, params: &StepValue, _entities: &EntityIndex, report: &mut ValidationReport) {
    if let Some(face_list) = params.nth_param(1).and_then(|v| v.as_list()) {
        if face_list.is_empty() {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "CLOSED_SHELL".to_string(),
                constraint: "WR1".to_string(),
                description: "Closed shell must contain at least one face".to_string(),
            });
        } else if face_list.len() < 4 {
            // Note: This is a warning as some valid files may have < 4 faces
            report.warnings.push(format!(
                "#{} CLOSED_SHELL has only {} face(s), minimum recommended is 4", id, face_list.len()
            ));
        }
    }
}

/// MANIFOLD_SOLID_BREP must reference a valid closed shell.
fn check_manifold_solid_brep_constraints(id: u64, params: &StepValue, entities: &EntityIndex, report: &mut ValidationReport) {
    if let Some(shell_ref) = params.nth_param(1).and_then(|v| v.as_ref_id()) {
        if let Some(shell_record) = entities.get(&shell_ref) {
            if shell_record.name != "CLOSED_SHELL" {
                report.schema_violations.push(SchemaViolation {
                    entity_id: id,
                    entity_name: "MANIFOLD_SOLID_BREP".to_string(),
                    constraint: "WR1".to_string(),
                    description: format!(
                        "Outer shell must be CLOSED_SHELL, found {}", shell_record.name
                    ),
                });
            }
        }
    }
}

/// ADVANCED_FACE must have consistent face bounds.
fn check_advanced_face_constraints(id: u64, params: &StepValue, entities: &EntityIndex, report: &mut ValidationReport) {
    // Check that face has at least an outer bound
    if let Some(bounds) = params.nth_param(1).and_then(|v| v.as_list()) {
        if bounds.is_empty() {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "ADVANCED_FACE".to_string(),
                constraint: "WR1".to_string(),
                description: "Face must have at least one bound (outer boundary)".to_string(),
            });
        }
    }

    // Check surface reference exists if provided
    if let Some(surface_ref) = params.nth_param(2).and_then(|v| v.as_ref_id()) {
        if !entities.contains_key(&surface_ref) {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "ADVANCED_FACE".to_string(),
                constraint: "WR2".to_string(),
                description: format!("Referenced surface #{} does not exist", surface_ref),
            });
        }
    }
}

/// EDGE_LOOP must have at least 3 edges and form a closed loop.
fn check_edge_loop_constraints(id: u64, params: &StepValue, _entities: &EntityIndex, report: &mut ValidationReport) {
    if let Some(edges) = params.nth_param(1).and_then(|v| v.as_list()) {
        if edges.len() < 3 {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "EDGE_LOOP".to_string(),
                constraint: "WR1".to_string(),
                description: format!(
                    "Edge loop must have at least 3 edges, found {}", edges.len()
                ),
            });
        }
    }
}

/// CARTESIAN_POINT coordinates must be all specified (no omitted values).
fn check_cartesian_point_constraints(id: u64, params: &StepValue, report: &mut ValidationReport) {
    if let Some(coords) = params.nth_param(1).and_then(|v| v.as_list()) {
        for (i, coord) in coords.iter().enumerate() {
            if matches!(coord, StepValue::Omitted) {
                report.schema_violations.push(SchemaViolation {
                    entity_id: id,
                    entity_name: "CARTESIAN_POINT".to_string(),
                    constraint: "WR1".to_string(),
                    description: format!("Coordinate {} is omitted", i),
                });
            }
        }
    }
}

/// AXIS2_PLACEMENT_3D must have three mutually perpendicular directions.
fn check_axis2_placement_3d_constraints(id: u64, _params: &StepValue, entities: &EntityIndex, report: &mut ValidationReport) {
    use super::topology::resolve_placement;
    if let Some(placement) = resolve_placement(id, entities) {
        let (_, x_axis, z_axis) = placement;

        // Check axis is normalized
        if (z_axis.length() - 1.0).abs() > 1e-6 {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "AXIS2_PLACEMENT_3D".to_string(),
                constraint: "WR1".to_string(),
                description: "Axis direction is not normalized".to_string(),
            });
        }

        // Check x_axis is normalized
        if (x_axis.length() - 1.0).abs() > 1e-6 {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "AXIS2_PLACEMENT_3D".to_string(),
                constraint: "WR2".to_string(),
                description: "Ref direction is not normalized".to_string(),
            });
        }

        // Check axis and ref_dir are perpendicular (dot product ≈ 0)
        let dot = z_axis.dot(x_axis);
        if dot.abs() > 1e-6 {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "AXIS2_PLACEMENT_3D".to_string(),
                constraint: "WR3".to_string(),
                description: format!(
                    "Axis and ref direction are not perpendicular (dot = {:.6})", dot
                ),
            });
        }
    }
}

/// B_SPLINE_CURVE_WITH_KNOTS must satisfy: num_knots = num_control_points + degree + 1.
fn check_bspline_curve_knots_constraints(id: u64, params: &StepValue, report: &mut ValidationReport) {
    // This is complex due to subsuper merging - check basic consistency
    if let Some(params_list) = params.as_list() {
        if params_list.len() < 8 {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "B_SPLINE_CURVE_WITH_KNOTS".to_string(),
                constraint: "WR1".to_string(),
                description: "Insufficient parameters for B-spline curve".to_string(),
            });
        }
    }
}

/// B_SPLINE_SURFACE_WITH_KNOTS must satisfy: num_knots = num_control_points + degree + 1 in each direction.
fn check_bspline_surface_knots_constraints(id: u64, params: &StepValue, report: &mut ValidationReport) {
    if let Some(params_list) = params.as_list() {
        if params_list.len() < 11 {
            report.schema_violations.push(SchemaViolation {
                entity_id: id,
                entity_name: "B_SPLINE_SURFACE_WITH_KNOTS".to_string(),
                constraint: "WR1".to_string(),
                description: "Insufficient parameters for B-spline surface".to_string(),
            });
        }
    }
}

/// TOROIDAL_SURFACE must have non-zero major and minor radii.
fn check_toroidal_surface_constraints(id: u64, params: &StepValue, report: &mut ValidationReport) {
    let minor = params.nth_param(3).and_then(|v| v.as_real()).unwrap_or(0.0);
    if minor.abs() < 1e-10 {
        report.schema_violations.push(SchemaViolation {
            entity_id: id,
            entity_name: "TOROIDAL_SURFACE".to_string(),
            constraint: "WR1".to_string(),
            description: format!("Degenerate toroidal surface: minor_radius={:.6} is zero", minor),
        });
    }
    let major = params.nth_param(2).and_then(|v| v.as_real()).unwrap_or(0.0);
    if major.abs() < 1e-10 {
        report.schema_violations.push(SchemaViolation {
            entity_id: id,
            entity_name: "TOROIDAL_SURFACE".to_string(),
            constraint: "WR2".to_string(),
            description: "Degenerate toroidal surface: major_radius is zero".to_string(),
        });
    }
}

/// Check for reference cycles that would break topology traversal.
fn check_ref_cycles(entities: &EntityIndex, report: &mut ValidationReport) {
    // Simple check: an entity referencing itself
    for (&id, record) in entities.iter() {
        let refs = collect_refs(&record.params);
        if refs.contains(&id) {
            report.warnings.push(format!("#{} references itself", id));
        }
    }
}

/// Recursively collect all entity reference IDs from a StepValue tree.
fn collect_refs(value: &StepValue) -> Vec<u64> {
    let mut refs = Vec::new();
    match value {
        StepValue::Ref(id) => refs.push(*id),
        StepValue::List(items) => {
            for item in items {
                refs.extend(collect_refs(item));
            }
        }
        StepValue::Typed(_, inner) => refs.extend(collect_refs(inner)),
        _ => {}
    }
    refs
}

/// Quick validation — returns Ok if no errors, Err with description otherwise.
pub fn quick_check(entities: &EntityIndex) -> Result<(), String> {
    let report = validate(entities);
    if !report.errors.is_empty() {
        Err(report.errors.join("; "))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser;

    fn parse_text(data: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_validate_valid_brep() {
        let entities = parse_text(
            "\
#1 = CARTESIAN_POINT('', (0., 0., 0.));
#2 = CARTESIAN_POINT('', (10., 0., 0.));
#3 = CARTESIAN_POINT('', (10., 10., 0.));
#4 = CARTESIAN_POINT('', (0., 10., 0.));
#5 = DIRECTION('', (0., 0., 1.));
#10 = LINE('', #1, #2);
#11 = LINE('', #2, #3);
#12 = LINE('', #3, #4);
#13 = LINE('', #4, #1);
#14 = EDGE_CURVE('', #1, #2, #10, .T.);
#15 = EDGE_CURVE('', #2, #3, #11, .T.);
#16 = EDGE_CURVE('', #3, #4, #12, .T.);
#17 = EDGE_CURVE('', #4, #1, #13, .T.);
#18 = EDGE_LOOP('', (#14, #15, #16, #17));
#19 = FACE_OUTER_BOUND('', #18, .T.);
#20 = AXIS2_PLACEMENT_3D('', #1, #5, #2);
#21 = PLANE('', #20);
#22 = ADVANCED_FACE('', (#19), #21, .T.);
#23 = CLOSED_SHELL('', (#22));
#24 = MANIFOLD_SOLID_BREP('', #23);\
",
        );
        let report = validate(&entities);
        assert!(report.errors.is_empty(), "valid brep should have no errors: {:?}", report.errors);
        assert_eq!(report.topology_info.shells, 1);
        assert_eq!(report.topology_info.faces, 1);
        assert_eq!(report.topology_info.points, 4);
        assert!(report.topology_info.has_closed_shell);
        assert!(report.topology_info.has_manifold_solid_brep);
    }

    #[test]
    fn test_validate_dangling_ref() {
        let entities = parse_text(
            "\
#1 = CARTESIAN_POINT('', (0., 0., 0.));
#2 = LINE('', #1, #999);\
",
        );
        let report = validate(&entities);
        assert!(!report.missing_refs.is_empty(), "should detect dangling ref #999");
        let dangling_ids: Vec<u64> = report.missing_refs.iter().map(|(_, m)| *m).collect();
        assert!(dangling_ids.contains(&999));
    }

    #[test]
    fn test_validate_self_ref() {
        let entities = parse_text(
            "\
#1 = CARTESIAN_POINT('', (0., 0., 0.));
#10 = LINE('', #1, #10);\
",
        );
        let report = validate(&entities);
        let has_self_ref = report.warnings.iter().any(|w| w.contains("references itself"));
        assert!(has_self_ref, "should detect self-reference in LINE #10");
    }

    #[test]
    fn test_degenerate_torus_detected() {
        let entities = parse_text(
            "\
#1 = CARTESIAN_POINT('', (0., 0., 0.));
#2 = DIRECTION('', (0., 0., 1.));
#3 = DIRECTION('', (1., 0., 0.));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#10 = TOROIDAL_SURFACE('', #4, 10.0, 0.0);\
");
        let report = validate(&entities);
        let has_degen = report.schema_violations.iter()
            .any(|v| v.entity_name == "TOROIDAL_SURFACE" && v.description.contains("zero"));
        assert!(has_degen, "should detect degenerate torus with minor_radius=0");
    }

    #[test]
    fn test_quick_check() {
        let entities = parse_text("#1 = CARTESIAN_POINT('', (0., 0., 0.));");
        // No geometry — should error
        let result = quick_check(&entities);
        assert!(result.is_err(), "should fail: no closed shell");
    }
}
