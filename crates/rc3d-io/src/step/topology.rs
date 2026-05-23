//! B-rep topology traversal: Shell → Face → Loop → Edge.

use super::parser::EntityIndex;
use super::value::StepValue;
use rc3d_core::math::Vec3;

#[derive(Debug, Clone)]
pub struct StepFace {
    pub bounds: Vec<StepLoop>,
    pub surface_id: Option<u64>,
    pub same_sense: bool,
}

#[derive(Debug, Clone)]
pub struct StepLoop {
    pub edges: Vec<StepEdge>,
}

#[derive(Debug, Clone)]
pub struct StepEdge {
    pub start: Vec3,
    pub end: Vec3,
    pub curve_id: u64,
    pub curve_type: String,
    pub reversed: bool,
}

/// A shell with its entity ID and extracted faces.
#[derive(Debug, Clone)]
pub struct StepShell {
    pub id: u64,
    pub faces: Vec<StepFace>,
}

/// Collect all Shell entities and extract their faces, keeping shells separate.
pub fn collect_shells(entities: &EntityIndex) -> Vec<StepShell> {
    use std::collections::HashSet;
    let mut shells = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();

    // First pass: collect shells referenced by MANIFOLD_SOLID_BREP / BREP_WITH_VOIDS
    for (&_id, record) in entities.iter() {
        match record.name.as_str() {
            "BREP_WITH_VOIDS" | "MANIFOLD_SOLID_BREP" => {
                if let Some(outer_id) = nth_ref(&record.params, 1) {
                    if seen.insert(outer_id) {
                        let faces = extract_shell_faces(outer_id, entities);
                        if !faces.is_empty() {
                            shells.push(StepShell { id: outer_id, faces });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Second pass: collect standalone shells (not already covered by brep)
    for (&id, record) in entities.iter() {
        match record.name.as_str() {
            "CLOSED_SHELL" | "OPEN_SHELL" | "SHELL" => {
                if seen.insert(id) {
                    let faces = extract_shell_faces(id, entities);
                    if !faces.is_empty() {
                        shells.push(StepShell { id, faces });
                    }
                }
            }
            _ => {}
        }
    }

    shells
}

/// Collect all Shell entities and extract their faces into a flat list.
pub fn collect_shell_faces(entities: &EntityIndex) -> Vec<StepFace> {
    collect_shells(entities).into_iter().flat_map(|s| s.faces).collect()
}

fn extract_shell_faces(shell_id: u64, entities: &EntityIndex) -> Vec<StepFace> {
    let record = match entities.get(&shell_id) {
        Some(r) => r,
        None => return vec![],
    };
    // Shell args: (name, (face1, face2, ...))
    let face_ids = match nth_list_refs(&record.params, 1) {
        Some(ids) => ids,
        None => return vec![],
    };

    let mut faces = Vec::new();
    for face_id in face_ids {
        if let Some(face) = resolve_face(face_id, entities) {
            faces.push(face);
        }
    }
    faces
}

fn resolve_face(face_id: u64, entities: &EntityIndex) -> Option<StepFace> {
    let record = entities.get(&face_id)?;
    match record.name.as_str() {
        "ADVANCED_FACE" | "FACE" | "FACE_SURFACE" => {
            resolve_face_surface(face_id, &record.params, entities)
        }
        "ORIENTED_FACE" => {
            let inner_id = nth_ref(&record.params, 3).or_else(|| nth_ref(&record.params, 1))?;
            resolve_face(inner_id, entities)
        }
        _ => None,
    }
}

fn resolve_face_surface(
    _face_id: u64,
    params: &StepValue,
    entities: &EntityIndex,
) -> Option<StepFace> {
    // Face args: (name, (bound1, bound2, ...))
    // ADVANCED_FACE args: (name, (bound1, ...), #surface, same_sense)
    let bound_ids = nth_list_refs(params, 1)?;

    // Try to get surface reference (position 2 for some face types)
    let surface_id = nth_ref(params, 2);
    let same_sense = match nth_enum(params, 3) {
        Some(s) => s == ".T.",
        None => true,
    };

    let mut bounds = Vec::new();
    for bound_id in bound_ids {
        if let Some(bloop) = resolve_bound(bound_id, entities) {
            bounds.push(bloop);
        }
    }

    if bounds.is_empty() {
        return None;
    }

    Some(StepFace { bounds, surface_id, same_sense })
}

fn resolve_bound(bound_id: u64, entities: &EntityIndex) -> Option<StepLoop> {
    let record = entities.get(&bound_id)?;
    match record.name.as_str() {
        "FACE_OUTER_BOUND" | "FACE_BOUND" => {
            // FaceBound args: (name, #loop, orientation)
            let loop_id = nth_ref(&record.params, 1)?;
            resolve_edge_loop(loop_id, entities)
        }
        _ => None,
    }
}

fn resolve_edge_loop(loop_id: u64, entities: &EntityIndex) -> Option<StepLoop> {
    let record = entities.get(&loop_id)?;
    if record.name != "EDGE_LOOP" && record.name != "POLY_LOOP" {
        return None;
    }

    if record.name == "POLY_LOOP" {
        return resolve_poly_loop(loop_id, entities);
    }

    // EDGE_LOOP args: (name, (edge1, edge2, ...))
    let edge_ids = nth_list_refs(&record.params, 1)?;
    let mut edges = Vec::new();
    for edge_id in edge_ids {
        if let Some(edge) = resolve_edge(edge_id, entities) {
            edges.push(edge);
        }
    }
    if edges.is_empty() { None } else { Some(StepLoop { edges }) }
}

fn resolve_poly_loop(loop_id: u64, entities: &EntityIndex) -> Option<StepLoop> {
    let record = entities.get(&loop_id)?;
    let pt_ids = nth_list_refs(&record.params, 1)?;
    let points: Vec<Vec3> = pt_ids.iter()
        .filter_map(|&id| resolve_point(id, entities))
        .collect();

    if points.len() < 3 { return None; }

    let mut edges = Vec::new();
    for i in 0..points.len() {
        let start = points[i];
        let end = points[(i + 1) % points.len()];
        edges.push(StepEdge {
            start,
            end,
            curve_id: 0,
            curve_type: "LINE".into(),
            reversed: false,
        });
    }
    Some(StepLoop { edges })
}

fn resolve_edge(edge_id: u64, entities: &EntityIndex) -> Option<StepEdge> {
    let record = entities.get(&edge_id)?;
    match record.name.as_str() {
        "EDGE_CURVE" => resolve_edge_curve(edge_id, entities, false),
        "ORIENTED_EDGE" => {
            // ORIENTED_EDGE args: (name, *, *, #edge_element, orientation)
            let inner_id = nth_ref(&record.params, 3)
                .or_else(|| nth_ref(&record.params, 1))?;
            let orient = nth_bool(&record.params, 4);
            resolve_edge_curve(inner_id, entities, !orient)
        }
        _ => None,
    }
}

fn resolve_edge_curve(edge_id: u64, entities: &EntityIndex, reversed: bool) -> Option<StepEdge> {
    let record = entities.get(&edge_id)?;
    // EDGE_CURVE args: (name, #start, #end, #curve, same_sense)
    let start_id = nth_ref(&record.params, 1)?;
    let end_id = nth_ref(&record.params, 2)?;
    let curve_id = nth_ref(&record.params, 3)?;

    let start_raw = resolve_point(start_id, entities)?;
    let end_raw = resolve_point(end_id, entities)?;

    let curve_type = entities.get(&curve_id)
        .map(|r| r.name.clone())
        .unwrap_or_else(|| "LINE".into());

    let (start, end) = if reversed { (end_raw, start_raw) } else { (start_raw, end_raw) };

    Some(StepEdge { start, end, curve_id, curve_type, reversed: false })
}

pub fn resolve_point(point_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let record = entities.get(&point_id)?;
    if record.name != "CARTESIAN_POINT" && record.name != "VERTEX_POINT" {
        return None;
    }
    if record.name == "VERTEX_POINT" {
        let inner = nth_ref(&record.params, 1)?;
        return resolve_point(inner, entities);
    }
    // CARTESIAN_POINT args: (name, (x, y, z))
    let coords = nth_list_params(&record.params, 1)?;
    if coords.len() < 3 { return None; }
    Some(Vec3::new(
        coords[0].as_real()? as f32,
        coords[1].as_real()? as f32,
        coords[2].as_real()? as f32,
    ))
}

pub fn resolve_placement(point_id: u64, entities: &EntityIndex) -> Option<(Vec3, Vec3, Vec3)> {
    let record = entities.get(&point_id)?;
    if record.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let origin_id = nth_ref(&record.params, 1)?;
    let axis_id = nth_ref(&record.params, 2)?;
    let refdir_id = nth_ref(&record.params, 3);

    let origin = resolve_point(origin_id, entities)?;
    let axis = resolve_direction(axis_id, entities).unwrap_or(Vec3::Z);
    let ref_dir = refdir_id.and_then(|id| resolve_direction(id, entities)).unwrap_or(Vec3::X);

    // Gram-Schmidt orthogonalization
    let z = axis.normalize();
    // Project ref_dir onto z's perpendicular plane
    let x_raw = ref_dir - z * ref_dir.dot(z);
    let x = if x_raw.length() > 1e-10 {
        x_raw.normalize()
    } else {
        // ref_dir is parallel to z, pick an arbitrary perpendicular X
        Vec3::Y.cross(z).normalize()
    };
    let _y = z.cross(x).normalize();

    Some((origin, x, z))
}

fn resolve_direction(dir_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let record = entities.get(&dir_id)?;
    if record.name != "DIRECTION" { return None; }
    let coords = nth_list_params(&record.params, 1)?;
    if coords.len() < 3 { return None; }
    let v = Vec3::new(
        coords[0].as_real()? as f32,
        coords[1].as_real()? as f32,
        coords[2].as_real()? as f32,
    );
    let len = v.length();
    if len > 1e-10 { Some(v / len) } else { None }
}

/// Public direction resolution for use by geom and other modules.
pub fn resolve_direction_public(dir_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    resolve_direction(dir_id, entities)
}

// ── Helpers ─────────────────────────────────────────────────

fn nth_ref(params: &StepValue, index: usize) -> Option<u64> {
    params.nth_param(index)?.as_ref_id()
}

fn nth_bool(params: &StepValue, index: usize) -> bool {
    match params.nth_param(index) {
        Some(StepValue::Enum(s)) => s == ".T.",
        _ => true,
    }
}

fn nth_enum(params: &StepValue, index: usize) -> Option<String> {
    match params.nth_param(index) {
        Some(StepValue::Enum(s)) => Some(s.clone()),
        _ => None,
    }
}

fn nth_list_refs(params: &StepValue, index: usize) -> Option<Vec<u64>> {
    let list = params.nth_param(index)?.as_list()?;
    Some(list.iter().filter_map(|v| v.as_ref_id()).collect())
}

fn nth_list_params(params: &StepValue, index: usize) -> Option<Vec<StepValue>> {
    match params.nth_param(index) {
        Some(StepValue::List(v)) => Some(v.clone()),
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser;

    fn make_exchange(data_section: &str) -> parser::EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data_section
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_advanced_face_resolution() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = DIRECTION('', (0.0, 0.0, 1.0));
#6 = AXIS2_PLACEMENT_3D('', #1, #5, #2);
#7 = PLANE('', #6);
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #7, .T.);
#17 = CLOSED_SHELL('', (#16));
#20 = LINE('', #1, #2);\
",
        );

        let shells = collect_shells(&entities);
        assert!(!shells.is_empty());
        let face = &shells[0].faces[0];
        assert!(face.surface_id.is_some());
        assert!(face.same_sense);
    }

    #[test]
    fn test_open_shell_collection() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = OPEN_SHELL('', (#16));
#20 = LINE('', #1, #2);\
",
        );

        let shells = collect_shells(&entities);
        assert_eq!(shells.len(), 1);
        assert_eq!(shells[0].faces.len(), 1);
    }

    #[test]
    fn test_brep_with_voids_extracts_outer() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = CLOSED_SHELL('', (#16));
#18 = BREP_WITH_VOIDS('', #17, ());
#20 = LINE('', #1, #2);\
",
        );

        let shells = collect_shells(&entities);
        // Should extract outer shell #17 from BREP_WITH_VOIDS #18
        assert!(!shells.is_empty());
        assert!(shells.iter().any(|s| s.id == 17));
    }
}
