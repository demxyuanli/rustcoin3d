//! B-rep topology traversal: Shell → Face → Loop → Edge.

use std::collections::HashMap;
use super::parser::EntityIndex;
use super::value::StepValue;
use rc3d_core::math::Vec3;

#[derive(Debug, Clone)]
pub struct StepFace {
    pub bounds: Vec<StepLoop>,
    pub surface_id: Option<u64>,
    pub same_sense: bool,
    /// STEP entity ID of this face (for color/material lookup).
    pub face_id: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct StepLoop {
    pub edges: Vec<StepEdge>,
    /// Set for STEP `VERTEX_LOOP`: anchor point (OCC degenerated-edge wire).
    pub vertex_loop_point: Option<Vec3>,
}

#[derive(Debug, Clone)]
pub struct StepEdge {
    pub start: Vec3,
    pub end: Vec3,
    pub curve_id: u64,
    pub curve_type: String,
    pub reversed: bool,
    pub tolerance: f32,
}

/// A shell with its entity ID and extracted faces.
#[derive(Debug, Clone)]
pub struct StepShell {
    pub id: u64,
    pub faces: Vec<StepFace>,
}

/// Solid model: outer shell plus optional void shells from `BREP_WITH_VOIDS`.
#[derive(Debug, Clone)]
pub struct StepSolidModel {
    pub outer: StepShell,
    pub voids: Vec<StepShell>,
}

fn collect_void_shell_ids(record: &super::parser::EntityRecord) -> Vec<u64> {
    let mut voids = Vec::new();
    if let Some(list) = record.params.nth_param(2).and_then(|v| v.as_list()) {
        for item in list {
            if let Some(id) = item.as_ref_id() {
                voids.push(id);
            }
        }
    }
    voids
}

/// Collect manifold solids with void shells when present.
pub fn collect_solid_models(entities: &EntityIndex) -> Vec<StepSolidModel> {
    use std::collections::HashSet;
    let mut models = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();

    for (_, record) in entities.iter() {
        match record.name.as_str() {
            "BREP_WITH_VOIDS" | "MANIFOLD_SOLID_BREP" => {
                if let Some(outer_id) = nth_ref(&record.params, 1) {
                    if seen.insert(outer_id) {
                        let faces = extract_shell_faces(outer_id, entities);
                        if faces.is_empty() {
                            continue;
                        }
                        let voids = if record.name == "BREP_WITH_VOIDS" {
                            collect_void_shell_ids(record)
                                .into_iter()
                                .filter_map(|void_id| {
                                    if seen.insert(void_id) {
                                        let vf = extract_shell_faces(void_id, entities);
                                        if vf.is_empty() {
                                            None
                                        } else {
                                            Some(StepShell { id: void_id, faces: vf })
                                        }
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };
                        models.push(StepSolidModel {
                            outer: StepShell { id: outer_id, faces },
                            voids,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    for (&id, record) in entities.iter() {
        match record.name.as_str() {
            "CLOSED_SHELL" | "OPEN_SHELL" | "SHELL" => {
                if seen.insert(id) {
                    let faces = extract_shell_faces(id, entities);
                    if !faces.is_empty() {
                        models.push(StepSolidModel {
                            outer: StepShell { id, faces },
                            voids: Vec::new(),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    models
}

/// Extract global distance tolerance from UNCERTAINTY_MEASURE_WITH_UNIT entities.
pub fn global_tolerance(entities: &EntityIndex) -> f32 {
    for (_id, record) in entities.iter() {
        if record.name == "UNCERTAINTY_MEASURE_WITH_UNIT" {
            if let Some(typed) = record.params.nth_param(0) {
                if let StepValue::Typed(tag, inner) = typed {
                    if tag == "LENGTH_MEASURE" {
                        if let StepValue::Real(v) = inner.as_ref() {
                            return (*v as f32).clamp(1e-5, 0.01);
                        }
                    }
                }
            }
        }
    }
    1e-4
}

/// Collect all Shell entities and extract their faces, keeping shells separate.
pub fn collect_shells(entities: &EntityIndex) -> Vec<StepShell> {
    collect_solid_models(entities)
        .into_iter()
        .map(|m| m.outer)
        .collect()
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
            let mut face = resolve_face(inner_id, entities)?;
            face.face_id = Some(face_id);
            Some(face)
        }
        _ => None,
    }
}

fn resolve_face_surface(
    face_id: u64,
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

    Some(StepFace { bounds, surface_id, same_sense, face_id: Some(face_id) })
}

fn resolve_bound(bound_id: u64, entities: &EntityIndex) -> Option<StepLoop> {
    let record = entities.get(&bound_id)?;
    match record.name.as_str() {
        "FACE_OUTER_BOUND" | "FACE_BOUND" => {
            // FaceBound args: (name, #loop, orientation)
            let loop_id = nth_ref(&record.params, 1)?;
            // Orientation (.T./.F.) describes loop direction vs face normal; edge
            // connectivity is already encoded by ORIENTED_EDGE in the EDGE_LOOP.
            resolve_loop(loop_id, entities)
        }
        _ => None,
    }
}

/// Resolve EDGE_LOOP, POLY_LOOP, or VERTEX_LOOP (closed analytic surfaces).
fn resolve_loop(loop_id: u64, entities: &EntityIndex) -> Option<StepLoop> {
    let record = entities.get(&loop_id)?;
    match record.name.as_str() {
        "VERTEX_LOOP" => resolve_vertex_loop(loop_id, entities),
        "EDGE_LOOP" | "POLY_LOOP" => resolve_edge_loop(loop_id, entities),
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
    if edges.is_empty() {
        None
    } else {
        Some(StepLoop {
            edges,
            vertex_loop_point: None,
        })
    }
}

/// VERTEX_LOOP: (name, #loop_vertex) where loop_vertex is VERTEX_POINT.
fn resolve_vertex_loop(loop_id: u64, entities: &EntityIndex) -> Option<StepLoop> {
    let record = entities.get(&loop_id)?;
    if record.name != "VERTEX_LOOP" {
        return None;
    }
    let vertex_id = nth_ref(&record.params, 1)?;
    let anchor = resolve_point(vertex_id, entities)?;
    Some(StepLoop {
        edges: vec![],
        vertex_loop_point: Some(anchor),
    })
}

fn resolve_poly_loop(loop_id: u64, entities: &EntityIndex) -> Option<StepLoop> {
    let record = entities.get(&loop_id)?;
    let tol = global_tolerance(entities);
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
            tolerance: tol,
        });
    }
    Some(StepLoop {
        edges,
        vertex_loop_point: None,
    })
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
    let tol = global_tolerance(entities);
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

    Some(StepEdge { start, end, curve_id, curve_type, reversed: false, tolerance: tol })
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

/// AXIS1_PLACEMENT: rotation axis as (origin, unit direction).
pub fn resolve_axis1_placement(place_id: u64, entities: &EntityIndex) -> Option<(Vec3, Vec3)> {
    let record = entities.get(&place_id)?;
    if record.name != "AXIS1_PLACEMENT" {
        return None;
    }
    let origin_id = nth_ref(&record.params, 1)?;
    let axis_id = nth_ref(&record.params, 2)?;
    let origin = resolve_point(origin_id, entities)?;
    let axis = resolve_direction(axis_id, entities).unwrap_or(Vec3::Z);
    Some((origin, axis.normalize()))
}

/// Sweep/revolution axis from AXIS1_PLACEMENT or AXIS2_PLACEMENT_3D (Z axis).
pub fn resolve_sweep_axis(place_id: u64, entities: &EntityIndex) -> Option<(Vec3, Vec3)> {
    if let Some((origin, axis)) = resolve_axis1_placement(place_id, entities) {
        return Some((origin, axis));
    }
    resolve_placement(place_id, entities).map(|(origin, _, z)| (origin, z.normalize()))
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

pub fn resolve_direction(dir_id: u64, entities: &EntityIndex) -> Option<Vec3> {
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

/// Extract per-face colors from STYLED_ITEM → SURFACE_STYLE_FILL_AREA → COLOUR_RGB chain.
/// Returns a map from face entity ID to (R, G, B) color.
pub fn collect_face_colors(entities: &EntityIndex) -> HashMap<u64, [f32; 3]> {
    let mut colors: HashMap<u64, [f32; 3]> = HashMap::new();

    for (_, record) in entities.iter() {
        if record.name != "STYLED_ITEM" {
            continue;
        }
        // STYLED_ITEM(name, (#psa1, ...), #item)
        let style_list = record.params.nth_param(1).and_then(|v| v.as_list());
        let item_ref = record.params.nth_param(2).and_then(|v| v.as_ref_id());

        let color = extract_color_from_styles(style_list, entities);
        if let (Some(item_id), Some(rgb)) = (item_ref, color) {
            // The item may be a face or a shape representation containing faces
            for face_id in resolve_item_to_faces(item_id, entities) {
                colors.insert(face_id, rgb);
            }
        }
    }

    colors
}

/// Walk PRESENTATION_STYLE_ASSIGNMENT → SURFACE_STYLE_USAGE → SURFACE_STYLE_FILL_AREA
/// → FILL_AREA_STYLE → COLOUR_RGB chain to extract RGB triplet.
fn extract_color_from_styles(style_list: Option<&[StepValue]>, entities: &EntityIndex) -> Option<[f32; 3]> {
    let list = style_list?;
    for psa_val in list {
        let psa_id = psa_val.as_ref_id()?;
        let psa_record = entities.get(&psa_id)?;
        if psa_record.name != "PRESENTATION_STYLE_ASSIGNMENT" {
            continue;
        }
        let inner_styles = psa_record.params.nth_param(1).and_then(|v| v.as_list())?;
        for style_val in inner_styles {
            let style_id = style_val.as_ref_id()?;
            let style_record = entities.get(&style_id)?;
            match style_record.name.as_str() {
                "SURFACE_STYLE_USAGE" => {
                    // SURFACE_STYLE_USAGE(usage_type, #side_style)
                    let side_id = nth_ref(&style_record.params, 1)?;
                    if let Some(rgb) = resolve_surface_style_to_rgb(side_id, entities) {
                        return Some(rgb);
                    }
                }
                "SURFACE_SIDE_STYLE" => {
                    if let Some(rgb) = resolve_surface_style_to_rgb(style_id, entities) {
                        return Some(rgb);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn resolve_surface_style_to_rgb(side_style_id: u64, entities: &EntityIndex) -> Option<[f32; 3]> {
    let record = entities.get(&side_style_id)?;
    match record.name.as_str() {
        "SURFACE_SIDE_STYLE" => {
            // SURFACE_SIDE_STYLE(name, #fill_area_style)
            let fill_id = nth_ref(&record.params, 1)?;
            resolve_surface_style_to_rgb(fill_id, entities)
        }
        "SURFACE_STYLE_FILL_AREA" => {
            // SURFACE_STYLE_FILL_AREA(#fill_area)
            let fill_id = nth_ref(&record.params, 0)?;
            resolve_surface_style_to_rgb(fill_id, entities)
        }
        "FILL_AREA_STYLE" => {
            // FILL_AREA_STYLE(name, #fill_colour)
            let colour_id = nth_ref(&record.params, 1)?;
            resolve_colour_rgb(colour_id, entities)
        }
        _ => None,
    }
}

fn resolve_colour_rgb(colour_id: u64, entities: &EntityIndex) -> Option<[f32; 3]> {
    let record = entities.get(&colour_id)?;
    if record.name != "COLOUR_RGB" && record.name != "COLOUR" {
        return None;
    }
    let r = record.params.nth_param(1).and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
    let g = record.params.nth_param(2).and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
    let b = record.params.nth_param(3).and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
    Some([r, g, b])
}

/// Resolve a STYLED_ITEM target to a set of face entity IDs.
fn resolve_item_to_faces(item_id: u64, entities: &EntityIndex) -> Vec<u64> {
    // Direct: the item itself is a face
    if let Some(record) = entities.get(&item_id) {
        match record.name.as_str() {
            "ADVANCED_FACE" | "FACE" | "FACE_SURFACE" => {
                return vec![item_id];
            }
            "CLOSED_SHELL" | "OPEN_SHELL" | "SHELL" => {
                // Shell: return its face IDs
                let face_ids = nth_list_refs(&record.params, 1).unwrap_or_default();
                return face_ids.iter().flat_map(|&fid| resolve_face_recursive(fid, entities)).collect();
            }
            _ => {}
        }
    }
    // Indirect: look for this item as a member of a shape representation
    for (&eid, record) in entities.iter() {
        if record.name == "SHAPE_REPRESENTATION" || record.name == "ADVANCED_BREP_SHAPE_REPRESENTATION" {
            let items = record.params.nth_param(1).and_then(|v| v.as_list());
            if let Some(item_list) = items {
                for item in item_list {
                    if let Some(iid) = item.as_ref_id() {
                        if iid == item_id {
                            // Found the representation containing this item
                            return find_all_faces_in_representation(eid, entities);
                        }
                    }
                }
            }
        }
    }
    vec![]
}

fn resolve_face_recursive(face_id: u64, entities: &EntityIndex) -> Vec<u64> {
    let record = match entities.get(&face_id) {
        Some(r) => r,
        None => return vec![],
    };
    match record.name.as_str() {
        "ADVANCED_FACE" | "FACE" | "FACE_SURFACE" => vec![face_id],
        "ORIENTED_FACE" => {
            let inner_id = nth_ref(&record.params, 3).or_else(|| nth_ref(&record.params, 1));
            inner_id.map(|id| resolve_face_recursive(id, entities)).unwrap_or_default()
        }
        _ => vec![],
    }
}

fn find_all_faces_in_representation(rep_id: u64, entities: &EntityIndex) -> Vec<u64> {
    let record = match entities.get(&rep_id) {
        Some(r) => r,
        None => return vec![],
    };
    let items: Vec<StepValue> = record.params.nth_param(1).and_then(|v| v.as_list()).map(|s| s.to_vec()).unwrap_or_default();
    let mut faces = Vec::new();
    for item in &items {
        let id: u64 = match item.as_ref_id() {
            Some(x) => x,
            None => continue,
        };
        if let Some(r) = entities.get(&id) {
            match r.name.as_str() {
                "MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS" => {
                    if let Some(outer_shell_id) = nth_ref(&r.params, 1) {
                        if let Some(shell) = entities.get(&outer_shell_id) {
                            let face_ids = nth_list_refs(&shell.params, 1).unwrap_or_default();
                            faces.extend(face_ids.iter().flat_map(|&fid| resolve_face_recursive(fid, entities)));
                        }
                    }
                }
                "SHELL_BASED_SURFACE_MODEL" => {
                    let shell_ids: Vec<u64> = r.params.nth_param(1)
                        .and_then(|v| v.as_list())
                        .map(|l| l.iter().filter_map(|x| x.as_ref_id()).collect())
                        .unwrap_or_default();
                    for sid in shell_ids {
                        if let Some(shell) = entities.get(&sid) {
                            let face_ids = nth_list_refs(&shell.params, 1).unwrap_or_default();
                            faces.extend(face_ids.iter().flat_map(|&fid| resolve_face_recursive(fid, entities)));
                        }
                    }
                }
                "CLOSED_SHELL" | "OPEN_SHELL" | "SHELL" => {
                    let face_ids = nth_list_refs(&r.params, 1).unwrap_or_default();
                    faces.extend(face_ids.iter().flat_map(|&fid| resolve_face_recursive(fid, entities)));
                }
                _ => {}
            }
        }
    }
    faces
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

    #[test]
    fn test_brep_with_voids_extracts_void_shells() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = CARTESIAN_POINT('', (2.0, 2.0, 0.0));
#6 = CARTESIAN_POINT('', (8.0, 2.0, 0.0));
#7 = CARTESIAN_POINT('', (8.0, 8.0, 0.0));
#8 = CARTESIAN_POINT('', (2.0, 8.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = CLOSED_SHELL('', (#16));
#30 = EDGE_CURVE('', #5, #6, #20, .T.);
#31 = EDGE_CURVE('', #6, #7, #20, .T.);
#32 = EDGE_CURVE('', #7, #8, #20, .T.);
#33 = EDGE_CURVE('', #8, #5, #20, .T.);
#34 = EDGE_LOOP('', (#30, #31, #32, #33));
#35 = FACE_OUTER_BOUND('', #34, .T.);
#36 = FACE_SURFACE('', (#35));
#37 = CLOSED_SHELL('', (#36));
#18 = BREP_WITH_VOIDS('', #17, (#37));
#20 = LINE('', #1, #2);\
",
        );

        let models = collect_solid_models(&entities);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].outer.id, 17);
        assert_eq!(models[0].voids.len(), 1);
        assert_eq!(models[0].voids[0].id, 37);
    }
}
