//! Topology collection and face/edge resolution.
use super::super::parser::EntityIndex;
use super::super::value::StepValue;
use super::helpers::*;
use super::placement::resolve_point;
use rc3d_core::math::Vec3;

#[derive(Debug, Clone)]
pub struct StepFace {
    pub bounds: Vec<StepLoop>,
    pub surface_id: Option<u64>,
    pub same_sense: bool,
    /// ORIENTED_FACE orientation flag relative to the underlying FACE.
    /// true => forward, false => reversed.
    pub oriented_forward: bool,
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
fn collect_void_shell_ids(record: &crate::step::parser::EntityRecord) -> Vec<u64> {
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
            // STEP ORIENTED_FACE orientation enum may appear at different indices
            // depending on schema flavor/writer.
            face.oriented_forward = nth_enum(&record.params, 4)
                .map(|v| v == ".T.")
                .or_else(|| nth_enum(&record.params, 2).map(|v| v == ".T."))
                .unwrap_or(true);
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

    Some(StepFace {
        bounds,
        surface_id,
        same_sense,
        oriented_forward: true,
        face_id: Some(face_id),
    })
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
