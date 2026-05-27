//! Gap closing via vertex merging. T3.2

use super::super::topo::{WireKey, EdgeKey, FaceKey, Orientation, VertexKey};
use super::super::registry::BRepRegistry;
use rc3d_core::math::Vec3;

/// Close gaps between consecutive wire edges by merging nearby vertices.
/// Returns number of gaps closed.
pub fn close_wire_gaps(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
) -> usize {
    let edges = {
        let wire = match reg.wires.get(wire_key) {
            Some(w) => w.edges.clone(),
            None => return 0,
        };
        wire
    };

    if edges.len() <= 1 { return 0; }

    let mut closed = 0;
    let n = edges.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let (ek_i, _) = edges[i];
        let (ek_j, _) = edges[j];

        let (end_vk_i, start_vk_j) = match (
            get_endpoint_vertex(ek_i, reg, false),
            get_endpoint_vertex(ek_j, reg, true),
        ) {
            (Some(vi), Some(vj)) => (vi, vj),
            _ => continue,
        };

        if end_vk_i == start_vk_j {
            continue; // already connected
        }

        let pos_i = reg.vertices.get(end_vk_i).map(|v| v.position);
        let pos_j = reg.vertices.get(start_vk_j).map(|v| v.position);

        if let (Some(pi), Some(pj)) = (pos_i, pos_j) {
            let gap = (pi - pj).length();
            if gap > 0.0 && gap < tolerance {
                // Merge: replace start_vk_j with end_vk_i across all edges
                merge_vertex_references(reg, end_vk_i, start_vk_j);
                closed += 1;
            }
        }
    }

    closed
}

fn get_endpoint_vertex(ek: EdgeKey, reg: &BRepRegistry, is_start: bool) -> Option<VertexKey> {
    let edge = reg.edges.get(ek)?;
    Some(if is_start { edge.v_low } else { edge.v_high })
}

/// Replace all references to `replace` vertex with `keep` across all edges,
/// then remove `replace` from the registry.
fn merge_vertex_references(reg: &mut BRepRegistry, keep: VertexKey, replace: VertexKey) {
    for (_, edge) in reg.edges.iter_mut() {
        if edge.v_low == replace {
            edge.v_low = keep;
        }
        if edge.v_high == replace {
            edge.v_high = keep;
        }
    }
    reg.vertices.remove(replace);
}

/// Close gaps between PCurve endpoints of adjacent edges in UV space.
///
/// For each adjacent edge pair (including last to first for closed wires), if the
/// 3D endpoints are connected (gap < tol_3d) but the UV endpoints differ by
/// more than tol_uv, nudge the PCurve endpoint of edge_i to match edge_{i+1}.
pub fn close_wire_gaps_2d(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepRegistry,
    tol_3d: f32,
    tol_uv: f32,
) -> usize {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return 0; };
        wire.edges.clone()
    };

    if edges.len() < 2 {
        return 0;
    }

    let n = edges.len();
    let mut closed = 0usize;

    for i in 0..n {
        let j = if i + 1 < n { i + 1 } else { 0 };
        let (ek_i, orient_i) = edges[i];
        let (ek_j, orient_j) = edges[j];

        // Check 3D connectivity
        let (end_i_3d, start_j_3d) = match (
            get_oriented_endpoint(ek_i, orient_i, false, reg),
            get_oriented_endpoint(ek_j, orient_j, true, reg),
        ) {
            (Some(pe), Some(ps)) => (pe, ps),
            _ => continue,
        };

        let gap_3d = (end_i_3d - start_j_3d).length();
        if gap_3d > tol_3d {
            continue;
        }

        // Check UV connectivity
        let uv_end_i = pcurve_endpoint(ek_i, orient_i, false, face_key, reg);
        let uv_start_j = pcurve_endpoint(ek_j, orient_j, true, face_key, reg);

        if let (Some((u1, v1)), Some((u2, v2))) = (uv_end_i, uv_start_j) {
            let gap_uv = ((u1 - u2).powi(2) + (v1 - v2).powi(2)).sqrt();
            if gap_uv > tol_uv && gap_uv < tol_uv * 100.0 {
                if let Some(pc) = reg.pcurve_mut(ek_i, face_key) {
                    let du = u2 - u1;
                    let dv = v2 - v1;
                    *pc = translate_pcurve_endpoint(pc, du, dv);
                    closed += 1;
                }
            }
        }
    }

    closed
}

/// Get the 3D position of an edge's start or end, accounting for orientation.
fn get_oriented_endpoint(
    ek: EdgeKey,
    orient: Orientation,
    is_start: bool,
    reg: &BRepRegistry,
) -> Option<Vec3> {
    let edge = reg.edges.get(ek)?;
    let vk = match (orient, is_start) {
        (Orientation::Forward, true) | (Orientation::Reversed, false) => edge.v_low,
        _ => edge.v_high,
    };
    reg.vertices.get(vk).map(|v| v.position)
}

/// Get the UV coordinates at the start or end of an edge's PCurve for a face.
fn pcurve_endpoint(
    ek: EdgeKey,
    orient: Orientation,
    is_start: bool,
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Option<(f32, f32)> {
    let edge = reg.edges.get(ek)?;
    let pc = edge.pcurves.get(&face_key)?;
    let t = if (orient == Orientation::Forward) == is_start { 0.0 } else { 1.0 };
    let uv = pc.d0(t);
    Some((uv.x, uv.y))
}

/// Nudge a PCurve's endpoint by (du, dv) in UV space.
/// For Line-based PCurves: adjust the origin. For other types: clone unchanged.
fn translate_pcurve_endpoint(
    pc: &crate::step::brep::geom::CurveGeom,
    du: f32,
    dv: f32,
) -> crate::step::brep::geom::CurveGeom {
    use crate::step::brep::geom::CurveGeom;
    match pc {
        CurveGeom::Line { origin, direction } => CurveGeom::Line {
            origin: *origin + Vec3::new(du, dv, 0.0),
            direction: *direction,
        },
        other => other.clone(),
    }
}
