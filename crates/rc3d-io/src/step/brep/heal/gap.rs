//! Gap closing via vertex merging. T3.2

use super::super::topo::{WireKey, EdgeKey, FaceKey, Orientation, VertexKey};
use super::super::registry::BRepRegistry;
use rc3d_core::math::Vec3;

/// Close gaps between consecutive wire edges by merging nearby vertices.
/// When `closed` is false (open wire), the last→first pair is skipped.
pub fn close_wire_gaps(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
    closed: bool,
) -> usize {
    let edges = {
        let wire = match reg.wires.get(wire_key) {
            Some(w) => w.edges.clone(),
            None => return 0,
        };
        wire
    };

    if edges.len() <= 1 { return 0; }

    let mut gaps_closed = 0;
    let n = edges.len();
    let limit = if closed { n } else { n.saturating_sub(1) };

    for i in 0..limit {
        let j = (i + 1) % n;
        let (ek_i, orient_i) = edges[i];
        let (ek_j, orient_j) = edges[j];

        let (end_vk_i, start_vk_j) = match (
            get_oriented_endpoint(ek_i, orient_i, false, reg),
            get_oriented_endpoint(ek_j, orient_j, true, reg),
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
                gaps_closed += 1;
            }
        }
    }

    gaps_closed
}

fn get_oriented_endpoint(
    ek: EdgeKey,
    orient: Orientation,
    is_start: bool,
    reg: &BRepRegistry,
) -> Option<VertexKey> {
    let edge = reg.edges.get(ek)?;
    let forward = orient == Orientation::Forward;
    let at_start = is_start == forward;
    Some(if at_start { edge.v_low } else { edge.v_high })
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
    let mut gaps_closed = 0usize;

    for i in 0..n {
        let j = if i + 1 < n { i + 1 } else { 0 };
        let (ek_i, orient_i) = edges[i];
        let (ek_j, orient_j) = edges[j];

        // Check 3D connectivity
        let (end_i_3d, start_j_3d) = match (
            oriented_endpoint_position(ek_i, orient_i, false, reg),
            oriented_endpoint_position(ek_j, orient_j, true, reg),
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
                    gaps_closed += 1;
                }
            }
        }
    }

    gaps_closed
}

/// Get the 3D position of an edge's start or end, accounting for orientation.
fn oriented_endpoint_position(
    ek: EdgeKey,
    orient: Orientation,
    is_start: bool,
    reg: &BRepRegistry,
) -> Option<Vec3> {
    let vk = get_oriented_endpoint(ek, orient, is_start, reg)?;
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
fn translate_pcurve_endpoint(
    pc: &crate::step::brep::geom::CurveGeom,
    du: f32,
    dv: f32,
) -> crate::step::brep::geom::CurveGeom {
    use crate::step::brep::geom::CurveGeom;
    let shift = Vec3::new(du, dv, 0.0);
    match pc {
        CurveGeom::Line { origin, direction } => CurveGeom::Line {
            origin: *origin + shift,
            direction: *direction,
        },
        CurveGeom::Circle { center, axis, radius } => CurveGeom::Circle {
            center: *center + shift,
            axis: *axis,
            radius: *radius,
        },
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor } => CurveGeom::Ellipse {
            center: *center + shift,
            axis: *axis,
            semi_major: *semi_major,
            semi_minor: *semi_minor,
        },
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::CurveGeom;
    use crate::step::brep::geom::SurfaceGeom;
    use crate::step::brep::topo::{BRepFace, BRepVertex, BRepWire};
    use rc3d_core::math::Vec3;

    #[test]
    fn test_close_3d_gap_merges_nearby_vertices() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let gap_v = reg.vertices.insert(BRepVertex {
            position: Vec3::new(1.0, 0.00005, 0.0),
            tolerance: 1e-6,
        });
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = |a, b| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line(Vec3::ZERO, Vec3::X), 1e-4, fk, line(Vec3::ZERO, Vec3::X));
        let e2 = reg.add_edge_with_pcurve(gap_v, v2, line(Vec3::new(1.0, 0.00005, 0.0), Vec3::new(1.0, 1.0, 0.0)), 1e-4, fk, line(Vec3::X, Vec3::Y));
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
        ];
        let closed = close_wire_gaps(wk, &mut reg, 1e-3, false);
        assert!(closed > 0, "expected 3D gap merge, got {closed}");
    }

    #[test]
    fn test_close_3d_gap_quad_wire() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 1.0, 0.0), 1e-4);
        let gap_v = reg.vertices.insert(BRepVertex {
            position: Vec3::new(1.0, 0.00005, 0.0),
            tolerance: 1e-6,
        });
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = |a, b| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line(Vec3::ZERO, Vec3::X), 1e-4, fk, line(Vec3::ZERO, Vec3::X));
        let e2 = reg.add_edge_with_pcurve(gap_v, v2, line(Vec3::new(1.0, 0.00005, 0.0), Vec3::new(1.0, 1.0, 0.0)), 1e-4, fk, line(Vec3::new(1.0, 0.1, 0.0), Vec3::new(1.0, 1.0, 0.0)));
        let e2_orient = if gap_v < v2 {
            Orientation::Forward
        } else {
            Orientation::Reversed
        };
        let e3 = reg.add_edge_with_pcurve(v2, v3, line(Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)), 1e-4, fk, line(Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)));
        let e5 = reg.add_edge_with_pcurve(v3, v0, line(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO), 1e-4, fk, line(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO));
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, e2_orient),
            (e3, Orientation::Forward),
            (e5, Orientation::Forward),
        ];
        let closed = close_wire_gaps(wk, &mut reg, 1e-3, false);
        assert!(closed > 0, "quad wire 3D gap merge, got {closed}");
    }
}
