//! Wire junction repair: gap closing + vertex merging.

use crate::geom::CurveGeom;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, VertexKey, WireKey};
use rc3d_core::math::Vec3;

// ── 3D gap closing ─────────────────────────────────────────────────

/// Close gaps between consecutive wire edges by merging nearby vertices.
/// When `closed` is false (open wire), the last→first pair is skipped.
pub(crate) fn close_wire_gaps(
    wire_key: WireKey,
    reg: &mut BRepStore,
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
    reg: &BRepStore,
) -> Option<VertexKey> {
    let edge = reg.edges.get(ek)?;
    let forward = orient == Orientation::Forward;
    let at_start = is_start == forward;
    Some(if at_start { edge.v_low } else { edge.v_high })
}

/// Replace all references to `replace` vertex with `keep` across all edges,
/// then remove `replace` from the registry.
fn merge_vertex_references(reg: &mut BRepStore, keep: VertexKey, replace: VertexKey) {
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

// ── UV gap closing ─────────────────────────────────────────────────

/// Close gaps between PCurve endpoints of adjacent edges in UV space.
pub(crate) fn close_wire_gaps_2d(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
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

fn oriented_endpoint_position(
    ek: EdgeKey,
    orient: Orientation,
    is_start: bool,
    reg: &BRepStore,
) -> Option<Vec3> {
    let vk = get_oriented_endpoint(ek, orient, is_start, reg)?;
    reg.vertices.get(vk).map(|v| v.position)
}

fn pcurve_endpoint(
    ek: EdgeKey,
    orient: Orientation,
    is_start: bool,
    face_key: FaceKey,
    reg: &BRepStore,
) -> Option<(f32, f32)> {
    let edge = reg.edges.get(ek)?;
    let pc = edge.pcurves.get(&face_key)?;
    let t = if (orient == Orientation::Forward) == is_start { 0.0 } else { 1.0 };
    let uv = pc.d0(t);
    Some((uv.x, uv.y))
}

fn translate_pcurve_endpoint(pc: &CurveGeom, du: f32, dv: f32) -> CurveGeom {
    let shift = Vec3::new(du, dv, 0.0);
    match pc {
        CurveGeom::Line { origin, direction } => CurveGeom::Line {
            origin: *origin + shift,
            direction: *direction,
        },
        CurveGeom::Circle { center, axis, radius, .. } => CurveGeom::circle(
            *center + shift, *axis, *radius,
        ),
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor, .. } => CurveGeom::ellipse(
            *center + shift, *axis, *semi_major, *semi_minor,
        ),
        other => other.clone(),
    }
}

// ── FixConnected ───────────────────────────────────────────────────

/// Result of a FixConnected pass.
#[derive(Debug, Default)]
pub(crate) struct ConnectedReport {
    pub merged_vertices: usize,
    pub already_connected: usize,
}

/// Merge vertices at adjacent edge junctions within a wire.
pub(crate) fn fix_connected_wire(
    wire_key: WireKey,
    reg: &mut BRepStore,
    tolerance: f32,
) -> ConnectedReport {
    let mut report = ConnectedReport::default();

    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return report; };
        wire.edges.clone()
    };

    if edges.len() < 2 {
        return report;
    }

    let n = edges.len();
    let mut edge_verts: Vec<(VertexKey, VertexKey)> = Vec::with_capacity(n);
    for &(ek, orient) in &edges {
        let Some(edge) = reg.edges.get(ek) else { return report; };
        let (start, end) = if orient == Orientation::Reversed {
            (edge.v_high, edge.v_low)
        } else {
            (edge.v_low, edge.v_high)
        };
        edge_verts.push((start, end));
    }

    let mut merges: Vec<(VertexKey, VertexKey)> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let end_i = edge_verts[i].1;
        let start_j = edge_verts[j].0;

        if end_i == start_j {
            report.already_connected += 1;
            continue;
        }

        let pos_i = reg.vertices.get(end_i).map(|v| v.position);
        let pos_j = reg.vertices.get(start_j).map(|v| v.position);

        if let (Some(pi), Some(pj)) = (pos_i, pos_j) {
            let dist = (pi - pj).length();
            if dist < tolerance {
                merges.push((end_i, start_j));
            }
        }
    }

    let applied = apply_vertex_merges(reg, &merges);
    report.merged_vertices = applied;

    if applied > 0 {
        nudge_pcurve_endpoints(&edges, reg);
    }

    report
}

fn nudge_pcurve_endpoints(edges: &[(EdgeKey, Orientation)], reg: &mut BRepStore) {
    let n = edges.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ek_i, orient_i) = edges[i];
        let (ek_j, orient_j) = edges[j];

        let end_i_vk = match get_vertex_at(ek_i, orient_i, false, reg) {
            Some(v) => v,
            None => continue,
        };
        let start_j_vk = match get_vertex_at(ek_j, orient_j, true, reg) {
            Some(v) => v,
            None => continue,
        };
        if end_i_vk != start_j_vk {
            continue;
        }

        let face_keys: Vec<FaceKey> = {
            let edge_i = match reg.edges.get(ek_i) {
                Some(e) => e,
                None => continue,
            };
            edge_i.pcurves.keys().copied().collect()
        };
        for face_key in face_keys {
            let (uv_i, uv_j, at_end) = {
                let edge_i = match reg.edges.get(ek_i) {
                    Some(e) => e,
                    None => continue,
                };
                let edge_j = match reg.edges.get(ek_j) {
                    Some(e) => e,
                    None => continue,
                };
                let pc_i = match edge_i.pcurves.get(&face_key) {
                    Some(p) => p,
                    None => continue,
                };
                let pc_j = match edge_j.pcurves.get(&face_key) {
                    Some(p) => p,
                    None => continue,
                };
                let t_i = if orient_i == Orientation::Reversed { 0.0 } else { 1.0 };
                let t_j = if orient_j == Orientation::Reversed { 1.0 } else { 0.0 };
                (pc_i.d0(t_i), pc_j.d0(t_j), t_i > 1.0 - 1e-6)
            };

            let dist = ((uv_i.x - uv_j.x).powi(2) + (uv_i.y - uv_j.y).powi(2)).sqrt();
            if dist > 1e-10 && dist < 1e-3 {
                if let Some(pc) = reg.pcurve_mut(ek_i, face_key) {
                    *pc = nudge_line_pcurve(pc, uv_j.x - uv_i.x, uv_j.y - uv_i.y, at_end);
                }
            }
        }
    }
}

fn get_vertex_at(ek: EdgeKey, orient: Orientation, is_start: bool, reg: &BRepStore) -> Option<VertexKey> {
    let edge = reg.edges.get(ek)?;
    if (orient == Orientation::Forward) == is_start {
        Some(edge.v_low)
    } else {
        Some(edge.v_high)
    }
}

fn nudge_line_pcurve(pc: &CurveGeom, du: f32, dv: f32, at_end: bool) -> CurveGeom {
    match pc {
        CurveGeom::Line { origin, direction } => CurveGeom::Line {
            origin: if at_end { *origin + Vec3::new(du, dv, 0.0) } else { *origin },
            direction: *direction + Vec3::new(if at_end { 0.0 } else { du }, if at_end { 0.0 } else { dv }, 0.0),
        },
        other => other.clone(),
    }
}

fn apply_vertex_merges(reg: &mut BRepStore, merges: &[(VertexKey, VertexKey)]) -> usize {
    let mut applied = 0usize;

    for &(keep, replace) in merges {
        if keep == replace {
            continue;
        }
        let mut safe = true;
        for (_, edge) in reg.edges.iter() {
            let uses_replace = edge.v_low == replace || edge.v_high == replace;
            let uses_keep = edge.v_low == keep || edge.v_high == keep;
            if uses_replace && uses_keep && keep != replace {
                safe = false;
                break;
            }
        }
        if !safe {
            log::warn!(
                "[BRep heal] FixConnected: skipping merge {:?}->{:?} (would create self-loop)",
                replace, keep
            );
            continue;
        }

        for (_, edge) in reg.edges.iter_mut() {
            if edge.v_low == replace {
                edge.v_low = keep;
            }
            if edge.v_high == replace {
                edge.v_high = keep;
            }
        }

        reg.vertices.remove(replace);
        applied += 1;
    }

    applied
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepVertex, BRepWire};
    use rc3d_core::math::Vec3;

    // ── gap tests ──

    #[test]
    fn test_close_3d_gap_merges_nearby_vertices() {
        let mut reg = BRepStore::new();
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
        let mut reg = BRepStore::new();
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
        let e2_orient = if gap_v < v2 { Orientation::Forward } else { Orientation::Reversed };
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

    // ── connected tests ──

    fn make_registry_with_two_edges(gap: f32) -> (BRepStore, WireKey, VertexKey, VertexKey) {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.vertices.insert(BRepVertex { position: Vec3::new(1.0 + gap, 0.0, 0.0), tolerance: 1e-4 });
        let v3 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);

        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let face_key = reg.faces.insert(BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let pcurve = line.clone();
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, face_key, pcurve.clone());
        let e2 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, face_key, pcurve.clone());

        let wire_key = reg.wires.insert(BRepWire {
            edges: vec![(e1, Orientation::Forward), (e2, Orientation::Forward)],
        });

        (reg, wire_key, v1, v2)
    }

    #[test]
    fn test_fix_connected_adjacent_edges() {
        let (mut reg, wire_key, _v1, _v2) = make_registry_with_two_edges(0.0);
        let report = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert_eq!(report.merged_vertices, 1, "vertices at the same position should merge");
        let wire = reg.wires.get(wire_key).unwrap();
        let e1 = reg.edges.get(wire.edges[0].0).unwrap();
        let e2 = reg.edges.get(wire.edges[1].0).unwrap();
        assert_eq!(e1.v_high, e2.v_low, "merged: end of e1 == start of e2");
    }

    #[test]
    fn test_fix_connected_outside_tolerance() {
        let (mut reg, wire_key, _v1, _v2) = make_registry_with_two_edges(0.1);
        let report = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert_eq!(report.merged_vertices, 0, "gap > tolerance should not merge");
    }

    #[test]
    fn test_fix_connected_already_connected() {
        let (mut reg, wire_key, _v1, _v2) = make_registry_with_two_edges(0.0);
        let r1 = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert!(r1.merged_vertices > 0);
        let r2 = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert!(r2.already_connected > 0);
        assert_eq!(r2.merged_vertices, 0);
    }

    #[test]
    fn test_fix_connected_closed_wire() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(0.5, 0.866, 0.0), 1e-4);
        let v0_dup = reg.vertices.insert(BRepVertex {
            position: Vec3::ZERO,
            tolerance: 1e-4,
        });
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, line.clone());
        let e3 = reg.add_edge_with_pcurve(v2, v0_dup, line.clone(), 1e-4, fk, line.clone());
        let wk = reg.wires.insert(BRepWire {
            edges: vec![
                (e1, Orientation::Forward),
                (e2, Orientation::Forward),
                (e3, Orientation::Forward),
            ],
        });
        let report = fix_connected_wire(wk, &mut reg, 1e-3);
        assert!(report.merged_vertices > 0, "closed wire last->first gap should be merged");
    }

    #[test]
    fn test_fix_connected_non_manifold_skip() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);
        let v3 = reg.vertices.insert(BRepVertex {
            position: Vec3::new(1.0, 0.0, 0.0),
            tolerance: 1e-4,
        });
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line.clone());
        let e2 = reg.add_edge_with_pcurve(v3, v2, line.clone(), 1e-4, fk, line.clone());
        let wk = reg.wires.insert(BRepWire {
            edges: vec![(e1, Orientation::Forward), (e2, Orientation::Reversed)],
        });
        let report = fix_connected_wire(wk, &mut reg, 1e-3);
        assert!(report.merged_vertices > 0 || report.already_connected > 0);
    }
}
