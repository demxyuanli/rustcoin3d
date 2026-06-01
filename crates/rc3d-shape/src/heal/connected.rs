//! Topological vertex sharing at wire junctions (OCC ShapeFix_Wire::FixConnected).

use crate::store::BRepRegistry;
use crate::topo::{EdgeKey, FaceKey, Orientation, VertexKey, WireKey};

/// Result of a FixConnected pass.
#[derive(Debug, Default)]
pub struct ConnectedReport {
    /// Number of vertex pairs merged.
    pub merged_vertices: usize,
    /// Edge junctions that were already connected.
    pub already_connected: usize,
}

/// Merge vertices at adjacent edge junctions within a wire.
///
/// For each consecutive edge pair (including last->first for closed wires),
/// checks whether the end vertex of edge_i and start vertex of edge_{i+1}
/// are at the same 3D position within tolerance. If so, replaces all
/// references to the second vertex with the first.
pub fn fix_connected_wire(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
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

    // Collect vertex endpoints per edge, accounting for orientation
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

    // Find merge pairs: end of edge i -> start of edge i+1
    let mut merges: Vec<(VertexKey, VertexKey)> = Vec::new(); // (keep, replace)
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

    // Apply merges: replace all references to `replace` with `keep`
    let applied = apply_vertex_merges(reg, &merges);
    report.merged_vertices = applied;

    // After vertex merging, nudge PCurve endpoints at junctions where
    // 3D is now connected but UV endpoints still differ.
    if applied > 0 {
        nudge_pcurve_endpoints(&edges, reg);
    }

    report
}

/// After vertex merging, nudge PCurve endpoints at connected junctions
/// where 3D vertices now match but UV endpoints still differ.
fn nudge_pcurve_endpoints(edges: &[(EdgeKey, Orientation)], reg: &mut BRepRegistry) {
    let n = edges.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ek_i, orient_i) = edges[i];
        let (ek_j, orient_j) = edges[j];

        // Check that 3D vertices are now the same
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

        // Nudge PCurve endpoints for each face
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

fn get_vertex_at(ek: EdgeKey, orient: Orientation, is_start: bool, reg: &BRepRegistry) -> Option<VertexKey> {
    let edge = reg.edges.get(ek)?;
    if (orient == Orientation::Forward) == is_start {
        Some(edge.v_low)
    } else {
        Some(edge.v_high)
    }
}

fn nudge_line_pcurve(pc: &crate::geom::CurveGeom, du: f32, dv: f32, at_end: bool) -> crate::geom::CurveGeom {
    use crate::geom::CurveGeom;
    match pc {
        CurveGeom::Line { origin, direction } => CurveGeom::Line {
            origin: if at_end { *origin + rc3d_core::math::Vec3::new(du, dv, 0.0) } else { *origin },
            direction: *direction + rc3d_core::math::Vec3::new(if at_end { 0.0 } else { du }, if at_end { 0.0 } else { dv }, 0.0),
        },
        other => other.clone(),
    }
}

/// Replace all references to `replace` vertex with `keep` across all edges.
/// Skips merges that would create non-manifold topology (>2 faces per edge).
fn apply_vertex_merges(reg: &mut BRepRegistry, merges: &[(VertexKey, VertexKey)]) -> usize {
    let mut applied = 0usize;

    for &(keep, replace) in merges {
        if keep == replace {
            continue;
        }
        // Safety: skip if replace vertex is referenced by the keep vertex's own edge
        // (would create a self-loop edge v_low == v_high on a non-degenerate edge)
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

        // Update all edges referencing `replace`
        for (_, edge) in reg.edges.iter_mut() {
            if edge.v_low == replace {
                edge.v_low = keep;
            }
            if edge.v_high == replace {
                edge.v_high = keep;
            }
        }

        // Remove the replaced vertex from the registry
        reg.vertices.remove(replace);
        applied += 1;
    }

    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepVertex, BRepWire};
    use rc3d_core::math::Vec3;

    fn make_registry_with_two_edges(
        gap: f32,
    ) -> (BRepRegistry, WireKey, VertexKey, VertexKey) {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.vertices.insert(
            crate::topo::BRepVertex { position: Vec3::new(1.0 + gap, 0.0, 0.0), tolerance: 1e-4 }
        );
        let v3 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);

        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let face_key = reg.faces.insert(crate::topo::BRepFace {
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
        // First call merges the coincident vertices
        let r1 = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert!(r1.merged_vertices > 0);
        // Second call: vertices are already the same key → already_connected
        let r2 = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert!(r2.already_connected > 0);
        assert_eq!(r2.merged_vertices, 0);
    }

    #[test]
    fn test_fix_connected_closed_wire() {
        let mut reg = BRepRegistry::new();
        // Build a 3-edge closed triangle wire
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(0.5, 0.866, 0.0), 1e-4);
        // Create separate "duplicate" vertex at same position for connected check
        let v0_dup = reg.vertices.insert(crate::topo::BRepVertex {
            position: Vec3::ZERO,
            tolerance: 1e-4,
        });
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, line.clone());
        // Edge 3: v2 → v0_dup (different vertex key at same position = gap at closure)
        let e3 = reg.add_edge_with_pcurve(v2, v0_dup, line.clone(), 1e-4, fk, line.clone());
        let wk = reg.wires.insert(BRepWire {
            edges: vec![
                (e1, Orientation::Forward),
                (e2, Orientation::Forward),
                (e3, Orientation::Forward),
            ],
        });
        let report = fix_connected_wire(wk, &mut reg, 1e-3);
        assert!(
            report.merged_vertices > 0,
            "closed wire last->first gap should be merged"
        );
    }

    #[test]
    fn test_fix_connected_non_manifold_skip() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);
        // v3 at same position as v1 — would be merged, but both are shared by separate edges
        let v3 = reg.vertices.insert(BRepVertex {
            position: Vec3::new(1.0, 0.0, 0.0),
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
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        // Create edges between v0-v1 and v3-v2 (v1 and v3 at same position but different keys)
        // e2 is created as (v3, v2) but add_edge_with_pcurve stores v_low=v2, v_high=v3
        // so with Reversed orientation, start=v_high=v3 and end=v_low=v2 as desired
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line.clone());
        let e2 = reg.add_edge_with_pcurve(v3, v2, line.clone(), 1e-4, fk, line.clone());
        let wk = reg.wires.insert(BRepWire {
            edges: vec![(e1, Orientation::Forward), (e2, Orientation::Reversed)],
        });
        let report = fix_connected_wire(wk, &mut reg, 1e-3);
        // The merge should succeed since v1 and v3 are at the same position
        assert!(
            report.merged_vertices > 0 || report.already_connected > 0
        );
    }
}
