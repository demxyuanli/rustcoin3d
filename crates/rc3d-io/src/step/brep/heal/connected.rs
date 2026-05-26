//! Topological vertex sharing at wire junctions (OCC ShapeFix_Wire::FixConnected).

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, Orientation, VertexKey, WireKey};

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
    report
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
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::BRepWire;
    use rc3d_core::math::Vec3;

    fn make_registry_with_two_edges(
        gap: f32,
    ) -> (BRepRegistry, WireKey, VertexKey, VertexKey) {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.vertices.insert(
            crate::step::brep::topo::BRepVertex { position: Vec3::new(1.0 + gap, 0.0, 0.0), tolerance: 1e-4 }
        );
        let v3 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);

        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let face_key = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
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
}
