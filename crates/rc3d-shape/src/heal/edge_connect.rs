//! OCC ShapeFix_EdgeConnect — merge geometrically coincident edge endpoints
//! across adjacent faces to close shell gaps.
//!
//! When two faces share an edge in the STEP model but the topological vertices
//! differ (different VertexKeys at the same 3D position), the shell has gaps.
//! This pass finds such pairs and merges the vertices, then updates the edges
//! to reference the shared vertex.

use rc3d_core::math::Vec3;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, VertexKey};
use std::collections::HashMap;

/// Number of edges with vertex merges applied.
pub fn connect_shell_edges(reg: &mut BRepStore, shell_key: crate::topo::ShellKey, tolerance: f32) -> usize {
    let shell = match reg.shells.get(shell_key) { Some(s) => s, None => return 0 };
    let tol = tolerance.max(1e-4);

    // Collect all edges with their face keys and endpoint positions
    let mut edge_endpoints: Vec<(EdgeKey, FaceKey, Vec3, Vec3)> = Vec::new();
    for &(fk, _) in &shell.faces {
        let face = match reg.faces.get(fk) { Some(f) => f, None => continue };
        let wire_keys = std::iter::once(face.outer_wire).chain(face.inner_wires.iter().copied());
        for wk in wire_keys {
            let wire = match reg.wires.get(wk) { Some(w) => w, None => continue };
            for &(ek, _) in &wire.edges {
                let edge = match reg.edges.get(ek) { Some(e) => e, None => continue };
                let lo = reg.vertices.get(edge.v_low).map(|v| v.position).unwrap_or(Vec3::ZERO);
                let hi = reg.vertices.get(edge.v_high).map(|v| v.position).unwrap_or(Vec3::ZERO);
                edge_endpoints.push((ek, fk, lo, hi));
            }
        }
    }

    // Build spatial index of vertex positions → VertexKey
    let mut pos_to_vk: HashMap<[i32; 3], VertexKey> = HashMap::new();
    for (vk, v) in reg.vertices.iter() {
        let key = quantize(v.position, tol);
        pos_to_vk.entry(key).or_insert(vk);
    }

    let mut merges = 0usize;

    // For each edge, ensure its vertices are in the shared spatial index.
    // If an edge's vertex position maps to a different VertexKey, merge.
    for (ek, _fk, lo, hi) in &edge_endpoints {
        let edge = match reg.edges.get(*ek) { Some(e) => e, None => continue };
        let v_lo = edge.v_low;
        let v_hi = edge.v_high;

        let key_lo = quantize(*lo, tol);
        let key_hi = quantize(*hi, tol);

        if let Some(&shared_vk) = pos_to_vk.get(&key_lo) {
            if shared_vk != v_lo {
                // Merge v_lo → shared_vk
                if let Some(edge_mut) = reg.edges.get_mut(*ek) {
                    edge_mut.v_low = shared_vk;
                    merges += 1;
                }
                // Update vertex_to_edges index
                if let Some(edges) = reg.vertex_to_edges.remove(&v_lo) {
                    for e in &edges {
                        if *e != *ek {
                            reg.vertex_to_edges.entry(shared_vk).or_default().push(*e);
                        }
                    }
                }
                reg.vertex_to_edges.entry(shared_vk).or_default().push(*ek);
            }
        }

        if let Some(&shared_vk) = pos_to_vk.get(&key_hi) {
            if shared_vk != v_hi {
                if let Some(edge_mut) = reg.edges.get_mut(*ek) {
                    edge_mut.v_high = shared_vk;
                    merges += 1;
                }
                if let Some(edges) = reg.vertex_to_edges.remove(&v_hi) {
                    for e in &edges {
                        if *e != *ek {
                            reg.vertex_to_edges.entry(shared_vk).or_default().push(*e);
                        }
                    }
                }
                reg.vertex_to_edges.entry(shared_vk).or_default().push(*ek);
            }
        }
    }

    merges
}

fn quantize(p: Vec3, tol: f32) -> [i32; 3] {
    let s = 1.0 / tol.max(1e-6);
    [
        (p.x * s).round() as i32,
        (p.y * s).round() as i32,
        (p.z * s).round() as i32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::*;
    use crate::geom::{CurveGeom, Curve2d, SurfaceGeom};
    use rc3d_core::math::Vec3;

    #[test]
    fn merge_coincident_vertices() {
        let mut reg = BRepStore::new();

        // Face 0 with its own wire
        let wk0 = reg.wires.insert(BRepWire { edges: vec![] });
        let f0 = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X },
            outer_wire: wk0, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });
        // Face 1 with its own wire
        let wk1 = reg.wires.insert(BRepWire { edges: vec![] });
        let f1 = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: Vec3::new(10.0, 0.0, 0.0), normal: Vec3::Z, u_dir: Vec3::X },
            outer_wire: wk1, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });

        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(5.0, 0.0, 0.0), 1e-4);
        let v1b = reg.vertices.insert(BRepVertex { position: Vec3::new(5.0, 1e-6, 0.0), tolerance: 1e-4 });
        let v2 = reg.find_or_add_vertex(Vec3::new(10.0, 0.0, 0.0), 1e-4);

        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let c = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let e0 = reg.add_edge_with_pcurve(v0, v1, c.clone(), 1e-4, f0, pc.clone(), true);
        let e1 = reg.add_edge_with_pcurve(v1b, v2, c.clone(), 1e-4, f1, pc.clone(), true);

        // Wire faces with their edges
        reg.wires.get_mut(wk0).unwrap().edges = vec![(e0, Orientation::Forward)];
        reg.wires.get_mut(wk1).unwrap().edges = vec![(e1, Orientation::Forward)];

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(f0, Orientation::Forward), (f1, Orientation::Forward)],
            closed: false, step_id: None,
        });

        let n = connect_shell_edges(&mut reg, sk, 1e-3);
        assert!(n > 0, "should merge coincident vertices");
    }
}
