//! Small edge removal (OCC ShapeFix_Wire::FixSmall).

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, Orientation, WireKey};

/// Remove edges whose 3D curve length is below `min_length`.
///
/// Seam and degenerated edges are never removed — they may be zero-length in 3D
/// but are required for closed parametric faces (VERTEX_LOOP, poles).
///
/// Returns the updated edge list, or None if the wire should be removed entirely.
pub fn remove_small_edges(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
    protected_edges: &[EdgeKey],
    min_length: f32,
) -> Option<Vec<(EdgeKey, Orientation)>> {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return None; };
        wire.edges.clone()
    };

    if edges.is_empty() {
        return Some(edges);
    }
    // Keep single-edge wires (e.g. circular plane boundary from STEP) — do not invalidate the wire.
    if edges.len() == 1 {
        return Some(edges);
    }

    let mut keep: Vec<bool> = vec![true; edges.len()];
    let mut removed = 0usize;

    for (i, &(ek, _)) in edges.iter().enumerate() {
        if protected_edges.contains(&ek) {
            continue;
        }

        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };

        let len = edge_length_3d(edge, reg);
        if len < min_length {
            keep[i] = false;
            removed += 1;
        }
    }

    if removed == 0 {
        return Some(edges);
    }

    let mut result: Vec<(EdgeKey, Orientation)> = Vec::new();
    for (i, &(ek, orient)) in edges.iter().enumerate() {
        if keep[i] {
            result.push((ek, orient));
        }
    }

    if result.is_empty() {
        return None;
    }
    if result.len() == 1 {
        return Some(result);
    }

    if let Some(wire) = reg.wires.get_mut(wire_key) {
        wire.edges = result.clone();
    }

    Some(result)
}

fn edge_length_3d(edge: &crate::step::brep::topo::BRepEdge, reg: &BRepRegistry) -> f32 {
    let p0 = reg.vertices.get(edge.v_low).map(|v| v.position);
    let p1 = reg.vertices.get(edge.v_high).map(|v| v.position);
    match (p0, p1) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => f32::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::{BRepWire, FaceKey};
    use rc3d_core::math::Vec3;

    fn make_wire_with_edges(
        reg: &mut BRepRegistry,
        edge_lengths: &[f32],
    ) -> (WireKey, Vec<EdgeKey>, FaceKey) {
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let face_key = reg.faces.insert(crate::step::brep::topo::BRepFace {
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
        let mut edge_keys = Vec::new();
        let mut x = 0.0f32;
        for &len in edge_lengths {
            let v0 = reg.find_or_add_vertex(Vec3::new(x, 0.0, 0.0), 1e-4);
            x += len;
            let v1 = reg.find_or_add_vertex(Vec3::new(x, 0.0, 0.0), 1e-4);
            let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, face_key, line.clone());
            edge_keys.push(ek);
        }
        let wire_edges: Vec<(EdgeKey, Orientation)> = edge_keys
            .iter()
            .map(|&ek| (ek, Orientation::Forward))
            .collect();
        let wire_key = reg.wires.insert(BRepWire { edges: wire_edges });
        (wire_key, edge_keys, face_key)
    }

    #[test]
    fn test_remove_zero_length_edge() {
        let mut reg = BRepRegistry::new();
        let (wire_key, _edge_keys, _fk) = make_wire_with_edges(&mut reg, &[1.0, 0.0, 2.0]);
        let before_len = reg.wires.get(wire_key).unwrap().edges.len();
        let result = remove_small_edges(wire_key, &mut reg, &[], 1e-6);
        assert!(result.is_some());
        let after = result.unwrap().len();
        assert_eq!(after, 2, "zero-length edge should be removed");
        assert_eq!(before_len, 3, "should have started with 3 edges");
        assert!(reg.wires.get(wire_key).unwrap().edges.len() == 2);
    }

    #[test]
    fn test_skip_seam_edge() {
        let mut reg = BRepRegistry::new();
        let (wire_key, edge_keys, face_key) = make_wire_with_edges(&mut reg, &[0.0]);
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges = vec![edge_keys[0]];
        }
        let result = remove_small_edges(wire_key, &mut reg, &[edge_keys[0]], 1e-6);
        let edges = result.unwrap();
        assert_eq!(edges.len(), 1, "seam edge should not be removed even if zero-length");
    }

    #[test]
    fn test_single_edge_wire_preserved() {
        let mut reg = BRepRegistry::new();
        let (wire_key, _edge_keys, _fk) = make_wire_with_edges(&mut reg, &[0.0]);
        let result = remove_small_edges(wire_key, &mut reg, &[], 1e-3);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_all_small_returns_none() {
        let mut reg = BRepRegistry::new();
        let (wire_key, _edge_keys, _fk) = make_wire_with_edges(&mut reg, &[0.0, 0.0]);
        let result = remove_small_edges(wire_key, &mut reg, &[], 1e-3);
        assert!(result.is_none(), "wire with all edges removed should return None");
    }

    #[test]
    fn test_small_inner_wire_removed() {
        let mut reg = BRepRegistry::new();
        let (wire_key, _edge_keys, face_key) =
            make_wire_with_edges(&mut reg, &[0.0, 0.0]);
        // Make this an inner wire by updating the face
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.inner_wires.push(wire_key);
        }
        let result = remove_small_edges(wire_key, &mut reg, &[], 1e-3);
        assert!(
            result.is_none(),
            "inner wire with all small edges removed should return None"
        );
    }
}
