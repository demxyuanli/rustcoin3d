//! Wire-level operations: edge reordering + small edge removal.

use crate::store::BRepStore;
use crate::topo::{EdgeKey, Orientation, WireKey};
use rc3d_core::math::{Real, PVec3};
use rc3d_core::utils::hash::f64x3_quantized_bits;

/// 3-component hash key for vertex positions — no XOR collisions.
type VertHash = [u64; 3];

// ── Edge reordering (T3.1) ─────────────────────────────────────────

/// Reorder wire edges into a connected chain. Returns None if disconnected.
/// Uses collision-free [u64; 3] hash keys for vertex position matching
/// (replaces the previous XOR-packed u64 which had aliasing collisions).
pub(crate) fn reorder_wire_edges(
    edges: &[(EdgeKey, Orientation)],
    reg: &BRepStore,
) -> Option<Vec<(EdgeKey, Orientation)>> {
    if edges.len() <= 1 { return Some(edges.to_vec()); }

    let n = edges.len();
    let mut endpoints: Vec<Option<(VertHash, VertHash)>> = vec![None; n];
    for (i, (ek, orient)) in edges.iter().enumerate() {
        if let Some(edge) = reg.edges.get(*ek) {
            let (p0, p1) = if *orient == Orientation::Forward {
                (edge.v_low, edge.v_high)
            } else {
                (edge.v_high, edge.v_low)
            };
            let pos0 = reg.vertices.get(p0).map(|v| v.position);
            let pos1 = reg.vertices.get(p1).map(|v| v.position);
            if let (Some(p0), Some(p1)) = (pos0, pos1) {
                let h0 = f64x3_quantized_bits([p0.x, p0.y, p0.z]);
                let h1 = f64x3_quantized_bits([p1.x, p1.y, p1.z]);
                endpoints[i] = Some((h0, h1));
            }
        }
    }

    let mut used = vec![false; n];
    let mut result = Vec::with_capacity(n);

    let mut current = 0;
    used[current] = true;
    result.push(edges[current]);

    for _ in 1..n {
        let (_, ref_prev_end) = endpoints[current]?;
        let mut found = false;
        for j in 0..n {
            if used[j] { continue; }
            let (next_start, _) = endpoints[j]?;
            if next_start == ref_prev_end {
                current = j;
                used[current] = true;
                result.push(edges[current]);
                found = true;
                break;
            }
            let (_, next_end) = endpoints[j]?;
            if next_end == ref_prev_end {
                current = j;
                used[current] = true;
                result.push((edges[current].0, edges[current].1));
                found = true;
                break;
            }
        }
        if !found { return None; }
    }

    Some(result)
}

// ── Small edge removal ─────────────────────────────────────────────

/// Remove edges whose 3D curve length is below `min_length`.
///
/// Seam and degenerated edges are never removed — they may be zero-length in 3D
/// but are required for closed parametric faces (VERTEX_LOOP, poles).
///
/// Returns the updated edge list, or None if the wire should be removed entirely.
pub(crate) fn remove_small_edges(
    wire_key: WireKey,
    reg: &mut BRepStore,
    protected_edges: &[EdgeKey],
    min_length: Real,
) -> Option<Vec<(EdgeKey, Orientation)>> {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return None; };
        wire.edges.clone()
    };

    if edges.is_empty() {
        return Some(edges);
    }
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

fn edge_length_3d(edge: &crate::topo::BRepEdge, reg: &BRepStore) -> Real {
    let p0 = reg.vertices.get(edge.v_low).map(|v| v.position);
    let p1 = reg.vertices.get(edge.v_high).map(|v| v.position);
    match (p0, p1) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => f64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepWire, FaceKey};
    use rc3d_core::math::PVec3;

    fn make_wire_with_edges(
        reg: &mut BRepStore,
        edge_lengths: &[Real],
    ) -> (WireKey, Vec<EdgeKey>, FaceKey) {
        let surface = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
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

        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let mut edge_keys = Vec::new();
        let mut x = 0.0_f64;
        for &len in edge_lengths {
            let v0 = reg.find_or_add_vertex(PVec3::new(x, 0.0, 0.0), 1e-4);
            x += len;
            let v1 = reg.find_or_add_vertex(PVec3::new(x, 0.0, 0.0), 1e-4);
            let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, face_key, pc.clone(), true);
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
        let mut reg = BRepStore::new();
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
        let mut reg = BRepStore::new();
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
        let mut reg = BRepStore::new();
        let (wire_key, _edge_keys, _fk) = make_wire_with_edges(&mut reg, &[0.0]);
        let result = remove_small_edges(wire_key, &mut reg, &[], 1e-3);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_all_small_returns_none() {
        let mut reg = BRepStore::new();
        let (wire_key, _edge_keys, _fk) = make_wire_with_edges(&mut reg, &[0.0, 0.0]);
        let result = remove_small_edges(wire_key, &mut reg, &[], 1e-3);
        assert!(result.is_none(), "wire with all edges removed should return None");
    }

    #[test]
    fn test_small_inner_wire_removed() {
        let mut reg = BRepStore::new();
        let (wire_key, _edge_keys, face_key) =
            make_wire_with_edges(&mut reg, &[0.0, 0.0]);
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
