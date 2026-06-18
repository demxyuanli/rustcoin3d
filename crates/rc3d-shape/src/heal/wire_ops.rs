//! Wire-level operations: edge reordering + small edge removal +
//! notch/tail fixing.

use std::collections::HashMap;

use crate::store::BRepStore;
use crate::topo::{BRepEdge, EdgeKey, Orientation, VertexKey, WireKey};
use crate::geom::CurveGeom;
use rc3d_core::math::{PVec3, Real};
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

// ── Notched edge fixing ─────────────────────────────────────────────

/// Fix notched edges in a wire (OCC ShapeFix_Wire::FixNotchedEdges).
///
/// Detects edges whose 3D curve has a sharp internal bend (tangent angle
/// change > threshold) and splits the edge at the notch point, preserving
/// the original curve geometry on each segment.
///
/// Returns the number of notches fixed (edges split).
pub(crate) fn fix_notched_edges(
    wire_key: WireKey,
    reg: &mut BRepStore,
    angle_threshold_deg: Real,
) -> usize {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return 0 };
        if wire.edges.is_empty() { return 0; }
        wire.edges.clone()
    };

    let angle_threshold_rad = angle_threshold_deg.to_radians();
    // Cosine of the threshold: if the dot product of normalized tangents
    // is below this value, the angle between them exceeds the threshold.
    let cos_threshold = angle_threshold_rad.cos();
    let mut splits: Vec<(usize, Real, VertexKey)> = Vec::new(); // (edge_idx, split_t, new_vertex)

    // Detect notches WITHIN each edge by sampling the curve tangents
    // at 64 uniformly spaced points and checking for abrupt direction changes.
    for (i, &(ek, _orient)) in edges.iter().enumerate() {
        let Some(edge) = reg.edges.get(ek) else { continue };
        let n_pts = 64usize;
        let mut prev_dir: Option<PVec3> = None;
        for k in 0..=n_pts {
            let t = k as Real / n_pts as Real;
            let d1 = edge.curve.d1(t);
            let len = d1.length();
            if len < 1e-12 { prev_dir = None; continue; }
            let dir = d1 / len;
            if let Some(pd) = prev_dir {
                let cos_a = dir.dot(pd).clamp(-1.0, 1.0);
                if cos_a < cos_threshold {
                    // Sharp turn detected at this point — split here
                    let pos = edge.curve.d0(t);
                    let vk = reg.find_or_add_vertex(pos, edge.tolerance.max(1e-6));
                    splits.push((i, t, vk));
                    break; // one split per edge is sufficient
                }
            }
            prev_dir = Some(dir);
        }
    }

    if splits.is_empty() { return 0; }

    // Apply splits: replace each notched edge with two edges.
    let mut new_edges: Vec<(EdgeKey, Orientation)> = Vec::new();
    for (i, &(ek, orient)) in edges.iter().enumerate() {
        if let Some(&(_, split_t, vk)) = splits.iter().find(|(idx, _, _)| *idx == i) {
            // Extract data before mutable ops (avoid borrow conflict)
            let (curve_clone, tol, p0, p1, pcurves_clone) = {
                let Some(edge) = reg.edges.get(ek) else { continue };
                (edge.curve.clone(), edge.tolerance,
                 edge.curve.d0(0.0), edge.curve.d0(1.0),
                 edge.pcurves.clone())
            };
            let basis = Box::new(curve_clone);

            // Segment A: t ∈ [0, split_t]
            let seg_a = CurveGeom::Trimmed { basis: basis.clone(), t_min: 0.0, t_max: split_t };
            let v0 = reg.find_or_add_vertex(p0, tol);
            let (v_lo_a, v_hi_a) = if v0 < vk { (v0, vk) } else { (vk, v0) };
            let ek_a = reg.edges.insert(BRepEdge {
                curve: seg_a, tolerance: tol,
                v_low: v_lo_a, v_high: v_hi_a,
                t_min: 0.0, t_max: 1.0,
                cached_deflection: None,
                pcurves: pcurves_clone.clone(),
            });
            reg.vertex_to_edges.entry(v_lo_a).or_default().push(ek_a);
            if v_lo_a != v_hi_a { reg.vertex_to_edges.entry(v_hi_a).or_default().push(ek_a); }
            new_edges.push((ek_a, orient));

            // Segment B: t ∈ [split_t, 1.0]
            let seg_b = CurveGeom::Trimmed { basis, t_min: split_t, t_max: 1.0 };
            let v1 = reg.find_or_add_vertex(p1, tol);
            let (v_lo_b, v_hi_b) = if vk < v1 { (vk, v1) } else { (v1, vk) };
            let ek_b = reg.edges.insert(BRepEdge {
                curve: seg_b, tolerance: tol,
                v_low: v_lo_b, v_high: v_hi_b,
                t_min: 0.0, t_max: 1.0,
                cached_deflection: None,
                pcurves: pcurves_clone,
            });
            reg.vertex_to_edges.entry(v_lo_b).or_default().push(ek_b);
            if v_lo_b != v_hi_b { reg.vertex_to_edges.entry(v_hi_b).or_default().push(ek_b); }
            new_edges.push((ek_b, orient));
        } else {
            new_edges.push((ek, orient));
        }
    }

    let fixed = splits.len();
    if let Some(wire) = reg.wires.get_mut(wire_key) {
        wire.edges = new_edges;
    }
    fixed
}

// ── Tail edge removal ───────────────────────────────────────────────

/// Remove small "tail" edges (dead-end edges at wire junctions).
/// A tail has one vertex shared by only one other edge in the wire,
/// and is shorter than `min_length`.
///
/// Returns the number of tails removed.
pub(crate) fn fix_tails(
    wire_key: WireKey,
    reg: &mut BRepStore,
    min_length: Real,
) -> usize {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return 0 };
        if wire.edges.len() < 3 {
            return 0;
        }
        wire.edges.clone()
    };

    let n = edges.len();

    // Count vertex occurrences in the wire.
    let mut vertex_count: HashMap<VertexKey, usize> = HashMap::new();
    for &(ek, orient) in &edges {
        let Some(edge) = reg.edges.get(ek) else { continue };
        let (v0, v1) = if orient == Orientation::Forward {
            (edge.v_low, edge.v_high)
        } else {
            (edge.v_high, edge.v_low)
        };
        *vertex_count.entry(v0).or_default() += 1;
        *vertex_count.entry(v1).or_default() += 1;
    }

    // Find edges with a dead-end vertex (count == 1) and short length.
    let mut remove = vec![false; n];
    let mut removed = 0usize;

    for (i, &(ek, orient)) in edges.iter().enumerate() {
        let Some(edge) = reg.edges.get(ek) else { continue };
        let (v0, v1) = if orient == Orientation::Forward {
            (edge.v_low, edge.v_high)
        } else {
            (edge.v_high, edge.v_low)
        };

        let len = edge_length_3d(edge, reg);
        if len < min_length
            && (vertex_count.get(&v0) == Some(&1)
                || vertex_count.get(&v1) == Some(&1))
        {
            remove[i] = true;
            removed += 1;
        }
    }

    if removed > 0 {
        let new_edges: Vec<(EdgeKey, Orientation)> = edges
            .iter()
            .enumerate()
            .filter(|&(i, _)| !remove[i])
            .map(|(_, &e)| e)
            .collect();

        if let Some(wire) = reg.wires.get_mut(wire_key) {
            wire.edges = new_edges;
        }
    }

    removed
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

    // ── Notched edge tests ──────────────────────────────────────────

    /// Create a wire with edges at given vertex positions (open chain).
    /// Each pair of consecutive positions defines one edge with a Line curve.
    fn make_wire_from_positions(
        reg: &mut BRepStore,
        positions: &[PVec3],
    ) -> (WireKey, Vec<EdgeKey>) {
        assert!(positions.len() >= 2);
        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let mut edge_keys = Vec::new();
        for w in positions.windows(2) {
            let v0 = reg.find_or_add_vertex(w[0], 1e-4);
            let v1 = reg.find_or_add_vertex(w[1], 1e-4);
            // Direct edge insertion (no face/pcurve needed for topology-only tests).
            let (v_lo, v_hi) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            let curve = crate::geom::CurveGeom::Line {
                origin: w[0],
                direction: w[1] - w[0],
            };
            let ek = reg.edges.insert(BRepEdge {
                curve,
                tolerance: 1e-4,
                v_low: v_lo,
                v_high: v_hi,
                t_min: 0.0,
                t_max: 1.0,
                cached_deflection: None,
                pcurves: HashMap::new(),
            });
            reg.vertex_to_edges.entry(v_lo).or_default().push(ek);
            if v_lo != v_hi {
                reg.vertex_to_edges.entry(v_hi).or_default().push(ek);
            }
            edge_keys.push(ek);
        }

        let wire_edges: Vec<(EdgeKey, Orientation)> = edge_keys
            .iter()
            .enumerate()
            .map(|(i, &ek)| {
                let v_start = reg.find_or_add_vertex(positions[i], 1e-4);
                let edge = reg.edges.get(ek).unwrap();
                let orient = if edge.v_low == v_start {
                    Orientation::Forward
                } else {
                    Orientation::Reversed
                };
                (ek, orient)
            })
            .collect();
        let wire_key = reg.wires.insert(BRepWire {
            edges: wire_edges,
        });
        (wire_key, edge_keys)
    }

    #[test]
    fn test_fix_notched_edges_splits_sharp_bend() {
        let mut reg = BRepStore::new();
        // Single edge with a Polyline curve that has a sharp ~45° internal bend.
        // OCCT: the edge should be split at the bend into two edges.
        let pts = vec![
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(1.0, 1.0, 0.0), // 90° bend at t≈0.5
        ];
        let poly = CurveGeom::Polyline { points: pts.clone() };
        let v0 = reg.find_or_add_vertex(pts[0], 1e-4);
        let v1 = reg.find_or_add_vertex(pts[2], 1e-4);
        let (v_lo, v_hi) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        let ek = reg.edges.insert(BRepEdge {
            curve: poly, tolerance: 1e-4,
            v_low: v_lo, v_high: v_hi, t_min: 0.0, t_max: 1.0,
            cached_deflection: None, pcurves: HashMap::new(),
        });
        reg.vertex_to_edges.entry(v_lo).or_default().push(ek);
        if v_lo != v_hi { reg.vertex_to_edges.entry(v_hi).or_default().push(ek); }
        let wire_key = reg.wires.insert(BRepWire {
            edges: vec![(ek, Orientation::Forward)],
        });
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 1);

        let fixed = fix_notched_edges(wire_key, &mut reg, 45.0);
        assert_eq!(fixed, 1, "polyline with 90° bend should be split");
        assert_eq!(
            reg.wires.get(wire_key).unwrap().edges.len(),
            2,
            "single edge split into two"
        );
    }

    #[test]
    fn test_fix_notched_edges_noop_on_straight_line() {
        let mut reg = BRepStore::new();
        // Straight line — no internal bends.
        let p0 = PVec3::new(0.0, 0.0, 0.0);
        let p1 = PVec3::new(10.0, 0.0, 0.0);
        let (wire_key, _) = make_wire_from_positions(&mut reg, &[p0, p1]);
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 1);

        let fixed = fix_notched_edges(wire_key, &mut reg, 10.0);
        assert_eq!(fixed, 0, "straight line should not be split");
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 1);
    }

    // ── Tail edge tests ─────────────────────────────────────────────

    #[test]
    fn test_fix_tails_removes_short_dead_end() {
        let mut reg = BRepStore::new();
        // Wire: A → B (short tail) then B → C → D → A (main loop back)
        // The short edge A→B has B shared with only one other edge (B→C),
        // making A a dead-end vertex (count == 1).
        // This is actually a Y-junction, but in a simple wire chain,
        // we create a pattern where the short edge has a dead-end.
        let p0 = PVec3::new(0.0, 0.0, 0.0);
        let p1 = PVec3::new(0.01, 0.0, 0.0); // very short edge from p0→p1
        let p2 = PVec3::new(1.0, 0.0, 0.0);
        let p3 = PVec3::new(1.0, 1.0, 0.0);
        let (wire_key, _) = make_wire_from_positions(&mut reg, &[p0, p1, p2, p3]);
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 3);

        let removed = fix_tails(wire_key, &mut reg, 0.1);
        assert_eq!(removed, 1, "short tail edge should be removed");
        assert_eq!(
            reg.wires.get(wire_key).unwrap().edges.len(),
            2,
            "tail edge removed, 2 remain"
        );
    }

    #[test]
    fn test_fix_tails_keeps_normal_edges() {
        let mut reg = BRepStore::new();
        // All edges are long and each vertex is shared by 2 edges.
        let p0 = PVec3::new(0.0, 0.0, 0.0);
        let p1 = PVec3::new(1.0, 0.0, 0.0);
        let p2 = PVec3::new(1.0, 1.0, 0.0);
        let p3 = PVec3::new(0.0, 1.0, 0.0);
        let (wire_key, _) = make_wire_from_positions(&mut reg, &[p0, p1, p2, p3]);
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 3);

        let removed = fix_tails(wire_key, &mut reg, 0.1);
        assert_eq!(removed, 0, "normal edges should not be removed");
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 3);
    }

    #[test]
    fn test_fix_tails_wire_too_small() {
        let mut reg = BRepStore::new();
        // Wire with < 3 edges should not be modified.
        let p0 = PVec3::new(0.0, 0.0, 0.0);
        let p1 = PVec3::new(0.01, 0.0, 0.0);
        let (wire_key, _) = make_wire_from_positions(&mut reg, &[p0, p1]);
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 1);

        let removed = fix_tails(wire_key, &mut reg, 1.0);
        assert_eq!(removed, 0, "wire with < 3 edges should be skipped");
    }
}
