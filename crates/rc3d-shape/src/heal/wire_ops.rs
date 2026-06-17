//! Wire-level operations: edge reordering + small edge removal +
//! notch/tail fixing.

use std::collections::HashMap;

use crate::store::BRepStore;
use crate::topo::{BRepEdge, EdgeKey, Orientation, VertexKey, WireKey};
use rc3d_core::math::Real;
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

/// Fix V-shaped notches in a wire. A notch is two consecutive edges meeting
/// at a sharp angle where the middle vertex is redundant (the two edges are
/// nearly collinear). The two edges are merged into a single edge spanning
/// from the start of the first to the end of the second.
///
/// Returns the number of notches fixed.
pub(crate) fn fix_notched_edges(
    wire_key: WireKey,
    reg: &mut BRepStore,
    angle_threshold_deg: Real,
) -> usize {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return 0 };
        if wire.edges.len() < 2 {
            return 0;
        }
        wire.edges.clone()
    };

    let n = edges.len();
    let angle_threshold_rad = angle_threshold_deg.to_radians();

    // Collect vertex positions for all edge endpoints.
    let mut edge_endpoints: Vec<Option<(VertexKey, VertexKey)>> = vec![None; n];
    for (i, &(ek, orient)) in edges.iter().enumerate() {
        let Some(edge) = reg.edges.get(ek) else { continue };
        let (v_start, v_end) = if orient == Orientation::Forward {
            (edge.v_low, edge.v_high)
        } else {
            (edge.v_high, edge.v_low)
        };
        edge_endpoints[i] = Some((v_start, v_end));
    }

    // Detect notches: for each consecutive pair (i, i+1), check the angle
    // between the tangent at the END of edge[i] and the tangent at the START
    // of edge[i+1] at their shared vertex.
    let mut skip = vec![false; n];
    let mut fixed = 0usize;
    let mut merged_edges: Vec<(usize, usize, EdgeKey)> = Vec::new();

    for i in 0..n {
        if skip[i] {
            continue;
        }
        let j = (i + 1) % n;
        if skip[j] {
            continue;
        }

        let (ek_a, _orient_a) = edges[i];
        let (ek_b, _orient_b) = edges[j];

        let Some(edge_a) = reg.edges.get(ek_a) else { continue };
        let Some(edge_b) = reg.edges.get(ek_b) else { continue };

        let (a_start, a_end) = match edge_endpoints[i] {
            Some(ep) => ep,
            None => continue,
        };
        let (b_start, _b_end) = match edge_endpoints[j] {
            Some(ep) => ep,
            None => continue,
        };

        // Shared vertex: end of edge_a must equal start of edge_b.
        if a_end != b_start {
            continue;
        }

        // Tangent at shared vertex:
        //   edge_a: at t=1.0 (end, forwarding toward the shared vertex)
        //   edge_b: at t=0.0 (start, forwarding away from the shared vertex)
        let dir_a = edge_a.curve.d1(1.0);
        let dir_b = edge_b.curve.d1(0.0);

        // Skip zero-length tangents (degenerate).
        let len_a = dir_a.length();
        let len_b = dir_b.length();
        if len_a < 1e-12 || len_b < 1e-12 {
            continue;
        }

        let cos_angle = (dir_a.dot(dir_b) / (len_a * len_b)).clamp(-1.0, 1.0);
        let angle_rad = cos_angle.acos();

        if angle_rad < angle_threshold_rad {
            // Notch detected: merge edges i and j into a single edge
            // from a_start to b_end (the non-shared endpoints).
            let b_end = _b_end;
            let p_start = reg.vertices.get(a_start).map(|v| v.position);
            let p_end = reg.vertices.get(b_end).map(|v| v.position);

            if let (Some(ps), Some(pe)) = (p_start, p_end) {
                let new_curve = crate::geom::CurveGeom::Line {
                    origin: ps,
                    direction: pe - ps,
                };
                let (v_lo, v_hi) = if a_start < b_end {
                    (a_start, b_end)
                } else {
                    (b_end, a_start)
                };
                let new_ek = reg.edges.insert(BRepEdge {
                    curve: new_curve,
                    tolerance: edge_a.tolerance.max(edge_b.tolerance),
                    v_low: v_lo,
                    v_high: v_hi,
                    t_min: 0.0,
                    t_max: 1.0,
                    cached_deflection: None,
                    pcurves: HashMap::new(),
                });
                // Maintain vertex_to_edges index.
                reg.vertex_to_edges.entry(v_lo).or_default().push(new_ek);
                if v_lo != v_hi {
                    reg.vertex_to_edges.entry(v_hi).or_default().push(new_ek);
                }

                merged_edges.push((i, j, new_ek));
                skip[i] = true;
                skip[j] = true;
                fixed += 1;
            }
        }
    }

    if fixed > 0 {
        // Build a lookup from original index to merged edge.
        let mut merge_at: HashMap<usize, (usize, EdgeKey)> = HashMap::new();
        for &(i, j, new_ek) in &merged_edges {
            merge_at.insert(i, (j, new_ek));
        }

        // Rebuild: walk edges in order. When we hit a merge-start index,
        // emit the merged edge and skip both original edges.
        let mut new_edges: Vec<(EdgeKey, Orientation)> = Vec::new();
        let mut idx = 0usize;
        // Guard against infinite loop when a wrap-around merge (n-1, 0)
        // causes idx to reset below n.
        let mut passes = 0usize;
        while idx < n && passes < n + 1 {
            passes += 1;
            if let Some(&(j, new_ek)) = merge_at.get(&idx) {
                new_edges.push((new_ek, Orientation::Forward));
                if j < idx {
                    // Wrap-around merge at e.g. (n-1, 0): the new edge
                    // is emitted. Continue from j+1 = 1, but remove the
                    // wrap-around entry so we don't re-process it.
                    merge_at.remove(&idx);
                    idx = 1;
                } else {
                    idx = j + 1;
                }
            } else if skip[idx] {
                idx += 1;
            } else {
                new_edges.push(edges[idx]);
                idx += 1;
            }
        }

        if let Some(wire) = reg.wires.get_mut(wire_key) {
            wire.edges = new_edges;
        }
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
    fn test_fix_notched_edges_merges_v_shape() {
        let mut reg = BRepStore::new();
        // Two nearly-collinear edges forming a 5° angle at the middle vertex.
        let p0 = PVec3::new(0.0, 0.0, 0.0);
        let p1 = PVec3::new(1.0, 0.0, 0.0);
        let p2 = PVec3::new(2.0, 0.1, 0.0); // ~5.7° from horizontal
        let (wire_key, _) = make_wire_from_positions(&mut reg, &[p0, p1, p2]);
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 2);

        let fixed = fix_notched_edges(wire_key, &mut reg, 10.0);
        assert_eq!(fixed, 1, "V-shaped notch should be fixed");
        assert_eq!(
            reg.wires.get(wire_key).unwrap().edges.len(),
            1,
            "two edges merged into one"
        );
    }

    #[test]
    fn test_fix_notched_edges_noop_on_smooth() {
        let mut reg = BRepStore::new();
        // Two edges meeting at a 90° angle — not a notch.
        let p0 = PVec3::new(0.0, 0.0, 0.0);
        let p1 = PVec3::new(1.0, 0.0, 0.0);
        let p2 = PVec3::new(1.0, 1.0, 0.0); // 90° turn
        let (wire_key, _) = make_wire_from_positions(&mut reg, &[p0, p1, p2]);
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 2);

        let fixed = fix_notched_edges(wire_key, &mut reg, 10.0);
        assert_eq!(fixed, 0, "90° corner should not be treated as a notch");
        assert_eq!(reg.wires.get(wire_key).unwrap().edges.len(), 2);
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
