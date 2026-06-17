//! Face-level fixes: natural boundary, reversed 2d, small-face merge.

use rc3d_core::math::Real;
use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::{FaceKey, Orientation, WireKey, ShellKey};
use super::unify_same_domain::merge_face_pair;

// ── Add natural boundary ───────────────────────────────────────────

/// Add a natural UV rectangle wire on analytic surfaces with an empty outer wire.
pub(crate) fn fix_add_natural_bound(reg: &mut BRepStore, face_key: FaceKey) -> bool {
    let (surface, tolerance, outer_wire) = {
        let Some(face) = reg.faces.get(face_key) else {
            return false;
        };
        (
            face.surface.clone(),
            face.tolerance,
            face.outer_wire,
        )
    };

    let wire_empty = reg
        .wires
        .get(outer_wire)
        .map(|w| w.edges.is_empty())
        .unwrap_or(true);
    if !wire_empty {
        return false;
    }

    match surface {
        SurfaceGeom::Plane { .. }
        | SurfaceGeom::Cylinder { .. }
        | SurfaceGeom::Cone { .. } => {}
        _ => return false,
    }

    let pr = surface.param_range();
    let corners = [
        (pr.u_min, pr.v_min),
        (pr.u_max, pr.v_min),
        (pr.u_max, pr.v_max),
        (pr.u_min, pr.v_max),
    ];

    let mut edges = Vec::with_capacity(4);
    for i in 0..4 {
        let (u0, v0) = corners[i];
        let (u1, v1) = corners[(i + 1) % 4];
        let p0 = surface.d0_native(u0, v0);
        let p1 = surface.d0_native(u1, v1);
        let v0k = reg.find_or_add_vertex(p0, tolerance);
        let v1k = reg.find_or_add_vertex(p1, tolerance);
        let dir3 = p1 - p0;
        let curve_3d = CurveGeom::Line {
            origin: p0,
            direction: dir3,
        };
        let pcurve = Curve2d::Line {
            origin: (u0, v0),
            direction: (u1 - u0, v1 - v0),
        };
        let ek = reg.add_edge_with_pcurve(v0k, v1k, curve_3d, tolerance, face_key, pcurve, true);
        edges.push((ek, Orientation::Forward));
    }

    if let Some(w) = reg.wires.get_mut(outer_wire) {
        w.edges = edges;
        true
    } else {
        false
    }
}

// ── Fix reversed 2d ────────────────────────────────────────────────

/// Flip outer wire orientation when its UV projection winds clockwise.
pub(crate) fn fix_reversed_2d(reg: &mut BRepStore, face_key: FaceKey) -> bool {
    let outer_wire = match reg.faces.get(face_key) {
        Some(f) => f.outer_wire,
        None => return false,
    };
    let area = signed_uv_wire_area(reg, face_key, outer_wire);
    if area >= 0.0 {
        return false;
    }
    let Some(w) = reg.wires.get_mut(outer_wire) else {
        return false;
    };
    for (_, orient) in &mut w.edges {
        *orient = match *orient {
            Orientation::Forward => Orientation::Reversed,
            Orientation::Reversed => Orientation::Forward,
            other => other,
        };
    }
    true
}

fn signed_uv_wire_area(reg: &BRepStore, face_key: FaceKey, wire_key: WireKey) -> Real {
    if reg.faces.get(face_key).is_none() {
        return 0.0;
    }
    let wire = match reg.wires.get(wire_key) {
        Some(w) => w,
        None => return 0.0,
    };
    let mut poly: Vec<(Real, Real)> = Vec::new();
    for &(ek, orient) in &wire.edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        let pcurve = match edge.pcurves.get(&face_key) {
            Some(pc) => pc,
            None => continue,
        };
        let n = 8usize;
        let mut pts: Vec<(Real, Real)> = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = i as Real / n as Real;
            let uv = pcurve.d0(t);
            pts.push((uv.0, uv.1));
        }
        if orient == Orientation::Reversed {
            pts.reverse();
        }
        if let Some(last) = poly.last() {
            if let Some(first) = pts.first() {
                if (last.0 - first.0).abs() < 1e-6 && (last.1 - first.1).abs() < 1e-6 {
                    pts.remove(0);
                }
            }
        }
        poly.extend(pts);
    }
    if poly.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0_f64;
    for i in 0..poly.len() {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % poly.len()];
        area += x0 * y1 - x1 * y0;
    }
    area * 0.5
}

// ── Fix small faces ─────────────────────────────────────────────────

/// Find the best neighbor to merge a small face into.
/// "Best" = adjacent face with the longest total shared-edge boundary.
fn find_best_neighbor(
    fk: FaceKey,
    reg: &BRepStore,
    candidates: &[FaceKey],
) -> Option<FaceKey> {
    let mut best: Option<FaceKey> = None;
    let mut best_score: Real = 0.0;

    for &candidate in candidates {
        if candidate == fk {
            continue;
        }
        let shared = reg.find_shared_edges(fk, candidate);
        if shared.is_empty() {
            continue;
        }
        // Score = total length of shared edges (longer shared boundary
        // means better adjacency for merging).
        let score: Real = shared
            .iter()
            .filter_map(|&ek| reg.edges.get(ek))
            .map(|edge| edge.curve.arc_length(0.0, 1.0))
            .sum();
        if score > best_score {
            best_score = score;
            best = Some(candidate);
        }
    }
    best
}

/// Merge faces with area below `min_area` into adjacent neighbors.
///
/// * Small faces are processed in ascending area order (smallest first)
///   so that the merge of one does not create new small faces.
/// * An isolated small face (no shared edges with any other shell face)
///   is simply removed from the shell.
///
/// Returns the number of faces merged/removed.
pub fn fix_small_faces(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    min_area: Real,
) -> usize {
    // 1. Get all faces in shell
    let face_entries: Vec<(FaceKey, Orientation)> = match reg.shells.get(shell_key) {
        Some(s) => s.faces.clone(),
        None => return 0,
    };

    if face_entries.len() < 2 {
        return 0;
    }

    // 2. Compute area for each face (wire-enclosed UV area)
    let areas: Vec<(FaceKey, Real)> = face_entries
        .iter()
        .map(|&(fk, _)| {
            let face = match reg.faces.get(fk) {
                Some(f) => f,
                None => return (fk, 0.0),
            };
            let area = signed_uv_wire_area(reg, fk, face.outer_wire).abs();
            (fk, area)
        })
        .collect();

    // 3. Identify small faces; sort by area ascending
    let mut small: Vec<(FaceKey, Real)> = areas
        .iter()
        .filter(|(_, a)| *a < min_area)
        .map(|&(fk, a)| (fk, a))
        .collect();
    small.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if small.is_empty() {
        return 0;
    }

    let mut merged_count: usize = 0;

    for &(fk_small, _area) in &small {
        // Check that the small face is still in the shell (it may have
        // been removed by a previous merge).
        let still_in_shell = match reg.shells.get(shell_key) {
            Some(s) => s.faces.iter().any(|&(fk, _)| fk == fk_small),
            None => false,
        };
        if !still_in_shell {
            continue;
        }

        // Re-read current faces for candidate list
        let current_faces: Vec<FaceKey> = match reg.shells.get(shell_key) {
            Some(s) => s.faces.iter().map(|&(fk, _)| fk).collect(),
            None => continue,
        };

        // 4. Find best neighbor
        let maybe_neighbor = find_best_neighbor(fk_small, reg, &current_faces);

        match maybe_neighbor {
            Some(fk_neighbor) => {
                // Merge small face into neighbor
                if let Some((merged_fk, _removed)) =
                    merge_face_pair(fk_small, fk_neighbor, reg)
                {
                    // Update shell: replace fk_small and fk_neighbor with merged_fk
                    if let Some(shell) = reg.shells.get_mut(shell_key) {
                        let mut new_faces: Vec<(FaceKey, Orientation)> = Vec::new();
                        let mut seen = false;
                        for &(fk, orient) in &shell.faces {
                            if fk == fk_small || fk == fk_neighbor {
                                if !seen {
                                    new_faces.push((merged_fk, orient));
                                    seen = true;
                                }
                                // else: skip duplicate
                            } else {
                                new_faces.push((fk, orient));
                            }
                        }
                        shell.faces = new_faces;
                    }
                    merged_count += 1;
                }
            }
            None => {
                // Isolated small face — remove from shell
                if let Some(shell) = reg.shells.get_mut(shell_key) {
                    shell.faces.retain(|&(fk, _)| fk != fk_small);
                }
                merged_count += 1;
            }
        }
    }

    merged_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::geom::curve2d::Curve2d;
    use crate::topo::{BRepShell, Orientation};
    use rc3d_core::math::PVec3;

    /// Build two adjacent coplanar rectangles sharing one edge.
    /// Small face: (0,0)-(0.1,0)-(0.1,1)-(0,1)
    /// Large face: (0.1,0)-(1,0)-(1,1)-(0.1,1)
    /// Returns (shell_key, small_fk, large_fk).
    fn build_small_large_pair(reg: &mut BRepStore) -> (ShellKey, FaceKey, FaceKey) {
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };

        // Small face: rectangle [0, 0.1] x [0, 1]
        // Large face: rectangle [0.1, 1] x [0, 1]
        // Shared edge: vertical line at x=0.1 from y=0 to y=1
        let p00 = PVec3::new(0.0, 0.0, 0.0);
        let p01 = PVec3::new(0.0, 1.0, 0.0);
        let p10 = PVec3::new(0.1, 0.0, 0.0);
        let p11 = PVec3::new(0.1, 1.0, 0.0);
        let p20 = PVec3::new(1.0, 0.0, 0.0);
        let p21 = PVec3::new(1.0, 1.0, 0.0);

        let v00 = reg.find_or_add_vertex(p00, 1e-4);
        let v01 = reg.find_or_add_vertex(p01, 1e-4);
        let v10 = reg.find_or_add_vertex(p10, 1e-4);
        let v11 = reg.find_or_add_vertex(p11, 1e-4);
        let v20 = reg.find_or_add_vertex(p20, 1e-4);
        let v21 = reg.find_or_add_vertex(p21, 1e-4);

        let line = |a: PVec3, b: PVec3| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let pc = |a: PVec3, b: PVec3| Curve2d::Line {
            origin: (a.x, a.y),
            direction: (b.x - a.x, b.y - a.y),
        };

        let f_small = reg.add_face(surface.clone(), 1e-4);
        let f_large = reg.add_face(surface.clone(), 1e-4);

        // Small face CCW: bottom(p00→p10), right(p10→p11), top(p11→p01), left(p01→p00)
        let e_small_bottom = reg.add_edge_with_pcurve(
            v00, v10, line(p00, p10), 1e-4, f_small, pc(p00, p10), true);
        // Shared edge: p10→p11 (vertical at x=0.1, bottom to top)
        let e_shared = reg.add_edge_with_pcurve(
            v10, v11, line(p10, p11), 1e-4, f_small, pc(p10, p11), true);
        let e_small_top = reg.add_edge_with_pcurve(
            v11, v01, line(p11, p01), 1e-4, f_small, pc(p11, p01), true);
        let e_small_left = reg.add_edge_with_pcurve(
            v01, v00, line(p01, p00), 1e-4, f_small, pc(p01, p00), true);

        // Large face CCW: bottom(p10→p20), right(p20→p21), top(p21→p11), left(p11→p10)
        // Shared edge p10→p11 is used Reversed in large face (p11→p10 is the left edge)
        reg.set_pcurve(e_shared, f_large,
            Curve2d::Line { origin: (0.1, 0.0), direction: (0.0, 1.0) }, true);
        let e_large_bottom = reg.add_edge_with_pcurve(
            v10, v20, line(p10, p20), 1e-4, f_large, pc(p10, p20), true);
        let e_large_right = reg.add_edge_with_pcurve(
            v20, v21, line(p20, p21), 1e-4, f_large, pc(p20, p21), true);
        let e_large_top = reg.add_edge_with_pcurve(
            v21, v11, line(p21, p11), 1e-4, f_large, pc(p21, p11), true);

        // Compute correct orientations based on actual edge v_low/v_high ordering.
        // add_edge_with_pcurve swaps v_low/v_high by key, so we can't assume Forward.
        fn orient_of(reg: &BRepStore, ek: crate::topo::EdgeKey, from_vk: crate::topo::VertexKey) -> Orientation {
            let edge = reg.edges.get(ek).unwrap();
            if edge.v_low == from_vk {
                Orientation::Forward
            } else {
                Orientation::Reversed
            }
        }

        let o_small_bottom = orient_of(&reg, e_small_bottom, v00);
        let o_shared_small = orient_of(&reg, e_shared, v10);
        let o_small_top = orient_of(&reg, e_small_top, v11);
        let o_small_left = orient_of(&reg, e_small_left, v01);
        let o_large_bottom = orient_of(&reg, e_large_bottom, v10);
        let o_large_right = orient_of(&reg, e_large_right, v20);
        let o_large_top = orient_of(&reg, e_large_top, v21);
        let o_shared_large = orient_of(&reg, e_shared, v11); // p11→p10 direction

        // Wire small (CCW: p00→p10→p11→p01→p00)
        if let Some(w) = reg.wires.get_mut(reg.faces.get(f_small).unwrap().outer_wire) {
            w.edges = vec![
                (e_small_bottom, o_small_bottom),
                (e_shared, o_shared_small),
                (e_small_top, o_small_top),
                (e_small_left, o_small_left),
            ];
        }
        // Wire large (CCW: p10→p20→p21→p11→p10)
        if let Some(w) = reg.wires.get_mut(reg.faces.get(f_large).unwrap().outer_wire) {
            w.edges = vec![
                (e_large_bottom, o_large_bottom),
                (e_large_right, o_large_right),
                (e_large_top, o_large_top),
                (e_shared, o_shared_large),
            ];
        }

        reg.build_edge_to_faces_index();

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(f_small, Orientation::Forward), (f_large, Orientation::Forward)],
            closed: false,
            step_id: None,
        });

        (sk, f_small, f_large)
    }

    #[test]
    fn test_fix_small_faces_merges_below_threshold() {
        let mut reg = BRepStore::new();
        let (sk, fk_small, fk_large) = build_small_large_pair(&mut reg);

        // Small face area ≈ 0.1×1.0 = 0.1; threshold = 1.0 should merge it
        let merged = fix_small_faces(sk, &mut reg, 1.0);
        assert_eq!(merged, 1, "one small face should be merged");

        let shell = reg.shells.get(sk).unwrap();
        assert_eq!(shell.faces.len(), 1, "shell should have 1 face after merge");
        let remaining_fk = shell.faces[0].0;
        assert_ne!(remaining_fk, fk_small, "small face should be gone");
        let _ = fk_large;
    }

    #[test]
    fn test_fix_small_faces_no_merge_above_threshold() {
        let mut reg = BRepStore::new();
        let (sk, _, _) = build_small_large_pair(&mut reg);

        // Threshold too small (0.0001) — small face area is 0.1 > 0.0001
        let merged = fix_small_faces(sk, &mut reg, 0.0001);
        assert_eq!(merged, 0, "no face should be merged when area > threshold");

        let shell = reg.shells.get(sk).unwrap();
        assert_eq!(shell.faces.len(), 2, "shell should still have 2 faces");
    }

    #[test]
    fn test_fix_small_faces_removes_isolated() {
        let mut reg = BRepStore::new();

        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };

        let p0 = PVec3::ZERO;
        let p1 = PVec3::new(0.05, 0.0, 0.0);
        let p2 = PVec3::new(0.05, 0.05, 0.0);
        let p3 = PVec3::new(0.0, 0.05, 0.0);

        let v0 = reg.find_or_add_vertex(p0, 1e-4);
        let v1 = reg.find_or_add_vertex(p1, 1e-4);
        let v2 = reg.find_or_add_vertex(p2, 1e-4);
        let v3 = reg.find_or_add_vertex(p3, 1e-4);

        let line = |a: PVec3, b: PVec3| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let pc = |a: PVec3, b: PVec3| Curve2d::Line {
            origin: (a.x, a.y),
            direction: (b.x - a.x, b.y - a.y),
        };

        // Compute correct orientations based on actual edge v_low/v_high ordering
        fn orient_of(reg: &BRepStore, ek: crate::topo::EdgeKey, from_vk: crate::topo::VertexKey) -> Orientation {
            let edge = reg.edges.get(ek).unwrap();
            if edge.v_low == from_vk {
                Orientation::Forward
            } else {
                Orientation::Reversed
            }
        }

        // Tiny isolated face (CCW: p0→p1→p2→p3→p0)
        let f_tiny = reg.add_face(surface.clone(), 1e-4);
        let e1 = reg.add_edge_with_pcurve(v0, v1, line(p0, p1), 1e-4, f_tiny, pc(p0, p1), true);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line(p1, p2), 1e-4, f_tiny, pc(p1, p2), true);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line(p2, p3), 1e-4, f_tiny, pc(p2, p3), true);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line(p3, p0), 1e-4, f_tiny, pc(p3, p0), true);

        let ot1 = orient_of(&reg, e1, v0);
        let ot2 = orient_of(&reg, e2, v1);
        let ot3 = orient_of(&reg, e3, v2);
        let ot4 = orient_of(&reg, e4, v3);

        // Normal face: NOT adjacent to the tiny one (CCW: p4→p5→p6→p7→p4)
        let p4 = PVec3::new(1.0, 0.0, 0.0);
        let p5 = PVec3::new(2.0, 0.0, 0.0);
        let p6 = PVec3::new(2.0, 1.0, 0.0);
        let p7 = PVec3::new(1.0, 1.0, 0.0);
        let v4 = reg.find_or_add_vertex(p4, 1e-4);
        let v5 = reg.find_or_add_vertex(p5, 1e-4);
        let v6 = reg.find_or_add_vertex(p6, 1e-4);
        let v7 = reg.find_or_add_vertex(p7, 1e-4);

        let f_normal = reg.add_face(surface.clone(), 1e-4);
        let e5 = reg.add_edge_with_pcurve(v4, v5, line(p4, p5), 1e-4, f_normal, pc(p4, p5), true);
        let e6 = reg.add_edge_with_pcurve(v5, v6, line(p5, p6), 1e-4, f_normal, pc(p5, p6), true);
        let e7 = reg.add_edge_with_pcurve(v6, v7, line(p6, p7), 1e-4, f_normal, pc(p6, p7), true);
        let e8 = reg.add_edge_with_pcurve(v7, v4, line(p7, p4), 1e-4, f_normal, pc(p7, p4), true);

        let on1 = orient_of(&reg, e5, v4);
        let on2 = orient_of(&reg, e6, v5);
        let on3 = orient_of(&reg, e7, v6);
        let on4 = orient_of(&reg, e8, v7);

        if let Some(w) = reg.wires.get_mut(reg.faces.get(f_tiny).unwrap().outer_wire) {
            w.edges = vec![(e1, ot1), (e2, ot2), (e3, ot3), (e4, ot4)];
        }

        if let Some(w) = reg.wires.get_mut(reg.faces.get(f_normal).unwrap().outer_wire) {
            w.edges = vec![(e5, on1), (e6, on2), (e7, on3), (e8, on4)];
        }

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(f_tiny, Orientation::Forward), (f_normal, Orientation::Forward)],
            closed: false,
            step_id: None,
        });

        // Tiny face area ≈ 0.0025; threshold = 0.01
        let merged = fix_small_faces(sk, &mut reg, 0.01);
        assert_eq!(merged, 1, "isolated small face should be removed");

        let shell = reg.shells.get(sk).unwrap();
        assert_eq!(shell.faces.len(), 1, "shell should have 1 face after removal");
        assert_eq!(shell.faces[0].0, f_normal, "normal face should remain");
    }
}
