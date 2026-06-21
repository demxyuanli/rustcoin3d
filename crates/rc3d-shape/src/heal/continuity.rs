//! Geometric continuity check along shared edges (OCC ShapeAnalysis).
//! Detects G0 (positional gap) and G1 (tangential mismatch) defects.

use rc3d_core::math::{Real, PVec3};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, ShellKey, WireKey};

#[derive(Debug, Clone)]
pub struct ContinuityDefect {
    pub edge_key: EdgeKey,
    pub face_a: FaceKey,
    pub face_b: FaceKey,
    pub kind: ContinuityKind,
    pub max_deviation: Real,
}

#[derive(Debug, Clone)]
pub enum ContinuityKind {
    G0 { max_gap: Real },
    G1 { max_angle_deg: Real },
}

/// Check G0/G1 continuity along all shared edges in a shell.
pub fn check_shell_continuity(
    shell_key: ShellKey,
    reg: &BRepStore,
    g0_tolerance: Real,
    g1_angle_tolerance_deg: Real,
) -> Vec<ContinuityDefect> {
    let mut defects = Vec::new();

    let face_keys: Vec<FaceKey> = {
        let Some(shell) = reg.shells.get(shell_key) else {
            return defects;
        };
        shell.faces.iter().map(|&(fk, _)| fk).collect()
    };

    for i in 0..face_keys.len() {
        for j in (i + 1)..face_keys.len() {
            let shared = reg.find_shared_edges(face_keys[i], face_keys[j]);
            for ek in shared {
                let defect = check_edge_continuity(
                    ek,
                    face_keys[i],
                    face_keys[j],
                    reg,
                    g0_tolerance,
                    g1_angle_tolerance_deg,
                );
                if let Some(d) = defect {
                    defects.push(d);
                }
            }
        }
    }

    defects
}

fn check_edge_continuity(
    ek: EdgeKey,
    fa: FaceKey,
    fb: FaceKey,
    reg: &BRepStore,
    g0_tol: Real,
    g1_tol_deg: Real,
) -> Option<ContinuityDefect> {
    let edge = reg.edges.get(ek)?;
    let face_a = reg.faces.get(fa)?;
    let face_b = reg.faces.get(fb)?;

    let samples = 4;
    let mut max_gap = 0.0_f64;
    let mut max_angle = 0.0_f64;

    for s in 0..=samples {
        let t = s as Real / samples as Real;
        let p3d = edge.curve.d0(t);

        // Use PCurve UV coordinates for surface evaluation, not project().
        // project() may find a different point on highly curved surfaces
        // (e.g., opposite side of a torus), producing incorrect normals.
        let (ua, va) = match edge.pcurves.get(&fa) {
            Some(pc) => { let uv = pc.d0(t); (uv.0, uv.1) }
            None => face_a.surface.project(p3d)?,
        };
        let (ub, vb) = match edge.pcurves.get(&fb) {
            Some(pc) => { let uv = pc.d0(t); (uv.0, uv.1) }
            None => face_b.surface.project(p3d)?,
        };

        let pa = face_a.surface.d0_native(ua, va);
        let pb = face_b.surface.d0_native(ub, vb);

        let gap = (pa - pb).length();
        max_gap = max_gap.max(gap);

        let na = face_a.surface.normal_native(ua, va);
        let nb = face_b.surface.normal_native(ub, vb);
        let dot = na.normalize().dot(nb.normalize()).max(-1.0).min(1.0);
        let angle = dot.acos().to_degrees();
        max_angle = max_angle.max(angle);
    }

    if max_gap > g0_tol {
        Some(ContinuityDefect {
            edge_key: ek,
            face_a: fa,
            face_b: fb,
            kind: ContinuityKind::G0 { max_gap },
            max_deviation: max_gap,
        })
    } else if max_angle > g1_tol_deg {
        Some(ContinuityDefect {
            edge_key: ek,
            face_a: fa,
            face_b: fb,
            kind: ContinuityKind::G1 {
                max_angle_deg: max_angle,
            },
            max_deviation: max_angle,
        })
    } else {
        None
    }
}

/// Split an edge into C1-continuous segments at C0 discontinuity points.
///
/// Samples the 3D curve at 64 uniform points, computes the tangent
/// (first derivative) at each sample, and detects C0 discontinuities
/// where consecutive tangents change direction by more than
/// `angle_threshold_deg`.
///
/// Each interior breakpoint is placed at the midpoint between the two
/// samples that straddle the discontinuity.  The edge is then split via
/// [`super::curve_trim::split_edge_at_params`] and all wires that
/// reference the edge are updated.
///
/// Returns the new edge keys (empty if the edge has no PCurves, or the
/// original key if no splits are needed).
pub fn split_at_c0_discontinuities(
    edge_key: EdgeKey,
    reg: &mut BRepStore,
    angle_threshold_deg: Real,
) -> Vec<EdgeKey> {
    let edge = match reg.edges.get(edge_key) {
        Some(e) => e,
        None => return vec![],
    };

    // Sample 64 uniform points on the 3D curve.
    const N: usize = 64;
    let mut tangents: Vec<(Real, PVec3)> = Vec::with_capacity(N);
    for i in 0..N {
        let t = i as Real / (N - 1) as Real;
        let d = edge.curve.d1(t);
        tangents.push((t, d));
    }

    if tangents.len() < 3 {
        return vec![edge_key];
    }

    // Detect C0 discontinuities: angle between consecutive tangents.
    let mut split_ts: Vec<Real> = Vec::new();
    for i in 1..tangents.len() {
        let (t_prev, d_prev) = tangents[i - 1];
        let (t_curr, d_curr) = tangents[i];

        let d_prev_n = d_prev.normalize();
        let d_curr_n = d_curr.normalize();
        let dot = d_prev_n.dot(d_curr_n).max(-1.0).min(1.0);
        let angle_deg = dot.acos().to_degrees();

        if angle_deg > angle_threshold_deg {
            let t_mid = (t_prev + t_curr) * 0.5;
            split_ts.push(t_mid);
        }
    }

    // Deduplicate close splits.
    split_ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut deduped: Vec<Real> = Vec::new();
    for t in split_ts {
        if deduped.is_empty() || (t - deduped.last().unwrap()).abs() > 1.0 / N as Real {
            deduped.push(t);
        }
    }

    if deduped.is_empty() {
        return vec![edge_key];
    }

    // Collect face keys from the edge's PCurves (snapshot before mutation).
    let face_keys: Vec<FaceKey> = edge.pcurves.keys().copied().collect();

    let mut new_edge_keys: Vec<EdgeKey> = Vec::new();

    for fk in face_keys {
        // Find the wire orientation for this edge in each face wire.
        let face = match reg.faces.get(fk) {
            Some(f) => f,
            None => continue,
        };
        let wire_keys: Vec<(WireKey, Orientation)> = {
            let mut wks = Vec::new();
            // Collect (wire_key, orientation) for all occurrences of this edge
            // across outer + inner wires.
            for &wk in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
                if let Some(wire) = reg.wires.get(wk) {
                    for &(ek, orient) in &wire.edges {
                        if ek == edge_key {
                            wks.push((wk, orient));
                        }
                    }
                }
            }
            wks
        };

        // Use the first orientation found; split_edge_at_params handles
        // reversed orientation internally by swapping v_lo/v_hi.
        let orient = wire_keys
            .first()
            .copied()
            .map(|(_, o)| o)
            .unwrap_or(Orientation::Forward);

        let parts = super::curve_trim::split_edge_at_params(
            edge_key,
            fk,
            orient,
            &deduped,
            reg,
        );

        // Collect new edge keys.
        for &(nek, _) in &parts {
            if !new_edge_keys.contains(&nek) {
                new_edge_keys.push(nek);
            }
        }

        // Replace old edge with new segments in every wire that referenced it.
        for (wk, _) in &wire_keys {
            super::curve_trim::replace_wire_edge_with_splits(*wk, edge_key, &parts, reg);
        }
    }

    if new_edge_keys.is_empty() {
        vec![edge_key]
    } else {
        new_edge_keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepWire, BRepShell, Orientation};
    use rc3d_core::math::PVec3;

    fn make_plane_face(reg: &mut BRepStore, surface: SurfaceGeom) -> FaceKey {
        reg.add_face(surface, 1e-4)
    }

    #[test]
    fn test_g0_continuous_shell() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);

        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let f1 = make_plane_face(&mut reg, surface.clone());
        let f2 = make_plane_face(&mut reg, surface);
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone(), true);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, f1, pc.clone(), true);
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f2, pc, true);

        reg.wires
            .get_mut(
                reg.faces
                    .get(f1)
                    .unwrap()
                    .outer_wire,
            )
            .unwrap()
            .edges = vec![(e1, Orientation::Forward), (e2, Orientation::Forward)];
        reg.wires
            .get_mut(
                reg.faces
                    .get(f2)
                    .unwrap()
                    .outer_wire,
            )
            .unwrap()
            .edges = vec![(e1, Orientation::Forward)];

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(f1, Orientation::Forward), (f2, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let defects = check_shell_continuity(sk, &reg, 0.01, 1.0);
        assert!(
            defects.is_empty(),
            "coplanar faces should have no G0 defects, got {:?}",
            defects
        );
    }

    #[test]
    fn test_g1_discontinuity_at_seam() {
        let mut reg = BRepStore::new();
        // Two faces meeting at 90 degrees along a shared edge
        let surface1 = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let surface2 = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Y, u_dir: PVec3::X };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let wk1 = reg.wires.insert(BRepWire { edges: vec![] });
        let f1 = reg.faces.insert(crate::topo::BRepFace {
            surface: surface1, outer_wire: wk1, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let wk2 = reg.wires.insert(BRepWire { edges: vec![] });
        let f2 = reg.faces.insert(crate::topo::BRepFace {
            surface: surface2, outer_wire: wk2, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone(), true);
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f2, pc, true);
        reg.wires.get_mut(wk1).unwrap().edges = vec![(ek, Orientation::Forward)];
        reg.wires.get_mut(wk2).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(f1, Orientation::Forward), (f2, Orientation::Forward)],
            closed: false, step_id: None,
        });
        // With strict G1 tolerance, 90-degree faces should produce a G1 defect
        let defects = check_shell_continuity(sk, &reg, 0.01, 1.0);
        // 90 degrees > 1 degree, so G1 defect expected
        assert!(!defects.is_empty(), "90-degree faces should produce G1 defect");
        assert!(
            defects
                .iter()
                .any(|d| matches!(d.kind, ContinuityKind::G1 { .. })),
            "expected G1 defect kind"
        );
    }

    #[test]
    fn test_g1_only_no_g0() {
        let mut reg = BRepStore::new();
        let surface1 = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let surface2 = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Y,
            u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let wk1 = reg.wires.insert(BRepWire { edges: vec![] });
        let f1 = reg.faces.insert(crate::topo::BRepFace {
            surface: surface1,
            outer_wire: wk1,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let wk2 = reg.wires.insert(BRepWire { edges: vec![] });
        let f2 = reg.faces.insert(crate::topo::BRepFace {
            surface: surface2,
            outer_wire: wk2,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone(), true);
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f2, pc, true);
        reg.wires.get_mut(wk1).unwrap().edges = vec![(ek, Orientation::Forward)];
        reg.wires.get_mut(wk2).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(f1, Orientation::Forward), (f2, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let defects = check_shell_continuity(sk, &reg, 1e-6, 1.0);
        assert!(
            defects.iter().all(|d| !matches!(d.kind, ContinuityKind::G0 { .. })),
            "coplanar junction should not report G0 when gap is zero"
        );
    }

    #[test]
    fn test_split_c0_straight_line_no_split() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.add_edge_with_pcurve(v0, v1, line, 1e-4, fk, pc, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];

        let result = split_at_c0_discontinuities(ek, &mut reg, 30.0);
        // Straight line tangent is constant — no split.
        assert_eq!(result.len(), 1, "straight line should not be split");
        assert_eq!(result[0], ek, "should return original edge key");
    }

    #[test]
    fn test_split_c0_polyline_sharp_corner() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        // L-shaped polyline: (0,0,0) → (1,0,0) → (1,1,0) — 90 degree turn
        let pts = vec![
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(1.0, 1.0, 0.0),
        ];
        let poly = CurveGeom::Polyline { points: pts };
        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.add_edge_with_pcurve(v0, v1, poly, 1e-4, fk, pc, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];

        // 90-degree corner > 30 degree threshold → should split
        let result = split_at_c0_discontinuities(ek, &mut reg, 30.0);
        assert!(
            result.len() > 1,
            "L-shaped polyline should be split at the corner, got {} edges",
            result.len()
        );
        // Check that the wire was updated with the new edges.
        let wire = reg.wires.get(wk).unwrap();
        assert_eq!(
            wire.edges.len(),
            result.len(),
            "wire should contain the new split edges"
        );
    }
}
