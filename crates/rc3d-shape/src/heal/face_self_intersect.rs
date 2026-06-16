//! Face-level self-intersection detection and repair via surface-normal
//! inversion grid.
//!
//! Unlike `self_intersect.rs` which checks UV-boundary wire crossing, this module
//! checks whether the surface **itself** folds in 3D by detecting adjacent grid cells
//! whose normals point in opposite directions (dot product < 0).
//!
//! When self-intersections are detected, the fix pass subdivides the face's
//! boundary wires by inserting Steiner vertices at edge midpoints, which gives
//! the CDT mesher more constraint points and reduces self-intersecting triangles.

use crate::geom::SurfaceGeom;
use crate::nurbs::NurbsSurface;
use crate::store::BRepStore;
use crate::topo::{FaceKey, ShellKey};
use rc3d_core::math::{Real, PVec3};

/// Report from face self-intersection detection.
#[allow(dead_code)]
pub struct FaceSelfIntersectReport {
    pub suspect_faces: Vec<FaceKey>,
    pub total_inversion_cells: usize,
}

/// Check a single face for surface-level self-intersection.
///
/// Returns the count of normal-inversion cells. Returns 0 for non-BSpline surfaces
/// (analytical surfaces like Plane, Cylinder, etc. cannot self-intersect in this sense).
pub fn check_face_self_intersect(face_key: FaceKey, reg: &BRepStore, grid_res: usize) -> usize {
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return 0,
    };
    let nurbs = match &face.surface {
        SurfaceGeom::BSpline(n) => n,
        _ => return 0,
    };
    check_nurbs_self_intersect(nurbs, grid_res)
}

/// Check all faces of a shell for surface-level self-intersection.
#[allow(dead_code)]
pub fn check_shell_face_self_intersect(
    shell_key: ShellKey,
    reg: &BRepStore,
    grid_res: usize,
) -> FaceSelfIntersectReport {
    let mut report = FaceSelfIntersectReport {
        suspect_faces: Vec::new(),
        total_inversion_cells: 0,
    };

    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return report,
    };

    for &(face_key, _) in &shell.faces {
        let count = check_face_self_intersect(face_key, reg, grid_res);
        if count > 0 {
            report.suspect_faces.push(face_key);
            report.total_inversion_cells += count;
        }
    }

    report
}

/// Core algorithm: sample normals on a (grid_res+1) x (grid_res+1) grid over [0,1]x[0,1],
/// then count cells where adjacent normals flip direction (dot < 0).
fn check_nurbs_self_intersect(nurbs: &NurbsSurface, grid_res: usize) -> usize {
    if grid_res < 2 {
        return 0;
    }

    let n = grid_res + 1;

    // Build normal grid.
    let mut normals: Vec<Vec<PVec3>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        let u = i as Real / grid_res as Real;
        for j in 0..n {
            let v = j as Real / grid_res as Real;
            row.push(nurbs.normal(u, v));
        }
        normals.push(row);
    }

    // Count inversion cells.
    let mut inversions = 0usize;
    for i in 0..grid_res {
        for j in 0..grid_res {
            let n00 = normals[i][j];
            // Right neighbor
            let n10 = normals[i + 1][j];
            if n00.dot(n10) < 0.0 {
                inversions += 1;
            }
            // Upper neighbor
            let n01 = normals[i][j + 1];
            if n00.dot(n01) < 0.0 {
                inversions += 1;
            }
        }
    }

    inversions
}

/// Report from face self-intersection fix pass.
#[derive(Debug, Clone, Default)]
pub struct FaceSelfIntersectFixReport {
    pub faces_checked: usize,
    pub faces_fixed: usize,
    pub steiner_vertices_added: usize,
}

/// Attempt to resolve face self-intersections by subdividing boundary edges.
///
/// Strategy: for each edge in the face's outer wire with chord length above
/// the tolerance, split it at its midpoint. This inserts Steiner vertices into
/// the UV boundary, providing the CDT mesher with more constraint points and
/// reducing the likelihood of self-intersecting triangles.
///
/// The approach is conservative: it only splits edges long enough to warrant
/// subdivision (chord > 10 × tolerance), avoiding creating tiny edges that
/// would complicate downstream processing.
pub fn fix_face_self_intersections(
    face_key: FaceKey,
    reg: &mut BRepStore,
    tolerance: Real,
) -> FaceSelfIntersectFixReport {
    let mut report = FaceSelfIntersectFixReport { faces_checked: 1, ..Default::default() };

    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return report,
    };

    // Only BSpline surfaces benefit from boundary subdivision.
    if !matches!(&face.surface, SurfaceGeom::BSpline(_)) {
        return report;
    }

    let outer_wk = face.outer_wire;
    let wire_edges: Vec<(crate::topo::EdgeKey, crate::topo::Orientation)> = match reg.wires.get(outer_wk) {
        Some(w) => w.edges.clone(),
        None => return report,
    };

    let min_chord = tolerance * 10.0;
    let mut steiner_added = 0usize;

    for (ek, orient) in &wire_edges {
        let chord = {
            let edge = match reg.edges.get(*ek) {
                Some(e) => e,
                None => continue,
            };
            let p_lo = reg.vertices.get(edge.v_low).map(|v| v.position).unwrap_or(PVec3::ZERO);
            let p_hi = reg.vertices.get(edge.v_high).map(|v| v.position).unwrap_or(PVec3::ZERO);
            (p_hi - p_lo).length()
        };

        if chord < min_chord {
            continue;
        }

        // Split at midpoint (t = 0.5 in edge parameter space).
        let parts = super::curve_trim::split_edge_at_params(*ek, face_key, *orient, &[0.5], reg);
        if parts.len() > 1 {
            super::curve_trim::replace_wire_edge_with_splits(outer_wk, *ek, &parts, reg);
            steiner_added += parts.len() - 1;
        }
    }

    if steiner_added > 0 {
        report.faces_fixed = 1;
        report.steiner_vertices_added = steiner_added;
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use crate::topo::{BRepFace, BRepShell, BRepWire};
    use crate::nurbs::NurbsSurface;
    use rc3d_core::math::PVec3;

    fn make_face_with_nurbs(reg: &mut BRepStore, nurbs: NurbsSurface) -> (FaceKey, ShellKey) {
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::BSpline(nurbs),
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            degenerated_edges: vec![],
            color: None,
        });
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk, crate::topo::Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        (fk, sk)
    }

    #[test]
    fn test_flat_surface_no_self_intersect() {
        let mut reg = BRepStore::new();
        let grid: Vec<Vec<PVec3>> = (0..3)
            .map(|i| (0..3).map(|j| PVec3::new(i as Real, j as Real, 0.0)).collect())
            .collect();
        let nurbs = NurbsSurface::from_points_grid(&grid, 2, 2);
        let (fk, _) = make_face_with_nurbs(&mut reg, nurbs);
        assert_eq!(check_face_self_intersect(fk, &reg, 10), 0);
    }

    #[test]
    fn test_folded_surface_detects_self_intersect() {
        let mut reg = BRepStore::new();
        // Degree-1 surface in u that physically folds back: control points in
        // X go 0 -> 2 -> 0, creating two linear segments that form a "V" fold.
        // With v-degree=1, 2 control points in v, this makes a ruled surface
        // that folds sharply in the middle.
        //
        // degree_u=1, 3 ctrl pts in u; degree_v=1, 2 ctrl pts in v
        // knots_u = [0, 0, 0.5, 1, 1] (5 knots = 3 + 1 + 1)
        //   Span 0: [0, 0.5) — line from P0 to P1, going +X
        //   Span 1: [0.5, 1] — line from P1 to P2, going -X (fold!)
        // knots_v = [0, 0, 1, 1] (4 knots = 2 + 1 + 1)
        //
        // Normal = du × dv. du reverses sign across the fold. dv stays constant.
        // So the normal flips direction, giving dot(n_left, n_right) < 0.
        let nurbs = NurbsSurface {
            degree_u: 1,
            degree_v: 1,
            control_points: vec![
                vec![PVec3::new(0.0, 0.0, 0.0), PVec3::new(0.0, 0.0, 1.0)],  // P0
                vec![PVec3::new(2.0, 0.0, 1.0), PVec3::new(2.0, 0.0, 2.0)],  // P1 (peak)
                vec![PVec3::new(0.0, 0.0, 2.0), PVec3::new(0.0, 0.0, 3.0)],  // P2 (back to x=0)
            ],
            weights: vec![vec![1.0, 1.0]; 3],
            knots_u: vec![0.0, 0.0, 0.5, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
        };

        let (fk, _) = make_face_with_nurbs(&mut reg, nurbs);
        let count = check_face_self_intersect(fk, &reg, 10);
        assert!(
            count > 0,
            "folded surface should detect normal inversions, got {count}"
        );
    }

    #[test]
    fn test_analytical_surface_returns_zero() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            degenerated_edges: vec![],
            color: None,
        });
        assert_eq!(check_face_self_intersect(fk, &reg, 10), 0);
    }
}
