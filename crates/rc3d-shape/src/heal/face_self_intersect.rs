//! Face-level self-intersection detection via surface-normal inversion grid.
//!
//! Unlike `self_intersect.rs` which checks UV-boundary wire crossing, this module
//! checks whether the surface **itself** folds in 3D by detecting adjacent grid cells
//! whose normals point in opposite directions (dot product < 0).

use crate::geom::SurfaceGeom;
use crate::nurbs::NurbsSurface;
use crate::store::BRepStore;
use crate::topo::{FaceKey, ShellKey};
use rc3d_core::math::Vec3;

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
    let mut normals: Vec<Vec<Vec3>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        let u = i as f32 / grid_res as f32;
        for j in 0..n {
            let v = j as f32 / grid_res as f32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use crate::topo::{BRepFace, BRepShell, BRepWire};
    use crate::nurbs::NurbsSurface;
    use rc3d_core::math::Vec3;

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
        let grid: Vec<Vec<Vec3>> = (0..3)
            .map(|i| (0..3).map(|j| Vec3::new(i as f32, j as f32, 0.0)).collect())
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
                vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)],  // P0
                vec![Vec3::new(2.0, 0.0, 1.0), Vec3::new(2.0, 0.0, 2.0)],  // P1 (peak)
                vec![Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 3.0)],  // P2 (back to x=0)
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
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
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
