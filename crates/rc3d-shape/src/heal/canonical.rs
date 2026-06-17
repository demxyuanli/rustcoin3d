//! Canonical recognition: detect when a BSpline surface is actually an analytic form.
//!
//! STEP importers often emit free-form NURBS for surfaces that are geometrically
//! planar, cylindrical, or spherical. Recognising these allows downstream code to
//! use faster analytic evaluators and improves accuracy.
//!
//! Reference: OCC `ShapeCustom_RestrictionParameters` (BSpline → elementary).

use rc3d_core::math::{Real, PVec3};
use crate::geom::curve_eval::build_ortho_axes;
use crate::geom::SurfaceGeom;
use crate::nurbs::NurbsSurface;

/// Try to recognize a BSpline surface as an analytic form.
///
/// Returns `Some(analytic_surface)` if the NURBS control polygon is exactly one of
/// the elementary forms within `tolerance`. Returns `None` if the surface is a
/// genuine free-form shape (or already analytic).
pub fn recognize_canonical(surface: &SurfaceGeom, tolerance: Real) -> Option<SurfaceGeom> {
    match surface {
        SurfaceGeom::BSpline(nurbs) => {
            if let Some(plane) = try_as_plane(nurbs, tolerance) {
                return Some(plane);
            }
            // Future: try_as_cylinder, try_as_sphere, etc.
            None
        }
        _ => None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Plane detection — PCA on control points
// ══════════════════════════════════════════════════════════════════════════════

/// Check if all control points lie on the same plane.
///
/// Uses PCA (Principal Component Analysis) on the de-homogenised control points:
/// 1. Compute centroid.
/// 2. Build 3x3 covariance matrix.
/// 3. The eigenvector with smallest eigenvalue is the plane normal.
/// 4. Verify that every control point is within `tol` of the fitted plane.
fn try_as_plane(nurbs: &NurbsSurface, tol: Real) -> Option<SurfaceGeom> {
    let cps: Vec<PVec3> = nurbs
        .control_points
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .collect();

    if cps.len() < 3 {
        return None;
    }

    let n = cps.len() as Real;

    // Centroid
    let centroid: PVec3 = cps.iter().fold(PVec3::ZERO, |acc, &p| acc + p) / n;

    // Covariance matrix (3x3 symmetric, stored as 6 scalars)
    let mut c00 = 0.0_f64;
    let mut c01 = 0.0_f64;
    let mut c02 = 0.0_f64;
    let mut c11 = 0.0_f64;
    let mut c12 = 0.0_f64;
    let mut c22 = 0.0_f64;

    for &p in &cps {
        let dx = p.x - centroid.x;
        let dy = p.y - centroid.y;
        let dz = p.z - centroid.z;
        c00 += dx * dx;
        c01 += dx * dy;
        c02 += dx * dz;
        c11 += dy * dy;
        c12 += dy * dz;
        c22 += dz * dz;
    }

    // Find the eigenvector with smallest eigenvalue via power iteration on the
    // inverse of the covariance matrix. Since this is a 3x3 PSD matrix we can
    // solve it analytically: compute the normal as the nullspace direction that
    // minimises the quadratic form v^T C v.
    let normal = smallest_eigenvector_symmetric(c00, c01, c02, c11, c12, c22);

    // Even a flat grid has some floating noise — ignore.
    let len2 = normal.length_squared();
    if len2 < 1e-20 {
        return None;
    }
    let normal = normal / len2.sqrt();

    // Verify all control points are within tolerance of the fitted plane.
    for &p in &cps {
        let dist = (p - centroid).dot(normal).abs();
        if dist > tol {
            return None;
        }
    }

    // Build orthonormal u_dir from the centroid+normal.
    let (u_dir, _v_dir) = build_ortho_axes(normal);

    Some(SurfaceGeom::Plane {
        origin: centroid,
        normal,
        u_dir,
    })
}

/// Return the eigenvector corresponding to the smallest eigenvalue of the 3x3
/// symmetric matrix with entries (c00,c01,c02; c01,c11,c12; c02,c12,c22).
///
/// Strategy: for each index i ∈ {0,1,2}, build the 2×2 minor obtained by deleting
/// row i and column i, solve the 2×2 eigenvalue problem analytically, pick the
/// smaller eigenvalue, then back-substitute to get the full 3-vector. Return the
/// one whose associated eigenvalue is globally the smallest.
fn smallest_eigenvector_symmetric(
    c00: f64, c01: f64, c02: f64,
          c11: f64, c12: f64,
                    c22: f64,
) -> PVec3 {
    // Compute the three candidates by solving each 2x2 minor.
    // For a 3x3 PSD matrix, the smallest eigenvalue can be found by
    // picking the candidate with minimal Rayleigh quotient.

    let candidates = [
        // Eliminate row/col 0 — solve 2x2 [[c11,c12],[c12,c22]]
        {
            let (ev, evec_2d) = smallest_2x2_eigenpair(c11, c12, c22);
            // Back-substitute: M * [0; evec_2d] gives the 3-vector
            let x = evec_2d.0;
            let y = evec_2d.1;
            let v = PVec3::new(0.0, x, y);
            (ev, v)
        },
        // Eliminate row/col 1 — solve 2x2 [[c00,c02],[c02,c22]]
        {
            let (ev, evec_2d) = smallest_2x2_eigenpair(c00, c02, c22);
            let x = evec_2d.0;
            let y = evec_2d.1;
            let v = PVec3::new(x, 0.0, y);
            (ev, v)
        },
        // Eliminate row/col 2 — solve 2x2 [[c00,c01],[c01,c11]]
        {
            let (ev, evec_2d) = smallest_2x2_eigenpair(c00, c01, c11);
            let x = evec_2d.0;
            let y = evec_2d.1;
            let v = PVec3::new(x, y, 0.0);
            (ev, v)
        },
    ];

    candidates
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, v)| *v)
        .unwrap_or(PVec3::Z)
}

/// Compute the smallest eigenvalue and corresponding eigenvector of a 2×2
/// symmetric matrix [[a, b], [b, c]].
///
/// Closed-form: eigenvalues λ = (a + c ± √((a-c)² + 4b²)) / 2.
/// Returns (λ_min, (x, y)) where (x, y) is the eigenvector for λ_min.
fn smallest_2x2_eigenpair(a: f64, b: f64, c: f64) -> (f64, (f64, f64)) {
    let trace = a + c;
    let disc = (a - c) * (a - c) + 4.0 * b * b;
    let sqrt_disc = disc.sqrt();
    let lambda_min = 0.5 * (trace - sqrt_disc);

    // Eigenvector for λ_min solves: [[a-λ, b],[b, c-λ]] * [x;y] = 0
    // Use the row with the larger norm for numerical stability.
    let r0_norm = (a - lambda_min).abs() + b.abs();
    let r1_norm = b.abs() + (c - lambda_min).abs();

    if r0_norm > r1_norm {
        // Use row 0: (a-λ)*x + b*y = 0 → y ∝ -(a-λ), x ∝ b
        let (x, y) = (-b, a - lambda_min);
        let len = (x * x + y * y).sqrt();
        if len > 1e-20 {
            (lambda_min, (x / len, y / len))
        } else {
            (lambda_min, (1.0, 0.0))
        }
    } else {
        // Use row 1: b*x + (c-λ)*y = 0 → x ∝ -(c-λ), y ∝ b
        let (x, y) = (-(c - lambda_min), b);
        let len = (x * x + y * y).sqrt();
        if len > 1e-20 {
            (lambda_min, (x / len, y / len))
        } else {
            (lambda_min, (1.0, 0.0))
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// BSpline of planar control points should be recognized as a Plane.
    #[test]
    fn test_planar_bspline_recognized_as_plane() {
        // Build a planar BSpline surface: all control points on z = 3.0.
        let cps: Vec<Vec<PVec3>> = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| PVec3::new(i as f64, j as f64, 3.0))
                    .collect()
            })
            .collect();
        let nurbs = NurbsSurface::from_points_grid(&cps, 2, 2);
        let surf = SurfaceGeom::BSpline(nurbs);

        let result = recognize_canonical(&surf, 1e-6);
        assert!(result.is_some(), "planar BSpline should be recognized");
        match result.unwrap() {
            SurfaceGeom::Plane { origin, normal, .. } => {
                // Normal should be roughly +Z or -Z
                let n_abs = normal.dot(PVec3::Z).abs();
                assert!(n_abs > 0.99, "plane normal should align with Z, got {:?}", normal);
                // Origin should be near the centroid (1.5, 1.5, 3.0)
                assert!((origin.z - 3.0).abs() < 1e-6,
                    "plane origin z should be 3.0, got {}", origin.z);
            }
            other => panic!("expected Plane, got {:?}", other),
        }
    }

    /// A genuinely curved BSpline should not be recognized as analytic.
    #[test]
    fn test_curved_bspline_not_recognized() {
        let cps: Vec<Vec<PVec3>> = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let x = i as f64;
                        let y = j as f64;
                        let z = (x - 1.5).powi(2) + (y - 1.5).powi(2);
                        PVec3::new(x, y, z)
                    })
                    .collect()
            })
            .collect();
        let nurbs = NurbsSurface::from_points_grid(&cps, 2, 2);
        let surf = SurfaceGeom::BSpline(nurbs);

        let result = recognize_canonical(&surf, 1e-6);
        assert!(result.is_none(), "curved BSpline should not be recognized as analytic");
    }

    /// Already-analytic surfaces pass through unchanged (None).
    #[test]
    fn test_already_analytic_returns_none() {
        let plane = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        assert!(recognize_canonical(&plane, 1e-6).is_none());

        let cyl = SurfaceGeom::Cylinder {
            origin: PVec3::ZERO,
            axis: PVec3::Z,
            radius: 1.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        assert!(recognize_canonical(&cyl, 1e-6).is_none());
    }

    /// Single-row (degenerate) BSpline: not enough points, should return None.
    #[test]
    fn test_degenerate_bspline_returns_none() {
        let cps = vec![vec![PVec3::ZERO]];
        let nurbs = NurbsSurface {
            degree_u: 0,
            degree_v: 0,
            control_points: cps,
            weights: vec![vec![1.0]],
            knots_u: vec![0.0, 1.0],
            knots_v: vec![0.0, 1.0],
        };
        let surf = SurfaceGeom::BSpline(nurbs);
        assert!(recognize_canonical(&surf, 1e-6).is_none());
    }

    /// Tilted plane: control points form a plane not parallel to any coordinate axis.
    #[test]
    fn test_tilted_planar_bspline_recognized() {
        // Plane: x + y + z = 1, so control points like (i, j, 1 - i - j).
        let cps: Vec<Vec<PVec3>> = (0..5)
            .map(|i| {
                (0..5)
                    .map(|j| {
                        let x = i as f64 * 0.25;
                        let y = j as f64 * 0.25;
                        let z = 1.0 - x - y;
                        PVec3::new(x, y, z)
                    })
                    .collect()
            })
            .collect();
        let nurbs = NurbsSurface::from_points_grid(&cps, 3, 3);
        let surf = SurfaceGeom::BSpline(nurbs);

        let result = recognize_canonical(&surf, 1e-6);
        assert!(result.is_some(), "tilted planar BSpline should be recognized");
        match result.unwrap() {
            SurfaceGeom::Plane { normal, .. } => {
                // Normal should be parallel to (1,1,1)
                let expected = PVec3::new(1.0, 1.0, 1.0).normalize();
                let dot = normal.dot(expected).abs();
                assert!(dot > 0.99, "normal should be parallel to (1,1,1), got {:?}", normal);
            }
            other => panic!("expected Plane, got {:?}", other),
        }
    }
}
