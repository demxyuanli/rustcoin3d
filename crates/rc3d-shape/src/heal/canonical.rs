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
            if let Some(cylinder) = try_as_cylinder(nurbs, tolerance) {
                return Some(cylinder);
            }
            if let Some(sphere) = try_as_sphere(nurbs, tolerance) {
                return Some(sphere);
            }
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
/// Strategy: the characteristic polynomial is a cubic whose roots are the
/// eigenvalues. For a real symmetric matrix all eigenvalues are real, so we
/// solve the depressed cubic trigonometrically, pick the smallest eigenvalue,
/// then back-substitute to get the corresponding eigenvector via row cross-
/// products of (M - λI).
fn smallest_eigenvector_symmetric(
    c00: f64, c01: f64, c02: f64,
          c11: f64, c12: f64,
                    c22: f64,
) -> PVec3 {
    // Characteristic polynomial:  λ³ + A·λ² + B·λ + C = 0
    let a = -(c00 + c11 + c22);
    let b = c00 * c11 + c00 * c22 + c11 * c22 - c01 * c01 - c02 * c02 - c12 * c12;
    let c = -(c00 * c11 * c22 + 2.0 * c01 * c12 * c02
        - c00 * c12 * c12
        - c11 * c02 * c02
        - c22 * c01 * c01);

    // Depressed cubic:  x³ + p·x + q = 0   where λ = x - a/3
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;

    let lambda_min = if p.abs() < 1e-15 {
        // p ≈ 0: triple (or near-triple) root  x³ + q = 0  → x = -∛q
        let x = -q.cbrt();
        x - a / 3.0
    } else {
        // Three real roots via trigonometric formula (discriminant ≤ 0 for
        // symmetric matrices; if floating noise pushes it positive, clamp).
        let r = (-p / 3.0).sqrt();
        let phi_arg = (3.0 * q / (2.0 * p * r)).clamp(-1.0, 1.0);
        let phi = phi_arg.acos() / 3.0;

        let x0 = 2.0 * r * phi.cos();
        let x1 = 2.0 * r * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
        let x2 = 2.0 * r * (phi + 4.0 * std::f64::consts::PI / 3.0).cos();

        let l0 = x0 - a / 3.0;
        let l1 = x1 - a / 3.0;
        let l2 = x2 - a / 3.0;
        l0.min(l1).min(l2)
    };

    // Eigenvector for λ_min: solve (M - λI)·v = 0.
    // Compute v as the cross product of two rows — robust for rank-2 matrices.
    let r0 = PVec3::new(c00 - lambda_min, c01, c02);
    let r1 = PVec3::new(c01, c11 - lambda_min, c12);
    let r2 = PVec3::new(c02, c12, c22 - lambda_min);

    let v = r0.cross(r1);
    let len2 = v.length_squared();
    if len2 > 1e-20 {
        return v / len2.sqrt();
    }
    let v = r1.cross(r2);
    let len2 = v.length_squared();
    if len2 > 1e-20 {
        return v / len2.sqrt();
    }
    let v = r2.cross(r0);
    let len2 = v.length_squared();
    if len2 > 1e-20 {
        return v / len2.sqrt();
    }
    // Degenerate — pick the first row direction
    r0.normalize()
}

/// Compute the smallest eigenvalue and corresponding eigenvector of a 2×2
/// symmetric matrix [[a, b], [b, c]].
///
/// Closed-form: eigenvalues λ = (a + c ± √((a-c)² + 4b²)) / 2.
/// Returns (λ_min, (x, y)) where (x, y) is the eigenvector for λ_min.
#[allow(dead_code)]
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
// Cylinder detection — circle fitting per u-row
// ══════════════════════════════════════════════════════════════════════════════

/// Try to recognize a BSpline surface whose control points lie on a cylinder.
///
/// Algorithm:
/// 1. For each u-direction row of control points, fit a 3D circle.
/// 2. Verify that all successfully-fitted rows have the same radius (within tol).
/// 3. Verify that all circle normals (axis directions) are parallel.
/// 4. If consistent, return a `SurfaceGeom::Cylinder`.
fn try_as_cylinder(nurbs: &NurbsSurface, tol: Real) -> Option<SurfaceGeom> {
    let cps = &nurbs.control_points;
    let u_count = cps.len();
    let v_count = cps.first().map(|r| r.len()).unwrap_or(0);
    if u_count < 3 || v_count < 2 {
        return None;
    }

    // For each u-row, fit a circle to the control points
    let mut axis_dirs = Vec::new();
    let mut centers = Vec::new();
    let mut radii = Vec::new();

    for i in 0..u_count {
        let pts: Vec<PVec3> = cps[i].iter().copied().collect();
        if let Some((center, axis, radius)) = fit_circle_3d(&pts, 100) {
            centers.push(center);
            axis_dirs.push(axis);
            radii.push(radius);
        }
    }
    if centers.len() < u_count / 2 {
        return None;
    }

    // Check consistency: all radii within tolerance
    let avg_radius = radii.iter().sum::<Real>() / radii.len() as Real;
    if radii
        .iter()
        .any(|r| (r - avg_radius).abs() > tol * avg_radius.max(1.0))
    {
        return None;
    }

    // Check axis directions are parallel (account for sign ambiguity)
    let avg_axis = axis_dirs
        .iter()
        .fold(PVec3::ZERO, |a, &b| a + b)
        .normalize();
    if axis_dirs
        .iter()
        .any(|&a| (1.0 - a.dot(avg_axis).abs()) > 0.01)
    {
        return None;
    }

    // Use the first centre as the origin point on the axis
    let origin = centers[0];

    Some(SurfaceGeom::cylinder(origin, avg_axis, avg_radius))
}

// ══════════════════════════════════════════════════════════════════════════════
// Sphere detection — constant-distance check
// ══════════════════════════════════════════════════════════════════════════════

/// Try to recognize a BSpline surface whose control points lie on a sphere.
///
/// Algorithm:
/// 1. Compute the centroid of all control points.
/// 2. Compute the average distance from centroid to each control point.
/// 3. Verify all control points are within `tol` of that average distance.
/// 4. If yes, return `SurfaceGeom::Sphere { center, radius }`.
fn try_as_sphere(nurbs: &NurbsSurface, tol: Real) -> Option<SurfaceGeom> {
    let cps: Vec<PVec3> = nurbs
        .control_points
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    if cps.len() < 4 {
        return None;
    }

    // Centroid as candidate centre
    let center = cps.iter().fold(PVec3::ZERO, |a, &b| a + b) / cps.len() as Real;
    let dists: Vec<Real> = cps.iter().map(|p| (*p - center).length()).collect();
    let avg_r = dists.iter().sum::<Real>() / dists.len() as Real;

    // Check all points are within tolerance of the sphere surface
    if dists.iter().any(|&d| (d - avg_r).abs() > tol) {
        return None;
    }

    Some(SurfaceGeom::Sphere {
        center,
        radius: avg_r,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Circle fitting helper — 3D → plane projection → algebraic 2D fit
// ══════════════════════════════════════════════════════════════════════════════

/// Fit a 3D circle to a set of points.
///
/// Returns `(center, normal, radius)` when successful.
///
/// Algorithm:
/// 1. Fit a plane to the points via PCA (same as `try_as_plane`).
/// 2. Project all points onto that plane.
/// 3. Fit a 2D circle algebraically in the plane coordinates.
/// 4. Convert the 2D centre back to 3D.
///
/// `_max_iter` is accepted for API compatibility; the algebraic method is
/// non-iterative so the parameter is unused.
fn fit_circle_3d(points: &[PVec3], _max_iter: usize) -> Option<(PVec3, PVec3, Real)> {
    if points.len() < 3 {
        return None;
    }

    let n = points.len() as Real;

    // Centroid
    let centroid: PVec3 = points.iter().fold(PVec3::ZERO, |acc, &p| acc + p) / n;

    // Covariance matrix (3x3 symmetric)
    let mut c00 = 0.0_f64;
    let mut c01 = 0.0_f64;
    let mut c02 = 0.0_f64;
    let mut c11 = 0.0_f64;
    let mut c12 = 0.0_f64;
    let mut c22 = 0.0_f64;

    for &p in points {
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

    let normal = smallest_eigenvector_symmetric(c00, c01, c02, c11, c12, c22);
    let len2 = normal.length_squared();
    if len2 < 1e-20 {
        return None;
    }
    let normal = normal / len2.sqrt();

    // Orthonormal basis on the fitted plane
    let (x_dir, y_dir) = build_ortho_axes(normal);

    // Project points to 2D
    let pts_2d: Vec<(Real, Real)> = points
        .iter()
        .map(|p| {
            let v = *p - centroid;
            (v.dot(x_dir), v.dot(y_dir))
        })
        .collect();

    // Fit 2D circle algebraically in plane coordinates
    let (cx, cy, radius) = fit_circle_2d(&pts_2d)?;

    // Convert 2D centre back to 3D
    let center = centroid + x_dir * cx + y_dir * cy;

    Some((center, normal, radius))
}

/// Algebraic (Kasa) 2D circle fit using centred-data formulation.
///
/// Returns `(center_x, center_y, radius)`.
///
/// Minimises Σ[(x_i - a)² + (y_i - b)² - R²]² via linear least squares.
/// After subtracting the centroid, the normal equations decouple into a 2×2
/// system for the centre offset and a scalar equation for R².
fn fit_circle_2d(points: &[(Real, Real)]) -> Option<(Real, Real, Real)> {
    if points.len() < 3 {
        return None;
    }

    let n = points.len() as Real;

    // Centroid of 2D points
    let xm: Real = points.iter().map(|p| p.0).sum::<Real>() / n;
    let ym: Real = points.iter().map(|p| p.1).sum::<Real>() / n;

    // Accumulate centred moments
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut sxu = 0.0_f64;
    let mut syu = 0.0_f64;
    let mut su = 0.0_f64;

    for &(x, y) in points {
        let xc = x - xm;
        let yc = y - ym;
        let u = xc * xc + yc * yc;
        sxx += xc * xc;
        syy += yc * yc;
        sxy += xc * yc;
        sxu += xc * u;
        syu += yc * u;
        su += u;
    }

    // Solve the 2×2 system for centred centre (a, b)
    let det = sxx * syy - sxy * sxy;
    if det.abs() < 1e-20 {
        // Points are (nearly) collinear — cannot fit a circle
        return None;
    }

    let a = (sxu * syy - syu * sxy) / (2.0 * det);
    let b = (sxx * syu - sxy * sxu) / (2.0 * det);

    let cx = a + xm;
    let cy = b + ym;
    let r2 = a * a + b * b + su / n;
    if r2 <= 0.0 {
        return None;
    }
    let radius = r2.sqrt();

    Some((cx, cy, radius))
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
        // Wavy surface: z varies non-uniformly with x and y — no row forms
        // a circle so neither cylinder nor sphere detection triggers.
        let cps: Vec<Vec<PVec3>> = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let x = i as f64;
                        let y = j as f64;
                        let z = (x - 1.5).powi(2) * (y - 1.5)
                            + 0.5 * (x * y).sin();
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

    /// BSpline with control points on a cylinder surface should be recognized.
    #[test]
    fn test_cylindrical_bspline_recognized() {
        let radius = 2.0;
        let u_count = 4; // 4 rows at different Z heights
        let v_count = 6; // 6 points around the full circle per row
        let cps: Vec<Vec<PVec3>> = (0..u_count)
            .map(|i| {
                let z = i as f64 * 1.5;
                (0..v_count)
                    .map(|j| {
                        let angle = j as f64 * std::f64::consts::TAU / v_count as f64;
                        PVec3::new(radius * angle.cos(), radius * angle.sin(), z)
                    })
                    .collect()
            })
            .collect();
        let nurbs = NurbsSurface::from_points_grid(&cps, 2, 2);
        let surf = SurfaceGeom::BSpline(nurbs);

        let result = recognize_canonical(&surf, 1e-6);
        assert!(
            result.is_some(),
            "cylindrical BSpline should be recognized"
        );
        match result.unwrap() {
            SurfaceGeom::Cylinder {
                origin: _,
                axis,
                radius: r,
                ..
            } => {
                // Axis should be roughly along Z
                let axis_dot = axis.dot(PVec3::Z).abs();
                assert!(
                    axis_dot > 0.99,
                    "cylinder axis should align with Z, got {:?}",
                    axis
                );
                assert!(
                    (r - radius).abs() < 1e-6,
                    "radius should be {}, got {}",
                    radius,
                    r
                );
            }
            other => panic!("expected Cylinder, got {:?}", other),
        }
    }

    /// BSpline with control points on a sphere surface should be recognized.
    #[test]
    fn test_spherical_bspline_recognized() {
        let radius = 2.0;
        let center = PVec3::new(1.0, 2.0, 3.0);
        // Generate control points on sphere surface (avoiding poles)
        let u_count = 5;
        let v_count = 8;
        let cps: Vec<Vec<PVec3>> = (0..u_count)
            .map(|i| {
                let phi = 0.3 + (std::f64::consts::PI - 0.6) * i as f64 / (u_count - 1) as f64;
                (0..v_count)
                    .map(|j| {
                        let theta =
                            j as f64 * std::f64::consts::TAU / v_count as f64;
                        let x = radius * phi.sin() * theta.cos();
                        let y = radius * phi.sin() * theta.sin();
                        let z = radius * phi.cos();
                        center + PVec3::new(x, y, z)
                    })
                    .collect()
            })
            .collect();
        let nurbs = NurbsSurface::from_points_grid(&cps, 2, 2);
        let surf = SurfaceGeom::BSpline(nurbs);

        let result = recognize_canonical(&surf, 1e-6);
        assert!(
            result.is_some(),
            "spherical BSpline should be recognized"
        );
        match result.unwrap() {
            SurfaceGeom::Sphere {
                center: c,
                radius: r,
            } => {
                assert!(
                    (c - center).length() < 1e-6,
                    "sphere center should be {:?}, got {:?}",
                    center,
                    c
                );
                assert!(
                    (r - radius).abs() < 1e-6,
                    "sphere radius should be {}, got {}",
                    radius,
                    r
                );
            }
            other => panic!("expected Sphere, got {:?}", other),
        }
    }

    /// A non-cylindrical/non-spherical BSpline should not be falsely recognized.
    #[test]
    fn test_non_cylindrical_bspline_not_recognized() {
        // Wavy surface: z varies non-uniformly with x and y
        let cps: Vec<Vec<PVec3>> = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let x = i as f64;
                        let y = j as f64;
                        let z = (x - 1.5).powi(2) * (y - 1.5)
                            + 0.5 * (x * y).sin();
                        PVec3::new(x, y, z)
                    })
                    .collect()
            })
            .collect();
        let nurbs = NurbsSurface::from_points_grid(&cps, 2, 2);
        let surf = SurfaceGeom::BSpline(nurbs);

        let result = recognize_canonical(&surf, 1e-6);
        assert!(
            result.is_none(),
            "non-cylindrical BSpline should not be recognized"
        );
    }
}
