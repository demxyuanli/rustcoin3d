//! Newton refinement for surface-surface intersection (SSI).
//!
//! Given two surfaces S_a(u,v) and S_b(s,t), solves the 3-equation system
//! F(u,v,s,t) = S_a(u,v) - S_b(s,t) = 0 using Gauss-Newton iteration.
//!
//! The 3×4 Jacobian has a 1D nullspace (tangent direction of intersection curve),
//! so we use the pseudo-inverse J^+ = J^T (J J^T)^{-1}.
//!
//! OCC alignment: IntPatch_TheIWalking + math_Gauss integration

use rc3d_core::math::Vec3;
use crate::geom::SurfaceGeom;

/// Refine UV parameters on both surfaces so that the 3D distance between
/// S_a(u,v) and S_b(s,t) is minimized below `tol`.
///
/// Uses Gauss-Newton with pseudo-inverse of the 3×4 Jacobian.
/// Returns refined (uv_a, uv_b) or None if the iteration diverges.
pub fn newton_refine_ssi(
    surf_a: &SurfaceGeom,
    surf_b: &SurfaceGeom,
    mut uv_a: (f32, f32),
    mut uv_b: (f32, f32),
    tol: f32,
    max_iter: usize,
) -> Option<((f32, f32), (f32, f32))> {
    for _ in 0..max_iter {
        // Current 3D positions
        let pa = surf_a.d0_native(uv_a.0, uv_a.1);
        let pb = surf_b.d0_native(uv_b.0, uv_b.1);
        let residual = pa - pb;

        let dist = residual.length();
        if dist < tol {
            return Some((uv_a, uv_b));
        }

        // First derivatives
        let (du_a, dv_a) = surf_a.d1_native(uv_a.0, uv_a.1);
        let (du_b, dv_b) = surf_b.d1_native(uv_b.0, uv_b.1);

        // Jacobian J = [du_a, dv_a, -du_b, -dv_b]  (3×4 matrix)
        // Build 3×3 matrix M = J * J^T
        let m00 = du_a.dot(du_a) + dv_a.dot(dv_a) + du_b.dot(du_b) + dv_b.dot(dv_b);
        // Actually JJ^T = [du_a dv_a -du_b -dv_b] * [du_a^T; dv_a^T; -du_b^T; -dv_b^T]
        // = du_a*du_a^T + dv_a*dv_a^T + du_b*du_b^T + dv_b*dv_b^T
        // But we need the 3×3 symmetric matrix:
        let j00 = du_a.x.powi(2) + dv_a.x.powi(2) + du_b.x.powi(2) + dv_b.x.powi(2);
        let j01 = du_a.x * du_a.y + dv_a.x * dv_a.y + du_b.x * du_b.y + dv_b.x * dv_b.y;
        let j02 = du_a.x * du_a.z + dv_a.x * dv_a.z + du_b.x * du_b.z + dv_b.x * dv_b.z;
        let j11 = du_a.y.powi(2) + dv_a.y.powi(2) + du_b.y.powi(2) + dv_b.y.powi(2);
        let j12 = du_a.y * du_a.z + dv_a.y * dv_a.z + du_b.y * du_b.z + dv_b.y * dv_b.z;
        let j22 = du_a.z.powi(2) + dv_a.z.powi(2) + du_b.z.powi(2) + dv_b.z.powi(2);

        // Solve (JJ^T) * lambda = -residual for the 3D correction lambda
        // Then delta_4d = J^T * lambda
        let det = j00 * (j11 * j22 - j12 * j12)
                - j01 * (j01 * j22 - j12 * j02)
                + j02 * (j01 * j12 - j11 * j02);

        if det.abs() < 1e-20 {
            return None;
        }

        let inv_det = 1.0 / det;
        let r0 = -residual.x;
        let r1 = -residual.y;
        let r2 = -residual.z;

        let lambda_x = inv_det * (r0 * (j11 * j22 - j12 * j12)
                                - j01 * (r1 * j22 - j12 * r2)
                                + j02 * (r1 * j12 - j11 * r2));
        let lambda_y = inv_det * (j00 * (r1 * j22 - j12 * r2)
                                - r0 * (j01 * j22 - j12 * j02)
                                + j02 * (j01 * r2 - r1 * j02));
        let lambda_z = inv_det * (j00 * (j11 * r2 - r1 * j12)
                                - j01 * (j01 * r2 - r1 * j02)
                                + r0 * (j01 * j12 - j11 * j02));

        // delta_4d = J^T * lambda
        let du_a_dot = du_a.x * lambda_x + du_a.y * lambda_y + du_a.z * lambda_z;
        let dv_a_dot = dv_a.x * lambda_x + dv_a.y * lambda_y + dv_a.z * lambda_z;
        let du_b_dot = du_b.x * lambda_x + du_b.y * lambda_y + du_b.z * lambda_z;
        let dv_b_dot = dv_b.x * lambda_x + dv_b.y * lambda_y + dv_b.z * lambda_z;

        // Update UV parameters with damping
        let step_scale = 0.8; // damped step for stability
        uv_a.0 += du_a_dot * step_scale;
        uv_a.1 += dv_a_dot * step_scale;
        uv_b.0 -= du_b_dot * step_scale; // note the -du_b in Jacobian means we add to uv_b
        uv_b.1 -= dv_b_dot * step_scale;

        // Clamp to reasonable parameter ranges
        let range_a = surf_a.param_range();
        let range_b = surf_b.param_range();
        uv_a.0 = uv_a.0.clamp(range_a.u_min - range_a.u_span(), range_a.u_max + range_a.u_span());
        uv_a.1 = uv_a.1.clamp(range_a.v_min - range_a.v_span(), range_a.v_max + range_a.v_span());
        uv_b.0 = uv_b.0.clamp(range_b.u_min - range_b.u_span(), range_b.u_max + range_b.u_span());
        uv_b.1 = uv_b.1.clamp(range_b.v_min - range_b.v_span(), range_b.v_max + range_b.v_span());
    }

    None
}

/// Refine a 3D seed point to lie exactly on the intersection of two surfaces.
/// Projects the seed to surface A, then runs Newton refinement.
///
/// Returns the refined 3D point if successful, None otherwise.
pub fn refine_seed_to_intersection(
    surf_a: &SurfaceGeom,
    surf_b: &SurfaceGeom,
    seed_3d: Vec3,
    tol: f32,
) -> Option<(Vec3, (f32, f32), (f32, f32))> {
    // Initial projection to both surfaces
    let uv_a = surf_a.project(seed_3d)?;
    let uv_b = surf_b.project(seed_3d)?;

    let ((ua, va), (ub, vb)) = newton_refine_ssi(surf_a, surf_b, uv_a, uv_b, tol, 20)?;

    // Re-evaluate at refined UV for exact 3D position
    let point = surf_a.d0_native(ua, va);
    Some((point, (ua, va), (ub, vb)))
}

/// Compute the intersection curve tangent direction at a point on the intersection.
/// Tangent = normal_a × normal_b (normalized).
///
/// The sign determines the marching direction.
pub fn ssi_tangent(
    surf_a: &SurfaceGeom,
    surf_b: &SurfaceGeom,
    uv_a: (f32, f32),
    uv_b: (f32, f32),
) -> Option<Vec3> {
    let (du_a, dv_a) = surf_a.d1_native(uv_a.0, uv_a.1);
    let (du_b, dv_b) = surf_b.d1_native(uv_b.0, uv_b.1);
    let normal_a = du_a.cross(dv_a);
    let normal_b = du_b.cross(dv_b);

    if normal_a.length_squared() < 1e-20 || normal_b.length_squared() < 1e-20 {
        return None;
    }

    let tangent = normal_a.cross(normal_b);
    let len = tangent.length();
    if len < 1e-12 {
        None
    } else {
        Some(tangent * (1.0 / len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_newton_refine_ssi_planes_converges() {
        let s1 = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let s2 = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            u_dir: Vec3::X,
        };
        // Start with slightly offset UV: point should be near the X-axis
        let result = newton_refine_ssi(
            &s1, &s2,
            (0.49, 0.0),  // S_a: near x=0.5
            (0.51, 0.0),  // S_b: near x=0.5
            1e-4, 10,
        );
        assert!(result.is_some(), "Newton should converge for planes");
        let ((ua, _va), (ub, _vb)) = result.unwrap();
        // Both should converge to x≈0.5 on the intersection line
        let pa = s1.d0_native(ua, 0.0);
        let pb = s2.d0_native(ub, 0.0);
        let dist = (pa - pb).length();
        assert!(dist < 1e-3, "refined points should coincide, dist={}", dist);
    }

    #[test]
    fn test_newton_refine_ssi_cylinder_plane() {
        let cyl = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 1.0);
        let plane = SurfaceGeom::Plane {
            origin: Vec3::new(0.5, 0.0, 0.0),
            normal: Vec3::X,
            u_dir: Vec3::Y,
        };
        // Intersection: two vertical lines at x=0.5, y=±0.866.
        // Seed near the upper intersection: angle ~60° on cylinder gives x=cos(60°)=0.5
        let angle_60 = std::f32::consts::FRAC_PI_3; // 60° in radians
        let result = newton_refine_ssi(
            &cyl, &plane,
            (angle_60, 0.5),     // cylinder: angle 60°, z=0.5 → near (0.5, 0.866, 0.5)
            (0.866, 0.5),        // plane: y=0.866, z=0.5 → near (0.5, 0.866, 0.5)
            1e-3, 20,
        );
        assert!(result.is_some(), "Newton should converge for cylinder-plane");
        let ((ua, _va), (ub, _vb)) = result.unwrap();
        let pa = cyl.d0_native(ua, 0.5);
        let pb = plane.d0_native(ub, 0.5);
        let dist = (pa - pb).length();
        assert!(dist < 0.01, "refined points should coincide, dist={}", dist);
    }

    #[test]
    fn test_ssi_tangent_planes() {
        let s1 = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };
        let s2 = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Y, u_dir: Vec3::X,
        };
        // Intersection is the X-axis; tangent should be ±X
        let tangent = ssi_tangent(&s1, &s2, (0.5, 0.0), (0.5, 0.0));
        assert!(tangent.is_some());
        let t = tangent.unwrap();
        // Should be parallel to X-axis
        assert!(t.x.abs() > 0.99, "tangent should be along X, got {:?}", t);
    }
}
