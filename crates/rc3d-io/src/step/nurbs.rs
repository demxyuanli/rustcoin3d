//! NURBS surface evaluator: unified geometric representation for STEP surfaces.
//!
//! All analytic surfaces (plane, cylinder, cone, sphere, torus) and free-form
//! surfaces (B-spline, NURBS) are converted to this representation for uniform
//! evaluation and derivative computation.

use rc3d_core::math::Vec3;

/// NURBS (Non-Uniform Rational B-Spline) surface.
///
/// Control points are stored as [u][v] where u is the first parametric direction
/// and v is the second. For non-rational surfaces all weights are 1.0.
#[derive(Debug, Clone)]
pub struct NurbsSurface {
    pub degree_u: usize,
    pub degree_v: usize,
    /// Control points [u_count][v_count]
    pub control_points: Vec<Vec<Vec3>>,
    /// Weights [u_count][v_count], 1.0 for non-rational B-spline
    pub weights: Vec<Vec<f32>>,
    /// Knot vector for u direction, length = u_count + degree_u + 1
    pub knots_u: Vec<f32>,
    /// Knot vector for v direction, length = v_count + degree_v + 1
    pub knots_v: Vec<f32>,
}

impl NurbsSurface {
    /// Number of control points in u direction.
    pub fn u_count(&self) -> usize {
        self.control_points.len()
    }

    /// Number of control points in v direction.
    pub fn v_count(&self) -> usize {
        self.control_points.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Evaluate the surface at (u, v). Returns the 3D point.
    pub fn evaluate(&self, u: f32, v: f32) -> Vec3 {
        let span_u = find_span(self.degree_u, &self.knots_u, u);
        let span_v = find_span(self.degree_v, &self.knots_v, v);
        let basis_u = bspline_bases(span_u, self.degree_u, u, &self.knots_u);
        let basis_v = bspline_bases(span_v, self.degree_v, v, &self.knots_v);

        let mut point = Vec3::ZERO;
        let mut weight_sum = 0.0f32;

        for &(i, nu) in &basis_u {
            for &(j, nv) in &basis_v {
                let w = self.weights[i][j];
                let cp = self.control_points[i][j];
                let coeff = nu * nv * w;
                point = point + cp * coeff;
                weight_sum += coeff;
            }
        }

        if weight_sum > 1e-10 {
            point * (1.0 / weight_sum)
        } else {
            Vec3::ZERO
        }
    }

    /// Compute first-order partial derivatives ∂S/∂u and ∂S/∂v at (u, v).
    /// Uses central finite differences for robustness (works for both rational
    /// and non-rational surfaces without complex quotient-rule derivations).
    pub fn derivative(&self, u: f32, v: f32) -> (Vec3, Vec3) {
        let eps = 1e-4f32;

        let u_min = self.knots_u[self.degree_u];
        let u_max = self.knots_u[self.knots_u.len() - self.degree_u - 1];
        let v_min = self.knots_v[self.degree_v];
        let v_max = self.knots_v[self.knots_v.len() - self.degree_v - 1];

        let du = (u_max - u_min).max(eps);
        let dv = (v_max - v_min).max(eps);
        let hu = eps.min(du * 0.01);
        let hv = eps.min(dv * 0.01);

        let du_vec = if u - hu >= u_min && u + hu <= u_max {
            let p_plus = self.evaluate(u + hu, v);
            let p_minus = self.evaluate(u - hu, v);
            (p_plus - p_minus) * (1.0 / (2.0 * hu))
        } else if u + hu <= u_max {
            let p_plus = self.evaluate(u + hu, v);
            let p_here = self.evaluate(u, v);
            (p_plus - p_here) * (1.0 / hu)
        } else if u - hu >= u_min {
            let p_here = self.evaluate(u, v);
            let p_minus = self.evaluate(u - hu, v);
            (p_here - p_minus) * (1.0 / hu)
        } else {
            Vec3::X
        };

        let dv_vec = if v - hv >= v_min && v + hv <= v_max {
            let p_plus = self.evaluate(u, v + hv);
            let p_minus = self.evaluate(u, v - hv);
            (p_plus - p_minus) * (1.0 / (2.0 * hv))
        } else if v + hv <= v_max {
            let p_plus = self.evaluate(u, v + hv);
            let p_here = self.evaluate(u, v);
            (p_plus - p_here) * (1.0 / hv)
        } else if v - hv >= v_min {
            let p_here = self.evaluate(u, v);
            let p_minus = self.evaluate(u, v - hv);
            (p_here - p_minus) * (1.0 / hv)
        } else {
            Vec3::Y
        };

        (du_vec, dv_vec)
    }

    /// Compute surface normal at (u, v) = ∂S/∂u × ∂S/∂v (normalized).
    /// Falls back to nearby probes when the derivative is degenerate
    /// (e.g. at NURBS circle knot-multiplicity points).
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        let (du, dv) = self.derivative(u, v);
        let n = du.cross(dv);
        let len = n.length();
        if len > 1e-6 {
            return n * (1.0 / len);
        }

        // Degenerate — sample a small cross around (u,v) inside valid domain.
        let eps = 1e-3f32;
        let u_min = self.knots_u[self.degree_u];
        let u_max = self.knots_u[self.knots_u.len() - self.degree_u - 1];
        let v_min = self.knots_v[self.degree_v];
        let v_max = self.knots_v[self.knots_v.len() - self.degree_v - 1];

        let probes = [
            (u + eps, v),
            (u - eps, v),
            (u, v + eps),
            (u, v - eps),
        ];
        let mut best = Vec3::Z;
        let mut best_len = 0.0f32;
        for (uc, vc) in probes {
            if uc < u_min || uc > u_max || vc < v_min || vc > v_max {
                continue;
            }
            let (du2, dv2) = self.derivative(uc, vc);
            let n2 = du2.cross(dv2);
            let l2 = n2.length();
            if l2 > best_len {
                best_len = l2;
                best = n2 * (1.0 / l2);
            }
        }
        best
    }
}

// ── B-spline basis functions (Cox-de Boor recursion) ──────────────

fn find_span(degree: usize, knots: &[f32], t: f32) -> usize {
    let n = knots.len() - degree - 1;
    if t >= knots[n] {
        return n - 1;
    }
    for i in (degree..n).rev() {
        if t >= knots[i] {
            return i;
        }
    }
    degree
}

/// Evaluate B-spline basis functions at parameter t, returning (index, value)
/// pairs for the non-zero basis functions in the support of span.
fn bspline_bases(span: usize, degree: usize, t: f32, knots: &[f32]) -> Vec<(usize, f32)> {
    let mut basis = vec![vec![0.0f32; degree + 1]; degree + 1];
    basis[0][0] = 1.0;
    for j in 1..=degree {
        for i in 0..=j {
            let left = if i >= 1 && span + i >= j {
                let idx_lo = span + i - j;
                if idx_lo + j < knots.len() {
                    let denom = knots[idx_lo + j] - knots[idx_lo];
                    if denom > 1e-10 {
                        (t - knots[idx_lo]) / denom * basis[j - 1][i - 1]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let right = if i < j && span + i + 1 >= j && span + i + 1 < knots.len() {
                let idx_lo = span + i + 1 - j;
                let denom = knots[idx_lo + j] - knots[idx_lo];
                if denom > 1e-10 {
                    (knots[idx_lo + j] - t) / denom * basis[j - 1][i]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            basis[j][i] = left + right;
        }
    }
    let start = span.saturating_sub(degree);
    (0..=degree)
        .map(|i| (start + i, basis[degree][i]))
        .collect()
}

// ── Construction helpers: analytic surfaces → NURBS ───────────────

impl NurbsSurface {
    /// Create a NURBS plane (degree 1×1) covering the given UV domain.
    pub fn plane(u_min: f32, u_max: f32, v_min: f32, v_max: f32) -> Self {
        NurbsSurface {
            degree_u: 1,
            degree_v: 1,
            control_points: vec![
                vec![Vec3::new(u_min, v_min, 0.0), Vec3::new(u_min, v_max, 0.0)],
                vec![Vec3::new(u_max, v_min, 0.0), Vec3::new(u_max, v_max, 0.0)],
            ],
            weights: vec![vec![1.0; 2]; 2],
            knots_u: vec![u_min, u_min, u_max, u_max],
            knots_v: vec![v_min, v_min, v_max, v_max],
        }
    }

    /// Create a NURBS cylinder (degree 2×1) with given radius and height range.
    /// u is the circular direction (degree 2 rational circle), v is the axial direction.
    /// u ∈ [0, 1] maps to angle [0, 2π).
    pub fn cylinder(radius: f32, v_min: f32, v_max: f32) -> Self {
        // Rational circle as NURBS: 9 control points for full 360°
        // Control points arranged so that u=0 → angle=0 (+X axis)
        let w = 0.5f32.sqrt(); // weight for 45° arc control points
        let r = radius;
        NurbsSurface {
            degree_u: 2,
            degree_v: 1,
            control_points: vec![
                vec![Vec3::new(r, 0.0, v_min), Vec3::new(r, 0.0, v_max)],   // u=0: angle 0
                vec![Vec3::new(r, r, v_min), Vec3::new(r, r, v_max)],       // u=0.25: angle π/2
                vec![Vec3::new(0.0, r, v_min), Vec3::new(0.0, r, v_max)],   // u=0.5: angle π
                vec![Vec3::new(-r, r, v_min), Vec3::new(-r, r, v_max)],     // u=0.75: angle 3π/2
                vec![Vec3::new(-r, 0.0, v_min), Vec3::new(-r, 0.0, v_max)],
                vec![Vec3::new(-r, -r, v_min), Vec3::new(-r, -r, v_max)],
                vec![Vec3::new(0.0, -r, v_min), Vec3::new(0.0, -r, v_max)],
                vec![Vec3::new(r, -r, v_min), Vec3::new(r, -r, v_max)],
                vec![Vec3::new(r, 0.0, v_min), Vec3::new(r, 0.0, v_max)],   // u=1: angle 2π (same as 0)
            ],
            weights: vec![
                vec![1.0, 1.0],
                vec![w, w],
                vec![1.0, 1.0],
                vec![w, w],
                vec![1.0, 1.0],
                vec![w, w],
                vec![1.0, 1.0],
                vec![w, w],
                vec![1.0, 1.0],
            ],
            knots_u: vec![0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
            knots_v: vec![v_min, v_min, v_max, v_max],
        }
    }

    /// Create a NURBS cone (degree 2×1) with given base radius, semi-angle and height range.
    /// u is the circular direction, v is the axial direction.
    /// Radius varies linearly with v: r(v) = radius + v * tan(semi_angle).
    pub fn cone(radius: f32, semi_angle: f32, v_min: f32, v_max: f32) -> Self {
        let w = 0.5f32.sqrt();
        let tan_a = semi_angle.tan();
        let r_min = radius + v_min * tan_a;
        let r_max = radius + v_max * tan_a;
        NurbsSurface {
            degree_u: 2,
            degree_v: 1,
            control_points: vec![
                vec![Vec3::new(r_min, 0.0, v_min), Vec3::new(r_max, 0.0, v_max)],
                vec![Vec3::new(r_min, r_min, v_min), Vec3::new(r_max, r_max, v_max)],
                vec![Vec3::new(0.0, r_min, v_min), Vec3::new(0.0, r_max, v_max)],
                vec![Vec3::new(-r_min, r_min, v_min), Vec3::new(-r_max, r_max, v_max)],
                vec![Vec3::new(-r_min, 0.0, v_min), Vec3::new(-r_max, 0.0, v_max)],
                vec![Vec3::new(-r_min, -r_min, v_min), Vec3::new(-r_max, -r_max, v_max)],
                vec![Vec3::new(0.0, -r_min, v_min), Vec3::new(0.0, -r_max, v_max)],
                vec![Vec3::new(r_min, -r_min, v_min), Vec3::new(r_max, -r_max, v_max)],
                vec![Vec3::new(r_min, 0.0, v_min), Vec3::new(r_max, 0.0, v_max)],
            ],
            weights: vec![
                vec![1.0, 1.0],
                vec![w, w],
                vec![1.0, 1.0],
                vec![w, w],
                vec![1.0, 1.0],
                vec![w, w],
                vec![1.0, 1.0],
                vec![w, w],
                vec![1.0, 1.0],
            ],
            knots_u: vec![0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
            knots_v: vec![v_min, v_min, v_max, v_max],
        }
    }

    /// Create a NURBS torus (degree 2×2) with given major and minor radius.
    /// u is the major circular direction, v is the minor circular direction.
    pub fn torus(major_r: f32, minor_r: f32) -> Self {
        let w = 0.5f32.sqrt();
        // Unit circle control points in XY (u direction) and YZ (v direction)
        let u_cp: &[(f32, f32, f32)] = &[
            (1.0, 0.0, 1.0),
            (1.0, 1.0, w),
            (0.0, 1.0, 1.0),
            (-1.0, 1.0, w),
            (-1.0, 0.0, 1.0),
            (-1.0, -1.0, w),
            (0.0, -1.0, 1.0),
            (1.0, -1.0, w),
            (1.0, 0.0, 1.0),
        ];
        let v_cp: &[(f32, f32, f32)] = &[
            (1.0, 0.0, 1.0),   // (y, z, w) — angle π/2 so v=0 → z=0
            (1.0, 1.0, w),     // angle π/4
            (0.0, 1.0, 1.0),   // angle 0
            (-1.0, 1.0, w),    // angle -π/4
            (-1.0, 0.0, 1.0),  // angle -π/2
            (-1.0, -1.0, w),
            (0.0, -1.0, 1.0),
            (1.0, -1.0, w),
            (1.0, 0.0, 1.0),   // back to π/2
        ];

        let mut control_points = Vec::with_capacity(9);
        let mut weights = Vec::with_capacity(9);

        for (ux, uy, uw) in u_cp.iter().copied() {
            let mut row_cp = Vec::with_capacity(9);
            let mut row_w = Vec::with_capacity(9);
            for (vy, vz, vw) in v_cp.iter().copied() {
                let r = major_r + minor_r * vy;
                row_cp.push(Vec3::new(r * ux, r * uy, minor_r * vz));
                row_w.push(uw * vw);
            }
            control_points.push(row_cp);
            weights.push(row_w);
        }

        let knots = vec![0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0];
        NurbsSurface {
            degree_u: 2,
            degree_v: 2,
            control_points,
            weights,
            knots_u: knots.clone(),
            knots_v: knots,
        }
    }

    /// Create a NURBS sphere (degree 2×2) with given radius.
    pub fn sphere(radius: f32) -> Self {
        // Approximate sphere using rational B-spline patches.
        // For simplicity, use a single bi-quadratic patch (approximate).
        let r = radius;
        let w = 0.5f32.sqrt();
        NurbsSurface {
            degree_u: 2,
            degree_v: 2,
            control_points: vec![
                vec![
                    Vec3::new(-r, -r, -r), Vec3::new(-r, -r, 0.0), Vec3::new(-r, -r, r),
                ],
                vec![
                    Vec3::new(0.0, -r, -r), Vec3::new(0.0, -r, 0.0), Vec3::new(0.0, -r, r),
                ],
                vec![
                    Vec3::new(r, -r, -r), Vec3::new(r, -r, 0.0), Vec3::new(r, -r, r),
                ],
            ],
            weights: vec![
                vec![1.0, w, 1.0],
                vec![w, 0.5, w],
                vec![1.0, w, 1.0],
            ],
            knots_u: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_evaluate_corners() {
        let surf = NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        assert!((surf.evaluate(0.0, 0.0) - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-6);
        assert!((surf.evaluate(1.0, 0.0) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((surf.evaluate(0.0, 1.0) - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        assert!((surf.evaluate(1.0, 1.0) - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_plane_normal() {
        let surf = NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        let n = surf.normal(0.5, 0.5);
        assert!((n - Vec3::Z).length() < 1e-6, "plane normal should be +Z, got {:?}", n);
    }

    #[test]
    fn test_cylinder_evaluate() {
        let surf = NurbsSurface::cylinder(1.0, 0.0, 1.0);
        let pt = surf.evaluate(0.0, 0.5);
        // At u=0 (angle 0), point should be on +X axis
        assert!((pt - Vec3::new(1.0, 0.0, 0.5)).length() < 0.1,
            "cylinder point at u=0 should be near (1,0,0.5), got {:?}", pt);
    }

    #[test]
    fn test_cylinder_normal() {
        let surf = NurbsSurface::cylinder(1.0, 0.0, 1.0);
        let n = surf.normal(0.0, 0.5);
        // At u=0, normal should point outward in +X direction
        assert!((n - Vec3::X).length() < 0.1,
            "cylinder normal at u=0 should be near +X, got {:?}", n);
    }

    #[test]
    fn test_cone_evaluate_radius_increases() {
        let surf = NurbsSurface::cone(1.0, std::f32::consts::FRAC_PI_6, 0.0, 1.0);
        // At v=0 (base), radius should be ~1.0
        let pt_base = surf.evaluate(0.0, 0.0);
        let r_base = (pt_base.x * pt_base.x + pt_base.y * pt_base.y).sqrt();
        assert!((r_base - 1.0).abs() < 0.1, "cone base radius should be ~1.0, got {}", r_base);

        // At v=1 (top), radius should be larger
        let pt_top = surf.evaluate(0.0, 1.0);
        let r_top = (pt_top.x * pt_top.x + pt_top.y * pt_top.y).sqrt();
        let expected_top_r = 1.0 + 1.0 * (std::f32::consts::FRAC_PI_6).tan();
        assert!((r_top - expected_top_r).abs() < 0.1,
            "cone top radius should be ~{}, got {}", expected_top_r, r_top);
    }

    #[test]
    fn test_torus_evaluate_major_minor_radius() {
        let major = 3.0f32;
        let minor = 1.0f32;
        let surf = NurbsSurface::torus(major, minor);
        // At u=0, v=0: should be at (major+minor, 0, 0) - outer top
        let pt = surf.evaluate(0.0, 0.0);
        let r_xy = (pt.x * pt.x + pt.y * pt.y).sqrt();
        assert!((r_xy - (major + minor)).abs() < 0.2,
            "torus outer radius should be ~{}, got {}", major + minor, r_xy);
        assert!(pt.z.abs() < 0.2, "torus z at v=0 should be ~0, got {}", pt.z);
    }

    #[test]
    fn test_torus_normal_points_outward() {
        let surf = NurbsSurface::torus(3.0, 1.0);
        let n = surf.normal(0.0, 0.0);
        // At u=0, v=0 (outer top), normal should point roughly outward (+X)
        assert!(n.x > 0.5, "torus normal at outer top should point +X, got {:?}", n);
    }
}
