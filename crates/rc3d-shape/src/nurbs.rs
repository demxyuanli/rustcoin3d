//! NURBS surface evaluator: unified geometric representation for STEP surfaces.
//!
//! All analytic surfaces (plane, cylinder, cone, sphere, torus) and free-form
//! surfaces (B-spline, NURBS) are converted to this representation for uniform
//! evaluation and derivative computation.

use rc3d_core::math::Vec3;
use crate::geom::bspline::{bspline_bases, find_span};

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

    /// True when the surface is closed/periodic in the u direction (for UV unwrap).
    pub fn is_u_closed(&self, tol: f32) -> bool {
        let n = self.u_count();
        if n < 2 {
            return false;
        }
        let first = &self.control_points[0];
        let last = &self.control_points[n - 1];
        if first.len() != last.len() {
            return false;
        }
        first
            .iter()
            .zip(last.iter())
            .all(|(a, b)| (*a - *b).length() <= tol)
    }

    /// True when the surface is closed/periodic in the v direction.
    pub fn is_v_closed(&self, tol: f32) -> bool {
        let nv = self.v_count();
        if nv < 2 || self.u_count() == 0 {
            return false;
        }
        (0..self.u_count()).all(|i| {
            let a = self.control_points[i][0];
            let b = self.control_points[i][nv - 1];
            (a - b).length() <= tol
        })
    }

    /// Parametric period in u when the NURBS is closed in u.
    pub fn u_period(&self) -> Option<f32> {
        if !self.is_u_closed(1e-3) {
            return None;
        }
        let u0 = self.knots_u[self.degree_u];
        let u1 = self.knots_u[self.u_count()];
        let span = u1 - u0;
        if span > 1e-8 {
            Some(span)
        } else {
            None
        }
    }

    /// Parametric period in v when the NURBS is closed in v.
    pub fn v_period(&self) -> Option<f32> {
        if !self.is_v_closed(1e-3) {
            return None;
        }
        let v0 = self.knots_v[self.degree_v];
        let v1 = self.knots_v[self.v_count()];
        let span = v1 - v0;
        if span > 1e-8 {
            Some(span)
        } else {
            None
        }
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

    /// Evaluate surface point and first-order partial derivatives together at (u, v).
    /// Returns (point, ∂S/∂u, ∂S/∂v). Shares span and basis computation between
    /// position and derivative evaluation for better performance.
    pub fn evaluate_with_derivative(&self, u: f32, v: f32) -> (Vec3, Vec3, Vec3) {
        // Shared span + basis computation (computed once, used for both position and derivatives)
        let span_u = find_span(self.degree_u, &self.knots_u, u);
        let span_v = find_span(self.degree_v, &self.knots_v, v);
        let basis_u = bspline_bases(span_u, self.degree_u, u, &self.knots_u);
        let basis_v = bspline_bases(span_v, self.degree_v, v, &self.knots_v);

        // Analytical basis derivatives (computed once, used for both ∂u and ∂v)
        let du_basis = analytical_basis_derivatives(span_u, self.degree_u, u, &self.knots_u);
        let dv_basis = analytical_basis_derivatives(span_v, self.degree_v, v, &self.knots_v);

        // Accumulate weighted position and partial derivatives in a single pass
        let mut w_sum = 0.0f32;
        let mut w_u = 0.0f32;
        let mut w_v = 0.0f32;
        let mut p = Vec3::ZERO;
        let mut p_u = Vec3::ZERO;
        let mut p_v = Vec3::ZERO;

        for &(i, nu) in &basis_u {
            let dn_du = lookup_basis_value(&du_basis, i);
            for &(j, nv) in &basis_v {
                let wgt = self.weights[i][j];
                let cp = self.control_points[i][j];
                let coeff = nu * nv * wgt;
                w_sum += coeff;
                p = p + cp * coeff;
                // ∂/∂u
                let c_u = dn_du * nv * wgt;
                w_u += c_u;
                p_u = p_u + cp * c_u;
                // ∂/∂v
                let dn_dv = lookup_basis_value(&dv_basis, j);
                let c_v = nu * dn_dv * wgt;
                w_v += c_v;
                p_v = p_v + cp * c_v;
            }
        }

        if w_sum.abs() < 1e-10 {
            return (Vec3::ZERO, Vec3::X, Vec3::Y);
        }

        let inv_w = 1.0 / w_sum;
        let pos = p * inv_w;
        // Quotient rule: d/dx (A/W) = (A' * W - A * W') / W²
        let inv_w2 = inv_w * inv_w;
        let du = (p_u * w_sum - p * w_u) * inv_w2;
        let dv = (p_v * w_sum - p * w_v) * inv_w2;
        (pos, du, dv)
    }

    /// Combined position + first + second derivatives in one pass.
    ///
    /// Shares `find_span` + `bspline_bases` between position and first derivatives.
    /// Second derivatives use finite differences of the first derivatives (4 extra
    /// `evaluate_with_derivative` calls at u±eps, v±eps), but the position evaluation
    /// is done only once (vs 5+ times when calling `evaluate` + `derivative` + `d2`
    /// separately through `SurfaceGeom::*` methods).
    pub fn evaluate_with_hessian(
        &self, u: f32, v: f32,
    ) -> (Vec3, Vec3, Vec3, Vec3, Vec3, Vec3) {
        let (pos, du, dv) = self.evaluate_with_derivative(u, v);

        // Finite-difference second derivatives from first derivatives
        let eps = 1e-4f32;
        let u_min = self.knots_u[self.degree_u];
        let u_max = self.knots_u[self.knots_u.len() - self.degree_u - 1];
        let v_min = self.knots_v[self.degree_v];
        let v_max = self.knots_v[self.knots_v.len() - self.degree_v - 1];

        let up = (u + eps).clamp(u_min, u_max);
        let um = (u - eps).clamp(u_min, u_max);
        let vp = (v + eps).clamp(v_min, v_max);
        let vm = (v - eps).clamp(v_min, v_max);

        let (_pu, du_p, dv_p) = self.evaluate_with_derivative(up, v);
        let (_mu, du_m, dv_m) = self.evaluate_with_derivative(um, v);
        let (_pv, du_vp, dv_vp) = self.evaluate_with_derivative(u, vp);
        let (_mv, du_vm, dv_vm) = self.evaluate_with_derivative(u, vm);

        let inv_2eps = 1.0 / (2.0 * eps);
        let duu = (du_p - du_m) * inv_2eps;
        let duv = (du_vp - du_vm) * inv_2eps;
        let dvv = (dv_vp - dv_vm) * inv_2eps;

        (pos, du, dv, duu, duv, dvv)
    }

    /// Compute first-order partial derivatives ∂S/∂u and ∂S/∂v at (u, v).
    /// Uses analytical B-spline derivative formulas for accuracy and efficiency.
    pub fn derivative(&self, u: f32, v: f32) -> (Vec3, Vec3) {
        let (_, du, dv) = self.evaluate_with_derivative(u, v);
        (du, dv)
    }

    /// Compute surface normal at (u, v) = ∂S/∂u × ∂S/∂v (normalized).
    /// Uses analytical derivatives for accuracy.
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

/// Linear scan lookup in a short basis derivative list (3-6 entries).
/// Faster than HashMap for these small sizes.
#[inline]
fn lookup_basis_value(basis: &[(usize, f32)], idx: usize) -> f32 {
    for &(i, v) in basis {
        if i == idx { return v; }
    }
    0.0
}

/// Compute first-order B-spline basis function derivatives using the analytical
/// B-spline derivative recurrence (Piegl & Tiller, The NURBS Book, Eq. 2.10):
///
///   N'_{i,p}(t) = p / (knots[i+p] - knots[i])     * N_{i,  p-1}(t)
///               - p / (knots[i+p+1] - knots[i+1]) * N_{i+1,p-1}(t)
///
/// Returns (index, derivative_value) pairs for the active basis functions
/// at the given span. For degree 0, all derivatives are zero.
fn analytical_basis_derivatives(span: usize, degree: usize, t: f32, knots: &[f32]) -> Vec<(usize, f32)> {
    if degree == 0 {
        return vec![(span, 0.0)];
    }

    // Degree p-1 basis functions: active for j = span-(p-1) to span (p functions)
    let lower_bases = bspline_bases(span, degree - 1, t, knots);

    let p = degree as f32;
    let mut result = Vec::with_capacity(degree + 1);

    // Degree-p basis functions are active for i = span-p to span (p+1 functions)
    let i_start = span.saturating_sub(degree);
    let i_end = span;

    for i in i_start..=i_end {
        // N_{i, p-1}(t): nonzero only if i is in [span-(p-1), span] = [span-p+1, span]
        let n_i = if i >= span - degree + 1 {
            lookup_basis_value(&lower_bases, i)
        } else {
            0.0
        };

        // N_{i+1, p-1}(t): nonzero only if i+1 is in [span-p+1, span]
        let n_ip1 = if i + 1 <= span {
            lookup_basis_value(&lower_bases, i + 1)
        } else {
            0.0
        };

        let denom1 = knots[i + degree] - knots[i];
        let denom2 = knots[i + degree + 1] - knots[i + 1];

        let term1 = if denom1.abs() > 1e-12 { n_i / denom1 } else { 0.0 };
        let term2 = if denom2.abs() > 1e-12 { n_ip1 / denom2 } else { 0.0 };

        let deriv = p * (term1 - term2);
        if deriv.abs() > 1e-12 {
            result.push((i, deriv));
        }
    }

    result
}

// B-spline basis functions imported from super::geom.

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

    /// Create a NURBS sphere using 6 bi-quadratic patches (one per face).
    /// Much more accurate than the single-patch approximation.
    ///
    /// Each patch is a non-rational (weights=1) biquadratic Bezier with
    /// control points projected to the sphere surface from a tangent plane.
    /// Using `half = r/3` keeps the control polygon close enough that the
    /// polynomial interior stays within 5% of the true radius.
    pub fn sphere_six_patch(radius: f32) -> Vec<NurbsSurface> {
        let r = radius;
        let faces = [
            (Vec3::Z, Vec3::X, Vec3::Y),
            (-Vec3::Z, Vec3::X, -Vec3::Y),
            (Vec3::X, Vec3::Y, Vec3::Z),
            (-Vec3::X, -Vec3::Y, Vec3::Z),
            (Vec3::Y, Vec3::Z, Vec3::X),
            (-Vec3::Y, -Vec3::Z, Vec3::X),
        ];

        let mut patches = Vec::with_capacity(6);
        for &(normal, u_dir, v_dir) in &faces {
            let nb = normal.normalize();
            let ub = u_dir.normalize();
            let vb = v_dir.normalize();
            let center = nb * r;
            let half = r / 3.0;

            let control_points: Vec<Vec<Vec3>> = (0..3).map(|i| {
                (0..3).map(|j| {
                    let u = (i as f32 - 1.0) * half;
                    let v = (j as f32 - 1.0) * half;
                    let pt = center + ub * u + vb * v;
                    let len = pt.length();
                    if len > 1e-6 { pt * (r / len) } else { pt }
                }).collect()
            }).collect();

            // Non-rational: all weights = 1. With tight tangent-plane spread,
            // the polynomial Bezier interior stays within 5% of true radius.
            let weights: Vec<Vec<f32>> = vec![vec![1.0; 3]; 3];

            patches.push(NurbsSurface {
                degree_u: 2, degree_v: 2,
                control_points, weights,
                knots_u: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                knots_v: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            });
        }
        patches
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

    #[test]
    fn test_evaluate_with_derivative() {
        // Test with cylinder: position matches evaluate(), derivatives match derivative()
        let surf = NurbsSurface::cylinder(1.0, 0.0, 1.0);
        let u = 0.25f32;
        let v = 0.5f32;

        let (pos_evd, du_evd, dv_evd) = surf.evaluate_with_derivative(u, v);
        let pos_eval = surf.evaluate(u, v);
        let (du_deriv, dv_deriv) = surf.derivative(u, v);

        assert!((pos_evd - pos_eval).length() < 1e-6,
            "evaluate_with_derivative position should match evaluate: {:?} vs {:?}", pos_evd, pos_eval);
        assert!((du_evd - du_deriv).length() < 1e-6,
            "evaluate_with_derivative du should match derivative: {:?} vs {:?}", du_evd, du_deriv);
        assert!((dv_evd - dv_deriv).length() < 1e-6,
            "evaluate_with_derivative dv should match derivative: {:?} vs {:?}", dv_evd, dv_deriv);

        // Test with plane: du should be +X, dv should be +Y
        let plane = NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        let (_, du, dv) = plane.evaluate_with_derivative(0.5, 0.5);
        assert!((du - Vec3::X).length() < 1e-5, "plane du should be +X, got {:?}", du);
        assert!((dv - Vec3::Y).length() < 1e-5, "plane dv should be +Y, got {:?}", dv);

        // Test degree-0 surface (degenerate): derivatives should still be computed without panic
        let deg0 = NurbsSurface {
            degree_u: 0,
            degree_v: 0,
            control_points: vec![vec![Vec3::new(1.0, 2.0, 3.0)]],
            weights: vec![vec![1.0]],
            knots_u: vec![0.0, 1.0],
            knots_v: vec![0.0, 1.0],
        };
        let (pos, _, _) = deg0.evaluate_with_derivative(0.5, 0.5);
        assert!((pos - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);

        // Test with torus: derivatives should produce a non-degenerate normal
        let torus = NurbsSurface::torus(3.0, 1.0);
        let (_, du, dv) = torus.evaluate_with_derivative(0.0, 0.0);
        let n = du.cross(dv);
        assert!(n.length() > 0.1, "torus derivatives should produce non-degenerate normal");
    }

    #[test]
    fn test_sphere_six_patch_radius_accuracy() {
        let r = 5.0;
        let patches = NurbsSurface::sphere_six_patch(r);
        assert_eq!(patches.len(), 6);
        for patch in &patches {
            for u in [0.0, 0.5, 1.0] {
                for v in [0.0, 0.5, 1.0] {
                    let pt = patch.evaluate(u, v);
                    let dist = pt.length();
                    assert!((dist - r).abs() < 0.05 * r,
                        "point at ({u},{v}) has distance {dist}, expected {r}");
                }
            }
        }
    }
}
