//! NURBS surface evaluator: unified geometric representation for STEP surfaces.
//!
//! All analytic surfaces (plane, cylinder, cone, sphere, torus) and free-form
//! surfaces (B-spline, NURBS) are converted to this representation for uniform
//! evaluation and derivative computation.

use rc3d_core::math::Vec3;
use crate::geom::bspline::{bspline_bases, find_span};
use rc3d_nurbs::NurbsCurve;

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

// ── NURBS type conversions ──────────────────────────────────────

impl From<NurbsSurface> for rc3d_nurbs::NurbsRenderSurface {
    /// Convert from separated control points + weights to homogeneous [x,y,z,w] format.
    fn from(s: NurbsSurface) -> Self {
        let control_points: Vec<Vec<[f32; 4]>> = s
            .control_points
            .iter()
            .zip(s.weights.iter())
            .map(|(cp_row, w_row)| {
                cp_row
                    .iter()
                    .zip(w_row.iter())
                    .map(|(&cp, &w)| [cp.x * w, cp.y * w, cp.z * w, w])
                    .collect()
            })
            .collect();
        rc3d_nurbs::NurbsRenderSurface {
            control_points,
            u_knots: s.knots_u,
            v_knots: s.knots_v,
            u_degree: s.degree_u,
            v_degree: s.degree_v,
        }
    }
}

impl From<rc3d_nurbs::NurbsRenderSurface> for NurbsSurface {
    /// Convert from homogeneous [x,y,z,w] to separated control points + weights.
    fn from(s: rc3d_nurbs::NurbsRenderSurface) -> Self {
        let n_u = s.u_count();
        let n_v = s.v_count();
        let mut control_points = Vec::with_capacity(n_u);
        let mut weights = Vec::with_capacity(n_u);
        for i in 0..n_u {
            let mut cp_row = Vec::with_capacity(n_v);
            let mut w_row = Vec::with_capacity(n_v);
            for j in 0..n_v {
                let [wx, wy, wz, w] = s.control_points[i][j];
                let inv_w = if w.abs() > 1e-10 { 1.0 / w } else { 1.0 };
                cp_row.push(Vec3::new(wx * inv_w, wy * inv_w, wz * inv_w));
                w_row.push(w);
            }
            control_points.push(cp_row);
            weights.push(w_row);
        }
        NurbsSurface {
            degree_u: s.u_degree,
            degree_v: s.v_degree,
            control_points,
            weights,
            knots_u: s.u_knots,
            knots_v: s.v_knots,
        }
    }
}

impl NurbsSurface {
    /// Number of control points in u direction.
    /// Find the knot span index in U direction: knots[i] <= t < knots[i+1].
    pub fn find_span_u(&self, t: f32) -> usize {
        let n = self.knots_u.len() - self.degree_u - 2;
        for i in (self.degree_u..=n).rev() {
            if t >= self.knots_u[i] { return i; }
        }
        self.degree_u
    }

    /// Find the knot span index in V direction.
    pub fn find_span_v(&self, t: f32) -> usize {
        let n = self.knots_v.len() - self.degree_v - 2;
        for i in (self.degree_v..=n).rev() {
            if t >= self.knots_v[i] { return i; }
        }
        self.degree_v
    }

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
                point += cp * coeff;
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
                p += cp * coeff;
                // ∂/∂u
                let c_u = dn_du * nv * wgt;
                w_u += c_u;
                p_u += cp * c_u;
                // ∂/∂v
                let dn_dv = lookup_basis_value(&dv_basis, j);
                let c_v = nu * dn_dv * wgt;
                w_v += c_v;
                p_v += cp * c_v;
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
    /// Uses analytical B-spline second derivatives (Piegl & Tiller recurrence
    /// applied twice), avoiding the 4 extra `evaluate_with_derivative` calls
    /// that finite differences would require. The analytical path is both
    /// faster and more accurate (no numerical cancellation from eps steps).
    pub fn evaluate_with_hessian(
        &self, u: f32, v: f32,
    ) -> (Vec3, Vec3, Vec3, Vec3, Vec3, Vec3) {
        // Shared span computation
        let span_u = find_span(self.degree_u, &self.knots_u, u);
        let span_v = find_span(self.degree_v, &self.knots_v, v);

        // Basis functions + first derivatives + second derivatives
        let basis_u = bspline_bases(span_u, self.degree_u, u, &self.knots_u);
        let basis_v = bspline_bases(span_v, self.degree_v, v, &self.knots_v);
        let d1_u = analytical_basis_derivatives(span_u, self.degree_u, u, &self.knots_u);
        let d1_v = analytical_basis_derivatives(span_v, self.degree_v, v, &self.knots_v);
        let d2_u = analytical_basis_second_derivatives(span_u, self.degree_u, u, &self.knots_u);
        let d2_v = analytical_basis_second_derivatives(span_v, self.degree_v, v, &self.knots_v);

        // Single-pass accumulation of weighted sums
        let mut w_sum = 0.0f32;
        let mut w_u = 0.0f32;    let mut w_v = 0.0f32;
        let mut w_uu = 0.0f32;   let mut w_uv = 0.0f32;   let mut w_vv = 0.0f32;

        let mut p = Vec3::ZERO;
        let mut p_u = Vec3::ZERO;    let mut p_v = Vec3::ZERO;
        let mut p_uu = Vec3::ZERO;   let mut p_uv = Vec3::ZERO;   let mut p_vv = Vec3::ZERO;

        for &(i, nu) in &basis_u {
            let dn_du = lookup_basis_value(&d1_u, i);
            let d2n_du = lookup_basis_value(&d2_u, i);
            for &(j, nv) in &basis_v {
                let wgt = self.weights[i][j];
                let cp = self.control_points[i][j];
                let dn_dv = lookup_basis_value(&d1_v, j);
                let d2n_dv = lookup_basis_value(&d2_v, j);

                // Position: nu * nv * wgt
                let c0 = nu * nv * wgt;
                w_sum += c0;
                p += cp * c0;

                // ∂/∂u: dn_du * nv * wgt
                let c_u = dn_du * nv * wgt;
                w_u += c_u;
                p_u += cp * c_u;

                // ∂/∂v: nu * dn_dv * wgt
                let c_v = nu * dn_dv * wgt;
                w_v += c_v;
                p_v += cp * c_v;

                // ∂²/∂u²: d2n_du * nv * wgt
                let c_uu = d2n_du * nv * wgt;
                w_uu += c_uu;
                p_uu += cp * c_uu;

                // ∂²/∂u∂v: dn_du * dn_dv * wgt
                let c_uv = dn_du * dn_dv * wgt;
                w_uv += c_uv;
                p_uv += cp * c_uv;

                // ∂²/∂v²: nu * d2n_dv * wgt
                let c_vv = nu * d2n_dv * wgt;
                w_vv += c_vv;
                p_vv += cp * c_vv;
            }
        }

        if w_sum.abs() < 1e-10 {
            return (Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        }

        let inv_w = 1.0 / w_sum;
        let inv_w2 = inv_w * inv_w;

        // Position (rational quotient rule)
        let pos = p * inv_w;

        // First derivatives (quotient rule)
        let du = (p_u * w_sum - p * w_u) * inv_w2;
        let dv = (p_v * w_sum - p * w_v) * inv_w2;

        // Second derivatives (full quotient rule for rational surfaces)
        let duu = (p_uu * w_sum - 2.0 * p_u * w_u - p * w_uu
                   + 2.0 * p * w_u * w_u * inv_w) * inv_w2;
        let duv = (p_uv * w_sum - p_u * w_v - p_v * w_u - p * w_uv
                   + 2.0 * p * w_u * w_v * inv_w) * inv_w2;
        let dvv = (p_vv * w_sum - 2.0 * p_v * w_v - p * w_vv
                   + 2.0 * p * w_v * w_v * inv_w) * inv_w2;

        // NaN safety
        let safe = |v: Vec3| if v.is_nan() { Vec3::ZERO } else { v };
        (safe(pos), safe(du), safe(dv), safe(duu), safe(duv), safe(dvv))
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

    /// Insert a knot at parameter `t` in the U direction (Boehm's algorithm).
    ///
    /// The surface shape is invariant — only the representation changes
    /// (one additional control point per V column, refined knot vector).
    pub fn insert_knot_u(&mut self, t: f32) {
        let n_u = self.u_count();
        let n_v = self.v_count();
        let p = self.degree_u;

        let span = find_span(p, &self.knots_u, t);

        // Check if knot already has max multiplicity (degree + 1)
        let mult = self.knots_u.iter().filter(|&&k| (k - t).abs() < 1e-10).count();
        if mult > p {
            return;
        }

        // Build new knot vector
        let mut new_knots_u = self.knots_u.clone();
        new_knots_u.insert(span + 1, t);

        // Build new control points and weights by applying Boehm per v-column
        let mut new_cp = Vec::with_capacity(n_u + 1);
        let mut new_w = Vec::with_capacity(n_u + 1);

        let start = span.saturating_sub(p);

        for i in 0..=n_u {
            if i <= start {
                // Copy unchanged
                new_cp.push(self.control_points[i].clone());
                new_w.push(self.weights[i].clone());
            } else if i > span {
                // Copy unchanged (shifted by 1)
                new_cp.push(self.control_points[i - 1].clone());
                new_w.push(self.weights[i - 1].clone());
            } else {
                // Affected range: blend (i-1) and (i)
                let alpha = {
                    let denom = self.knots_u[i + p] - self.knots_u[i];
                    if denom.abs() > 1e-10 {
                        ((t - self.knots_u[i]) / denom).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                };
                let prev_cp = &self.control_points[i - 1];
                let curr_cp = &self.control_points[i];
                let prev_w = &self.weights[i - 1];
                let curr_w = &self.weights[i];

                let mut row_cp = Vec::with_capacity(n_v);
                let mut row_w = Vec::with_capacity(n_v);
                for j in 0..n_v {
                    row_cp.push(alpha * curr_cp[j] + (1.0 - alpha) * prev_cp[j]);
                    row_w.push(alpha * curr_w[j] + (1.0 - alpha) * prev_w[j]);
                }
                new_cp.push(row_cp);
                new_w.push(row_w);
            }
        }

        self.control_points = new_cp;
        self.weights = new_w;
        self.knots_u = new_knots_u;
    }

    /// Insert a knot at parameter `t` in the V direction (Boehm's algorithm).
    ///
    /// The surface shape is invariant — only the representation changes
    /// (one additional control point per U row, refined knot vector).
    pub fn insert_knot_v(&mut self, t: f32) {
        let p = self.degree_v;

        let span = find_span(p, &self.knots_v, t);

        // Check if knot already has max multiplicity (degree + 1)
        let mult = self.knots_v.iter().filter(|&&k| (k - t).abs() < 1e-10).count();
        if mult > p {
            return;
        }

        // Build new knot vector
        let mut new_knots_v = self.knots_v.clone();
        new_knots_v.insert(span + 1, t);

        // Apply Boehm per U row
        let start = span.saturating_sub(p);

        for u in 0..self.control_points.len() {
            let row_cp = &self.control_points[u];
            let row_w = &self.weights[u];
            let n = row_cp.len();

            let mut new_row_cp = Vec::with_capacity(n + 1);
            let mut new_row_w = Vec::with_capacity(n + 1);

            for j in 0..=n {
                if j <= start {
                    new_row_cp.push(row_cp[j]);
                    new_row_w.push(row_w[j]);
                } else if j > span {
                    new_row_cp.push(row_cp[j - 1]);
                    new_row_w.push(row_w[j - 1]);
                } else {
                    let alpha = {
                        let denom = self.knots_v[j + p] - self.knots_v[j];
                        if denom.abs() > 1e-10 {
                            ((t - self.knots_v[j]) / denom).clamp(0.0, 1.0)
                        } else {
                            0.0
                        }
                    };
                    new_row_cp.push(alpha * row_cp[j] + (1.0 - alpha) * row_cp[j - 1]);
                    new_row_w.push(alpha * row_w[j] + (1.0 - alpha) * row_w[j - 1]);
                }
            }

            self.control_points[u] = new_row_cp;
            self.weights[u] = new_row_w;
        }

        self.knots_v = new_knots_v;
    }

    /// Create a NURBS surface from a grid of points with clamped uniform knot vectors.
    ///
    /// `points[i][j]` is row i (u direction), column j (v direction).
    /// Produces a non-rational surface (all weights = 1.0).
    pub fn from_points_grid(points: &[Vec<Vec3>], u_degree: usize, v_degree: usize) -> Self {
        let n_u = points.len();
        let n_v = points.first().map(|r| r.len()).unwrap_or(0);

        // Clamped uniform knot vectors
        let knots_u = clamped_uniform_knots(u_degree, n_u);
        let knots_v = clamped_uniform_knots(v_degree, n_v);

        let weights = vec![vec![1.0f32; n_v]; n_u];

        NurbsSurface {
            degree_u: u_degree,
            degree_v: v_degree,
            control_points: points.to_vec(),
            weights,
            knots_u,
            knots_v,
        }
    }

    /// Elevate the degree in the U direction by 1 (Bezier decomposition method).
    ///
    /// Each V-column is treated as a NURBS curve, degree-elevated independently,
    /// then reassembled back into the surface. Shape is invariant.
    pub fn elevate_degree_u(&mut self) {
        let n_v = self.v_count();

        let mut new_control_points = Vec::new();
        let mut new_weights = Vec::new();
        let mut new_knots_u = None;

        for j in 0..n_v {
            // Column j: collect control points along u direction
            let cp: Vec<[f32; 4]> = (0..self.u_count()).map(|i| {
                let pt = self.control_points[i][j];
                let w = self.weights[i][j];
                [pt.x * w, pt.y * w, pt.z * w, w]
            }).collect();
            let curve = NurbsCurve {
                control_points: cp,
                knots: self.knots_u.clone(),
                degree: self.degree_u,
            };
            let elevated = curve.elevate_degree();

            // Convert back from homogeneous
            let pts: Vec<Vec3> = elevated.control_points.iter().map(|c| {
                let w = c[3];
                if w.abs() > 1e-10 { Vec3::new(c[0]/w, c[1]/w, c[2]/w) }
                else { Vec3::new(c[0], c[1], c[2]) }
            }).collect();
            let ws: Vec<f32> = elevated.control_points.iter().map(|c| c[3]).collect();

            if new_knots_u.is_none() {
                new_knots_u = Some(elevated.knots);
            }

            // First column initializes the rows; subsequent columns append to each row
            if new_control_points.is_empty() {
                new_control_points = pts.into_iter().map(|p| vec![p]).collect();
                new_weights = ws.into_iter().map(|w| vec![w]).collect();
            } else {
                for (i, p) in pts.into_iter().enumerate() {
                    new_control_points[i].push(p);
                    new_weights[i].push(ws[i]);
                }
            }
        }

        self.control_points = new_control_points;
        self.weights = new_weights;
        self.knots_u = new_knots_u.unwrap();
        self.degree_u += 1;
    }

    /// Elevate the degree in the V direction by 1 (Bezier decomposition method).
    ///
    /// Each U-row is treated as a NURBS curve, degree-elevated independently,
    /// then reassembled back into the surface. Shape is invariant.
    pub fn elevate_degree_v(&mut self) {
        let n_u = self.u_count();

        for i in 0..n_u {
            let row_cp = &self.control_points[i];
            let row_w = &self.weights[i];

            // Row i: collect control points along v direction
            let cp: Vec<[f32; 4]> = (0..row_cp.len()).map(|j| {
                let pt = row_cp[j];
                let w = row_w[j];
                [pt.x * w, pt.y * w, pt.z * w, w]
            }).collect();
            let curve = NurbsCurve {
                control_points: cp,
                knots: self.knots_v.clone(),
                degree: self.degree_v,
            };
            let elevated = curve.elevate_degree();

            // Convert back from homogeneous
            let pts: Vec<Vec3> = elevated.control_points.iter().map(|c| {
                let w = c[3];
                if w.abs() > 1e-10 { Vec3::new(c[0]/w, c[1]/w, c[2]/w) }
                else { Vec3::new(c[0], c[1], c[2]) }
            }).collect();
            let ws: Vec<f32> = elevated.control_points.iter().map(|c| c[3]).collect();

            self.control_points[i] = pts;
            self.weights[i] = ws;

            // All rows produce the same knot vector — update on first iteration
            if i == 0 {
                self.knots_v = elevated.knots;
            }
        }

        self.degree_v += 1;
    }

    /// Try to reduce degree in U by 1 using approximate inverse of elevation.
    /// Returns false if shape deviation exceeds tolerance.
    /// OCC: BSplCLib::ReduceDegree
    pub fn reduce_degree_u(&mut self, tolerance: f32) -> bool {
        if self.degree_u <= 1 { return false; }
        let new_deg = self.degree_u - 1;
        let n_u = self.u_count();
        let n_v = self.v_count();
        if n_u < new_deg + 2 { return false; }

        // Approximate inverse of degree elevation:
        // elevated_cp[i] = α*original[i-1] + (1-α)*original[i]
        // → original[i] ≈ (elevated[i] - α*original[i-1]) / (1-α)
        // For each u-row, try reduction independently.
        let mut new_cps = Vec::with_capacity(n_u - 1);
        let mut new_ws = Vec::with_capacity(n_u - 1);
        for i in 0..n_u {
            if i == 0 {
                new_cps.push(self.control_points[0].clone());
                new_ws.push(self.weights[0].clone());
            } else if i == n_u - 1 {
                new_cps.push(self.control_points[n_u - 1].clone());
                new_ws.push(self.weights[n_u - 1].clone());
            } else {
                let alpha = i as f32 / self.degree_u as f32;
                let one_minus_a = 1.0 - alpha;
                let prev_cp = &new_cps[new_cps.len() - 1];
                let prev_w = &new_ws[new_ws.len() - 1];
                let mut row_cp = Vec::with_capacity(n_v);
                let mut row_w = Vec::with_capacity(n_v);
                for j in 0..n_v {
                    let cp = if one_minus_a.abs() > 1e-10 {
                        (self.control_points[i][j] - prev_cp[j] * alpha) / one_minus_a
                    } else { self.control_points[i][j] };
                    let w = if one_minus_a.abs() > 1e-10 {
                        (self.weights[i][j] - prev_w[j] * alpha) / one_minus_a
                    } else { self.weights[i][j] };
                    row_cp.push(cp); row_w.push(w);
                }
                new_cps.push(row_cp); new_ws.push(row_w);
            }
        }

        // Deviation check: sample and compare
        if !self.reduce_deviation_ok(&new_cps, &new_ws, tolerance) { return false; }
        self.control_points = new_cps;
        self.weights = new_ws;
        self.degree_u = new_deg;
        let n = self.u_count();
        self.knots_u = clamped_uniform_knots(new_deg, n);
        true
    }

    /// Try to reduce degree in V by 1.
    pub fn reduce_degree_v(&mut self, tolerance: f32) -> bool {
        if self.degree_v <= 1 { return false; }
        let new_deg = self.degree_v - 1;
        let n_u = self.u_count();
        let n_v = self.v_count();
        if n_v < new_deg + 2 { return false; }

        let mut new_cp_per_row: Vec<Vec<Vec3>> = (0..n_u).map(|_| Vec::with_capacity(n_v - 1)).collect();
        let mut new_w_per_row: Vec<Vec<f32>> = (0..n_u).map(|_| Vec::with_capacity(n_v - 1)).collect();

        for i_u in 0..n_u {
            let mut row_cps = Vec::with_capacity(n_v - 1);
            let mut row_ws = Vec::with_capacity(n_v - 1);
            for j in 0..n_v {
                if j == 0 {
                    row_cps.push(self.control_points[i_u][0]);
                    row_ws.push(self.weights[i_u][0]);
                } else if j == n_v - 1 {
                    row_cps.push(self.control_points[i_u][n_v - 1]);
                    row_ws.push(self.weights[i_u][n_v - 1]);
                } else {
                    let alpha = j as f32 / self.degree_v as f32;
                    let one_minus_a = 1.0 - alpha;
                    let prev_cp = *row_cps.last().unwrap();
                    let prev_w = *row_ws.last().unwrap();
                    let cp = if one_minus_a.abs() > 1e-10 {
                        (self.control_points[i_u][j] - prev_cp * alpha) / one_minus_a
                    } else { self.control_points[i_u][j] };
                    let w = if one_minus_a.abs() > 1e-10 {
                        (self.weights[i_u][j] - prev_w * alpha) / one_minus_a
                    } else { self.weights[i_u][j] };
                    row_cps.push(cp); row_ws.push(w);
                }
            }
            new_cp_per_row[i_u] = row_cps;
            new_w_per_row[i_u] = row_ws;
        }

        if !self.reduce_deviation_ok(&new_cp_per_row, &new_w_per_row, tolerance) { return false; }
        self.control_points = new_cp_per_row;
        self.weights = new_w_per_row;
        self.degree_v = new_deg;
        let n = self.v_count();
        self.knots_v = clamped_uniform_knots(new_deg, n);
        true
    }

    /// Check deviation of proposed CPs/weights by sampling and comparing.
    fn reduce_deviation_ok(&self, new_cps: &[Vec<Vec3>], new_ws: &[Vec<f32>], tol: f32) -> bool {
        for i in 0..=8 {
            for j in 0..=8 {
                let u = i as f32 / 8.0;
                let v = j as f32 / 8.0;
                let orig = self.evaluate(u, v);
                let new_val = self.evaluate_with_cps(new_cps, new_ws, u, v);
                if (new_val - orig).length() > tol { return false; }
            }
        }
        true
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

/// Build a clamped uniform knot vector for `n` control points of given `degree`.
/// E.g. degree=3, n=4 → [0,0,0,0, 1,1,1,1] (Bezier).
/// degree=3, n=6 → [0,0,0,0, 0.333, 0.667, 1,1,1,1].
fn clamped_uniform_knots(degree: usize, n: usize) -> Vec<f32> {
    let n_knots = n + degree + 1;
    let n_interior = n_knots.saturating_sub(2 * (degree + 1));
    let mut knots = Vec::with_capacity(n_knots);
    for _ in 0..=degree {
        knots.push(0.0);
    }
    for i in 1..=n_interior {
        knots.push(i as f32 / (n_interior + 1) as f32);
    }
    for _ in 0..=degree {
        knots.push(1.0);
    }
    knots
}

// ── Knot removal (inverse Boehm, OCC BSplCLib::RemoveKnot) ─────────

impl NurbsSurface {
    /// Try to remove one copy of knot `t` in the U direction.
    pub fn remove_knot_u(&mut self, t: f32, tolerance: f32) -> bool {
        let p = self.degree_u;
        let mult = knot_multiplicity(&self.knots_u, t);
        if mult == 0 { return false; }
        let is_end = (t - self.knots_u[0]).abs() < 1e-10
            || (t - self.knots_u[self.knots_u.len() - 1]).abs() < 1e-10;
        if is_end && mult <= p + 1 { return false; }

        let span = self.find_span_u(t);
        let start = span.saturating_sub(p + 1) + 1;

        let n_cp = self.u_count();
        let n_v = self.v_count();
        let mut new_cps: Vec<Vec<Vec3>> = Vec::with_capacity(n_cp - 1);
        let mut new_ws: Vec<Vec<f32>> = Vec::with_capacity(n_cp - 1);

        for i_u in 0..n_cp {
            let is_affected = i_u >= start && i_u < span;
            if !is_affected {
                new_cps.push(self.control_points[i_u].clone());
                new_ws.push(self.weights[i_u].clone());
                continue;
            }
            let alpha = {
                let denom = self.knots_u[i_u + p + 1] - self.knots_u[i_u];
                if denom.abs() > 1e-10 { ((t - self.knots_u[i_u]) / denom).clamp(0.0, 1.0) } else { 0.0 }
            };
            let one_minus_a = 1.0 - alpha;
            let prev_cp = new_cps.last().unwrap();
            let prev_w = new_ws.last().unwrap();
            let mut row_cp = Vec::with_capacity(n_v);
            let mut row_w = Vec::with_capacity(n_v);
            for j in 0..n_v {
                let cp_new = if alpha.abs() > 1e-10 {
                    (self.control_points[i_u][j] - prev_cp[j] * one_minus_a) / alpha
                } else { self.control_points[i_u][j] };
                let w_new = if alpha.abs() > 1e-10 {
                    (self.weights[i_u][j] - prev_w[j] * one_minus_a) / alpha
                } else { self.weights[i_u][j] };
                row_cp.push(cp_new); row_w.push(w_new);
            }
            new_cps.push(row_cp); new_ws.push(row_w);
        }

        // Deviation check at sample points
        if !self.knot_removal_deviation_ok(&new_cps, &new_ws, tolerance) { return false; }

        self.control_points = new_cps;
        self.weights = new_ws;
        let pos = self.knots_u.iter().position(|&k| (k - t).abs() < 1e-10).unwrap_or(span);
        self.knots_u.remove(pos);
        true
    }

    /// Try to remove one copy of knot `t` in the V direction.
    pub fn remove_knot_v(&mut self, t: f32, tolerance: f32) -> bool {
        let p = self.degree_v;
        let mult = knot_multiplicity(&self.knots_v, t);
        if mult == 0 { return false; }
        let is_end = (t - self.knots_v[0]).abs() < 1e-10
            || (t - self.knots_v[self.knots_v.len() - 1]).abs() < 1e-10;
        if is_end && mult <= p + 1 { return false; }

        let span = self.find_span_v(t);
        let start = span.saturating_sub(p + 1) + 1;
        let n_u = self.u_count();
        let n_cp = self.v_count();

        let mut new_cp_per_row: Vec<Vec<Vec3>> = (0..n_u).map(|_| Vec::with_capacity(n_cp - 1)).collect();
        let mut new_w_per_row: Vec<Vec<f32>> = (0..n_u).map(|_| Vec::with_capacity(n_cp - 1)).collect();

        for i_u in 0..n_u {
            for j_v in 0..n_cp {
                let is_affected = j_v >= start && j_v < span;
                if !is_affected {
                    new_cp_per_row[i_u].push(self.control_points[i_u][j_v]);
                    new_w_per_row[i_u].push(self.weights[i_u][j_v]);
                    continue;
                }
                let alpha = {
                    let denom = self.knots_v[j_v + p + 1] - self.knots_v[j_v];
                    if denom.abs() > 1e-10 { ((t - self.knots_v[j_v]) / denom).clamp(0.0, 1.0) } else { 0.0 }
                };
                let one_minus_a = 1.0 - alpha;
                let prev_cp = new_cp_per_row[i_u].last().copied().unwrap_or(Vec3::ZERO);
                let prev_w = new_w_per_row[i_u].last().copied().unwrap_or(1.0);
                let curr_cp = self.control_points[i_u][j_v];
                let curr_w = self.weights[i_u][j_v];
                let cp_new = if alpha.abs() > 1e-10 {
                    (curr_cp - prev_cp * one_minus_a) / alpha
                } else { curr_cp };
                let w_new = if alpha.abs() > 1e-10 {
                    (curr_w - prev_w * one_minus_a) / alpha
                } else { curr_w };
                new_cp_per_row[i_u].push(cp_new); new_w_per_row[i_u].push(w_new);
            }
        }

        if !self.knot_removal_deviation_ok(&new_cp_per_row, &new_w_per_row, tolerance) { return false; }

        self.control_points = new_cp_per_row;
        self.weights = new_w_per_row;
        let pos = self.knots_v.iter().position(|&k| (k - t).abs() < 1e-10).unwrap_or(span);
        self.knots_v.remove(pos);
        true
    }

    /// Check whether proposed new CPs/weights deviate within tolerance.
    fn knot_removal_deviation_ok(&self, new_cps: &[Vec<Vec3>], new_ws: &[Vec<f32>], tol: f32) -> bool {
        let samples = 8usize;
        for i in 0..=samples {
            for j in 0..=samples {
                let u = i as f32 / samples as f32;
                let v = j as f32 / samples as f32;
                let orig = self.evaluate_with_cps(&self.control_points, &self.weights, u, v);
                let new_val = self.evaluate_with_cps(new_cps, new_ws, u, v);
                if (new_val - orig).length() > tol { return false; }
            }
        }
        true
    }

    /// Evaluate surface with explicit control points and weights (for deviation checking).
    fn evaluate_with_cps(&self, cps: &[Vec<Vec3>], ws: &[Vec<f32>], u: f32, v: f32) -> Vec3 {
        let span_u = self.find_span_u(u);
        let span_v = self.find_span_v(v);
        let basis_u = self.basis_funs(span_u, u, self.degree_u, &self.knots_u);
        let basis_v = self.basis_funs(span_v, v, self.degree_v, &self.knots_v);

        let mut p = Vec3::ZERO;
        let mut w_sum = 0.0f32;
        for (iu, &nu) in basis_u.iter().enumerate() {
            let ci = span_u.saturating_sub(self.degree_u) + iu;
            if ci >= cps.len() { continue; }
            for (iv, &nv) in basis_v.iter().enumerate() {
                let cj = span_v.saturating_sub(self.degree_v) + iv;
                if cj >= ws[ci].len() { continue; }
                let w = ws[ci][cj] * nu * nv;
                p += cps[ci][cj] * w;
                w_sum += w;
            }
        }
        if w_sum.abs() > 1e-10 { p / w_sum } else { p }
    }

    /// Compute B-spline basis functions for given span and parameter.
    fn basis_funs(&self, span: usize, t: f32, degree: usize, knots: &[f32]) -> Vec<f32> {
        let mut n = vec![0.0f32; degree + 1];
        n[0] = 1.0;
        let mut left = vec![0.0f32; degree + 1];
        let mut right = vec![0.0f32; degree + 1];
        for j in 1..=degree {
            left[j] = t - knots[span + 1 - j];
            right[j] = knots[span + j] - t;
            let mut saved = 0.0f32;
            for r in 0..j {
                let temp = n[r] / (right[r + 1] + left[j - r] + 1e-12);
                n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n[j] = saved;
        }
        n
    }
}

fn knot_multiplicity(knots: &[f32], t: f32) -> usize {
    knots.iter().filter(|&&k| (k - t).abs() < 1e-10).count()
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
        let n_i = if i > span - degree {
            lookup_basis_value(&lower_bases, i)
        } else {
            0.0
        };

        // N_{i+1, p-1}(t): nonzero only if i+1 is in [span-p+1, span]
        let n_ip1 = if i < span {
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

/// Compute second-order B-spline basis function derivatives using the
/// analytical recurrence applied twice (Piegl & Tiller, The NURBS Book):
///
///   N''_{i,p}(t) = p/(k_{i+p}-k_i) * N'_{i,p-1}(t)
///                - p/(k_{i+p+1}-k_{i+1}) * N'_{i+1,p-1}(t)
///
/// where N'_{j,p-1}(t) is the first derivative of the degree p-1 basis,
/// computed via `analytical_basis_derivatives`.
///
/// Returns (index, second_derivative) pairs. For degree ≤ 1, all are zero.
fn analytical_basis_second_derivatives(span: usize, degree: usize, t: f32, knots: &[f32]) -> Vec<(usize, f32)> {
    if degree <= 1 {
        // Degree 0: constant basis, derivatives = 0
        // Degree 1: linear basis, second derivative = 0
        let i_start = span.saturating_sub(degree);
        return (i_start..=span).map(|i| (i, 0.0f32)).collect();
    }

    // First derivatives of degree (p-1) basis functions
    let d1_lower = analytical_basis_derivatives(span, degree - 1, t, knots);

    let p = degree as f32;
    let mut result = Vec::with_capacity(degree + 1);

    let i_start = span.saturating_sub(degree);
    let i_end = span;

    for i in i_start..=i_end {
        // N'_{i, p-1}(t): nonzero only if i in [span-(p-1), span]
        let d1_i = if i >= span.saturating_sub(degree - 1) {
            lookup_basis_value(&d1_lower, i)
        } else {
            0.0
        };

        // N'_{i+1, p-1}(t): nonzero only if i+1 in [span-(p-1), span]
        let d1_ip1 = if i < span {
            lookup_basis_value(&d1_lower, i + 1)
        } else {
            0.0
        };

        let denom1 = knots[i + degree] - knots[i];
        let denom2 = knots[i + degree + 1] - knots[i + 1];

        let term1 = if denom1.abs() > 1e-12 { d1_i / denom1 } else { 0.0 };
        let term2 = if denom2.abs() > 1e-12 { d1_ip1 / denom2 } else { 0.0 };

        let d2 = p * (term1 - term2);
        if d2.abs() > 1e-12 {
            result.push((i, d2));
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

    #[test]
    fn test_insert_knot_u_shape_invariance() {
        let mut surf = NurbsSurface::cylinder(1.0, 0.0, 2.0);
        let test_pts: Vec<(f32, f32)> = vec![
            (0.0, 0.0), (0.25, 0.5), (0.5, 0.3), (0.75, 0.8), (1.0, 1.0),
        ];
        let before: Vec<Vec3> = test_pts.iter().map(|&(u, v)| surf.evaluate(u, v)).collect();
        surf.insert_knot_u(0.5);
        for (i, &(u, v)) in test_pts.iter().enumerate() {
            let after = surf.evaluate(u, v);
            assert!((after - before[i]).length() < 1e-5,
                "shape changed at ({}, {}): {:?} vs {:?}", u, v, before[i], after);
        }
    }

    #[test]
    fn test_insert_knot_v_shape_invariance() {
        let mut surf = NurbsSurface::cylinder(1.0, 0.0, 2.0);
        let test_pts: Vec<(f32, f32)> = vec![
            (0.0, 0.0), (0.25, 0.5), (0.5, 0.3), (0.75, 0.8), (1.0, 1.0),
        ];
        let before: Vec<Vec3> = test_pts.iter().map(|&(u, v)| surf.evaluate(u, v)).collect();
        surf.insert_knot_v(0.5);
        for (i, &(u, v)) in test_pts.iter().enumerate() {
            let after = surf.evaluate(u, v);
            assert!((after - before[i]).length() < 1e-5,
                "shape changed at ({}, {}): {:?} vs {:?}", u, v, before[i], after);
        }
    }

    #[test]
    fn test_insert_knot_u_increases_control_points() {
        let mut surf = NurbsSurface::cylinder(1.0, 0.0, 2.0);
        let before = surf.u_count();
        surf.insert_knot_u(0.5);
        assert_eq!(surf.u_count(), before + 1);
    }

    #[test]
    fn test_insert_knot_v_increases_control_points() {
        let mut surf = NurbsSurface::cylinder(1.0, 0.0, 2.0);
        let before = surf.v_count();
        surf.insert_knot_v(0.5);
        assert_eq!(surf.v_count(), before + 1);
    }

    #[test]
    fn test_insert_knot_multiple() {
        // Use a non-rational plane for tighter numerical tolerance
        let mut surf = NurbsSurface::plane(-1.0, 1.0, -1.0, 1.0);
        let test_pts: Vec<(f32, f32)> = vec![
            (0.0, 0.0), (0.3, 0.4), (0.6, 0.7), (1.0, 1.0),
        ];
        let before: Vec<Vec3> = test_pts.iter().map(|&(u, v)| surf.evaluate(u, v)).collect();
        let orig_u = surf.u_count();

        surf.insert_knot_u(0.3);
        surf.insert_knot_u(0.6);
        surf.insert_knot_u(0.8);

        assert_eq!(surf.u_count(), orig_u + 3);
        for (i, &(u, v)) in test_pts.iter().enumerate() {
            let after = surf.evaluate(u, v);
            assert!((after - before[i]).length() < 1e-6,
                "shape changed at ({}, {}): {:?} vs {:?}", u, v, before[i], after);
        }
    }

    #[test]
    fn test_insert_knot_at_existing() {
        // Cylinder knots_u: [0,0,0, 0.25,0.25, 0.5,0.5, 0.75,0.75, 1.0,1.0,1.0]
        // t=0.25 has multiplicity 2, degree_u=2, so mult < degree+1 → can insert
        let mut surf = NurbsSurface::cylinder(1.0, 0.0, 2.0);
        let before_count = surf.u_count();
        surf.insert_knot_u(0.25);
        assert_eq!(surf.u_count(), before_count + 1,
            "inserting at existing knot with mult < degree+1 should increase multiplicity");
    }

    #[test]
    fn test_elevate_degree_u_shape_invariance() {
        let grid: Vec<Vec<Vec3>> = (0..4)
            .map(|i| (0..4).map(|j| Vec3::new(i as f32, j as f32, (i*j) as f32 * 0.3)).collect())
            .collect();
        let mut surface = NurbsSurface::from_points_grid(&grid, 3, 3);
        let mut before = Vec::new();
        for iu in 0..=5 {
            for iv in 0..=5 {
                before.push(surface.evaluate(iu as f32 / 5.0, iv as f32 / 5.0));
            }
        }
        surface.elevate_degree_u();
        assert_eq!(surface.degree_u, 4);
        assert_eq!(surface.degree_v, 3);
        for iu in 0..=5 {
            for iv in 0..=5 {
                let p = surface.evaluate(iu as f32 / 5.0, iv as f32 / 5.0);
                let diff = (before[iu * 6 + iv] - p).length();
                assert!(diff < 1e-3, "shape changed at iu={iu}, iv={iv}: diff={diff}");
            }
        }
    }

    #[test]
    fn test_elevate_degree_v_shape_invariance() {
        let grid: Vec<Vec<Vec3>> = (0..4)
            .map(|i| (0..4).map(|j| Vec3::new(i as f32, j as f32, (i*j) as f32 * 0.3)).collect())
            .collect();
        let mut surface = NurbsSurface::from_points_grid(&grid, 3, 3);
        let mut before = Vec::new();
        for iu in 0..=5 {
            let mut row = Vec::new();
            for iv in 0..=5 {
                row.push(surface.evaluate(iu as f32 / 5.0, iv as f32 / 5.0));
            }
            before.push(row);
        }
        surface.elevate_degree_v();
        assert_eq!(surface.degree_v, 4);
        assert_eq!(surface.degree_u, 3);
        for iu in 0..=5 {
            for iv in 0..=5 {
                let p = surface.evaluate(iu as f32 / 5.0, iv as f32 / 5.0);
                let diff = (before[iu][iv] - p).length();
                assert!(diff < 1e-3, "shape changed at iu={iu}, iv={iv}: diff={diff}");
            }
        }
    }

    #[test]
    fn test_elevate_degree_both_directions() {
        let grid: Vec<Vec<Vec3>> = (0..3)
            .map(|i| (0..3).map(|j| Vec3::new(i as f32, j as f32, 0.0)).collect())
            .collect();
        let mut surface = NurbsSurface::from_points_grid(&grid, 2, 2);
        let p_center = surface.evaluate(0.5, 0.5);
        surface.elevate_degree_u();
        surface.elevate_degree_v();
        assert_eq!(surface.degree_u, 3);
        assert_eq!(surface.degree_v, 3);
        let p_after = surface.evaluate(0.5, 0.5);
        assert!((p_center - p_after).length() < 1e-3);
    }

    #[test]
    fn test_nurbs_type_roundtrip() {
        let surf = NurbsSurface::cylinder(2.0, 0.0, 3.0);
        let test_pts: Vec<(f32, f32)> = vec![
            (0.0, 0.0), (0.25, 0.5), (0.5, 0.3), (0.75, 0.8), (1.0, 1.0),
        ];
        let before: Vec<Vec3> = test_pts.iter().map(|&(u, v)| surf.evaluate(u, v)).collect();

        // Convert to render surface and back
        let render: rc3d_nurbs::NurbsRenderSurface = surf.into();
        let roundtripped: NurbsSurface = render.into();

        assert_eq!(roundtripped.degree_u, 2);
        assert_eq!(roundtripped.degree_v, 1);
        assert_eq!(roundtripped.u_count(), 9);
        assert_eq!(roundtripped.v_count(), 2);

        // Shape must be preserved
        for (i, &(u, v)) in test_pts.iter().enumerate() {
            let after = roundtripped.evaluate(u, v);
            let dist = (after - before[i]).length();
            assert!(dist < 1e-5, "shape changed at ({u},{v}): {dist}");
        }
    }
}
