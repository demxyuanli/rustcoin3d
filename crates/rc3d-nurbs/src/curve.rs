use glam::Vec3;
use crate::basis::bspline_basis;
use crate::knot::{find_span, open_uniform_knots};

/// A NURBS curve defined by homogeneous control points, knot vector, and degree.
///
/// Control points are stored as (x, y, z, w) homogeneous coordinates.
#[derive(Clone, Debug)]
pub struct NurbsCurve {
    pub control_points: Vec<[f32; 4]>, // (wx, wy, wz, w) in homogeneous space
    pub knots: Vec<f32>,
    pub degree: usize,
}

impl NurbsCurve {
    /// Create a NURBS curve from homogeneous control points.
    /// Builds a clamped knot vector automatically.
    pub fn new(control_points: Vec<[f32; 4]>, degree: usize) -> Self {
        let knots = open_uniform_knots(degree, control_points.len());
        Self { control_points, knots, degree }
    }

    /// Create a NURBS curve from non-rational (w=1) control points.
    pub fn from_points(points: &[Vec3], degree: usize) -> Self {
        let cp: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 1.0]).collect();
        Self::new(cp, degree)
    }

    /// Number of control points.
    pub fn n_control_points(&self) -> usize {
        self.control_points.len()
    }

    /// Evaluate the curve at parameter t, returning a 3D point.
    pub fn evaluate(&self, t: f32) -> Vec3 {
        let span = find_span(self.degree, &self.knots, t);
        let mut point = Vec3::ZERO;
        let mut w_sum = 0.0;

        for i in 0..=self.degree {
            let idx = span.saturating_sub(self.degree) + i;
            if idx >= self.control_points.len() {
                continue;
            }
            let basis = bspline_basis(idx, self.degree, t, &self.knots);
            let cp = &self.control_points[idx];
            let w = cp[3] * basis;
            point += Vec3::new(cp[0], cp[1], cp[2]) * w;
            w_sum += w;
        }

        if w_sum.abs() > 1e-10 {
            point / w_sum
        } else {
            point
        }
    }

    /// Evaluate the unit tangent vector at parameter t (finite difference approximation).
    pub fn tangent(&self, t: f32) -> Vec3 {
        let eps = 1e-4;
        let p0 = self.evaluate((t - eps).max(0.0));
        let p1 = self.evaluate((t + eps).min(1.0));
        let d = p1 - p0;
        if d.length() > 1e-10 {
            d.normalize()
        } else {
            Vec3::X
        }
    }

    /// Approximate arc length using n_samples.
    pub fn arc_length(&self, n_samples: usize) -> f32 {
        let mut len = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=n_samples {
            let t = i as f32 / n_samples as f32;
            let pt = self.evaluate(t);
            len += (pt - prev).length();
            prev = pt;
        }
        len
    }

    /// Adaptive tessellation: sample at curvature-dependent intervals.
    /// Uses chord-height deviation + tangent angle deflection to decide subdivision.
    pub fn tessellate(&self, tolerance: f32) -> Vec<Vec3> {
        // Default angle tolerance: ~2 degrees in radians
        self.tessellate_adaptive(tolerance, 0.035)
    }

    /// Adaptive tessellation with explicit angle tolerance (radians).
    /// Smaller angle_tol = more samples in high-curvature regions.
    pub fn tessellate_adaptive(&self, tolerance: f32, angle_tol: f32) -> Vec<Vec3> {
        let mut points = vec![self.evaluate(0.0)];
        self.tessellate_recursive(0.0, 1.0, tolerance, angle_tol, &mut points);
        points
    }

    fn tessellate_recursive(&self, t0: f32, t1: f32, tol: f32, angle_tol: f32, out: &mut Vec<Vec3>) {
        let tm = (t0 + t1) * 0.5;
        let p0 = self.evaluate(t0);
        let p1 = self.evaluate(t1);
        let pm = self.evaluate(tm);

        let chord = p1 - p0;
        let chord_len_sq = chord.length_squared();
        if chord_len_sq < 1e-12 {
            out.push(p1);
            return;
        }

        // Chord-height deviation check
        let t_proj = (pm - p0).dot(chord) / chord_len_sq;
        let proj = p0 + chord * t_proj.clamp(0.0, 1.0);
        let dist = (pm - proj).length();

        // Tangent angle deflection check
        let tan0 = self.tangent(t0);
        let tan1 = self.tangent(t1);
        let dot = (tan0.dot(tan1)).clamp(-1.0, 1.0);
        let angle = dot.acos();

        let needs_subdiv = (dist > tol || angle > angle_tol) && (t1 - t0) > 1e-5;

        if needs_subdiv {
            self.tessellate_recursive(t0, tm, tol, angle_tol, out);
            self.tessellate_recursive(tm, t1, tol, angle_tol, out);
        } else {
            out.push(p1);
        }
    }

    /// Insert a knot value into the knot vector (Boehm's algorithm).
    /// Refines the control polygon without changing the curve shape.
    pub fn insert_knot(&mut self, t: f32) {
        let span = find_span(self.degree, &self.knots, t);
        let p = self.degree;

        let mut new_cp = Vec::with_capacity(self.control_points.len() + 1);

        // Control points before the insertion span are unchanged
        for i in 0..=span.saturating_sub(p) {
            new_cp.push(self.control_points[i]);
        }

        // Compute new control points for the affected span
        for i in (span.saturating_sub(p) + 1)..=span {
            if i >= self.control_points.len() {
                break;
            }
            let alpha = if i + p < self.knots.len() {
                let denom = self.knots[i + p] - self.knots[i];
                if denom.abs() > 1e-10 {
                    (t - self.knots[i]) / denom
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let prev = self.control_points[i - 1];
            let curr = self.control_points[i];
            let alpha = alpha.clamp(0.0, 1.0);
            let new = [
                alpha * curr[0] + (1.0 - alpha) * prev[0],
                alpha * curr[1] + (1.0 - alpha) * prev[1],
                alpha * curr[2] + (1.0 - alpha) * prev[2],
                alpha * curr[3] + (1.0 - alpha) * prev[3],
            ];
            new_cp.push(new);
        }

        // Remaining control points
        for i in span + 1..self.control_points.len() {
            new_cp.push(self.control_points[i]);
        }

        // Insert the knot
        let mut new_knots = self.knots.clone();
        new_knots.insert(span + 1, t);

        self.control_points = new_cp;
        self.knots = new_knots;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_approximation() {
        // NURBS 180-degree arc via 3 quadratic rational control points
        let w = 2f32.sqrt() / 2.0;
        let cp = vec![
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, w],
            [0.0, 1.0, 0.0, 1.0],
        ];
        let curve = NurbsCurve::new(cp, 2);
        let p = curve.evaluate(0.5);
        // Midpoint of 90-degree arc should be near (cos45°, sin45°)
        let expected = 2f32.sqrt() / 2.0;
        assert!((p.x - expected).abs() < 0.05, "x={}", p.x);
        assert!((p.y - expected).abs() < 0.05, "y={}", p.y);
    }

    #[test]
    fn test_knot_insertion_invariance() {
        let cp = vec![
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];
        let mut curve = NurbsCurve::new(cp, 3); // degree 3 with 4 CPs = cubic Bezier

        let pt_before = curve.evaluate(0.5);
        curve.insert_knot(0.5);
        let pt_after = curve.evaluate(0.5);

        assert!(
            (pt_before - pt_after).length() < 1e-4,
            "knot insertion changed curve: {:?} vs {:?}",
            pt_before,
            pt_after
        );
    }

    #[test]
    fn test_tangent_normalized() {
        let curve = NurbsCurve::from_points(
            &[Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y],
            3,
        );
        let t = curve.tangent(0.5);
        assert!((t.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_tessellate_count() {
        let curve = NurbsCurve::from_points(
            &[Vec3::ZERO, Vec3::X * 0.5, Vec3::X, Vec3::X * 1.5],
            3,
        );
        let pts = curve.tessellate(0.1);
        assert!(pts.len() >= 2);
        assert!(pts.len() <= 20); // straight line shouldn't explode
    }
}
