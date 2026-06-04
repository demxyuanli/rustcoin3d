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

        // Remaining control points (shifted by 1: Q_{i+1} = P_i)
        for i in span..self.control_points.len() {
            new_cp.push(self.control_points[i]);
        }

        // Insert the knot
        let mut new_knots = self.knots.clone();
        new_knots.insert(span + 1, t);

        self.control_points = new_cp;
        self.knots = new_knots;
    }

    /// Elevate the curve degree by 1 using Bezier decomposition.
    ///
    /// Algorithm:
    /// 1. Decompose into Bezier segments by inserting interior knots to full multiplicity.
    /// 2. Degree-elevate each Bezier segment independently.
    /// 3. Reassemble into a single NURBS curve with degree+1.
    pub fn elevate_degree(&self) -> Self {
        let p = self.degree;
        let new_degree = p + 1;
        let mut curve = self.clone();

        // Step 1: Bezier decomposition — insert interior knots to reach full multiplicity (p).
        let unique = crate::knot::unique_knots(&curve.knots);
        for (idx, &(val, mult)) in unique.iter().enumerate() {
            if idx == 0 || idx == unique.len() - 1 {
                continue; // skip end knots
            }
            let needed = p.saturating_sub(mult);
            for _ in 0..needed {
                curve.insert_knot(val);
            }
        }

        // Step 2: Count Bezier segments.
        // After decomposition: unique interior knots + 1 segments.
        // Each segment has p+1 control points; adjacent segments share boundary CPs.
        let unique_after = crate::knot::unique_knots(&curve.knots);
        let n_segments = unique_after.len() - 1; // gaps between unique knots
        let cps = &curve.control_points;

        // Step 3: Degree-elevate each Bezier segment.
        let mut new_cp = Vec::new();
        for seg in 0..n_segments {
            let base = seg * p; // first CP index of this segment
            // Q[0] = P[0] — skip for seg>0 (shared with prev segment's last CP)
            if seg == 0 {
                new_cp.push(cps[base]);
            }
            // Q[i] = α * P[i-1] + (1-α) * P[i], α = i/(p+1), for i=1..p
            for i in 1..=p {
                let alpha = i as f32 / new_degree as f32;
                let prev = cps[base + i - 1];
                let curr = cps[base + i];
                new_cp.push([
                    alpha * prev[0] + (1.0 - alpha) * curr[0],
                    alpha * prev[1] + (1.0 - alpha) * curr[1],
                    alpha * prev[2] + (1.0 - alpha) * curr[2],
                    alpha * prev[3] + (1.0 - alpha) * curr[3],
                ]);
            }
            // Q[p+1] = P[p] — last CP of segment
            new_cp.push(cps[base + p]);
        }

        // Step 4: Build new knot vector.
        let mut new_knots = Vec::new();
        for (idx, &(val, mult)) in unique_after.iter().enumerate() {
            let new_mult = if idx == 0 || idx == unique_after.len() - 1 {
                new_degree + 1 // end knots: full multiplicity for new degree
            } else {
                mult + 1 // interior knots: original multiplicity + 1
            };
            for _ in 0..new_mult {
                new_knots.push(val);
            }
        }

        NurbsCurve {
            control_points: new_cp,
            knots: new_knots,
            degree: new_degree,
        }
    }

    /// Try to remove one copy of knot `t`. Returns `true` if the resulting curve
    /// matches the original within `tolerance` at sample points.
    ///
    /// Uses the inverse of the Boehm insertion formula. Works exactly (within
    /// floating-point) for knots that were previously inserted.
    pub fn remove_knot(&mut self, t: f32, tolerance: f32) -> bool {
        let p = self.degree;
        let mult = crate::knot::knot_multiplicity(&self.knots, t);
        if mult == 0 {
            return false;
        }
        // Guard: don't remove end knots (they must have multiplicity p+1)
        let is_end = (t - self.knots[0]).abs() < 1e-10
            || (t - self.knots[self.knots.len() - 1]).abs() < 1e-10;
        if is_end && mult <= p + 1 {
            return false;
        }

        let span = crate::knot::find_span(p, &self.knots, t);
        // For an interior knot, span is the index of the last copy (find_span
        // returns i such that knots[i] <= t < knots[i+1]).
        let start = span.saturating_sub(p);

        // Reverse-Boehm: compute what CPs were before insertion
        let mut new_cp = Vec::with_capacity(self.control_points.len() - 1);
        // Control points before the affected range stay unchanged
        for i in 0..start {
            new_cp.push(self.control_points[i]);
        }
        // For the affected range, solve: new_cp[i] = alpha * orig[i] + (1-alpha) * orig[i-1]
        // => orig[i] = (new_cp[i] - (1-alpha) * orig[i-1]) / alpha
        // But we already have orig[i-1] from the previous iteration.
        for i in start..span {
            if i >= self.control_points.len() || i.saturating_sub(1) >= self.control_points.len() {
                return false;
            }
            // The original insertion alpha used OLD knot vector (before t was inserted).
            // Now that t is in the new knot vector, all indices >= insertion_point
            // are shifted by +1. Since i+p >= start+p = r+1 > r, we use i+p+1.
            let alpha = {
                let denom = self.knots[i + p + 1] - self.knots[i];
                if denom.abs() > 1e-10 {
                    ((t - self.knots[i]) / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            };
            let prev = if i == start {
                // First affected point: the original before CP is from new_cp
                // Actually orig[start-1] = new_cp[start-1] (unchanged), which is
                // already pushed into new_cp above.
                self.control_points[start]
            } else {
                self.control_points[i]
            };
            // Wait — we need to solve for the ORIGINAL control point.
            // Let me reconsider...

            // Boehm insertion:
            //   Q_i = alpha * P_i + (1-alpha) * P_{i-1}    for i = start+1 .. span
            //   (with Q being the new CPs after insertion)
            // To reverse: given Q_i and P_{i-1} (already computed), find P_i:
            //   P_i = (Q_i - (1-alpha) * P_{i-1}) / alpha

            // P_i maps to position i in the output (no shift yet)
            // We need P_{i-1} which we just pushed to new_cp
            let prev_cp = new_cp.last().copied().unwrap_or([0.0; 4]);
            let curr = self.control_points[i];
            let alpha_clamped = alpha.clamp(1e-10, 1.0);
            let new_pt: [f32; 4] = [
                (curr[0] - (1.0 - alpha_clamped) * prev_cp[0]) / alpha_clamped,
                (curr[1] - (1.0 - alpha_clamped) * prev_cp[1]) / alpha_clamped,
                (curr[2] - (1.0 - alpha_clamped) * prev_cp[2]) / alpha_clamped,
                (curr[3] - (1.0 - alpha_clamped) * prev_cp[3]) / alpha_clamped,
            ];
            new_cp.push(new_pt);
        }
        // Remaining: skip first point after blends (it equals the last blend result).
        // In Boehm: Q_{span+1} = P_{span}, and P_{span} was already recovered.
        for i in (span + 1)..self.control_points.len() {
            new_cp.push(self.control_points[i]);
        }

        // Remove one copy of t from knots
        let mut new_knots = Vec::with_capacity(self.knots.len() - 1);
        let mut removed = false;
        for &k in &self.knots {
            if !removed && (k - t).abs() < 1e-10 {
                removed = true;
                continue;
            }
            new_knots.push(k);
        }

        let candidate = Self { control_points: new_cp, knots: new_knots, degree: p };

        // Verify shape preservation
        for i in 0..=10 {
            let ti = i as f32 / 10.0;
            // Skip evaluation at the removed knot if it would cause issues
            let td = if (ti - t).abs() < 1e-6 { (ti + 0.01).min(1.0) } else { ti };
            if (self.evaluate(td) - candidate.evaluate(td)).length() > tolerance {
                return false;
            }
        }

        *self = candidate;
        true
    }

    /// Try to reduce the curve degree from p to p-1 while staying within tolerance.
    ///
    /// Returns the reduced curve if successful, or `None` if the approximation
    /// error exceeds tolerance. Uses Bezier decomposition → segment reduction
    /// → reassembly.
    pub fn reduce_degree(&self, tolerance: f32) -> Option<Self> {
        let p = self.degree;
        if p <= 1 {
            return None; // can't reduce linear
        }
        let new_p = p - 1;

        // Step 1: Bezier decomposition
        let mut bez = self.clone();
        let uq = crate::knot::unique_knots(&self.knots);
        let knot_first = uq.first().map(|(v, _)| *v).unwrap_or(0.0);
        let knot_last = uq.last().map(|(v, _)| *v).unwrap_or(1.0);
        for &(val, mult) in &uq {
            if (val - knot_first).abs() < 1e-10 || (val - knot_last).abs() < 1e-10 {
                continue;
            }
            let needed = p.saturating_sub(mult);
            for _ in 0..needed {
                bez.insert_knot(val);
            }
        }

        let n_interior = uq.iter().filter(|(v, _)| {
            (v - knot_first).abs() > 1e-10 && (v - knot_last).abs() > 1e-10
        }).count();
        let n_segments = n_interior + 1;

        // Step 2: Reduce each Bezier segment
        let mut new_cp: Vec<[f32; 4]> = Vec::with_capacity(n_segments * p);
        for seg in 0..n_segments {
            let offset = seg * p;
            // Forward pass
            let mut fwd = vec![[0.0f32; 4]; p];
            fwd[0] = bez.control_points[offset];
            for i in 1..=(p - 1) {
                let prev = fwd[i - 1];
                let curr = bez.control_points[offset + i];
                let denom = (p - i) as f32;
                if denom.abs() < 1e-10 {
                    return None;
                }
                for d in 0..4 {
                    fwd[i][d] = (p as f32 * curr[d] - i as f32 * prev[d]) / denom;
                }
            }
            // Backward pass
            let mut bwd = vec![[0.0f32; 4]; p];
            bwd[p - 1] = bez.control_points[offset + p];
            for i in (1..=(p - 1)).rev() {
                let next = bwd[i];
                let curr = bez.control_points[offset + i];
                let denom = i as f32;
                if denom.abs() < 1e-10 {
                    return None;
                }
                for d in 0..4 {
                    bwd[i - 1][d] = (p as f32 * curr[d] - (p - i) as f32 * next[d]) / denom;
                }
            }
            // Average forward + backward for interior points
            if seg == 0 {
                new_cp.push(bez.control_points[offset]); // first point
            }
            for i in 1..=(p - 1) {
                let avg: [f32; 4] = [
                    (fwd[i][0] + bwd[i][0]) * 0.5,
                    (fwd[i][1] + bwd[i][1]) * 0.5,
                    (fwd[i][2] + bwd[i][2]) * 0.5,
                    (fwd[i][3] + bwd[i][3]) * 0.5,
                ];
                new_cp.push(avg);
            }
            new_cp.push(bez.control_points[offset + p]); // last point of segment
        }

        // Step 3: Build new knot vector (end multiplicities = new_p+1, interior = 1)
        let mut new_knots: Vec<f32> = Vec::new();
        for &(val, _) in &uq {
            let is_end = (val - knot_first).abs() < 1e-10
                || (val - knot_last).abs() < 1e-10;
            let mult = if is_end { new_p + 1 } else { 1 };
            for _ in 0..mult {
                new_knots.push(val);
            }
        }

        let result = Self {
            control_points: new_cp,
            knots: new_knots,
            degree: new_p,
        };

        // Step 4: Verify within tolerance
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let diff = (self.evaluate(t) - result.evaluate(t)).length();
            if diff > tolerance {
                return None;
            }
        }

        Some(result)
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

    #[test]
    fn test_elevate_degree_line_invariance() {
        let curve = NurbsCurve::from_points(&[Vec3::ZERO, Vec3::X], 1);
        let pt_before = curve.evaluate(0.5);
        let elevated = curve.elevate_degree();
        let pt_after = elevated.evaluate(0.5);
        assert_eq!(elevated.degree, 2);
        assert!((pt_before - pt_after).length() < 1e-4,
            "shape changed: {:?} vs {:?}", pt_before, pt_after);
    }

    #[test]
    fn test_elevate_degree_cubic_bezier() {
        let cp = vec![
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 2.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
        ];
        let curve = NurbsCurve::new(cp, 3);
        let before: Vec<Vec3> = (0..=10).map(|i| curve.evaluate(i as f32 / 10.0)).collect();
        let elevated = curve.elevate_degree();
        assert_eq!(elevated.degree, 4);
        for i in 0..=10 {
            let p = elevated.evaluate(i as f32 / 10.0);
            assert!((before[i] - p).length() < 1e-3, "sample {i} mismatch");
        }
    }

    #[test]
    fn test_elevate_degree_multi_segment() {
        let cp = vec![
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, -1.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
            [4.0, 1.0, 0.0, 1.0],
        ];
        let mut knots = vec![0.0f32; 4];
        knots.push(0.5);
        knots.extend_from_slice(&[1.0f32; 4]);
        let curve = NurbsCurve { control_points: cp, knots, degree: 3 };
        let before: Vec<Vec3> = (0..=20).map(|i| curve.evaluate(i as f32 / 20.0)).collect();
        let elevated = curve.elevate_degree();
        assert_eq!(elevated.degree, 4);
        for i in 0..=20 {
            let p = elevated.evaluate(i as f32 / 20.0);
            assert!((before[i] - p).length() < 1e-3, "sample {i} mismatch at t={}", i as f32 / 20.0);
        }
    }

    #[test]
    fn test_remove_knot_cubic() {
        // Insert a knot, then remove it — should be exact (0 tolerance needed)
        let cp = vec![
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, -1.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
        ];
        let mut curve = NurbsCurve::new(cp, 3);
        let pt_before = curve.evaluate(0.5);
        curve.insert_knot(0.3);
        assert_eq!(curve.n_control_points(), 5);
        let ok = curve.remove_knot(0.3, 0.01);
        assert!(ok, "knot removal should succeed");
        assert_eq!(curve.n_control_points(), 4);
        let pt_after = curve.evaluate(0.5);
        assert!((pt_before - pt_after).length() < 1e-4, "shape changed after remove");
    }

    #[test]
    fn test_remove_knot_nonexistent() {
        let cp = vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
        let mut curve = NurbsCurve::new(cp, 1);
        assert!(!curve.remove_knot(0.5, 1e-6));
    }

    #[test]
    fn test_reduce_degree_cubic_to_quad() {
        // Almost-parabolic cubic should reduce to quadratic
        let cp = vec![
            [0.0, 0.0, 0.0, 1.0],
            [0.33, 0.5, 0.0, 1.0],
            [0.66, 0.5, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ];
        let curve = NurbsCurve::new(cp, 3);
        let reduced = curve.reduce_degree(0.1);
        assert!(reduced.is_some(), "near-parabolic cubic should reduce");
        let r = reduced.unwrap();
        assert_eq!(r.degree, 2);
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let diff = (curve.evaluate(t) - r.evaluate(t)).length();
            assert!(diff < 0.1, "deviation too large at t={t}: {diff}");
        }
    }

    #[test]
    fn test_reduce_degree_linear_is_none() {
        let curve = NurbsCurve::from_points(&[Vec3::ZERO, Vec3::X], 1);
        assert!(curve.reduce_degree(0.1).is_none());
    }
}
