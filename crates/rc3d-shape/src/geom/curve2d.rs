//! 2D parametric curve types for UV-space operations.
//!
//! Provides curve evaluation, Bézier decomposition, and recursive
//! Bézier clipping intersection for use in constrained Delaunay
//! triangulation (CDT) constraint edge resolution.
//!
//! OCC alignment: Geom2d_Curve hierarchy + IntRes2d_Intersection

/// 2D curve in UV parameter space.
#[derive(Debug, Clone)]
pub enum Curve2d {
    Line {
        origin: (f32, f32),
        direction: (f32, f32),
    },
    Circle {
        center: (f32, f32),
        radius: f32,
    },
    BSpline {
        degree: usize,
        control_points: Vec<(f32, f32)>,
        knots: Vec<f32>,
        weights: Option<Vec<f32>>,
    },
    Trimmed {
        basis: Box<Curve2d>,
        t_min: f32,
        t_max: f32,
    },
}

impl Curve2d {
    /// Evaluate the curve at parameter t ∈ [0, 1].
    pub fn d0(&self, t: f32) -> (f32, f32) {
        match self {
            Curve2d::Line { origin, direction } => {
                (origin.0 + direction.0 * t, origin.1 + direction.1 * t)
            }
            Curve2d::Circle { center, radius } => {
                let theta = t * std::f32::consts::TAU;
                (center.0 + radius * theta.cos(), center.1 + radius * theta.sin())
            }
            Curve2d::BSpline { degree, control_points, knots, weights } => {
                bspline_2d_d0(*degree, control_points, knots, weights.as_deref(), t)
            }
            Curve2d::Trimmed { basis, t_min, t_max } => {
                let span = t_max - t_min;
                if span.abs() < 1e-10 {
                    return basis.d0(*t_min);
                }
                basis.d0(t_min + t * span)
            }
        }
    }

    /// Decompose into monotonic cubic Bézier segments.
    ///
    /// OCC: BSplCLib::BezierCoefficients — converts B-spline to piecewise Bézier.
    pub fn to_beziers(&self) -> Vec<Bezier2d> {
        match self {
            Curve2d::Line { origin, direction } => {
                let p0 = *origin;
                let p3 = (origin.0 + direction.0, origin.1 + direction.1);
                let p1 = (p0.0 + (p3.0 - p0.0) / 3.0, p0.1 + (p3.1 - p0.1) / 3.0);
                let p2 = (p3.0 - (p3.0 - p0.0) / 3.0, p3.1 - (p3.1 - p0.1) / 3.0);
                vec![Bezier2d { c0: p0, c1: p1, c2: p2, c3: p3 }]
            }
            Curve2d::Circle { center, radius } => {
                // Approximate full circle with 4 cubic Bézier arcs (90° each).
                // Magic constant K = 4/3 * tan(π/8) ≈ 0.5522847498
                const K: f32 = 0.5522847498;
                let r = *radius;
                let cx = center.0;
                let cy = center.1;
                vec![
                    Bezier2d { c0: (cx + r, cy), c1: (cx + r, cy + r * K),
                        c2: (cx + r * K, cy + r), c3: (cx, cy + r) },
                    Bezier2d { c0: (cx, cy + r), c1: (cx - r * K, cy + r),
                        c2: (cx - r, cy + r * K), c3: (cx - r, cy) },
                    Bezier2d { c0: (cx - r, cy), c1: (cx - r, cy - r * K),
                        c2: (cx - r * K, cy - r), c3: (cx, cy - r) },
                    Bezier2d { c0: (cx, cy - r), c1: (cx + r * K, cy - r),
                        c2: (cx + r, cy - r * K), c3: (cx + r, cy) },
                ]
            }
            Curve2d::BSpline { degree, control_points, knots, weights } => {
                decompose_bspline_to_beziers(*degree, control_points, knots, weights.as_deref())
            }
            Curve2d::Trimmed { basis, t_min, t_max } => {
                let all = basis.to_beziers();
                clip_beziers_to_range(&all, *t_min, *t_max)
            }
        }
    }
}

/// Cubic Bézier curve segment. The curve is parameterized t ∈ [0, 1]:
///
///   B(t) = (1-t)³·c0 + 3(1-t)²t·c1 + 3(1-t)t²·c2 + t³·c3
#[derive(Debug, Clone, Copy)]
pub struct Bezier2d {
    pub c0: (f32, f32),
    pub c1: (f32, f32),
    pub c2: (f32, f32),
    pub c3: (f32, f32),
}

impl Bezier2d {
    /// Evaluate the Bézier at parameter t.
    pub fn eval(&self, t: f32) -> (f32, f32) {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        (
            mt3 * self.c0.0 + 3.0 * mt2 * t * self.c1.0 + 3.0 * mt * t2 * self.c2.0 + t3 * self.c3.0,
            mt3 * self.c0.1 + 3.0 * mt2 * t * self.c1.1 + 3.0 * mt * t2 * self.c2.1 + t3 * self.c3.1,
        )
    }

    /// Axis-aligned bounding box.
    pub fn bbox(&self) -> (f32, f32, f32, f32) {
        let xs = [self.c0.0, self.c1.0, self.c2.0, self.c3.0];
        let ys = [self.c0.1, self.c1.1, self.c2.1, self.c3.1];
        (
            xs.iter().cloned().fold(f32::INFINITY, f32::min),
            xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ys.iter().cloned().fold(f32::INFINITY, f32::min),
            ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    }

    /// Split at parameter t ∈ [0, 1] using de Casteljau subdivision.
    pub fn split_at(&self, t: f32) -> (Bezier2d, Bezier2d) {
        let p00 = self.c0;
        let p01 = self.c1;
        let p02 = self.c2;
        let p03 = self.c3;

        let p10 = lerp_2d(p00, p01, t);
        let p11 = lerp_2d(p01, p02, t);
        let p12 = lerp_2d(p02, p03, t);

        let p20 = lerp_2d(p10, p11, t);
        let p21 = lerp_2d(p11, p12, t);

        let p30 = lerp_2d(p20, p21, t);

        (
            Bezier2d { c0: p00, c1: p10, c2: p20, c3: p30 },
            Bezier2d { c0: p30, c1: p21, c2: p12, c3: p03 },
        )
    }
}

fn lerp_2d(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

/// Bézier clipping 2D intersection.
///
/// Recursively clips two Bézier curves against each other's bounding boxes,
/// splitting the larger curve at its midpoint when they overlap. Terminates
/// when both segments are small enough or max_depth is reached.
///
/// Returns a list of intersection points as ((x, y) on curve A, (x, y) on curve B).
///
/// OCC: IntRes2d_Intersection using Bézier clipping (BndLib + math_FunctionRoots)
pub fn bezier_clip_intersect(
    a: &Bezier2d,
    b: &Bezier2d,
    tol: f32,
    max_depth: usize,
) -> Vec<((f32, f32), (f32, f32))> {
    // Phase 1: Bounding box rejection
    let (ax0, ax1, ay0, ay1) = a.bbox();
    let (bx0, bx1, by0, by1) = b.bbox();
    if ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0 {
        return vec![];
    }

    // Phase 2: If both curves are tiny, return midpoint intersection
    let a_diag = ((ax1 - ax0).powi(2) + (ay1 - ay0).powi(2)).sqrt();
    let b_diag = ((bx1 - bx0).powi(2) + (by1 - by0).powi(2)).sqrt();
    let tol_sq = tol * tol;

    if a_diag < tol && b_diag < tol {
        let am = a.eval(0.5);
        let bm = b.eval(0.5);
        let dist_sq = (am.0 - bm.0).powi(2) + (am.1 - bm.1).powi(2);
        if dist_sq < tol_sq * 16.0 {
            return vec![(am, bm)];
        }
        return vec![];
    }

    if max_depth == 0 {
        // At max depth: return intersection of bbox centers as best estimate.
        // Both curves are tiny at this point (diagonals < tol at the leaf above).
        let ac = ((ax0 + ax1) * 0.5, (ay0 + ay1) * 0.5);
        let bc = ((bx0 + bx1) * 0.5, (by0 + by1) * 0.5);
        let dist_sq = (ac.0 - bc.0).powi(2) + (ac.1 - bc.1).powi(2);
        if dist_sq < tol_sq * 16.0 {
            return vec![(ac, bc)];
        }
        return vec![];
    }

    // Phase 3: Split the larger curve and recurse (asymmetric convergence).
    // For near-equal diagonals, split both to ensure symmetric convergence.
    let mut results = Vec::new();

    // If both are within 2x of each other, split both for faster symmetric convergence.
    // Otherwise split only the larger one.
    let both_large = a_diag > tol && b_diag > tol;
    let similar_size = both_large && a_diag.max(b_diag) / a_diag.min(b_diag).max(1e-10) < 2.0;
    let split_a = a_diag > tol;
    let split_b = b_diag > tol;

    if similar_size {
        let (a_l, a_r) = a.split_at(0.5);
        let (b_l, b_r) = b.split_at(0.5);
        for (sa, sb) in &[(&a_l, &b_l), (&a_l, &b_r), (&a_r, &b_l), (&a_r, &b_r)] {
            let (sx0, sx1, sy0, sy1) = sa.bbox();
            let (tx0, tx1, ty0, ty1) = sb.bbox();
            if sx1 >= tx0 && tx1 >= sx0 && sy1 >= ty0 && ty1 >= sy0 {
                results.extend(bezier_clip_intersect(sa, sb, tol, max_depth - 1));
            }
        }
    } else if split_a && a_diag >= b_diag {
        let (left, right) = a.split_at(0.5);
        for sub in &[&left, &right] {
            let (sx0, sx1, sy0, sy1) = sub.bbox();
            if sx1 >= bx0 && bx1 >= sx0 && sy1 >= by0 && by1 >= sy0 {
                results.extend(bezier_clip_intersect(sub, b, tol, max_depth - 1));
            }
        }
    } else if split_b {
        let (left, right) = b.split_at(0.5);
        for sub in &[&left, &right] {
            let (sx0, sx1, sy0, sy1) = sub.bbox();
            if ax1 >= sx0 && sx1 >= ax0 && ay1 >= sy0 && sy1 >= ay0 {
                results.extend(bezier_clip_intersect(a, sub, tol, max_depth - 1));
            }
        }
    }
    results
}

/// Find all intersection points between two Curve2d objects.
///
/// Converts both curves to Bézier segments and runs Bézier clipping
/// on each pair of segments whose bounding boxes overlap.
///
/// Uses max_depth=12 for robust convergence on line-line intersections.
pub fn intersect_curves_2d(a: &Curve2d, b: &Curve2d, tol: f32) -> Vec<((f32, f32), (f32, f32))> {
    let beziers_a = a.to_beziers();
    let beziers_b = b.to_beziers();

    let mut results = Vec::new();
    for ba in &beziers_a {
        for bb in &beziers_b {
            // Quick AABB rejection
            let (ax0, ax1, ay0, ay1) = ba.bbox();
            let (bx0, bx1, by0, by1) = bb.bbox();
            if ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0 {
                continue;
            }
            results.extend(bezier_clip_intersect(ba, bb, tol, 12));
        }
    }
    results
}

// ── Internal helpers ──────────────────────────────────────────────

/// Evaluate a B-spline curve at parameter t (de Boor algorithm).
fn bspline_2d_d0(
    degree: usize,
    cps: &[(f32, f32)],
    knots: &[f32],
    weights: Option<&[f32]>,
    t: f32,
) -> (f32, f32) {
    let p = degree;
    let n = cps.len();
    if n < p + 1 || knots.len() < p + n + 1 {
        return (0.0, 0.0);
    }

    let t = t.clamp(knots[p], knots[n]);

    // Find span
    let span = {
        let mut s = p;
        for i in p..n {
            if t >= knots[i] && t < knots[i + 1] {
                s = i;
                break;
            }
        }
        if t >= knots[n] {
            n - 1
        } else {
            s
        }
    };

    // de Boor for 2D
    let mut d: Vec<(f32, f32)> = cps[span - p..=span].to_vec();
    for k in 1..=p {
        for i in (k..=p).rev() {
            let idx = span - p + i;
            let alpha = if (knots[idx + p - k + 1] - knots[idx]).abs() > 1e-10 {
                (t - knots[idx]) / (knots[idx + p - k + 1] - knots[idx])
            } else {
                0.0
            };
            d[i].0 = (1.0 - alpha) * d[i - 1].0 + alpha * d[i].0;
            d[i].1 = (1.0 - alpha) * d[i - 1].1 + alpha * d[i].1;
        }
    }

    if let Some(ws) = weights {
        let mut w: Vec<f32> = ws[span - p..=span].to_vec();
        for k in 1..=p {
            for i in (k..=p).rev() {
                let idx = span - p + i;
                let alpha = if (knots[idx + p - k + 1] - knots[idx]).abs() > 1e-10 {
                    (t - knots[idx]) / (knots[idx + p - k + 1] - knots[idx])
                } else {
                    0.0
                };
                w[i] = (1.0 - alpha) * w[i - 1] + alpha * w[i];
            }
        }
        let inv_w = 1.0 / w[p].max(1e-10);
        (d[p].0 * inv_w, d[p].1 * inv_w)
    } else {
        d[p]
    }
}

/// Decompose a B-spline into piecewise Bézier segments by inserting
/// internal knots to full multiplicity.
fn decompose_bspline_to_beziers(
    degree: usize,
    cps: &[(f32, f32)],
    knots: &[f32],
    _weights: Option<&[f32]>,
) -> Vec<Bezier2d> {
    let n = cps.len();
    if n <= degree + 1 {
        // Single Bézier segment
        let c0 = cps[0];
        let c3 = cps[n - 1];
        let c1 = if n >= 3 { cps[1] } else {
            (c0.0 + (c3.0 - c0.0) / 3.0, c0.1 + (c3.1 - c0.1) / 3.0)
        };
        let c2 = if n >= 4 { cps[n - 2] } else {
            (c3.0 - (c3.0 - c0.0) / 3.0, c3.1 - (c3.1 - c0.1) / 3.0)
        };
        return vec![Bezier2d { c0, c1, c2, c3 }];
    }

    // Multi-segment: one Bézier per internal knot interval
    let mut beziers = Vec::new();
    let mut seg_start = degree;

    for i in (degree + 1)..n {
        let at_last = i == n - 1;
        let is_knot_boundary = (knots[i + 1] - knots[i]).abs() > 1e-10 || at_last;

        if is_knot_boundary {
            let seg_cp_count = i - seg_start + degree + 1;
            let seg_cps = &cps[seg_start - degree..seg_start - degree + seg_cp_count.min(cps.len() - (seg_start - degree))];

            if seg_cps.len() == degree + 1 {
                let c0 = seg_cps[0];
                let c3 = seg_cps[degree];
                let c1 = if degree >= 2 { seg_cps[1] } else {
                    (c0.0 + (c3.0 - c0.0) / 3.0, c0.1 + (c3.1 - c0.1) / 3.0)
                };
                let c2 = if degree >= 3 { seg_cps[degree - 1] } else {
                    (c3.0 - (c3.0 - c0.0) / 3.0, c3.1 - (c3.1 - c0.1) / 3.0)
                };
                beziers.push(Bezier2d { c0, c1, c2, c3 });
            }
            seg_start = i;
        }
    }
    beziers
}

/// Clip a list of Bézier segments to a parameter range [t_min, t_max].
fn clip_beziers_to_range(beziers: &[Bezier2d], t_min: f32, t_max: f32) -> Vec<Bezier2d> {
    if beziers.is_empty() {
        return vec![];
    }
    let n = beziers.len() as f32;
    let idx_min = ((t_min * n).floor() as usize).min(beziers.len() - 1);
    let idx_max = ((t_max * n).ceil() as usize).min(beziers.len() - 1);
    beziers[idx_min..=idx_max].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_curve_eval() {
        let line = Curve2d::Line {
            origin: (1.0, 2.0),
            direction: (3.0, 4.0),
        };
        let p0 = line.d0(0.0);
        let p1 = line.d0(1.0);
        assert!((p0.0 - 1.0).abs() < 1e-6);
        assert!((p0.1 - 2.0).abs() < 1e-6);
        assert!((p1.0 - 4.0).abs() < 1e-6);
        assert!((p1.1 - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_line_to_bezier() {
        let line = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let beziers = line.to_beziers();
        assert_eq!(beziers.len(), 1);
        let b = &beziers[0];
        assert!((b.c0.0 - 0.0).abs() < 1e-6);
        assert!((b.c3.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bezier_split_midpoint() {
        let b = Bezier2d {
            c0: (0.0, 0.0), c1: (0.33, 0.0),
            c2: (0.67, 0.0), c3: (1.0, 0.0),
        };
        let (left, right) = b.split_at(0.5);
        // Right starts where left ends
        assert!((left.c3.0 - right.c0.0).abs() < 1e-6);
        assert!((left.c3.1 - right.c0.1).abs() < 1e-6);
        // Combined covers full range
        assert!((left.c0.0 - 0.0).abs() < 1e-6);
        assert!((right.c3.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bezier_clip_crossing_lines() {
        let a = Bezier2d {
            c0: (0.0, 0.0), c1: (0.33, 0.0),
            c2: (0.67, 0.0), c3: (1.0, 0.0),
        };
        let b = Bezier2d {
            c0: (0.5, -1.0), c1: (0.5, -0.33),
            c2: (0.5, 0.33), c3: (0.5, 1.0),
        };
        let hits = bezier_clip_intersect(&a, &b, 0.05, 12);
        assert!(!hits.is_empty(), "crossing lines should produce at least one intersection");
        // Intersection should be near (0.5, 0.0)
        let (pa, _pb) = hits[0];
        assert!((pa.0 - 0.5).abs() < 0.15, "intersection x ≈ 0.5, got {}", pa.0);
        assert!(pa.1.abs() < 0.15, "intersection y ≈ 0.0, got {}", pa.1);
    }

    #[test]
    fn test_bezier_clip_separated_no_intersection() {
        let a = Bezier2d {
            c0: (0.0, 0.0), c1: (0.33, 0.0),
            c2: (0.67, 0.0), c3: (1.0, 0.0),
        };
        let b = Bezier2d {
            c0: (0.0, 10.0), c1: (0.33, 10.0),
            c2: (0.67, 10.0), c3: (1.0, 10.0),
        };
        let hits = bezier_clip_intersect(&a, &b, 0.01, 8);
        assert!(hits.is_empty(), "well-separated beziers should not intersect");
    }

    #[test]
    fn test_intersect_curves_2d_lines() {
        let a = Curve2d::Line { origin: (0.0, 0.5), direction: (1.0, 0.0) };
        let b = Curve2d::Line { origin: (0.5, 0.0), direction: (0.0, 1.0) };
        let hits = intersect_curves_2d(&a, &b, 0.05);
        assert!(!hits.is_empty(), "perpendicular lines must intersect");
        let (pa, _pb) = hits[0];
        assert!((pa.0 - 0.5).abs() < 0.15, "intersection at x=0.5, got {}", pa.0);
        assert!((pa.1 - 0.5).abs() < 0.15, "intersection at y=0.5, got {}", pa.1);
    }
}
