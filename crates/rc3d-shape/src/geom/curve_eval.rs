//! Curve geometry evaluation (CurveGeom enum and methods).

use rc3d_core::math::{Real, PVec3};
use super::bspline::find_span;
use super::Curve2d;

// ── Helpers ────────────────────────────────────────────────────

/// Build perpendicular axes (x_dir, y_dir) from a direction vector,
/// forming a right-handed orthonormal basis (x_dir, y_dir, axis).
pub fn build_ortho_axes(axis: PVec3) -> (PVec3, PVec3) {
    let a = axis.normalize();
    let ref_dir = if a.x.abs() < 0.9 { PVec3::X } else { PVec3::Y };
    let y_dir = a.cross(ref_dir).normalize();
    let x_dir = y_dir.cross(a);
    (x_dir, y_dir)
}

/// Orthonormal (u, v) tangent basis for a plane; `u_dir` is projected onto the plane.
pub fn plane_tangent_basis(normal: PVec3, u_dir: PVec3) -> (PVec3, PVec3) {
    let n = normal.normalize();
    let mut u = u_dir - n * u_dir.dot(n);
    if u.length_squared() < 1e-20 {
        u = build_ortho_axes(n).0;
    } else {
        u = u.normalize();
    }
    let v = n.cross(u);
    (u, v)
}

/// Maximum B-spline degree expected in practice.
/// Typical STEP B-splines are degree 2–5, rarely exceeding 8.
/// Degrees above this threshold log a warning but are handled correctly
/// via dynamically sized working tables.
const MAX_DEGREE: usize = 16;

/// Index into the flattened `ndu` table: row `k` starts at offset k*(k+1)/2.
#[inline]
fn ndu_idx(k: usize, i: usize) -> usize {
    k * (k + 1) / 2 + i
}

/// Compute d0, d1, d2 for a (possibly rational) B-spline at parameter t.
/// Uses the Cox-de Boor recurrence for basis functions and their derivatives.
///
/// Working tables are sized dynamically to the B-spline degree.
/// A warning is logged when degree exceeds `MAX_DEGREE` (16).
fn bspline_d012(
    degree: usize,
    control_points: &[PVec3],
    knots: &[Real],
    weights: Option<&[Real]>,
    t: Real,
) -> (PVec3, PVec3, PVec3) {
    let p = degree;

    // Guard: insufficient data
    if control_points.len() < p + 1 || knots.len() < 2 * (p + 1) {
        return (PVec3::ZERO, PVec3::ZERO, PVec3::ZERO);
    }

    // Clamp t to valid knot domain [knots[p], knots[control_points.len()]]
    let t_min = knots[p];
    let t_max = knots[control_points.len()];
    let t = t.clamp(t_min, t_max);

    // Degree 0 special case
    if p == 0 {
        let span = find_span(0, knots, t);
        let pt = control_points[span];
        return (pt, PVec3::ZERO, PVec3::ZERO);
    }

    if p > MAX_DEGREE {
        log::warn!(
            "B-spline degree {} exceeds MAX_DEGREE ({}), using dynamic allocation",
            p,
            MAX_DEGREE
        );
    }

    let span = find_span(p, knots, t);
    let s = span;

    // ── Basis functions (triangular table) ──
    // ndu[k*(k+1)/2 + i] = N_{s-k+i, k}(t) for i = 0..k
    let ndu_size = (p + 1) * (p + 2) / 2;
    let mut ndu = vec![0.0_f64; ndu_size];
    ndu[ndu_idx(0, 0)] = 1.0;

    for k in 1..=p {
        for i in 0..=k {
            let ctrl_idx = s + i - k;

            let left = if i >= 1 {
                let denom = knots[ctrl_idx + k] - knots[ctrl_idx];
                if denom > 1e-10 {
                    (t - knots[ctrl_idx]) / denom * ndu[ndu_idx(k - 1, i - 1)]
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let right = if i < k {
                let denom = knots[ctrl_idx + k + 1] - knots[ctrl_idx + 1];
                if denom > 1e-10 {
                    (knots[ctrl_idx + k + 1] - t) / denom * ndu[ndu_idx(k - 1, i)]
                } else {
                    0.0
                }
            } else {
                0.0
            };

            ndu[ndu_idx(k, i)] = left + right;
        }
    }

    // ── First derivatives N'_{s-p+k, p} ──
    let mut ndu1 = vec![0.0_f64; p + 1];
    for k in 0..=p {
        let idx = s + k - p;
        let left = if k >= 1 {
            let denom = knots[idx + p] - knots[idx];
            if denom > 1e-10 {
                (p as Real) / denom * ndu[ndu_idx(p - 1, k - 1)]
            } else {
                0.0
            }
        } else {
            0.0
        };
        let right = if k < p {
            let denom = knots[idx + p + 1] - knots[idx + 1];
            if denom > 1e-10 {
                (p as Real) / denom * ndu[ndu_idx(p - 1, k)]
            } else {
                0.0
            }
        } else {
            0.0
        };
        ndu1[k] = left - right;
    }

    // ── Second derivatives N''_{s-p+k, p} ──
    let mut ndu2 = vec![0.0_f64; p + 1];
    if p >= 2 {
        // First compute N'_{s-(p-1)+k, p-1} for k = 0..p-1
        let mut ndu1_pm1 = vec![0.0_f64; p];
        for k in 0..p {
            let idx = s + k - (p - 1);
            let left = if k >= 1 {
                let denom = knots[idx + p - 1] - knots[idx];
                if denom > 1e-10 {
                    ((p - 1) as Real) / denom * ndu[ndu_idx(p - 2, k - 1)]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let right = if k < p - 1 {
                let denom = knots[idx + p] - knots[idx + 1];
                if denom > 1e-10 {
                    ((p - 1) as Real) / denom * ndu[ndu_idx(p - 2, k)]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            ndu1_pm1[k] = left - right;
        }

        // N''_{s-p+k, p} = p/denom_L * ndu1_pm1[k-1] - p/denom_R * ndu1_pm1[k]
        for k in 0..=p {
            let idx = s + k - p;
            let left = if k >= 1 {
                let denom = knots[idx + p] - knots[idx];
                if denom > 1e-10 {
                    (p as Real) / denom * ndu1_pm1[k - 1]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let right = if k < p {
                let denom = knots[idx + p + 1] - knots[idx + 1];
                if denom > 1e-10 {
                    (p as Real) / denom * ndu1_pm1[k]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            ndu2[k] = left - right;
        }
    }

    // ── Assemble weighted sums ──
    let mut a0 = PVec3::ZERO;
    let mut a1 = PVec3::ZERO;
    let mut a2 = PVec3::ZERO;
    let mut w_sum = 0.0_f64;
    let mut w_sum_1 = 0.0_f64;
    let mut w_sum_2 = 0.0_f64;

    for k in 0..=p {
        let idx = s + k - p;
        let cp = control_points[idx];
        let w = weights.map_or(1.0, |ws| ws[idx]);

        let n0 = ndu[ndu_idx(p, k)];
        let n1 = ndu1[k];
        let n2 = ndu2[k];

        a0 += cp * (n0 * w);
        a1 += cp * (n1 * w);
        a2 += cp * (n2 * w);
        w_sum += n0 * w;
        w_sum_1 += n1 * w;
        w_sum_2 += n2 * w;
    }

    let (d0, d1, d2) = if weights.is_some() {
        let w = w_sum;
        let w1 = w_sum_1;
        let w2 = w_sum_2;
        let w_inv = 1.0 / w.max(1e-12);
        let w_inv2 = w_inv * w_inv;
        let w_inv3 = w_inv2 * w_inv;

        let d0 = a0 * w_inv;
        let d1 = (a1 * w - a0 * w1) * w_inv2;
        let d2 = (a2 * w * w - a0 * w2 * w - 2.0 * a1 * w1 * w + 2.0 * a0 * w1 * w1) * w_inv3;
        (d0, d1, d2)
    } else {
        (a0, a1, a2)
    };

    // NaN guard: log warning and clamp to zero instead of silent propagation.
    // A NaN result indicates a numerical issue upstream (degenerate knot, near-zero weights).
    // OCC throws Geom_UndefinedDerivative here; we warn and return zero for robustness.
    let safe = |v: PVec3, which: &str| -> PVec3 {
        if v.is_nan() {
            log::warn!("[geom] bspline_d012: NaN in {} derivative — clamping to zero", which);
            PVec3::ZERO
        } else { v }
    };
    (safe(d0, "d0"), safe(d1, "d1"), safe(d2, "d2"))
}

/// De Casteljau evaluation of Bezier curve at parameter t ∈ [0, 1].
fn de_casteljau_d0(points: &[PVec3], weights: Option<&[Real]>, t: Real) -> PVec3 {
    let n = points.len();
    if n == 0 { return PVec3::ZERO; }
    if n == 1 { return points[0]; }
    if let Some(w) = weights {
        let mut p: Vec<(Real, PVec3)> = points.iter().zip(w.iter())
            .map(|(&pt, &wt)| (wt, pt * wt))
            .collect();
        for r in 1..n {
            for i in 0..(n - r) {
                let w_new = p[i].0 * (1.0 - t) + p[i + 1].0 * t;
                let v_new = p[i].1 * (1.0 - t) + p[i + 1].1 * t;
                p[i] = (w_new, v_new);
            }
        }
        if p[0].0.abs() < 1e-20 { PVec3::ZERO } else { p[0].1 / p[0].0 }
    } else {
        let mut p = points.to_vec();
        for r in 1..n {
            for i in 0..(n - r) {
                p[i] = p[i] * (1.0 - t) + p[i + 1] * t;
            }
        }
        p[0]
    }
}

/// First derivative via De Casteljau.
///
/// For unweighted curves: uses the analytical hodograph (degree·Δcp).
/// For rational curves: uses Richardson extrapolation with two finite-difference
/// step sizes for 4th-order accuracy (interior) / 2nd-order (endpoints).
/// This is a significant improvement over the previous single-step central
/// difference which had O(1e-4) endpoint error.
fn de_casteljau_d1(points: &[PVec3], weights: Option<&[Real]>, t: Real) -> PVec3 {
    let n = points.len();
    if n <= 1 { return PVec3::ZERO; }
    if let Some(w) = weights {
        if w.is_empty() { return PVec3::ZERO; }
        let h = 1e-4_f64;
        if t > h && t < 1.0 - h {
            // Interior: Richardson extrapolation on central differences
            let p_lo1 = de_casteljau_d0(points, weights, t - h);
            let p_hi1 = de_casteljau_d0(points, weights, t + h);
            let d1 = (p_hi1 - p_lo1) / (2.0 * h);
            let h2 = h * 0.5;
            let p_lo2 = de_casteljau_d0(points, weights, t - h2);
            let p_hi2 = de_casteljau_d0(points, weights, t + h2);
            let d2 = (p_hi2 - p_lo2) / (2.0 * h2);
            // 4th-order extrapolation: (4*d2 - d1) / 3
            (d2 * 4.0 - d1) / 3.0
        } else if t <= h {
            // Near t=0: one-sided Richardson
            let p0 = de_casteljau_d0(points, weights, 0.0);
            let p_h = de_casteljau_d0(points, weights, h);
            let d1 = (p_h - p0) / h;
            let h2 = h * 0.5;
            let p_h2 = de_casteljau_d0(points, weights, h2);
            let d2 = (p_h2 - p0) / h2;
            // 2nd-order extrapolation: 2*d2 - d1
            d2 * 2.0 - d1
        } else {
            // Near t=1: one-sided Richardson
            let p1 = de_casteljau_d0(points, weights, 1.0);
            let p_h = de_casteljau_d0(points, weights, 1.0 - h);
            let d1 = (p1 - p_h) / h;
            let h2 = h * 0.5;
            let p_h2 = de_casteljau_d0(points, weights, 1.0 - h2);
            let d2 = (p1 - p_h2) / h2;
            d2 * 2.0 - d1
        }
    } else {
        let degree = (n - 1) as Real;
        let diff: Vec<PVec3> = points.windows(2).map(|w| (w[1] - w[0]) * degree).collect();
        de_casteljau_d0(&diff, None, t)
    }
}

/// Approximate arc length of a curve by chordal sum with fixed sampling.
/// Avoids calling `arc_length` / `sample_adaptive` to break circular dependency
/// with `Composite` segment selection.
pub fn approx_chordal_length(curve: &CurveGeom) -> Real {
    match curve {
        CurveGeom::Composite { segments, .. } => {
            segments.iter().map(|(seg, _)| approx_chordal_length(seg)).sum()
        }
        _ => {
            const N: usize = 32;
            let mut len = 0.0;
            let mut prev = curve.d0(0.0);
            for i in 1..=N {
                let t = i as Real / N as Real;
                let curr = curve.d0(t);
                len += (curr - prev).length();
                prev = curr;
            }
            len
        }
    }
}

/// Find which segment of a composite curve a parameter t ∈ [0,1] falls in.
/// Returns (segment_index, t_mapped) where t_mapped ∈ [0,1] maps onto the segment.
/// If `cached_lengths` is present and matches segment count, uses it directly;
/// otherwise computes lengths on the fly.
fn find_composite_segment(
    segments: &[(CurveGeom, bool)],
    cached_lengths: &Option<Vec<Real>>,
    t: Real,
) -> Option<(usize, Real, Real)> {
    if segments.is_empty() {
        return None;
    }
    if segments.len() == 1 {
        return Some((0, t, 1.0));
    }

    // Use cached lengths if available and valid, otherwise compute
    let owned_lengths: Option<Vec<Real>> = match cached_lengths {
        Some(c) if c.len() == segments.len() => None, // use cache directly
        _ => Some(segments.iter().map(|(seg, _)| approx_chordal_length(seg).max(1e-10)).collect()),
    };
    let lengths: &[Real] = match &owned_lengths {
        Some(computed) => computed,
        None => cached_lengths.as_deref().unwrap(),
    };
    let total: Real = lengths.iter().sum();
    let inv_total = 1.0 / total.max(1e-10);

    let target = t.clamp(0.0, 1.0);
    let mut cumulative = 0.0_f64;
    for (i, &len) in lengths.iter().enumerate() {
        let seg_frac = len * inv_total;
        let next = cumulative + seg_frac;
        if target <= next || i == segments.len() - 1 {
            let t_local = if seg_frac > 1e-12 {
                (target - cumulative) / seg_frac
            } else {
                0.0
            };
            return Some((i, t_local.clamp(0.0, 1.0), seg_frac));
        }
        cumulative = next;
    }
    None
}

// ── CurveGeom ──────────────────────────────────────────────────

/// Parametric curve geometry (retained, not sampled).
#[derive(Debug, Clone)]
pub enum CurveGeom {
    Line { origin: PVec3, direction: PVec3 },
    Circle { center: PVec3, axis: PVec3, radius: Real, x_dir: PVec3, y_dir: PVec3 },
    Ellipse { center: PVec3, axis: PVec3, semi_major: Real, semi_minor: Real, x_dir: PVec3, y_dir: PVec3 },
    Hyperbola { center: PVec3, axis: PVec3, semi_major: Real, semi_minor: Real, x_dir: PVec3, y_dir: PVec3 },
    Parabola { center: PVec3, axis: PVec3, focal_dist: Real, x_dir: PVec3, y_dir: PVec3 },
    BSpline { degree: usize, control_points: Vec<PVec3>, knots: Vec<Real>, weights: Option<Vec<Real>> },
    /// Bezier curve of arbitrary degree with optional rational weights.
    /// Equivalent to BSpline with knot vector [0ⁿ⁺¹, 1ⁿ⁺¹] but evaluated
    /// via De Casteljau for better performance and OCC type-6 compatibility.
    BezierCurve {
        degree: usize,
        control_points: Vec<PVec3>,
        weights: Option<Vec<Real>>,
    },
    Trimmed { basis: Box<CurveGeom>, t_min: Real, t_max: Real },
    Composite { segments: Vec<(CurveGeom, bool)>, cached_lengths: Option<Vec<Real>> },
    Polyline { points: Vec<PVec3> },
    /// Offset curve at signed distance from basis curve.
    /// Points are computed as basis(t) + distance * normal_dir(t)
    /// where normal_dir is offset_dir projected onto the curve normal plane.
    Offset {
        basis: Box<CurveGeom>,
        offset_dir: PVec3,
        distance: Real,
    },
}

impl CurveGeom {
    /// Construct a Circle with pre-computed ortho axes.
    pub fn circle(center: PVec3, axis: PVec3, radius: Real) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Circle { center, axis, radius, x_dir, y_dir }
    }
    /// Construct an Ellipse with pre-computed ortho axes.
    pub fn ellipse(center: PVec3, axis: PVec3, semi_major: Real, semi_minor: Real) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir }
    }
    /// Construct a Hyperbola with pre-computed ortho axes.
    pub fn hyperbola(center: PVec3, axis: PVec3, semi_major: Real, semi_minor: Real) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Hyperbola { center, axis, semi_major, semi_minor, x_dir, y_dir }
    }
    /// Construct a Parabola with pre-computed ortho axes.
    pub fn parabola(center: PVec3, axis: PVec3, focal_dist: Real) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Parabola { center, axis, focal_dist, x_dir, y_dir }
    }

    /// Construct a Bezier curve from control points and optional rational weights.
    /// The degree is inferred from the number of control points (degree = len - 1).
    pub fn bezier(control_points: Vec<PVec3>, weights: Option<Vec<Real>>) -> Self {
        let degree = control_points.len().saturating_sub(1);
        CurveGeom::BezierCurve { degree, control_points, weights }
    }
}

/// Map edge parameter t in [0,1] to the basis curve parameter used by `d0`/`d1`.
fn trimmed_edge_to_basis(basis: &CurveGeom, t_min: Real, t_max: Real, t: Real) -> (Real, Real) {
    let span = (t_max - t_min).max(1e-12);
    let t_mapped = t_min + t * span;
    match basis {
        CurveGeom::Circle { .. } | CurveGeom::Ellipse { .. } => {
            (t_mapped / std::f64::consts::TAU, span / std::f64::consts::TAU)
        }
        _ => (t_mapped, span),
    }
}

impl CurveGeom {
    /// Native parameter interval for curve evaluation (STEP knot domain when applicable).
    pub fn native_param_range(&self) -> (Real, Real) {
        match self {
            CurveGeom::Trimmed { t_min, t_max, .. } => (*t_min, *t_max),
            CurveGeom::BSpline { degree, control_points, knots, .. } => {
                let n = control_points.len();
                if knots.len() > n + degree {
                    (knots[*degree], knots[n])
                } else {
                    (0.0, 1.0)
                }
            }
            _ => (0.0, 1.0),
        }
    }

    /// Evaluate position at parameter t ∈ [0, 1].
    pub fn d0(&self, t: Real) -> PVec3 {
        match self {
            CurveGeom::Line { origin, direction } => *origin + *direction * t,

            CurveGeom::Circle { center, x_dir, y_dir, radius, .. } => {
                let theta = t * std::f64::consts::TAU;
                *center + *x_dir * radius * theta.cos() + *y_dir * radius * theta.sin()
            }

            CurveGeom::Ellipse { center, x_dir, y_dir, semi_major, semi_minor, .. } => {
                let theta = t * std::f64::consts::TAU;
                *center + *x_dir * semi_major * theta.cos() + *y_dir * semi_minor * theta.sin()
            }

            CurveGeom::Hyperbola { center, x_dir, y_dir, semi_major, semi_minor, .. } => {
                let s = -2.0 + 4.0 * t;
                *center + *x_dir * (*semi_major * s.cosh()) + *y_dir * (*semi_minor * s.sinh())
            }

            CurveGeom::Parabola { center, x_dir, y_dir, focal_dist, .. } => {
                let s = -2.0 + 4.0 * t;
                *center + *x_dir * s + *y_dir * (s * s / (4.0 * *focal_dist))
            }

            CurveGeom::BSpline { degree, control_points, knots, weights } => {
                bspline_d012(*degree, control_points, knots, weights.as_deref(), t).0
            }

            CurveGeom::BezierCurve { control_points, weights, .. } => {
                de_casteljau_d0(control_points, weights.as_deref(), t)
            }

            CurveGeom::Trimmed { basis, t_min, t_max } => {
                let (t_eval, _) = trimmed_edge_to_basis(basis, *t_min, *t_max, t);
                basis.d0(t_eval)
            }

            CurveGeom::Composite { segments, cached_lengths } => {
                if let Some((idx, t_local, _)) = find_composite_segment(segments, cached_lengths, t) {
                    let (seg, reversed) = &segments[idx];
                    let t_eval = if *reversed { 1.0 - t_local } else { t_local };
                    seg.d0(t_eval)
                } else {
                    PVec3::ZERO
                }
            }

            CurveGeom::Polyline { points } => {
                if points.len() < 2 {
                    return points.first().copied().unwrap_or(PVec3::ZERO);
                }
                let n = points.len() - 1;
                let t_scaled = t.clamp(0.0, 1.0) * n as Real;
                let idx = (t_scaled as usize).min(n - 1);
                let frac = t_scaled - idx as Real;
                if idx >= n {
                    points[n]
                } else {
                    points[idx] + (points[idx + 1] - points[idx]) * frac
                }
            }

            CurveGeom::Offset { basis, offset_dir, distance } => {
                let p = basis.d0(t);
                let d = basis.d1(t);
                if d.length_squared() < 1e-12 {
                    return p;
                }
                let tangent = d.normalize();
                let offset_vec = (*offset_dir - tangent * offset_dir.dot(tangent)).normalize_or_zero();
                p + offset_vec * *distance
            }
        }
    }

    /// Evaluate first derivative (tangent) at parameter t ∈ [0, 1].
    pub fn d1(&self, t: Real) -> PVec3 {
        match self {
            CurveGeom::Line { direction, .. } => *direction,

            CurveGeom::Circle { x_dir, y_dir, radius, .. } => {
                let theta = t * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                twopi * radius * (-theta.sin() * *x_dir + theta.cos() * *y_dir)
            }

            CurveGeom::Ellipse { x_dir, y_dir, semi_major, semi_minor, .. } => {
                let theta = t * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                twopi * (-semi_major * theta.sin() * *x_dir + semi_minor * theta.cos() * *y_dir)
            }

            CurveGeom::Hyperbola { x_dir, y_dir, semi_major, semi_minor, .. } => {
                let s = -2.0 + 4.0 * t;
                4.0 * (semi_major * s.sinh() * *x_dir + semi_minor * s.cosh() * *y_dir)
            }

            CurveGeom::Parabola { x_dir, y_dir, focal_dist, .. } => {
                let s = -2.0 + 4.0 * t;
                4.0 * (*x_dir + *y_dir * (s / (2.0 * *focal_dist)))
            }

            CurveGeom::BSpline { degree, control_points, knots, weights } => {
                bspline_d012(*degree, control_points, knots, weights.as_deref(), t).1
            }

            CurveGeom::BezierCurve { control_points, weights, .. } => {
                de_casteljau_d1(control_points, weights.as_deref(), t)
            }

            CurveGeom::Trimmed { basis, t_min, t_max } => {
                let (t_eval, dt_dedge) = trimmed_edge_to_basis(basis, *t_min, *t_max, t);
                basis.d1(t_eval) * dt_dedge
            }

            CurveGeom::Composite { segments, cached_lengths } => {
                if let Some((idx, t_local, seg_width)) = find_composite_segment(segments, cached_lengths, t) {
                    let (seg, reversed) = &segments[idx];
                    let t_eval = if *reversed { 1.0 - t_local } else { t_local };
                    let chain_scale = if *reversed { -1.0 / seg_width } else { 1.0 / seg_width };
                    seg.d1(t_eval) * chain_scale
                } else {
                    PVec3::ZERO
                }
            }

            CurveGeom::Polyline { points } => {
                let n = points.len();
                if n < 2 {
                    return PVec3::ZERO;
                }
                let n_seg = n - 1;
                let t_scaled = t.clamp(0.0, 1.0) * n_seg as Real;
                let idx = (t_scaled as usize).min(n_seg - 1);
                if idx >= n_seg {
                    PVec3::ZERO
                } else {
                    (points[idx + 1] - points[idx]) * n_seg as Real
                }
            }

            CurveGeom::Offset { .. } => {
                let eps = 1e-4;
                let t0 = (t - eps).max(0.0);
                let t1 = (t + eps).min(1.0);
                (self.d0(t1) - self.d0(t0)) / (t1 - t0)
            }
        }
    }

    /// Evaluate second derivative at parameter t ∈ [0, 1].
    pub fn d2(&self, t: Real) -> PVec3 {
        match self {
            CurveGeom::Line { .. } => PVec3::ZERO,

            CurveGeom::Circle { x_dir, y_dir, radius, .. } => {
                let theta = t * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                -(twopi * twopi) * radius * (theta.cos() * *x_dir + theta.sin() * *y_dir)
            }

            CurveGeom::Ellipse { x_dir, y_dir, semi_major, semi_minor, .. } => {
                let theta = t * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                -(twopi * twopi) * (semi_major * theta.cos() * *x_dir + semi_minor * theta.sin() * *y_dir)
            }

            CurveGeom::Hyperbola { x_dir, y_dir, semi_major, semi_minor, .. } => {
                let s = -2.0 + 4.0 * t;
                16.0 * (semi_major * s.cosh() * *x_dir + semi_minor * s.sinh() * *y_dir)
            }

            CurveGeom::Parabola { y_dir, focal_dist, .. } => {
                16.0 * *y_dir / (2.0 * *focal_dist)
            }

            CurveGeom::BSpline { degree, control_points, knots, weights } => {
                bspline_d012(*degree, control_points, knots, weights.as_deref(), t).2
            }

            CurveGeom::BezierCurve { .. } => {
                let eps = 1e-4;
                (self.d1(t + eps) - self.d1(t - eps)) / (2.0 * eps)
            }

            CurveGeom::Trimmed { basis, t_min, t_max } => {
                let (t_eval, dt_dedge) = trimmed_edge_to_basis(basis, *t_min, *t_max, t);
                basis.d2(t_eval) * (dt_dedge * dt_dedge)
            }

            CurveGeom::Composite { segments, cached_lengths } => {
                if let Some((idx, t_local, seg_width)) = find_composite_segment(segments, cached_lengths, t) {
                    let (seg, reversed) = &segments[idx];
                    let t_eval = if *reversed { 1.0 - t_local } else { t_local };
                    // d² is divided by seg_width² (squared is always positive, reversal doesn't matter)
                    seg.d2(t_eval) / (seg_width * seg_width)
                } else {
                    PVec3::ZERO
                }
            }

            CurveGeom::Polyline { .. } => PVec3::ZERO,

            CurveGeom::Offset { .. } => {
                let eps = 1e-4;
                let t0 = (t - eps).max(0.0);
                let t1 = (t + eps).min(1.0);
                (self.d1(t1) - self.d1(t0)) / (t1 - t0)
            }
        }
    }

    /// Third derivative d³C/dt³ at parameter t via finite differences.
    /// Uses 5-point stencil for smooth curves, falling back to forward difference.
    pub fn d3(&self, t: Real) -> PVec3 {
        let eps = 2e-4;
        let t0 = (t - eps * 2.0).max(0.0);
        let t1 = (t - eps).max(0.0);
        let t2 = (t + eps).min(1.0);
        let t3 = (t + eps * 2.0).min(1.0);
        // 4-point central: (d0(t3) - 2*d0(t2) + 2*d0(t1) - d0(t0)) / (2*eps³)
        // Actually use: (-d0(t0) + 2*d0(t1) - 2*d0(t2) + d0(t3)) / (2 * eps³)
        let p0 = self.d0(t0);
        let p1 = self.d0(t1);
        let p2 = self.d0(t2);
        let p3 = self.d0(t3);
        let h3 = (t3 - t0).powi(3);
        if h3 < 1e-15 {
            return PVec3::ZERO;
        }
        (-p0 + 2.0 * p1 - 2.0 * p2 + p3) / h3
    }

    /// Curvature κ = |d1 × d2| / |d1|³ at parameter t.
    /// Returns 0.0 if the first derivative magnitude is below 1e-10.
    pub fn curvature(&self, t: Real) -> Real {
        let d1 = self.d1(t);
        let d1_len = d1.length();
        if d1_len < 1e-10 {
            return 0.0;
        }
        let d2 = self.d2(t);
        let cross_mag = d1.cross(d2).length();
        cross_mag / (d1_len * d1_len * d1_len)
    }

    /// Torsion τ = (d1 × d2) · d3 / |d1 × d2|² at parameter t.
    /// Uses finite-difference for d3. Returns 0.0 for degenerate cases.
    pub fn torsion(&self, t: Real) -> Real {
        let eps = 1e-4;
        let t_hi = (t + eps).min(1.0);
        let d1 = self.d1(t);
        let d2 = self.d2(t);
        let d2_hi = self.d2(t_hi);
        let d3 = (d2_hi - d2) / (t_hi - t);
        let cross = d1.cross(d2);
        let cross_len_sq = cross.length_squared();
        if cross_len_sq < 1e-20 {
            return 0.0;
        }
        cross.dot(d3) / cross_len_sq
    }

    /// Higher-order derivatives up to `order` (OCC Geom_Curve::DN).
    /// Returns Vec of [d0, d1, d2, ..., dN] where d0 is position.
    /// For orders > 2, uses finite-difference on lower-order derivatives.
    /// Line: d3+ = zero. Circle/Ellipse: trigonometric recurrence.
    pub fn dn(&self, t: Real, order: usize) -> Vec<PVec3> {
        if order == 0 { return vec![self.d0(t)]; }
        let mut result = Vec::with_capacity(order + 1);
        result.push(self.d0(t));
        if order >= 1 { result.push(self.d1(t)); }
        if order >= 2 { result.push(self.d2(t)); }
        if order <= 2 { return result; }

        match self {
            CurveGeom::Line { .. } => {
                for _ in 3..=order { result.push(PVec3::ZERO); }
            }
            CurveGeom::Circle { x_dir, y_dir, radius, .. } => {
                let twopi = std::f64::consts::TAU;
                for n in 3..=order {
                    let deriv = circle_deriv_n(t, n, *x_dir, *y_dir, *radius, twopi);
                    result.push(deriv);
                }
            }
            CurveGeom::Ellipse { x_dir, y_dir, semi_major, semi_minor, .. } => {
                let twopi = std::f64::consts::TAU;
                for n in 3..=order {
                    let deriv = ellipse_deriv_n(t, n, *x_dir, *y_dir, *semi_major, *semi_minor, twopi);
                    result.push(deriv);
                }
            }
            _ => {
                // Finite-difference: dN = (d(N-1)(t+eps) - d(N-1)(t-eps)) / (2*eps)
                let eps = 1e-4;
                for n in 3..=order {
                    let t_hi = (t + eps).min(1.0);
                    let t_lo = (t - eps).max(0.0);
                    let _prev = &result[n - 1];
                    // Recurse: compute d(n-1) at t_hi and t_lo
                    let fwd = self.dn(t_hi, n - 1);
                    let bwd = self.dn(t_lo, n - 1);
                    let d_n = if fwd.len() >= n && bwd.len() >= n {
                        (fwd[n - 1] - bwd[n - 1]) / (2.0 * eps)
                    } else {
                        PVec3::ZERO
                    };
                    result.push(d_n);
                }
            }
        }
        result
    }

    /// Combined position, first, and second derivative in one call.
    /// For BSpline curves, this avoids 3× redundant Cox-de Boor evaluation
    /// vs calling `d0`, `d1`, `d2` separately.
    pub fn d012(&self, t: Real) -> (PVec3, PVec3, PVec3) {
        match self {
            CurveGeom::BSpline { degree, control_points, knots, weights } => {
                bspline_d012(*degree, control_points, knots, weights.as_deref(), t)
            }
            CurveGeom::BezierCurve { control_points, weights, .. } => {
                let p = de_casteljau_d0(control_points, weights.as_deref(), t);
                let d1 = de_casteljau_d1(control_points, weights.as_deref(), t);
                let h = 1e-4_f64;
                let d2 = if t > h && t < 1.0 - h {
                    // Interior: Richardson extrapolation on central d1 differences
                    let dp1 = de_casteljau_d1(control_points, weights.as_deref(), t + h);
                    let dm1 = de_casteljau_d1(control_points, weights.as_deref(), t - h);
                    let a1 = (dp1 - dm1) / (2.0 * h);
                    let h2 = h * 0.5;
                    let dp2 = de_casteljau_d1(control_points, weights.as_deref(), t + h2);
                    let dm2 = de_casteljau_d1(control_points, weights.as_deref(), t - h2);
                    let a2 = (dp2 - dm2) / (2.0 * h2);
                    (a2 * 4.0 - a1) / 3.0
                } else {
                    let d1p = de_casteljau_d1(control_points, weights.as_deref(), (t + h).min(1.0));
                    let d1m = de_casteljau_d1(control_points, weights.as_deref(), (t - h).max(0.0));
                    (d1p - d1m) / ((t + h).min(1.0) - (t - h).max(0.0)).max(h)
                };
                (p, d1, d2)
            }
            CurveGeom::Circle { center, x_dir, y_dir, radius, .. } => {
                let theta = t * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                let (c, s) = (theta.cos(), theta.sin());
                let d0 = *center + *x_dir * (*radius * c) + *y_dir * (*radius * s);
                let d1 = twopi * *radius * (-s * *x_dir + c * *y_dir);
                let d2 = -(twopi * twopi) * *radius * (c * *x_dir + s * *y_dir);
                (d0, d1, d2)
            }
            CurveGeom::Ellipse { center, x_dir, y_dir, semi_major, semi_minor, .. } => {
                let theta = t * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                let (c, s) = (theta.cos(), theta.sin());
                let d0 = *center + *x_dir * (*semi_major * c) + *y_dir * (*semi_minor * s);
                let d1 = twopi * (-*semi_major * s * *x_dir + *semi_minor * c * *y_dir);
                let d2 = -(twopi * twopi) * (*semi_major * c * *x_dir + *semi_minor * s * *y_dir);
                (d0, d1, d2)
            }
            CurveGeom::Hyperbola { center, x_dir, y_dir, semi_major, semi_minor, .. } => {
                let s = -2.0 + 4.0 * t;
                let (ch, sh) = (s.cosh(), s.sinh());
                let d0 = *center + *x_dir * (*semi_major * ch) + *y_dir * (*semi_minor * sh);
                let d1 = 4.0 * (*semi_major * sh * *x_dir + *semi_minor * ch * *y_dir);
                let d2 = 16.0 * (*semi_major * ch * *x_dir + *semi_minor * sh * *y_dir);
                (d0, d1, d2)
            }
            CurveGeom::Parabola { center, x_dir, y_dir, focal_dist, .. } => {
                let s = -2.0 + 4.0 * t;
                let d0 = *center + *x_dir * s + *y_dir * (s * s / (4.0 * *focal_dist));
                let d1 = 4.0 * (*x_dir + *y_dir * (s / (2.0 * *focal_dist)));
                let d2 = 16.0 * *y_dir / (2.0 * *focal_dist);
                (d0, d1, d2)
            }
            CurveGeom::Line { origin, direction } => {
                (*origin + *direction * t, *direction, PVec3::ZERO)
            }
            CurveGeom::Trimmed { basis, t_min, t_max } => {
                let (t_eval, dt) = trimmed_edge_to_basis(basis, *t_min, *t_max, t);
                let (d0, d1, d2) = basis.d012(t_eval);
                (d0, d1 * dt, d2 * (dt * dt))
            }
            CurveGeom::Composite { segments, cached_lengths } => {
                if let Some((idx, t_local, w)) = find_composite_segment(segments, cached_lengths, t) {
                    let (seg, rev) = &segments[idx];
                    let t_eval = if *rev { 1.0 - t_local } else { t_local };
                    let (d0, d1, d2) = seg.d012(t_eval);
                    let scale = if *rev { -1.0 / w } else { 1.0 / w };
                    (d0, d1 * scale, d2 / (w * w))
                } else {
                    (PVec3::ZERO, PVec3::ZERO, PVec3::ZERO)
                }
            }
            CurveGeom::Polyline { .. } => (self.d0(t), self.d1(t), self.d2(t)),

            CurveGeom::Offset { .. } => (self.d0(t), self.d1(t), self.d2(t)),
        }
    }

    /// Adaptive chordal arc length between t0 and t1.
    /// Uses `sample_adaptive` and sums chord lengths of the result points.
    pub fn arc_length(&self, t0: Real, t1: Real) -> Real {
        let pts = self.sample_adaptive(t0, t1, 0.1);
        let mut len = 0.0;
        for w in pts.windows(2) {
            len += (w[1].1 - w[0].1).length();
        }
        len
    }

    /// Curvature-driven adaptive sampling in [t0, t1] with given tolerance.
    ///
    /// Starts with 8 uniform samples, then subdivides segments where either:
    /// - The chordal deviation exceeds `tolerance`, or
    /// - `curvature(midpoint) * segment_length > tolerance`.
    ///
    /// Stops at 512 points or when all segments are within tolerance.
    pub fn sample_adaptive(&self, t0: Real, t1: Real, tolerance: Real) -> Vec<(Real, PVec3)> {
        const MAX_POINTS: usize = 512;

        // Seed with 8 uniform samples
        let mut points: Vec<(Real, PVec3)> = Vec::with_capacity(MAX_POINTS);
        let initial = 8usize;
        for i in 0..=initial {
            let ti = t0 + (t1 - t0) * i as Real / initial as Real;
            points.push((ti, self.d0(ti)));
        }

        // Refine segments
        let mut i = 0;
        while i + 1 < points.len() && points.len() < MAX_POINTS {
            let t_a = points[i].0;
            let p_a = points[i].1;
            let t_b = points[i + 1].0;
            let p_b = points[i + 1].1;

            let seg_len = (p_b - p_a).length();
            let t_mid = (t_a + t_b) * 0.5;
            let p_mid = self.d0(t_mid);

            // Chordal deviation
            let midpoint_on_chord = (p_a + p_b) * 0.5;
            let deviation = (p_mid - midpoint_on_chord).length();

            // Curvature-based refinement
            let curv = self.curvature(t_mid);
            let need_refine = deviation > tolerance || curv * seg_len > tolerance;

            if need_refine {
                points.insert(i + 1, (t_mid, p_mid));
                // Don't advance i, refine this new segment next
            } else {
                i += 1;
            }
        }

        points
    }
}

/// Rebuild edge curve so `t=0` / `t=1` match canonical B-Rep vertex positions.
///
/// For Circle/Ellipse curves, find angular parameters matching vertex positions.
fn trim_circle_to_vertices(
    curve: &CurveGeom, p_lo: PVec3, p_hi: PVec3,
) -> Option<CurveGeom> {
    match curve {
        CurveGeom::Circle { center, axis, radius, x_dir, y_dir } => {
            let a0 = circle_angle_geom(center, *axis, *radius, *x_dir, *y_dir, p_lo)?;
            let a1 = circle_angle_geom(center, *axis, *radius, *x_dir, *y_dir, p_hi)?;
            let (t_min, t_max) = normalize_arc_params(a0, a1);
            Some(CurveGeom::Trimmed { basis: Box::new(curve.clone()), t_min, t_max })
        }
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
            let a0 = ellipse_angle_geom(center, *axis, *semi_major, *semi_minor, *x_dir, *y_dir, p_lo)?;
            let a1 = ellipse_angle_geom(center, *axis, *semi_major, *semi_minor, *x_dir, *y_dir, p_hi)?;
            let (t_min, t_max) = normalize_arc_params(a0, a1);
            Some(CurveGeom::Trimmed { basis: Box::new(curve.clone()), t_min, t_max })
        }
        CurveGeom::Trimmed { basis, .. } => trim_circle_to_vertices(basis, p_lo, p_hi),
        _ => None,
    }
}

/// Compute angular parameter [0,TAU) of a point on a circle/ellipse.
fn circle_angle_geom(
    center: &PVec3, _axis: PVec3, radius: Real, x_dir: PVec3, y_dir: PVec3, point: PVec3,
) -> Option<Real> {
    let rel = point - *center;
    let a = _axis.normalize();
    let proj = rel - a * rel.dot(a);
    let dist = proj.length();
    if (dist - radius).abs() > radius * 0.1 && (dist - radius).abs() > 0.5 { return None; }
    let u = Real::atan2(proj.dot(y_dir), proj.dot(x_dir));
    Some(if u < 0.0 { u + std::f64::consts::TAU } else { u })
}

/// Compute eccentric anomaly angle [0,TAU) of a point on an ellipse.
///
/// Unlike `circle_angle_geom`, this correctly handles high-eccentricity ellipses
/// (e.g. a=10, b=1) by computing θ = atan2(y/b, x/a) instead of assuming
/// constant distance from center.
fn ellipse_angle_geom(
    center: &PVec3, axis: PVec3, semi_major: Real, semi_minor: Real,
    x_dir: PVec3, y_dir: PVec3, point: PVec3,
) -> Option<Real> {
    let a = axis.normalize();
    let rel = point - *center;
    let proj = rel - a * rel.dot(a);
    let u = Real::atan2(proj.dot(y_dir) / semi_minor, proj.dot(x_dir) / semi_major);
    Some(if u < 0.0 { u + std::f64::consts::TAU } else { u })
}

/// Clamp arc params to [0,TAU) with correct t_min < t_max ordering.
fn normalize_arc_params(a0: Real, a1: Real) -> (Real, Real) {
    let diff = a1 - a0;
    if diff.abs() > std::f64::consts::PI {
        if a0 < a1 { (a1 - std::f64::consts::TAU, a0) }
        else { (a0 - std::f64::consts::TAU, a1) }
    } else { (a0.min(a1), a0.max(a1)) }
}

/// Find the parameter t∈[0,1] on a curve closest to the target point.
///
/// Delegates to multi-start Newton-Raphson curve projection.
pub fn find_param_on_curve(curve: &CurveGeom, target: PVec3) -> Real {
    let results = super::project::project_point_on_curve(curve, target);
    // Default to 0.5 (mid-point) if all projection methods fail, which is
    // safer than 0.0 (curve start) for most downstream uses like trim fitting.
    results.first().map(|(t, _)| *t).unwrap_or(0.5)
}

/// Compute angular parameter range for a circle or ellipse edge.
/// Uses eccentric anomaly for ellipses (divides by semi-axes).
/// Correctly handles the atan2 branch cut at ±π via `min(|d|, 2π-|d|)`.
fn angular_param_range(
    center: PVec3,
    x_dir: PVec3,
    y_dir: PVec3,
    scale_x: Real,
    scale_y: Real,
    v_low: PVec3,
    v_high: PVec3,
) -> (Real, Real) {
    let to_angle = |p: PVec3| -> Real {
        let d = p - center;
        let sx = scale_x.max(1e-12);
        let sy = scale_y.max(1e-12);
        Real::atan2(d.dot(y_dir) / sy, d.dot(x_dir) / sx)
    };
    let t0 = to_angle(v_low);
    let t1 = to_angle(v_high);
    let diff = (t0 - t1).abs();
    if diff.min(std::f64::consts::TAU - diff) < 1e-2 {
        return (0.0, std::f64::consts::TAU);
    }
    // Return the shorter arc that contains both vertices
    if diff > std::f64::consts::PI {
        // Vertices straddle the branch cut; the range wraps through ±π
        (t0.max(t1), t0.min(t1) + std::f64::consts::TAU)
    } else {
        (t0.min(t1), t0.max(t1))
    }
}

/// Compute the curve parameter range that corresponds to the edge
/// bounded by `v_low` and `v_high`. For a `Trimmed` curve this is the
/// trim bounds; for other curves we project the vertices onto the curve.
///
/// The returned `(t_min, t_max)` satisfies:
///   curve.d0(t_min) ≈ v_low   and   curve.d0(t_max) ≈ v_high
pub fn curve_param_range_from_vertices(
    curve: &CurveGeom,
    v_low: PVec3,
    v_high: PVec3,
) -> (Real, Real) {
    // Trimmed curves already carry the correct bounds.
    if let CurveGeom::Trimmed { t_min, t_max, .. } = curve {
        return (*t_min, *t_max);
    }

    // Closed (seam) edge: both vertices at same 3D point on a periodic curve.
    // Return one full period. Use a generous tolerance — vertex positions may
    // differ by FP noise when vertices are split across faces.
    let chord = (v_high - v_low).length();
    let is_closed = chord < 1e-4;
    if is_closed {
        match curve {
            CurveGeom::Circle { .. } | CurveGeom::Ellipse { .. } => {
                return (0.0, std::f64::consts::TAU);
            }
            CurveGeom::BSpline { .. } => {
                return curve.native_param_range();
            }
            _ => {}
        }
    }

    // For analytic curves use fast closed-form inversion.
    match curve {
        CurveGeom::Line { origin, direction } => {
            let len2 = direction.length_squared();
            if len2 < 1e-20 {
                return (0.0, 1.0);
            }
            let inv_len2 = 1.0 / len2;
            let t_lo = (v_low - origin).dot(*direction) * inv_len2;
            let t_hi = (v_high - origin).dot(*direction) * inv_len2;
            (t_lo.min(t_hi), t_lo.max(t_hi))
        }
        CurveGeom::Circle { center, radius, x_dir, y_dir, .. } => {
            angular_param_range(*center, *x_dir, *y_dir, *radius, *radius, v_low, v_high)
        }
        CurveGeom::Ellipse { center, semi_major, semi_minor, x_dir, y_dir, .. } => {
            angular_param_range(*center, *x_dir, *y_dir, *semi_major, *semi_minor, v_low, v_high)
        }
        _ => {
            // Generic fallback: project vertices onto curve via Newton.
            let t_lo = find_param_on_curve(curve, v_low);
            let t_hi = find_param_on_curve(curve, v_high);
            (t_lo.min(t_hi), t_lo.max(t_hi))
        }
    }
}

/// STEP `LINE` entities often reference a unit `VECTOR`; the actual edge span is
/// defined by `VERTEX_POINT` coordinates, not vector magnitude.
pub fn normalize_edge_curve_to_vertices(
    curve: CurveGeom,
    p_lo: PVec3,
    p_hi: PVec3,
    tol: Real,
) -> CurveGeom {
    let chord = p_hi - p_lo;
    let len = chord.length();
    if len < tol {
        // Closed curve (full circle etc.) — keep original geometry.
        return curve;
    }

    // Match heal's gap_tolerance. Add length component for long revolution arcs.
    let match_tol = (tol.max(1e-4) * 10.0).max(len * 0.001).min(0.1);
    let c0 = curve.d0(0.0);
    let c1 = curve.d0(1.0);
    if (c0 - p_lo).length() <= match_tol && (c1 - p_hi).length() <= match_tol { return curve; }
    if (c0 - p_hi).length() <= match_tol && (c1 - p_lo).length() <= match_tol { return curve; }

    // Circle/Ellipse: find angular parameters matching vertices.
    if let Some(t) = trim_circle_to_vertices(&curve, p_lo, p_hi) { return t; }

    // Generic: trim curve to correct parameter range.
    let t_lo = find_param_on_curve(&curve, p_lo);
    let t_hi = find_param_on_curve(&curve, p_hi);
    if (curve.d0(t_lo) - p_lo).length() <= match_tol && (curve.d0(t_hi) - p_hi).length() <= match_tol {
        let (t_min, t_max) = (t_lo.min(t_hi), t_lo.max(t_hi));
        return CurveGeom::Trimmed { basis: Box::new(curve), t_min, t_max };
    }

    // Keep original geometry — PCurve on each face provides correct surface trajectory.
    let c0_err = (curve.d0(0.0) - p_lo).length();
    let c1_err = (curve.d0(1.0) - p_hi).length();
    if c0_err > match_tol || c1_err > match_tol {
        log::warn!("edge curve mismatch: endpoints off by {:.4}/{:.4} (tol={:.4}, len={:.4})",
            c0_err, c1_err, match_tol, len);
    }
    curve
}

/// Signed area of a 2D polygon (shoelace formula). Returns positive for CCW winding.
pub fn signed_area_2d(uv: &[(Real, Real)]) -> f64 {
    let n = uv.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0f64;
    for i in 0..n {
        let (u0, v0) = uv[i];
        let (u1, v1) = uv[(i + 1) % n];
        a += u0 as f64 * v1 as f64 - u1 as f64 * v0 as f64;
    }
    a * 0.5
}

/// Evaluate PCurve on surface at t (OCCT BRepAdaptor_Curve with face context).
pub fn eval_pcurve_on_surface(pcurve: &Curve2d, surface: &super::SurfaceGeom, t: Real) -> PVec3 {
    let uv = pcurve.d0(t);
    surface.d0_native(uv.0, uv.1)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CurveGeom tests ────────────────────────────────

    #[test]
    fn test_line_d0() {
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let p = line.d0(0.5);
        assert!((p - PVec3::new(0.5, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_line_d1_is_direction() {
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::new(3.0, 4.0, 0.0) };
        let d = line.d1(0.5);
        assert!((d - PVec3::new(3.0, 4.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_circle_curvature() {
        let circle = CurveGeom::circle(PVec3::ZERO, PVec3::Z, 2.0);
        let k = circle.curvature(0.25);
        assert!((k - 0.5).abs() < 1e-4); // curvature = 1/r
    }

    #[test]
    fn test_circle_d1_orthogonal_to_radius() {
        let circle = CurveGeom::circle(PVec3::ZERO, PVec3::Z, 1.0);
        let pos = circle.d0(0.0); // at angle 0 -> (1,0,0)
        let tan = circle.d1(0.0);
        // tangent at angle 0 should be (0, 2pi, 0) -- orthogonal to radius
        assert!(pos.dot(tan).abs() < 1e-6);
    }

    #[test]
    fn test_sample_adaptive_line_minimal() {
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::new(10.0, 0.0, 0.0) };
        let pts = line.sample_adaptive(0.0, 1.0, 0.1);
        // A line should need very few points (low curvature)
        assert!(pts.len() >= 2); // at least start and end
        assert!(pts.len() <= 16);
    }

    // ── bspline_d012 stack-allocation tests ─────────────

    #[test]
    fn test_ndu_idx_triangular_layout() {
        // Row 0: index 0
        assert_eq!(ndu_idx(0, 0), 0);
        // Row 1: indices 1, 2
        assert_eq!(ndu_idx(1, 0), 1);
        assert_eq!(ndu_idx(1, 1), 2);
        // Row 2: indices 3, 4, 5
        assert_eq!(ndu_idx(2, 0), 3);
        assert_eq!(ndu_idx(2, 1), 4);
        assert_eq!(ndu_idx(2, 2), 5);
        // Row k starts at k*(k+1)/2
        assert_eq!(ndu_idx(5, 0), 15);
        assert_eq!(ndu_idx(5, 5), 20);
    }

    #[test]
    fn test_bspline_d012_degree3_collinear() {
        // Cubic B-spline with 4 collinear control points along X axis.
        // Clamped knot vector → curve starts at P0 and ends at P3.
        let cps = vec![
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(2.0, 0.0, 0.0),
            PVec3::new(3.0, 0.0, 0.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        // Endpoint interpolation (clamped B-spline)
        let (d0_start, _, _) = bspline_d012(3, &cps, &knots, None, 0.0);
        let (d0_end, _, _) = bspline_d012(3, &cps, &knots, None, 1.0);
        assert!((d0_start - cps[0]).length() < 1e-5, "start endpoint: {d0_start:?}");
        assert!((d0_end - cps[3]).length() < 1e-5, "end endpoint: {d0_end:?}");

        // Curve stays on X axis (y=0, z=0) for all t
        for i in 0..=10 {
            let t = i as Real / 10.0;
            let (d0, d1, d2) = bspline_d012(3, &cps, &knots, None, t);
            assert!(d0.y.abs() < 1e-5 && d0.z.abs() < 1e-5, "off-axis at t={t}: {d0:?}");
            // Tangent should be along X
            assert!(d1.y.abs() < 1e-4 && d1.z.abs() < 1e-4, "tangent off-axis at t={t}: {d1:?}");
            // Second derivative should also be along X for collinear CPs
            assert!(d2.y.abs() < 1e-3 && d2.z.abs() < 1e-3, "d2 off-axis at t={t}: {d2:?}");
            // X should be monotonically increasing
            assert!(d0.x >= -1e-5 && d0.x <= 3.0 + 1e-5, "x out of range at t={t}: {}", d0.x);
        }
    }

    #[test]
    fn test_bspline_d012_degree8_collinear() {
        // Degree-8 B-spline with 9 collinear control points along X axis.
        // Tests the stack path near the upper limit (8 < MAX_DEGREE=16).
        let cps: Vec<PVec3> = (0..9).map(|i| PVec3::new(i as Real, 0.0, 0.0)).collect();
        let knots = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];

        // Endpoint interpolation
        let (d0_start, _, _) = bspline_d012(8, &cps, &knots, None, 0.0);
        let (d0_end, _, _) = bspline_d012(8, &cps, &knots, None, 1.0);
        assert!((d0_start - cps[0]).length() < 1e-4, "start: {d0_start:?}");
        assert!((d0_end - cps[8]).length() < 1e-4, "end: {d0_end:?}");

        // Stays on X axis
        for i in 0..=10 {
            let t = i as Real / 10.0;
            let (d0, d1, _d2) = bspline_d012(8, &cps, &knots, None, t);
            assert!(d0.y.abs() < 1e-4 && d0.z.abs() < 1e-4, "off-axis at t={t}: {d0:?}");
            assert!(d1.y.abs() < 1e-3 && d1.z.abs() < 1e-3, "tangent off-axis at t={t}: {d1:?}");
        }
    }

    #[test]
    fn test_ellipse_trim_eccentric() {
        // High-eccentricity ellipse: a=10, b=1.  The old circle_angle_geom
        // rejects valid points because distance from center varies from 1 to 10.
        let center = PVec3::ZERO;
        let axis = PVec3::Z;
        let a = 10.0_f64;
        let b = 1.0_f64;
        let ellipse = CurveGeom::ellipse(center, axis, a, b);
        // Point at eccentric anomaly θ=0:  (a, 0, 0)
        let p_lo = PVec3::new(a, 0.0, 0.0);
        // Point at eccentric anomaly θ=π/2:  (0, b, 0)
        let p_hi = PVec3::new(0.0, b, 0.0);

        let trimmed = trim_circle_to_vertices(&ellipse, p_lo, p_hi);
        assert!(trimmed.is_some(), "high-eccentricity ellipse trim must succeed");

        // Verify the trimmed curve evaluates to the correct endpoints.
        let t = trimmed.unwrap();
        let e0 = (t.d0(0.0) - p_lo).length();
        let e1 = (t.d0(1.0) - p_hi).length();
        assert!(e0 < 0.1, "trimmed d0(0) error {e0} too large");
        assert!(e1 < 0.1, "trimmed d0(1) error {e1} too large");
    }

    #[test]
    fn test_composite_cached_lengths() {
        // Build a composite of two line segments: length 1 + length 2 = total 3.
        let seg1 = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let seg2 = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::new(2.0, 0.0, 0.0) };
        let segments = vec![(seg1, false), (seg2, false)];

        // Precompute cached lengths (as build_curve does)
        let cached: Vec<Real> = segments.iter()
            .map(|(seg, _)| approx_chordal_length(seg).max(1e-10))
            .collect();
        let comp = CurveGeom::Composite { segments, cached_lengths: Some(cached.clone()) };

        // Verify cache matches segment count
        assert_eq!(cached.len(), 2);
        assert!((cached[0] - 1.0).abs() < 0.05, "seg1 length ≈ 1.0, got {}", cached[0]);
        assert!((cached[1] - 2.0).abs() < 0.05, "seg2 length ≈ 2.0, got {}", cached[1]);

        // d0 at t=0 should be start of first segment
        let p0 = comp.d0(0.0);
        assert!(p0.length() < 1e-5, "d0(0) should be origin, got {:?}", p0);

        // d0 at t=1 should be end of second segment (seg2.d0(1.0) = (2,0,0))
        let p1 = comp.d0(1.0);
        assert!((p1 - PVec3::new(2.0, 0.0, 0.0)).length() < 0.1, "d0(1) ≈ (2,0,0), got {:?}", p1);

        // d0 at t=1/3 should be near the junction (end of seg1 = (1,0,0))
        let p_third = comp.d0(1.0 / 3.0);
        assert!((p_third - PVec3::new(1.0, 0.0, 0.0)).length() < 0.1,
            "d0(1/3) ≈ junction (1,0,0), got {:?}", p_third);

        // Also test Composite without cache (cached_lengths: None) still works
        let seg3 = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let seg4 = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::Y };
        let comp_no_cache = CurveGeom::Composite {
            segments: vec![(seg3, false), (seg4, false)],
            cached_lengths: None,
        };
        let p_mid = comp_no_cache.d0(0.5);
        // Should still evaluate without panic; midpoint should be near junction
        assert!(p_mid.x.is_finite() && p_mid.y.is_finite(), "no-cache eval must produce finite point");
    }

    #[test]
    fn test_bspline_d012_rational_degree3() {
        // Rational cubic B-spline (NURBS) — verify weights affect the curve.
        let cps = vec![
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 1.0, 0.0),
            PVec3::new(2.0, 1.0, 0.0),
            PVec3::new(3.0, 0.0, 0.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0, 2.0, 2.0, 1.0];

        // Endpoints should still match (clamped, weight=1 at ends)
        let (d0_start, _, _) = bspline_d012(3, &cps, &knots, Some(&weights), 0.0);
        let (d0_end, _, _) = bspline_d012(3, &cps, &knots, Some(&weights), 1.0);
        assert!((d0_start - cps[0]).length() < 1e-5, "rational start: {d0_start:?}");
        assert!((d0_end - cps[3]).length() < 1e-5, "rational end: {d0_end:?}");

        // Higher weights on middle CPs should pull curve toward them
        let (d0_mid_unweighted, _, _) = bspline_d012(3, &cps, &knots, None, 0.5);
        let (d0_mid_weighted, _, _) = bspline_d012(3, &cps, &knots, Some(&weights), 0.5);
        // The weighted midpoint should be pulled higher in Y
        assert!(
            d0_mid_weighted.y > d0_mid_unweighted.y,
            "weights should pull curve up: unweighted y={}, weighted y={}",
            d0_mid_unweighted.y, d0_mid_weighted.y
        );
    }

    #[test]
    fn test_hyperbola_d0_d1_d2() {
        let h = CurveGeom::Hyperbola {
            center: PVec3::ZERO, axis: PVec3::Z,
            semi_major: 2.0, semi_minor: 1.0,
            x_dir: PVec3::X, y_dir: PVec3::Y,
        };
        // At t=0.5, s=0: P = center + cosh(0)*2*X + sinh(0)*1*Y = (2, 0, 0)
        let p = h.d0(0.5);
        assert!((p.x - 2.0).abs() < 1e-5, "hyperbola at s=0 should be at (2,0,0), got {:?}", p);
        assert!(p.y.abs() < 1e-5);
        assert!(p.z.abs() < 1e-5);

        // d1 at s=0: 4*(sinh(0)*2*X + cosh(0)*1*Y) = 4*(0, 1) = (0, 4, 0)
        let d1 = h.d1(0.5);
        assert!(d1.x.abs() < 1e-4, "d1.x at s=0 should be 0, got {}", d1.x);
        assert!((d1.y - 4.0).abs() < 1e-4, "d1.y at s=0 should be 4, got {}", d1.y);

        // d2 at s=0: 16*(cosh(0)*2*X + sinh(0)*1*Y) = (32, 0, 0)
        let d2 = h.d2(0.5);
        assert!((d2.x - 32.0).abs() < 1e-4, "d2.x at s=0 should be 32, got {}", d2.x);
        assert!(d2.y.abs() < 1e-4);

        // d012 consistency
        let (p2, d1_2, d2_2) = h.d012(0.5);
        assert!((p - p2).length() < 1e-10);
        assert!((d1 - d1_2).length() < 1e-10);
        assert!((d2 - d2_2).length() < 1e-10);
    }

    #[test]
    fn test_parabola_d0_d1_d2() {
        let p = CurveGeom::Parabola {
            center: PVec3::ZERO, axis: PVec3::Z,
            focal_dist: 1.0,
            x_dir: PVec3::X, y_dir: PVec3::Y,
        };
        // At t=0.5, s=0: P = center + 0*X + 0*Y = (0, 0, 0)
        let pt = p.d0(0.5);
        assert!(pt.length() < 1e-5, "parabola at vertex should be origin, got {:?}", pt);

        // d1 at s=0: 4*(X + Y*0/(2*1)) = (4, 0, 0)
        let d1 = p.d1(0.5);
        assert!((d1.x - 4.0).abs() < 1e-4, "d1.x at s=0 should be 4, got {}", d1.x);
        assert!(d1.y.abs() < 1e-4);

        // d2 at s=0: 16*Y/(2*1) = (0, 8, 0)
        let d2 = p.d2(0.5);
        assert!(d2.x.abs() < 1e-4);
        assert!((d2.y - 8.0).abs() < 1e-4, "d2.y at s=0 should be 8, got {}", d2.y);

        // d012 consistency
        let (p2, d1_2, d2_2) = p.d012(0.5);
        assert!((pt - p2).length() < 1e-10);
        assert!((d1 - d1_2).length() < 1e-10);
        assert!((d2 - d2_2).length() < 1e-10);
    }

    #[test]
    fn test_hyperbola_constructor() {
        let h = CurveGeom::hyperbola(PVec3::new(1.0, 2.0, 3.0), PVec3::Z, 3.0, 1.5);
        // At t=0.5 (s=0), position should be center + semi_major * x_dir
        let p = h.d0(0.5);
        assert!((p.x - 4.0).abs() < 1e-5, "expected x=4.0, got {}", p.x);
        assert!((p.y - 2.0).abs() < 1e-5);
        assert!((p.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_parabola_constructor() {
        let p = CurveGeom::parabola(PVec3::new(1.0, 0.0, 0.0), PVec3::Z, 2.0);
        // At t=0.5 (s=0), position should be center
        let pt = p.d0(0.5);
        assert!((pt.x - 1.0).abs() < 1e-5);
        assert!(pt.y.abs() < 1e-5);
    }

    #[test]
    fn test_offset_curve_line() {
        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let offset = CurveGeom::Offset {
            basis: Box::new(line),
            offset_dir: PVec3::Y,
            distance: 1.0,
        };
        let p = offset.d0(0.5);
        assert!((p.x - 0.5).abs() < 1e-6, "x={}", p.x);
        assert!((p.y - 1.0).abs() < 0.01, "y={}", p.y);
        assert!(p.z.abs() < 1e-6, "z={}", p.z);
    }

    #[test]
    fn test_offset_curve_circle() {
        let circle = CurveGeom::circle(PVec3::ZERO, PVec3::Z, 1.0);
        let offset = CurveGeom::Offset {
            basis: Box::new(circle),
            offset_dir: PVec3::X,
            distance: 0.5,
        };
        let p = offset.d0(0.0);
        let r = (p.x * p.x + p.y * p.y).sqrt();
        assert!((r - 1.5).abs() < 0.02, "expected radius ~1.5, got {}", r);
    }

    #[test]
    fn test_line_dn_d3_is_zero() {
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let d = line.dn(0.5, 3);
        assert_eq!(d.len(), 4); // [d0, d1, d2, d3]
        assert!((d[3] - PVec3::ZERO).length() < 1e-6); // d3 of line = 0
    }

    #[test]
    fn test_circle_d3_matches_recurrence() {
        let circle = CurveGeom::circle(PVec3::ZERO, PVec3::Z, 2.0);
        // d3 of circle: (2π)^3 * r * (+sin * x_dir - cos * y_dir) at t=0
        let d = circle.dn(0.0, 3);
        assert_eq!(d.len(), 4);
        let twopi3 = (2.0 * std::f64::consts::PI).powi(3) * 2.0; // radius=2
        // At t=0: sin(0)=0, cos(0)=1 → d3 = twopi3 * (0*x_dir - 1*y_dir) = -twopi3 * y_dir
        assert!((d[3] - PVec3::new(0.0, -twopi3, 0.0)).length() < 1.0);
    }

    #[test]
    fn test_d012_off_center_circle() {
        // Circle centered at (10, 20, 30), radius=3, in XY plane
        let center = PVec3::new(10.0, 20.0, 30.0);
        let circle = CurveGeom::circle(center, PVec3::Z, 3.0);

        for i in 0..=8 {
            let t = i as Real / 8.0;
            let (d0, d1, d2) = circle.d012(t);
            // d0 should be at distance ~3 from center
            let rel = d0 - center;
            assert!((rel.length() - 3.0).abs() < 1e-4,
                "d012 d0 off-center circle: t={}, dist from center={:.6}, expected 3.0", t, rel.length());
            // d0 must agree with standalone d0()
            assert!((d0 - circle.d0(t)).length() < 1e-10,
                "d012 vs d0 mismatch for off-center circle at t={}", t);
            // d1 and d2 should also agree
            assert!((d1 - circle.d1(t)).length() < 1e-10,
                "d012 d1 mismatch for off-center circle at t={}", t);
            assert!((d2 - circle.d2(t)).length() < 1e-10,
                "d012 d2 mismatch for off-center circle at t={}", t);
        }
    }

    #[test]
    fn test_d012_off_center_ellipse() {
        // Ellipse centered at (5, -2, 1), semi_major=4, semi_minor=1.5, in XY plane
        let center = PVec3::new(5.0, -2.0, 1.0);
        let ellipse = CurveGeom::ellipse(center, PVec3::Z, 4.0, 1.5);

        for i in 0..=8 {
            let t = i as Real / 8.0;
            let (d0, d1, d2) = ellipse.d012(t);
            // d0 must agree with standalone d0()
            assert!((d0 - ellipse.d0(t)).length() < 1e-10,
                "d012 vs d0 mismatch for off-center ellipse at t={}", t);
            // d1 and d2 should also agree
            assert!((d1 - ellipse.d1(t)).length() < 1e-10,
                "d012 d1 mismatch for off-center ellipse at t={}", t);
            assert!((d2 - ellipse.d2(t)).length() < 1e-10,
                "d012 d2 mismatch for off-center ellipse at t={}", t);
            // d0 should stay in XY plane (z = center.z)
            assert!((d0.z - center.z).abs() < 1e-4,
                "d012 d0.z should stay at center.z for XY ellipse, t={}, z={}", t, d0.z);
        }
    }
}

// ── Trigonometric derivative recurrence for circle/ellipse ──────────

fn circle_deriv_n(t: Real, n: usize, x_dir: PVec3, y_dir: PVec3, r: Real, twopi: Real) -> PVec3 {
    let theta = t * twopi;
    let twopi_n = twopi.powi(n as i32);
    // d^n/dt^n [r*cos(θ)*x + r*sin(θ)*y]
    // = r * (2π)^n * [cos(θ+nπ/2)*x + sin(θ+nπ/2)*y]
    let phase = theta + n as Real * std::f64::consts::FRAC_PI_2;
    let scale = r * twopi_n;
    scale * (phase.cos() * x_dir + phase.sin() * y_dir)
}

fn ellipse_deriv_n(t: Real, n: usize, x_dir: PVec3, y_dir: PVec3, a: Real, b: Real, twopi: Real) -> PVec3 {
    let theta = t * twopi;
    let twopi_n = twopi.powi(n as i32);
    let phase = theta + n as Real * std::f64::consts::FRAC_PI_2;
    twopi_n * (a * phase.cos() * x_dir + b * phase.sin() * y_dir)
}
