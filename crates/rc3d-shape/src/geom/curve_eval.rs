//! Curve geometry evaluation (CurveGeom enum and methods).

use rc3d_core::math::Vec3;
use super::bspline::find_span;

// ── Helpers ────────────────────────────────────────────────────

/// Build perpendicular axes (x_dir, y_dir) from a direction vector,
/// forming a right-handed orthonormal basis (x_dir, y_dir, axis).
pub fn build_ortho_axes(axis: Vec3) -> (Vec3, Vec3) {
    let a = axis.normalize();
    let ref_dir = if a.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let y_dir = a.cross(ref_dir).normalize();
    let x_dir = y_dir.cross(a);
    (x_dir, y_dir)
}

/// Orthonormal (u, v) tangent basis for a plane; `u_dir` is projected onto the plane.
pub fn plane_tangent_basis(normal: Vec3, u_dir: Vec3) -> (Vec3, Vec3) {
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

/// Maximum B-spline degree handled with stack-allocated tables.
/// Typical STEP B-splines are degree 2–5, rarely exceeding 8.
const MAX_DEGREE: usize = 16;

/// Flattened triangular table size for degrees 0..=MAX_DEGREE.
/// Sum of (k+1) for k in 0..=MAX_DEGREE = (MAX_DEGREE+1)*(MAX_DEGREE+2)/2
const NDU_SIZE: usize = (MAX_DEGREE + 1) * (MAX_DEGREE + 2) / 2; // 153

/// Index into the flattened `ndu` table: row `k` starts at offset k*(k+1)/2.
#[inline]
fn ndu_idx(k: usize, i: usize) -> usize {
    k * (k + 1) / 2 + i
}

/// Compute d0, d1, d2 for a (possibly rational) B-spline at parameter t.
/// Uses the Cox-de Boor recurrence for basis functions and their derivatives.
///
/// For degree ≤ `MAX_DEGREE` (16), all working tables are stack-allocated.
/// Returns `Vec3::ZERO` for degree > `MAX_DEGREE` (not seen in practice).
fn bspline_d012(
    degree: usize,
    control_points: &[Vec3],
    knots: &[f32],
    weights: Option<&[f32]>,
    t: f32,
) -> (Vec3, Vec3, Vec3) {
    let p = degree;

    // Guard: insufficient data
    if control_points.len() < p + 1 || knots.len() < 2 * (p + 1) {
        return (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    }

    // Clamp t to valid knot domain [knots[p], knots[control_points.len()]]
    let t_min = knots[p];
    let t_max = knots[control_points.len()];
    let t = t.clamp(t_min, t_max);

    // Degree 0 special case
    if p == 0 {
        let span = find_span(0, knots, t);
        let pt = control_points[span];
        return (pt, Vec3::ZERO, Vec3::ZERO);
    }

    // STEP files never exceed degree ~8; MAX_DEGREE=16 is well above that.
    // Return zero for pathological degrees rather than maintaining a duplicate
    // heap-allocated code path.
    if p > MAX_DEGREE {
        return (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    }

    let span = find_span(p, knots, t);
    let s = span;

    // ── Basis functions (stack-allocated triangular table) ──
    // ndu[k*(k+1)/2 + i] = N_{s-k+i, k}(t) for i = 0..k
    let mut ndu = [0.0f32; NDU_SIZE];
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
    let mut ndu1 = [0.0f32; MAX_DEGREE + 1];
    for k in 0..=p {
        let idx = s + k - p;
        let left = if k >= 1 {
            let denom = knots[idx + p] - knots[idx];
            if denom > 1e-10 {
                (p as f32) / denom * ndu[ndu_idx(p - 1, k - 1)]
            } else {
                0.0
            }
        } else {
            0.0
        };
        let right = if k < p {
            let denom = knots[idx + p + 1] - knots[idx + 1];
            if denom > 1e-10 {
                (p as f32) / denom * ndu[ndu_idx(p - 1, k)]
            } else {
                0.0
            }
        } else {
            0.0
        };
        ndu1[k] = left - right;
    }

    // ── Second derivatives N''_{s-p+k, p} ──
    let mut ndu2 = [0.0f32; MAX_DEGREE + 1];
    if p >= 2 {
        // First compute N'_{s-(p-1)+k, p-1} for k = 0..p-1
        let mut ndu1_pm1 = [0.0f32; MAX_DEGREE];
        for k in 0..p {
            let idx = s + k - (p - 1);
            let left = if k >= 1 {
                let denom = knots[idx + p - 1] - knots[idx];
                if denom > 1e-10 {
                    ((p - 1) as f32) / denom * ndu[ndu_idx(p - 2, k - 1)]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let right = if k < p - 1 {
                let denom = knots[idx + p] - knots[idx + 1];
                if denom > 1e-10 {
                    ((p - 1) as f32) / denom * ndu[ndu_idx(p - 2, k)]
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
                    (p as f32) / denom * ndu1_pm1[k - 1]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let right = if k < p {
                let denom = knots[idx + p + 1] - knots[idx + 1];
                if denom > 1e-10 {
                    (p as f32) / denom * ndu1_pm1[k]
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
    let mut a0 = Vec3::ZERO;
    let mut a1 = Vec3::ZERO;
    let mut a2 = Vec3::ZERO;
    let mut w_sum = 0.0f32;
    let mut w_sum_1 = 0.0f32;
    let mut w_sum_2 = 0.0f32;

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

    // NaN safety: replace any NaN component with zero
    let safe = |v: Vec3| -> Vec3 {
        if v.is_nan() { Vec3::ZERO } else { v }
    };
    (safe(d0), safe(d1), safe(d2))
}

/// Approximate arc length of a curve by chordal sum with fixed sampling.
/// Avoids calling `arc_length` / `sample_adaptive` to break circular dependency
/// with `Composite` segment selection.
pub fn approx_chordal_length(curve: &CurveGeom) -> f32 {
    match curve {
        CurveGeom::Composite { segments, .. } => {
            segments.iter().map(|(seg, _)| approx_chordal_length(seg)).sum()
        }
        _ => {
            const N: usize = 32;
            let mut len = 0.0;
            let mut prev = curve.d0(0.0);
            for i in 1..=N {
                let t = i as f32 / N as f32;
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
    cached_lengths: &Option<Vec<f32>>,
    t: f32,
) -> Option<(usize, f32, f32)> {
    if segments.is_empty() {
        return None;
    }
    if segments.len() == 1 {
        return Some((0, t, 1.0));
    }

    // Use cached lengths if available and valid, otherwise compute
    let owned_lengths: Option<Vec<f32>> = match cached_lengths {
        Some(c) if c.len() == segments.len() => None, // use cache directly
        _ => Some(segments.iter().map(|(seg, _)| approx_chordal_length(seg).max(1e-10)).collect()),
    };
    let lengths: &[f32] = match &owned_lengths {
        Some(computed) => computed,
        None => cached_lengths.as_deref().unwrap(),
    };
    let total: f32 = lengths.iter().sum();
    let inv_total = 1.0 / total.max(1e-10);

    let target = t.clamp(0.0, 1.0);
    let mut cumulative = 0.0f32;
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
    Line { origin: Vec3, direction: Vec3 },
    Circle { center: Vec3, axis: Vec3, radius: f32, x_dir: Vec3, y_dir: Vec3 },
    Ellipse { center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32, x_dir: Vec3, y_dir: Vec3 },
    Hyperbola { center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32, x_dir: Vec3, y_dir: Vec3 },
    Parabola { center: Vec3, axis: Vec3, focal_dist: f32, x_dir: Vec3, y_dir: Vec3 },
    BSpline { degree: usize, control_points: Vec<Vec3>, knots: Vec<f32>, weights: Option<Vec<f32>> },
    Trimmed { basis: Box<CurveGeom>, t_min: f32, t_max: f32 },
    Composite { segments: Vec<(CurveGeom, bool)>, cached_lengths: Option<Vec<f32>> },
    Polyline { points: Vec<Vec3> },
    /// Offset curve at signed distance from basis curve.
    /// Points are computed as basis(t) + distance * normal_dir(t)
    /// where normal_dir is offset_dir projected onto the curve normal plane.
    Offset {
        basis: Box<CurveGeom>,
        offset_dir: Vec3,
        distance: f32,
    },
}

impl CurveGeom {
    /// Construct a Circle with pre-computed ortho axes.
    pub fn circle(center: Vec3, axis: Vec3, radius: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Circle { center, axis, radius, x_dir, y_dir }
    }
    /// Construct an Ellipse with pre-computed ortho axes.
    pub fn ellipse(center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir }
    }
    /// Construct a Hyperbola with pre-computed ortho axes.
    pub fn hyperbola(center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Hyperbola { center, axis, semi_major, semi_minor, x_dir, y_dir }
    }
    /// Construct a Parabola with pre-computed ortho axes.
    pub fn parabola(center: Vec3, axis: Vec3, focal_dist: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Parabola { center, axis, focal_dist, x_dir, y_dir }
    }
}

/// Map edge parameter t in [0,1] to the basis curve parameter used by `d0`/`d1`.
fn trimmed_edge_to_basis(basis: &CurveGeom, t_min: f32, t_max: f32, t: f32) -> (f32, f32) {
    let span = (t_max - t_min).max(1e-12);
    let t_mapped = t_min + t * span;
    match basis {
        CurveGeom::Circle { .. } | CurveGeom::Ellipse { .. } => {
            (t_mapped / std::f32::consts::TAU, span / std::f32::consts::TAU)
        }
        _ => (t_mapped, span),
    }
}

impl CurveGeom {
    /// Native parameter interval for curve evaluation (STEP knot domain when applicable).
    pub fn native_param_range(&self) -> (f32, f32) {
        match self {
            CurveGeom::Trimmed { t_min, t_max, .. } => (*t_min, *t_max),
            CurveGeom::BSpline { degree, control_points, knots, .. } => {
                let n = control_points.len();
                if knots.len() >= n + degree + 1 {
                    (knots[*degree], knots[n])
                } else {
                    (0.0, 1.0)
                }
            }
            _ => (0.0, 1.0),
        }
    }

    /// Evaluate position at parameter t ∈ [0, 1].
    pub fn d0(&self, t: f32) -> Vec3 {
        match self {
            CurveGeom::Line { origin, direction } => *origin + *direction * t,

            CurveGeom::Circle { center, x_dir, y_dir, radius, .. } => {
                let theta = t * std::f32::consts::TAU;
                *center + *x_dir * radius * theta.cos() + *y_dir * radius * theta.sin()
            }

            CurveGeom::Ellipse { center, x_dir, y_dir, semi_major, semi_minor, .. } => {
                let theta = t * std::f32::consts::TAU;
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
                    Vec3::ZERO
                }
            }

            CurveGeom::Polyline { points } => {
                if points.len() < 2 {
                    return points.first().copied().unwrap_or(Vec3::ZERO);
                }
                let n = points.len() - 1;
                let t_scaled = t.clamp(0.0, 1.0) * n as f32;
                let idx = (t_scaled as usize).min(n - 1);
                let frac = t_scaled - idx as f32;
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
    pub fn d1(&self, t: f32) -> Vec3 {
        match self {
            CurveGeom::Line { direction, .. } => *direction,

            CurveGeom::Circle { x_dir, y_dir, radius, .. } => {
                let theta = t * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
                twopi * radius * (-theta.sin() * *x_dir + theta.cos() * *y_dir)
            }

            CurveGeom::Ellipse { x_dir, y_dir, semi_major, semi_minor, .. } => {
                let theta = t * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
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
                    Vec3::ZERO
                }
            }

            CurveGeom::Polyline { points } => {
                let n = points.len();
                if n < 2 {
                    return Vec3::ZERO;
                }
                let n_seg = n - 1;
                let t_scaled = t.clamp(0.0, 1.0) * n_seg as f32;
                let idx = (t_scaled as usize).min(n_seg - 1);
                if idx >= n_seg {
                    Vec3::ZERO
                } else {
                    (points[idx + 1] - points[idx]) * n_seg as f32
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
    pub fn d2(&self, t: f32) -> Vec3 {
        match self {
            CurveGeom::Line { .. } => Vec3::ZERO,

            CurveGeom::Circle { x_dir, y_dir, radius, .. } => {
                let theta = t * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
                -(twopi * twopi) * radius * (theta.cos() * *x_dir + theta.sin() * *y_dir)
            }

            CurveGeom::Ellipse { x_dir, y_dir, semi_major, semi_minor, .. } => {
                let theta = t * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
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
                    Vec3::ZERO
                }
            }

            CurveGeom::Polyline { .. } => Vec3::ZERO,

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
    pub fn d3(&self, t: f32) -> Vec3 {
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
            return Vec3::ZERO;
        }
        (-p0 + 2.0 * p1 - 2.0 * p2 + p3) / h3
    }

    /// Curvature κ = |d1 × d2| / |d1|³ at parameter t.
    /// Returns 0.0 if the first derivative magnitude is below 1e-10.
    pub fn curvature(&self, t: f32) -> f32 {
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
    pub fn torsion(&self, t: f32) -> f32 {
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

    /// Combined position, first, and second derivative in one call.
    /// For BSpline curves, this avoids 3× redundant Cox-de Boor evaluation
    /// vs calling `d0`, `d1`, `d2` separately.
    pub fn d012(&self, t: f32) -> (Vec3, Vec3, Vec3) {
        match self {
            CurveGeom::BSpline { degree, control_points, knots, weights } => {
                bspline_d012(*degree, control_points, knots, weights.as_deref(), t)
            }
            CurveGeom::Circle { x_dir, y_dir, radius, .. } => {
                let theta = t * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
                let (c, s) = (theta.cos(), theta.sin());
                let d0 = *x_dir * (*radius * c) + *y_dir * (*radius * s);
                let d1 = twopi * *radius * (-s * *x_dir + c * *y_dir);
                let d2 = -(twopi * twopi) * *radius * (c * *x_dir + s * *y_dir);
                (d0, d1, d2)
            }
            CurveGeom::Ellipse { x_dir, y_dir, semi_major, semi_minor, .. } => {
                let theta = t * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
                let (c, s) = (theta.cos(), theta.sin());
                let d0 = *x_dir * (*semi_major * c) + *y_dir * (*semi_minor * s);
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
                (*origin + *direction * t, *direction, Vec3::ZERO)
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
                    (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO)
                }
            }
            CurveGeom::Polyline { .. } => (self.d0(t), self.d1(t), self.d2(t)),

            CurveGeom::Offset { .. } => (self.d0(t), self.d1(t), self.d2(t)),
        }
    }

    /// Adaptive chordal arc length between t0 and t1.
    /// Uses `sample_adaptive` and sums chord lengths of the result points.
    pub fn arc_length(&self, t0: f32, t1: f32) -> f32 {
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
    pub fn sample_adaptive(&self, t0: f32, t1: f32, tolerance: f32) -> Vec<(f32, Vec3)> {
        const MAX_POINTS: usize = 512;

        // Seed with 8 uniform samples
        let mut points: Vec<(f32, Vec3)> = Vec::with_capacity(MAX_POINTS);
        let initial = 8usize;
        for i in 0..=initial {
            let ti = t0 + (t1 - t0) * i as f32 / initial as f32;
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
    curve: &CurveGeom, p_lo: Vec3, p_hi: Vec3,
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
    center: &Vec3, _axis: Vec3, radius: f32, x_dir: Vec3, y_dir: Vec3, point: Vec3,
) -> Option<f32> {
    let rel = point - *center;
    let a = _axis.normalize();
    let proj = rel - a * rel.dot(a);
    let dist = proj.length();
    if (dist - radius).abs() > radius * 0.1 && (dist - radius).abs() > 0.5 { return None; }
    let u = f32::atan2(proj.dot(y_dir), proj.dot(x_dir));
    Some(if u < 0.0 { u + std::f32::consts::TAU } else { u })
}

/// Compute eccentric anomaly angle [0,TAU) of a point on an ellipse.
///
/// Unlike `circle_angle_geom`, this correctly handles high-eccentricity ellipses
/// (e.g. a=10, b=1) by computing θ = atan2(y/b, x/a) instead of assuming
/// constant distance from center.
fn ellipse_angle_geom(
    center: &Vec3, axis: Vec3, semi_major: f32, semi_minor: f32,
    x_dir: Vec3, y_dir: Vec3, point: Vec3,
) -> Option<f32> {
    let a = axis.normalize();
    let rel = point - *center;
    let proj = rel - a * rel.dot(a);
    let u = f32::atan2(proj.dot(y_dir) / semi_minor, proj.dot(x_dir) / semi_major);
    Some(if u < 0.0 { u + std::f32::consts::TAU } else { u })
}

/// Clamp arc params to [0,TAU) with correct t_min < t_max ordering.
fn normalize_arc_params(a0: f32, a1: f32) -> (f32, f32) {
    let diff = a1 - a0;
    if diff.abs() > std::f32::consts::PI {
        if a0 < a1 { (a1 - std::f32::consts::TAU, a0) }
        else { (a0 - std::f32::consts::TAU, a1) }
    } else { (a0.min(a1), a0.max(a1)) }
}

/// Find the parameter t∈[0,1] on a curve closest to the target point.
///
/// Uses uniform sampling followed by iterative step-halving refinement.
/// `n_samples` controls the initial grid resolution; `refine_iters` controls
/// the number of refinement passes.
pub fn find_param_on_curve(curve: &CurveGeom, target: Vec3, _n_samples: usize, _refine_iters: usize) -> f32 {
    let results = super::project::project_point_on_curve(curve, target);
    results.first().map(|(t, _)| *t).unwrap_or(0.0)
}

/// STEP `LINE` entities often reference a unit `VECTOR`; the actual edge span is
/// defined by `VERTEX_POINT` coordinates, not vector magnitude.
pub fn normalize_edge_curve_to_vertices(
    curve: CurveGeom,
    p_lo: Vec3,
    p_hi: Vec3,
    tol: f32,
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
    let t_lo = find_param_on_curve(&curve, p_lo, 24, 3);
    let t_hi = find_param_on_curve(&curve, p_hi, 24, 3);
    if (curve.d0(t_lo) - p_lo).length() <= match_tol && (curve.d0(t_hi) - p_hi).length() <= match_tol {
        let (t_min, t_max) = (t_lo.min(t_hi), t_lo.max(t_hi));
        return CurveGeom::Trimmed { basis: Box::new(curve), t_min, t_max };
    }

    // Keep original geometry — PCurve on each face provides correct surface trajectory.
    // Replacing with a straight line destroys geometric fidelity.
    let c0_err = (curve.d0(0.0) - p_lo).length();
    let c1_err = (curve.d0(1.0) - p_hi).length();
    if c0_err > match_tol || c1_err > match_tol {
        log::warn!("edge curve mismatch: endpoints off by {:.4}/{:.4} (tol={:.4}, len={:.4})",
            c0_err, c1_err, match_tol, len);
    }
    curve
}

/// Signed area of a 2D polygon (shoelace formula). Returns positive for CCW winding.
pub fn signed_area_2d(uv: &[(f32, f32)]) -> f64 {
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
pub fn eval_pcurve_on_surface(pcurve: &CurveGeom, surface: &super::SurfaceGeom, t: f32) -> Vec3 {
    let uv = pcurve.d0(t);
    surface.d0_native(uv.x, uv.y)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CurveGeom tests ────────────────────────────────

    #[test]
    fn test_line_d0() {
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let p = line.d0(0.5);
        assert!((p - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_line_d1_is_direction() {
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::new(3.0, 4.0, 0.0) };
        let d = line.d1(0.5);
        assert!((d - Vec3::new(3.0, 4.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_circle_curvature() {
        let circle = CurveGeom::circle(Vec3::ZERO, Vec3::Z, 2.0);
        let k = circle.curvature(0.25);
        assert!((k - 0.5).abs() < 1e-4); // curvature = 1/r
    }

    #[test]
    fn test_circle_d1_orthogonal_to_radius() {
        let circle = CurveGeom::circle(Vec3::ZERO, Vec3::Z, 1.0);
        let pos = circle.d0(0.0); // at angle 0 -> (1,0,0)
        let tan = circle.d1(0.0);
        // tangent at angle 0 should be (0, 2pi, 0) -- orthogonal to radius
        assert!(pos.dot(tan).abs() < 1e-6);
    }

    #[test]
    fn test_sample_adaptive_line_minimal() {
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::new(10.0, 0.0, 0.0) };
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
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        // Endpoint interpolation (clamped B-spline)
        let (d0_start, _, _) = bspline_d012(3, &cps, &knots, None, 0.0);
        let (d0_end, _, _) = bspline_d012(3, &cps, &knots, None, 1.0);
        assert!((d0_start - cps[0]).length() < 1e-5, "start endpoint: {d0_start:?}");
        assert!((d0_end - cps[3]).length() < 1e-5, "end endpoint: {d0_end:?}");

        // Curve stays on X axis (y=0, z=0) for all t
        for i in 0..=10 {
            let t = i as f32 / 10.0;
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
        let cps: Vec<Vec3> = (0..9).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
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
            let t = i as f32 / 10.0;
            let (d0, d1, _d2) = bspline_d012(8, &cps, &knots, None, t);
            assert!(d0.y.abs() < 1e-4 && d0.z.abs() < 1e-4, "off-axis at t={t}: {d0:?}");
            assert!(d1.y.abs() < 1e-3 && d1.z.abs() < 1e-3, "tangent off-axis at t={t}: {d1:?}");
        }
    }

    #[test]
    fn test_ellipse_trim_eccentric() {
        // High-eccentricity ellipse: a=10, b=1.  The old circle_angle_geom
        // rejects valid points because distance from center varies from 1 to 10.
        let center = Vec3::ZERO;
        let axis = Vec3::Z;
        let a = 10.0f32;
        let b = 1.0f32;
        let ellipse = CurveGeom::ellipse(center, axis, a, b);
        // Point at eccentric anomaly θ=0:  (a, 0, 0)
        let p_lo = Vec3::new(a, 0.0, 0.0);
        // Point at eccentric anomaly θ=π/2:  (0, b, 0)
        let p_hi = Vec3::new(0.0, b, 0.0);

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
        let seg1 = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let seg2 = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::new(2.0, 0.0, 0.0) };
        let segments = vec![(seg1, false), (seg2, false)];

        // Precompute cached lengths (as build_curve does)
        let cached: Vec<f32> = segments.iter()
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
        assert!((p1 - Vec3::new(2.0, 0.0, 0.0)).length() < 0.1, "d0(1) ≈ (2,0,0), got {:?}", p1);

        // d0 at t=1/3 should be near the junction (end of seg1 = (1,0,0))
        let p_third = comp.d0(1.0 / 3.0);
        assert!((p_third - Vec3::new(1.0, 0.0, 0.0)).length() < 0.1,
            "d0(1/3) ≈ junction (1,0,0), got {:?}", p_third);

        // Also test Composite without cache (cached_lengths: None) still works
        let seg3 = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let seg4 = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::Y };
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
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
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
            center: Vec3::ZERO, axis: Vec3::Z,
            semi_major: 2.0, semi_minor: 1.0,
            x_dir: Vec3::X, y_dir: Vec3::Y,
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
            center: Vec3::ZERO, axis: Vec3::Z,
            focal_dist: 1.0,
            x_dir: Vec3::X, y_dir: Vec3::Y,
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
        let h = CurveGeom::hyperbola(Vec3::new(1.0, 2.0, 3.0), Vec3::Z, 3.0, 1.5);
        // At t=0.5 (s=0), position should be center + semi_major * x_dir
        let p = h.d0(0.5);
        assert!((p.x - 4.0).abs() < 1e-5, "expected x=4.0, got {}", p.x);
        assert!((p.y - 2.0).abs() < 1e-5);
        assert!((p.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_parabola_constructor() {
        let p = CurveGeom::parabola(Vec3::new(1.0, 0.0, 0.0), Vec3::Z, 2.0);
        // At t=0.5 (s=0), position should be center
        let pt = p.d0(0.5);
        assert!((pt.x - 1.0).abs() < 1e-5);
        assert!(pt.y.abs() < 1e-5);
    }

    #[test]
    fn test_offset_curve_line() {
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let offset = CurveGeom::Offset {
            basis: Box::new(line),
            offset_dir: Vec3::Y,
            distance: 1.0,
        };
        let p = offset.d0(0.5);
        assert!((p.x - 0.5).abs() < 1e-6, "x={}", p.x);
        assert!((p.y - 1.0).abs() < 0.01, "y={}", p.y);
        assert!(p.z.abs() < 1e-6, "z={}", p.z);
    }

    #[test]
    fn test_offset_curve_circle() {
        let circle = CurveGeom::circle(Vec3::ZERO, Vec3::Z, 1.0);
        let offset = CurveGeom::Offset {
            basis: Box::new(circle),
            offset_dir: Vec3::X,
            distance: 0.5,
        };
        let p = offset.d0(0.0);
        let r = (p.x * p.x + p.y * p.y).sqrt();
        assert!((r - 1.5).abs() < 0.02, "expected radius ~1.5, got {}", r);
    }
}
