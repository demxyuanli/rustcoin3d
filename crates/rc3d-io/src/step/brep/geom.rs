//! Unified curve and surface geometry.
//! T1.1: CurveGeom  T1.2: SurfaceGeom

use rc3d_core::math::Vec3;
use crate::step::geom::find_span;

// ── Helpers ────────────────────────────────────────────────────

/// Build perpendicular axes (x_dir, y_dir) from a direction vector,
/// forming a right-handed orthonormal basis (x_dir, y_dir, axis).
fn build_ortho_axes(axis: Vec3) -> (Vec3, Vec3) {
    let a = axis.normalize();
    let ref_dir = if a.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let x_dir = ref_dir.cross(a).normalize();
    let y_dir = a.cross(x_dir);
    (x_dir, y_dir)
}

/// Compute d0, d1, d2 for a (possibly rational) B-spline at parameter t.
/// Uses the Cox-de Boor recurrence for basis functions and their derivatives.
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

    // Degree 0 special case
    if p == 0 {
        let span = find_span(0, knots, t);
        let pt = control_points[span];
        return (pt, Vec3::ZERO, Vec3::ZERO);
    }

    let span = find_span(p, knots, t);
    let s = span;

    // ── Basis functions: ndu[k][i] = N_{s-k+i, k}(t) for i = 0..k ──
    let mut ndu: Vec<Vec<f32>> = (0..=p).map(|k| vec![0.0f32; k + 1]).collect();
    ndu[0][0] = 1.0;

    for k in 1..=p {
        for i in 0..=k {
            let ctrl_idx = s + i - k;

            let left = if i >= 1 {
                let denom = knots[ctrl_idx + k] - knots[ctrl_idx];
                if denom > 1e-10 {
                    (t - knots[ctrl_idx]) / denom * ndu[k - 1][i - 1]
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let right = if i < k {
                let denom = knots[ctrl_idx + k + 1] - knots[ctrl_idx + 1];
                if denom > 1e-10 {
                    (knots[ctrl_idx + k + 1] - t) / denom * ndu[k - 1][i]
                } else {
                    0.0
                }
            } else {
                0.0
            };

            ndu[k][i] = left + right;
        }
    }

    // ── First derivatives N'_{s-p+k, p} ──
    // N'_{s-p+k, p} = p/denom_L * ndu[p-1][k-1] - p/denom_R * ndu[p-1][k]
    let mut ndu1: Vec<f32> = vec![0.0; p + 1];
    for k in 0..=p {
        let idx = s + k - p;
        let left = if k >= 1 {
            let denom = knots[idx + p] - knots[idx];
            if denom > 1e-10 {
                (p as f32) / denom * ndu[p - 1][k - 1]
            } else {
                0.0
            }
        } else {
            0.0
        };
        let right = if k < p {
            let denom = knots[idx + p + 1] - knots[idx + 1];
            if denom > 1e-10 {
                (p as f32) / denom * ndu[p - 1][k]
            } else {
                0.0
            }
        } else {
            0.0
        };
        ndu1[k] = left - right;
    }

    // ── Second derivatives N''_{s-p+k, p} ──
    let mut ndu2: Vec<f32> = vec![0.0; p + 1];
    if p >= 2 {
        // First compute N'_{s-(p-1)+k, p-1} for k = 0..p-1
        let mut ndu1_pm1: Vec<f32> = vec![0.0; p];
        for k in 0..p {
            let idx = s + k - (p - 1);
            let left = if k >= 1 {
                let denom = knots[idx + p - 1] - knots[idx];
                if denom > 1e-10 {
                    ((p - 1) as f32) / denom * ndu[p - 2][k - 1]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let right = if k < p - 1 {
                let denom = knots[idx + p] - knots[idx + 1];
                if denom > 1e-10 {
                    ((p - 1) as f32) / denom * ndu[p - 2][k]
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

        let n0 = ndu[p][k];
        let n1 = ndu1[k];
        let n2 = ndu2[k];

        a0 += cp * (n0 * w);
        a1 += cp * (n1 * w);
        a2 += cp * (n2 * w);
        w_sum += n0 * w;
        w_sum_1 += n1 * w;
        w_sum_2 += n2 * w;
    }

    if weights.is_some() {
        // Rational: C(t) = A(t) / w(t)
        // C' = (A'w - Aw') / w²
        // C'' = (A''w² - Aw''w - 2A'w'w + 2Aw'²) / w³
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
    }
}

/// Approximate arc length of a curve by chordal sum with fixed sampling.
/// Avoids calling `arc_length` / `sample_adaptive` to break circular dependency
/// with `Composite` segment selection.
fn approx_chordal_length(curve: &CurveGeom) -> f32 {
    match curve {
        CurveGeom::Composite { segments } => {
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
fn find_composite_segment(segments: &[(CurveGeom, bool)], t: f32) -> Option<(usize, f32, f32)> {
    if segments.is_empty() {
        return None;
    }
    if segments.len() == 1 {
        return Some((0, t, 1.0));
    }

    // Compute approximate lengths for each segment
    let lengths: Vec<f32> = segments.iter().map(|(seg, _)| approx_chordal_length(seg).max(1e-10)).collect();
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
    Circle { center: Vec3, axis: Vec3, radius: f32 },
    Ellipse { center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32 },
    BSpline { degree: usize, control_points: Vec<Vec3>, knots: Vec<f32>, weights: Option<Vec<f32>> },
    Trimmed { basis: Box<CurveGeom>, t_min: f32, t_max: f32 },
    Composite { segments: Vec<(CurveGeom, bool)> },
    Polyline { points: Vec<Vec3> },
}

impl CurveGeom {
    /// Evaluate position at parameter t ∈ [0, 1].
    pub fn d0(&self, t: f32) -> Vec3 {
        match self {
            CurveGeom::Line { origin, direction } => *origin + *direction * t,

            CurveGeom::Circle { center, axis, radius } => {
                let theta = t * std::f32::consts::TAU;
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                *center + x_dir * radius * theta.cos() + y_dir * radius * theta.sin()
            }

            CurveGeom::Ellipse { center, axis, semi_major, semi_minor } => {
                let theta = t * std::f32::consts::TAU;
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                *center + x_dir * semi_major * theta.cos() + y_dir * semi_minor * theta.sin()
            }

            CurveGeom::BSpline { degree, control_points, knots, weights } => {
                bspline_d012(*degree, control_points, knots, weights.as_deref(), t).0
            }

            CurveGeom::Trimmed { basis, t_min, t_max } => {
                let t_mapped = *t_min + t * (*t_max - *t_min);
                basis.d0(t_mapped)
            }

            CurveGeom::Composite { segments } => {
                if let Some((idx, t_local, _)) = find_composite_segment(segments, t) {
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
        }
    }

    /// Evaluate first derivative (tangent) at parameter t ∈ [0, 1].
    pub fn d1(&self, t: f32) -> Vec3 {
        match self {
            CurveGeom::Line { direction, .. } => *direction,

            CurveGeom::Circle { axis, radius, .. } => {
                let theta = t * std::f32::consts::TAU;
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let twopi = std::f32::consts::TAU;
                twopi * radius * (-theta.sin() * x_dir + theta.cos() * y_dir)
            }

            CurveGeom::Ellipse { axis, semi_major, semi_minor, .. } => {
                let theta = t * std::f32::consts::TAU;
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let twopi = std::f32::consts::TAU;
                twopi * (-semi_major * theta.sin() * x_dir + semi_minor * theta.cos() * y_dir)
            }

            CurveGeom::BSpline { degree, control_points, knots, weights } => {
                bspline_d012(*degree, control_points, knots, weights.as_deref(), t).1
            }

            CurveGeom::Trimmed { basis, t_min, t_max } => {
                let t_mapped = *t_min + t * (*t_max - *t_min);
                basis.d1(t_mapped) * (*t_max - *t_min)
            }

            CurveGeom::Composite { segments } => {
                if let Some((idx, t_local, seg_width)) = find_composite_segment(segments, t) {
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
        }
    }

    /// Evaluate second derivative at parameter t ∈ [0, 1].
    pub fn d2(&self, t: f32) -> Vec3 {
        match self {
            CurveGeom::Line { .. } => Vec3::ZERO,

            CurveGeom::Circle { axis, radius, .. } => {
                let theta = t * std::f32::consts::TAU;
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let twopi = std::f32::consts::TAU;
                -(twopi * twopi) * radius * (theta.cos() * x_dir + theta.sin() * y_dir)
            }

            CurveGeom::Ellipse { axis, semi_major, semi_minor, .. } => {
                let theta = t * std::f32::consts::TAU;
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let twopi = std::f32::consts::TAU;
                -(twopi * twopi) * (semi_major * theta.cos() * x_dir + semi_minor * theta.sin() * y_dir)
            }

            CurveGeom::BSpline { degree, control_points, knots, weights } => {
                bspline_d012(*degree, control_points, knots, weights.as_deref(), t).2
            }

            CurveGeom::Trimmed { basis, t_min, t_max } => {
                let t_mapped = *t_min + t * (*t_max - *t_min);
                let scale = *t_max - *t_min;
                basis.d2(t_mapped) * (scale * scale)
            }

            CurveGeom::Composite { segments } => {
                if let Some((idx, t_local, seg_width)) = find_composite_segment(segments, t) {
                    let (seg, reversed) = &segments[idx];
                    let t_eval = if *reversed { 1.0 - t_local } else { t_local };
                    // d² is divided by seg_width² (squared is always positive, reversal doesn't matter)
                    seg.d2(t_eval) / (seg_width * seg_width)
                } else {
                    Vec3::ZERO
                }
            }

            CurveGeom::Polyline { .. } => Vec3::ZERO,
        }
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

// ── SurfaceGeom ────────────────────────────────────────────

use crate::step::nurbs::NurbsSurface;

/// Parametric surface geometry (retained, not converted to NURBS).
#[derive(Debug, Clone)]
pub enum SurfaceGeom {
    Plane { origin: Vec3, normal: Vec3, u_dir: Vec3 },
    Cylinder { origin: Vec3, axis: Vec3, radius: f32 },
    Cone { apex: Vec3, axis: Vec3, semi_angle: f32, radius_at_apex: f32 },
    Sphere { center: Vec3, radius: f32 },
    Torus { center: Vec3, axis: Vec3, major_r: f32, minor_r: f32 },
    BSpline(NurbsSurface),
    Extrusion { generatrix: Box<CurveGeom>, direction: Vec3 },
    Revolution { generatrix: Box<CurveGeom>, axis_origin: Vec3, axis_dir: Vec3 },
    Offset { basis: Box<SurfaceGeom>, distance: f32 },
}

impl SurfaceGeom {
    pub fn d0(&self, _u: f32, _v: f32) -> Vec3 { todo!("T1.2") }
    pub fn d1(&self, _u: f32, _v: f32) -> (Vec3, Vec3) { todo!("T1.2") }
    pub fn normal(&self, _u: f32, _v: f32) -> Vec3 { todo!("T1.2") }
    pub fn project(&self, _point: Vec3) -> Option<(f32, f32)> { todo!("T1.2") }
    pub fn evaluate_grid(&self, _u_range: (f32,f32), _v_range: (f32,f32), _n_u: usize, _n_v: usize) -> Vec<Vec<Vec3>> { todo!("T1.2") }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        let circle = CurveGeom::Circle { center: Vec3::ZERO, axis: Vec3::Z, radius: 2.0 };
        let k = circle.curvature(0.25);
        assert!((k - 0.5).abs() < 1e-4); // curvature = 1/r
    }

    #[test]
    fn test_circle_d1_orthogonal_to_radius() {
        let circle = CurveGeom::Circle { center: Vec3::ZERO, axis: Vec3::Z, radius: 1.0 };
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
}
