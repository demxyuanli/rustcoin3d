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
    let y_dir = a.cross(ref_dir).normalize();
    let x_dir = y_dir.cross(a);
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

/// Rodrigues rotation: rotate point `p` around the line through `origin` along `axis`
/// by `angle` radians.  `axis` must be unit-length.
fn rotate_around_axis(p: Vec3, origin: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let rel = p - origin;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let dot = axis.dot(rel);
    let rot = rel * cos_a + axis.cross(rel) * sin_a + axis * dot * (1.0 - cos_a);
    origin + rot
}

/// Map a normalized parameter t ∈ [0, 1] into the B-spline knot-domain interval
/// [knots[degree], knots[count]].
fn map_to_knot_domain(knots: &[f32], degree: usize, count: usize, t: f32) -> f32 {
    let t_min = knots[degree];
    let t_max = knots[count];
    t_min + t * (t_max - t_min)
}

/// Width of the B-spline knot-domain interval.
fn knot_domain_width(knots: &[f32], degree: usize, count: usize) -> f32 {
    knots[count] - knots[degree]
}

impl SurfaceGeom {
    /// Evaluate position at parameter (u, v) ∈ [0, 1]^2.
    pub fn d0(&self, u: f32, v: f32) -> Vec3 {
        match self {
            SurfaceGeom::Plane { origin, normal, u_dir } => {
                let v_dir = normal.cross(*u_dir);
                *origin + *u_dir * u + v_dir * v
            }
            SurfaceGeom::Cylinder { origin, axis, radius } => {
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let theta = u * std::f32::consts::TAU;
                let r = *radius;
                *origin
                    + x_dir * r * theta.cos()
                    + y_dir * r * theta.sin()
                    + *axis * v
            }
            SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex } => {
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let theta = u * std::f32::consts::TAU;
                let r = *radius_at_apex + v * semi_angle.tan();
                *apex
                    + x_dir * r * theta.cos()
                    + y_dir * r * theta.sin()
                    + *axis * v
            }
            SurfaceGeom::Sphere { center, radius } => {
                let theta = u * std::f32::consts::TAU; // azimuth
                let phi = v * std::f32::consts::PI; // polar
                let r = *radius;
                *center
                    + r * Vec3::new(
                        phi.sin() * theta.cos(),
                        phi.sin() * theta.sin(),
                        phi.cos(),
                    )
            }
            SurfaceGeom::Torus { center, axis, major_r, minor_r } => {
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let theta = u * std::f32::consts::TAU; // major angle
                let phi = v * std::f32::consts::TAU; // minor angle
                let r = *major_r + *minor_r * phi.cos();
                *center
                    + x_dir * r * theta.cos()
                    + y_dir * r * theta.sin()
                    + *axis * *minor_r * phi.sin()
            }
            SurfaceGeom::BSpline(nurbs) => {
                let u_k = map_to_knot_domain(
                    &nurbs.knots_u, nurbs.degree_u, nurbs.u_count(), u,
                );
                let v_k = map_to_knot_domain(
                    &nurbs.knots_v, nurbs.degree_v, nurbs.v_count(), v,
                );
                nurbs.evaluate(u_k, v_k)
            }
            SurfaceGeom::Extrusion { generatrix, direction } => {
                generatrix.d0(u) + *direction * v
            }
            SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
                let axis = axis_dir.normalize();
                let angle = v * std::f32::consts::TAU;
                rotate_around_axis(generatrix.d0(u), *axis_origin, axis, angle)
            }
            SurfaceGeom::Offset { basis, distance } => {
                basis.d0(u, v) + basis.normal(u, v) * *distance
            }
        }
    }

    /// Evaluate first-order partial derivatives (∂S/∂u, ∂S/∂v).
    pub fn d1(&self, u: f32, v: f32) -> (Vec3, Vec3) {
        match self {
            SurfaceGeom::Plane { normal, u_dir, .. } => {
                let v_dir = normal.cross(*u_dir);
                (*u_dir, v_dir)
            }
            SurfaceGeom::Cylinder { axis, radius, .. } => {
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let r = *radius;
                let theta = u * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
                let du = twopi * r * (-theta.sin() * x_dir + theta.cos() * y_dir);
                (du, *axis)
            }
            SurfaceGeom::Cone { axis, semi_angle, radius_at_apex, .. } => {
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let tan_a = semi_angle.tan();
                let r = *radius_at_apex + v * tan_a;
                let theta = u * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
                let du = twopi * r * (-theta.sin() * x_dir + theta.cos() * y_dir);
                let dv = tan_a * (theta.cos() * x_dir + theta.sin() * y_dir) + *axis;
                (du, dv)
            }
            SurfaceGeom::Sphere { radius, .. } => {
                let r = *radius;
                let theta = u * std::f32::consts::TAU;
                let phi = v * std::f32::consts::PI;
                let twopi = std::f32::consts::TAU;
                let pi = std::f32::consts::PI;
                // ∂S/∂θ
                let du = twopi * r * phi.sin() * Vec3::new(-theta.sin(), theta.cos(), 0.0);
                // ∂S/∂φ
                let dv = pi * r * Vec3::new(
                    phi.cos() * theta.cos(),
                    phi.cos() * theta.sin(),
                    -phi.sin(),
                );
                (du, dv)
            }
            SurfaceGeom::Torus { axis, major_r, minor_r, .. } => {
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let mr = *major_r;
                let nr = *minor_r;
                let theta = u * std::f32::consts::TAU;
                let phi = v * std::f32::consts::TAU;
                let twopi = std::f32::consts::TAU;
                let r = mr + nr * phi.cos();
                // ∂S/∂u (major-angle derivative)
                let du = twopi * r * (-theta.sin() * x_dir + theta.cos() * y_dir);
                // ∂S/∂v (minor-angle derivative)
                let dv = twopi * (
                    x_dir * (-nr * phi.sin() * theta.cos())
                    + y_dir * (-nr * phi.sin() * theta.sin())
                    + *axis * nr * phi.cos()
                );
                (du, dv)
            }
            SurfaceGeom::BSpline(nurbs) => {
                let u_k = map_to_knot_domain(
                    &nurbs.knots_u, nurbs.degree_u, nurbs.u_count(), u,
                );
                let v_k = map_to_knot_domain(
                    &nurbs.knots_v, nurbs.degree_v, nurbs.v_count(), v,
                );
                let u_w = knot_domain_width(&nurbs.knots_u, nurbs.degree_u, nurbs.u_count());
                let v_w = knot_domain_width(&nurbs.knots_v, nurbs.degree_v, nurbs.v_count());
                let (du_k, dv_k) = nurbs.derivative(u_k, v_k);
                (du_k * u_w, dv_k * v_w)
            }
            SurfaceGeom::Extrusion { generatrix, direction } => {
                (generatrix.d1(u), *direction)
            }
            SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
                let axis = axis_dir.normalize();
                let angle = v * std::f32::consts::TAU;
                let gen_p = generatrix.d0(u);
                let gen_d1 = generatrix.d1(u);
                let du = rotate_around_axis(gen_d1, *axis_origin, axis, angle);
                let rotated_p = rotate_around_axis(gen_p, *axis_origin, axis, angle);
                let dv = axis.cross(rotated_p - *axis_origin) * std::f32::consts::TAU;
                (du, dv)
            }
            SurfaceGeom::Offset { basis, distance } => {
                let (db_du, db_dv) = basis.d1(u, v);
                let eps = 1e-4f32;
                let n_u1 = basis.normal(u + eps, v);
                let n_u0 = basis.normal(u - eps, v);
                let n_v1 = basis.normal(u, v + eps);
                let n_v0 = basis.normal(u, v - eps);
                let dn_du = (n_u1 - n_u0) / (2.0 * eps);
                let dn_dv = (n_v1 - n_v0) / (2.0 * eps);
                let d = *distance;
                (db_du + dn_du * d, db_dv + dn_dv * d)
            }
        }
    }

    /// Surface normal = (∂S/∂u × ∂S/∂v) normalized.
    /// Handles degeneracy by probing a neighborhood with small epsilon offsets.
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        let (du, dv) = self.d1(u, v);
        let n = du.cross(dv);
        let len = n.length();
        if len > 1e-6 {
            return n * (1.0 / len);
        }
        // Degenerate — probe surrounding neighborhood
        let eps = 1e-3f32;
        let probes = [(u + eps, v), (u - eps, v), (u, v + eps), (u, v - eps)];
        let mut best = Vec3::Z;
        let mut best_len = 0.0f32;
        for (up, vp) in probes {
            if up < 0.0 || up > 1.0 || vp < 0.0 || vp > 1.0 {
                continue;
            }
            let (du2, dv2) = self.d1(up, vp);
            let n2 = du2.cross(dv2);
            let l2 = n2.length();
            if l2 > best_len {
                best_len = l2;
                best = n2 * (1.0 / l2.max(1e-12));
            }
        }
        best
    }

    /// Sample a curve and find the parameter t ∈ [0,1] closest to a target 3D point.
    fn find_closest_t_on_curve(curve: &CurveGeom, target: Vec3) -> f32 {
    let n = 64;
    let mut best_t = 0.0f32;
    let mut best_dist = f32::MAX;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let d = (curve.d0(t) - target).length_squared();
        if d < best_dist { best_dist = d; best_t = t; }
    }
    // Local refinement with shrinking step size
    let mut step = 1.0 / (n as f32 * 2.0);
    for _ in 0..5 {
        for &dt in &[-step, step] {
            let t = (best_t + dt).clamp(0.0, 1.0);
            let d = (curve.d0(t) - target).length_squared();
            if d < best_dist { best_dist = d; best_t = t; }
        }
        step *= 0.5;
    }
    best_t
}

    /// Returns `None` for surfaces requiring numerical optimization
    /// (torus, B-spline, extrusion, offset).
    pub fn project(&self, point: Vec3) -> Option<(f32, f32)> {
        match self {
            SurfaceGeom::Plane { origin, normal, u_dir } => {
                let v_dir = normal.cross(*u_dir);
                let rel = point - *origin;
                Some((rel.dot(*u_dir), rel.dot(v_dir)))
            }
            SurfaceGeom::Cylinder { origin, axis, .. } => {
                let a = axis.normalize();
                let rel = point - *origin;
                let v = rel.dot(a);
                let radial = rel - a * v;
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let u_raw = f32::atan2(radial.dot(y_dir), radial.dot(x_dir));
                let u = if u_raw < 0.0 {
                    u_raw / std::f32::consts::TAU + 1.0
                } else {
                    u_raw / std::f32::consts::TAU
                };
                Some((u, v))
            }
            SurfaceGeom::Cone { apex, axis, .. } => {
                let a = axis.normalize();
                let rel = point - *apex;
                let v = rel.dot(a);
                let radial = rel - a * v;
                let (x_dir, y_dir) = build_ortho_axes(*axis);
                let u_raw = f32::atan2(radial.dot(y_dir), radial.dot(x_dir));
                let u = if u_raw < 0.0 {
                    u_raw / std::f32::consts::TAU + 1.0
                } else {
                    u_raw / std::f32::consts::TAU
                };
                Some((u, v))
            }
            SurfaceGeom::Sphere { center, .. } => {
                let rel = point - *center;
                let r = rel.length();
                if r < 1e-12 {
                    return Some((0.0, 0.0));
                }
                let phi = (rel.z / r).clamp(-1.0, 1.0).acos();
                let v = phi / std::f32::consts::PI;
                let u_raw = f32::atan2(rel.y, rel.x);
                let u = if u_raw < 0.0 {
                    u_raw / std::f32::consts::TAU + 1.0
                } else {
                    u_raw / std::f32::consts::TAU
                };
                Some((u, v))
            }
            SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
                let axis = axis_dir.normalize();
                let rel = point - *axis_origin;
                // Project onto axis to get the radial component
                let along = rel.dot(axis);
                let radial = rel - axis * along;
                let r = radial.length();
                if r < 1e-10 { return Some((0.5, 0.0)); }
                // Angle around axis
                let (x_dir, y_dir) = build_ortho_axes(axis);
                let u_raw = f32::atan2(radial.dot(y_dir), radial.dot(x_dir));
                let v = if u_raw < 0.0 {
                    u_raw / std::f32::consts::TAU + 1.0
                } else {
                    u_raw / std::f32::consts::TAU
                };
                // Unrotate the point to find where it lies on the generatrix
                let angle = v * std::f32::consts::TAU;
                let unrotated = rotate_around_axis(point, *axis_origin, axis, -angle);
                // Find closest point on generatrix
                let u = Self::find_closest_t_on_curve(generatrix, unrotated);
                Some((u, v))
            }
            SurfaceGeom::BSpline(nurbs) => {
                // Coarse grid search to find initial guess
                let grid = 16;
                let u_range = nurbs.knots_u[nurbs.degree_u];
                let u_end = nurbs.knots_u[nurbs.knots_u.len() - nurbs.degree_u - 1];
                let v_range = nurbs.knots_v[nurbs.degree_v];
                let v_end = nurbs.knots_v[nurbs.knots_v.len() - nurbs.degree_v - 1];

                let mut best_u = u_range;
                let mut best_v = v_range;
                let mut best_d2 = f32::MAX;

                for i in 0..=grid {
                    let u = u_range + (u_end - u_range) * i as f32 / grid as f32;
                    for j in 0..=grid {
                        let v = v_range + (v_end - v_range) * j as f32 / grid as f32;
                        let p = nurbs.evaluate(u, v);
                        let d2 = (p - point).length_squared();
                        if d2 < best_d2 { best_d2 = d2; best_u = u; best_v = v; }
                    }
                }

                // Local refinement: steepest descent on the parameter space
                let mut u = best_u;
                let mut v = best_v;
                let mut step = (u_end - u_range).max(v_end - v_range) / grid as f32 * 0.5;
                for _ in 0..8 {
                    for &(du, dv) in &[(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
                        let nu = (u + du).clamp(u_range, u_end);
                        let nv = (v + dv).clamp(v_range, v_end);
                        let d2 = (nurbs.evaluate(nu, nv) - point).length_squared();
                        if d2 < best_d2 { best_d2 = d2; u = nu; v = nv; }
                    }
                    step *= 0.5;
                }

                // Normalize UV to [0,1]
                let u_norm = if (u_end - u_range).abs() > 1e-8 {
                    (u - u_range) / (u_end - u_range)
                } else { 0.5 };
                let v_norm = if (v_end - v_range).abs() > 1e-8 {
                    (v - v_range) / (v_end - v_range)
                } else { 0.5 };
                Some((u_norm, v_norm))
            }
            SurfaceGeom::Torus { .. }
            | SurfaceGeom::Extrusion { .. }
            | SurfaceGeom::Offset { .. } => None,
        }
    }

    /// Evaluate a uniform grid of (n_u+1) × (n_v+1) 3D points over `u_range` × `v_range`.
    pub fn evaluate_grid(
        &self,
        u_range: (f32, f32),
        v_range: (f32, f32),
        n_u: usize,
        n_v: usize,
    ) -> Vec<Vec<Vec3>> {
        let mut grid = Vec::with_capacity(n_u + 1);
        for i in 0..=n_u {
            let u = u_range.0 + (u_range.1 - u_range.0) * i as f32 / n_u.max(1) as f32;
            let mut row = Vec::with_capacity(n_v + 1);
            for j in 0..=n_v {
                let v = v_range.0 + (v_range.1 - v_range.0) * j as f32 / n_v.max(1) as f32;
                row.push(self.d0(u, v));
            }
            grid.push(row);
        }
        grid
    }
}

// ── Tests ──────────────────────────────────────────────────────

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

    // ── SurfaceGeom tests ──────────────────────────────

    #[test]
    fn test_plane_d0() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let p = plane.d0(2.0, 3.0);
        assert!((p - Vec3::new(2.0, 3.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_plane_d1_matches_axes() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let (du, dv) = plane.d1(0.0, 0.0);
        assert!((du - Vec3::X).length() < 1e-6);
        assert!((dv - Vec3::Y).length() < 1e-6);
    }

    #[test]
    fn test_plane_normal() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let n = plane.normal(0.0, 0.0);
        assert!((n - Vec3::Z).length() < 1e-6);
    }

    #[test]
    fn test_plane_project() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::new(1.0, 0.0, 0.0),
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let (u, v) = plane.project(Vec3::new(5.0, 3.0, 0.0)).unwrap();
        assert!((u - 4.0).abs() < 1e-4);
        assert!((v - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_plane_evaluate_grid_shape() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let grid = plane.evaluate_grid((0.0, 1.0), (0.0, 1.0), 3, 2);
        assert_eq!(grid.len(), 4); // 3+1 rows
        assert_eq!(grid[0].len(), 3); // 2+1 cols
    }

    #[test]
    fn test_cylinder_d0_on_surface() {
        let cyl = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 2.0,
        };
        let p = cyl.d0(0.0, 5.0); // u=0 (angle 0 -> +X), v=5
        assert!((p - Vec3::new(2.0, 0.0, 5.0)).length() < 1e-4);
    }

    #[test]
    fn test_cylinder_d1_du_is_tangent_dv_is_axis() {
        let cyl = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 1.0,
        };
        let (du, dv) = cyl.d1(0.0, 1.0);
        // dv should be axis (+Z)
        assert!((dv - Vec3::Z).length() < 1e-6);
        // du should be orthogonal to axis
        assert!(du.dot(Vec3::Z).abs() < 1e-6);
        // du magnitude = 2π * r = 2π
        assert!((du.length() - std::f32::consts::TAU).abs() < 1e-4);
    }

    #[test]
    fn test_cylinder_normal_is_radial() {
        let cyl = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 1.0,
        };
        let n = cyl.normal(0.0, 0.0);
        // At u=0 (angle 0), normal should point +X
        assert!((n - Vec3::X).length() < 1e-4);
        // At u=0.25 (angle π/2), normal should point +Y
        let n2 = cyl.normal(0.25, 0.0);
        assert!((n2 - Vec3::Y).length() < 1e-4);
    }

    #[test]
    fn test_cylinder_project() {
        let cyl = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 2.0,
        };
        let (u, v) = cyl.project(Vec3::new(2.0, 0.0, 5.0)).unwrap();
        assert!(u.abs() < 1e-4); // angle 0
        assert!((v - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_cone_d0_at_apex() {
        let cone = SurfaceGeom::Cone {
            apex: Vec3::ZERO,
            axis: Vec3::Z,
            semi_angle: std::f32::consts::FRAC_PI_4,
            radius_at_apex: 0.0,
        };
        // At v=0 (apex), radius = 0, so point is at apex for any u
        let p = cone.d0(0.0, 0.0);
        assert!(p.length() < 1e-6);
    }

    #[test]
    fn test_cone_d0_radius_grows_with_v() {
        let cone = SurfaceGeom::Cone {
            apex: Vec3::ZERO,
            axis: Vec3::Z,
            semi_angle: std::f32::consts::FRAC_PI_4,
            radius_at_apex: 0.0,
        };
        // At v=1, z=1, radius = 1*tan(π/4) = 1
        let p = cone.d0(0.0, 1.0); // angle 0 -> +X
        let r = (p.x * p.x + p.y * p.y).sqrt();
        assert!((r - 1.0).abs() < 1e-4);
        assert!((p.z - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_cone_d1_dv_has_axis_component() {
        let cone = SurfaceGeom::Cone {
            apex: Vec3::ZERO,
            axis: Vec3::Z,
            semi_angle: std::f32::consts::FRAC_PI_4,
            radius_at_apex: 1.0,
        };
        let (_du, dv) = cone.d1(0.0, 0.5);
        // dv should have an axial (+Z) component
        assert!(dv.z > 0.5);
    }

    #[test]
    fn test_sphere_d0_at_equator() {
        let sphere = SurfaceGeom::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        };
        // u=0 (azimuth 0), v=0.5 (polar π/2 → equator), should be (2, 0, 0)
        let p = sphere.d0(0.0, 0.5);
        assert!((p - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn test_sphere_d0_at_poles() {
        let sphere = SurfaceGeom::Sphere {
            center: Vec3::new(1.0, 0.0, 0.0),
            radius: 3.0,
        };
        // v=0 → north pole (+Z)
        let p_north = sphere.d0(0.0, 0.0);
        assert!((p_north - Vec3::new(1.0, 0.0, 3.0)).length() < 1e-4);
        // v=1 → south pole (-Z)
        let p_south = sphere.d0(0.0, 1.0);
        assert!((p_south - Vec3::new(1.0, 0.0, -3.0)).length() < 1e-4);
    }

    #[test]
    fn test_sphere_radius_consistency() {
        let sphere = SurfaceGeom::Sphere {
            center: Vec3::new(10.0, 0.0, 0.0),
            radius: 5.0,
        };
        for u in [0.0, 0.25, 0.5, 0.75] {
            for v in [0.1, 0.3, 0.5, 0.7, 0.9] {
                let p = sphere.d0(u, v);
                let dist = (p - Vec3::new(10.0, 0.0, 0.0)).length();
                assert!((dist - 5.0).abs() < 1e-4,
                    "radius error at u={u}, v={v}: dist={dist}");
            }
        }
    }

    #[test]
    fn test_sphere_normal_is_outward() {
        let sphere = SurfaceGeom::Sphere {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 4.0,
        };
        for (u, v) in [(0.1, 0.25), (0.5, 0.5), (0.9, 0.75)] {
            let p = sphere.d0(u, v);
            let n = sphere.normal(u, v);
            let radial = (p - Vec3::new(1.0, 2.0, 3.0)).normalize();
            // Normal should be collinear with the radial direction
            assert!((n.dot(radial).abs() - 1.0).abs() < 1e-4,
                "normal not radial at u={u}, v={v}: n={:?}, radial={:?}", n, radial);
        }
    }

    #[test]
    fn test_sphere_project() {
        let sphere = SurfaceGeom::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };
        // Point at north pole
        let (u, v) = sphere.project(Vec3::new(0.0, 0.0, 5.0)).unwrap();
        assert!(v.abs() < 1e-4); // polar angle ≈ 0
    }

    #[test]
    fn test_torus_d0_outer_equator() {
        let torus = SurfaceGeom::Torus {
            center: Vec3::ZERO,
            axis: Vec3::Z,
            major_r: 3.0,
            minor_r: 1.0,
        };
        // u=0 (major angle 0), v=0 (minor angle 0): point on outer equator +X
        let p = torus.d0(0.0, 0.0);
        let r_xy = (p.x * p.x + p.y * p.y).sqrt();
        assert!((r_xy - 4.0).abs() < 1e-4); // major_r + minor_r
        assert!(p.z.abs() < 1e-4);
    }

    #[test]
    fn test_torus_d0_inner_top() {
        let torus = SurfaceGeom::Torus {
            center: Vec3::ZERO,
            axis: Vec3::Z,
            major_r: 3.0,
            minor_r: 1.0,
        };
        // u=0 (major angle 0), v=0.5 (minor angle π): inner equator
        let p = torus.d0(0.0, 0.5);
        let r_xy = (p.x * p.x + p.y * p.y).sqrt();
        assert!((r_xy - 2.0).abs() < 1e-4); // major_r - minor_r
    }

    #[test]
    fn test_torus_normal_outward() {
        let torus = SurfaceGeom::Torus {
            center: Vec3::ZERO,
            axis: Vec3::Z,
            major_r: 3.0,
            minor_r: 1.0,
        };
        // At outer equator (u=0, v=0), normal should be +X (radially outward)
        let n = torus.normal(0.0, 0.0);
        assert!(n.x > 0.5, "normal at outer equator should point +X, got {:?}", n);
    }

    #[test]
    fn test_torus_project_returns_none() {
        let torus = SurfaceGeom::Torus {
            center: Vec3::ZERO,
            axis: Vec3::Z,
            major_r: 3.0,
            minor_r: 1.0,
        };
        assert!(torus.project(Vec3::new(4.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn test_extrusion_d0() {
        let generatrix = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let extrusion = SurfaceGeom::Extrusion {
            generatrix: Box::new(generatrix),
            direction: Vec3::Z,
        };
        let p = extrusion.d0(2.0, 3.0); // u=2 → (2,0,0), v=3 → +3Z
        assert!((p - Vec3::new(2.0, 0.0, 3.0)).length() < 1e-4);
    }

    #[test]
    fn test_extrusion_project_returns_none() {
        let generatrix = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let extrusion = SurfaceGeom::Extrusion {
            generatrix: Box::new(generatrix),
            direction: Vec3::Z,
        };
        assert!(extrusion.project(Vec3::ZERO).is_none());
    }

    #[test]
    fn test_extrusion_d1() {
        let generatrix = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let extrusion = SurfaceGeom::Extrusion {
            generatrix: Box::new(generatrix),
            direction: Vec3::Z,
        };
        let (du, dv) = extrusion.d1(0.5, 0.5);
        // du = generatrix.d1 = +X
        assert!((du - Vec3::X).length() < 1e-6);
        // dv = direction = +Z
        assert!((dv - Vec3::Z).length() < 1e-6);
    }

    #[test]
    fn test_revolution_d0_circle_makes_torus() {
        use std::f32::consts::PI;
        // Circle of radius 1 in XZ plane, centered at (3, 0, 0), revolved around Z
        let circle = CurveGeom::Circle {
            center: Vec3::new(3.0, 0.0, 0.0),
            axis: Vec3::Y,
            radius: 1.0,
        };
        let rev = SurfaceGeom::Revolution {
            generatrix: Box::new(circle),
            axis_origin: Vec3::ZERO,
            axis_dir: Vec3::Z,
        };
        // u=0 (angle 0 on circle → (4,0,0)), v=0 (revolution angle 0 on Z)
        let p = rev.d0(0.0, 0.0);
        assert!((p - Vec3::new(4.0, 0.0, 0.0)).length() < 1e-4);
        // u=0 (angle 0 on circle → (4,0,0)), v=0.25 (revolution π/2 around Z → (0,4,0))
        let p2 = rev.d0(0.0, 0.25);
        assert!((p2 - Vec3::new(0.0, 4.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn test_revolution_project() {
        let circle = CurveGeom::Circle {
            center: Vec3::new(3.0, 0.0, 0.0),
            axis: Vec3::Y,
            radius: 1.0,
        };
        let rev = SurfaceGeom::Revolution {
            generatrix: Box::new(circle),
            axis_origin: Vec3::ZERO,
            axis_dir: Vec3::Z,
        };
        let proj = rev.project(Vec3::new(4.0, 0.0, 0.0));
        assert!(proj.is_some(), "revolution project should now work");
        let (u, v) = proj.unwrap();
        // The circle of radius 1 revolved around Z at distance 4 gives a torus-like shape
        // projection should give valid UV
        assert!((0.0..=1.0).contains(&u));
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_offset_d0() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let offset = SurfaceGeom::Offset {
            basis: Box::new(plane),
            distance: 2.0,
        };
        // Offset plane by 2 units along +Z
        let p = offset.d0(1.0, 1.0);
        assert!((p - Vec3::new(1.0, 1.0, 2.0)).length() < 1e-4);
    }

    #[test]
    fn test_offset_project_returns_none() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let offset = SurfaceGeom::Offset {
            basis: Box::new(plane),
            distance: 2.0,
        };
        assert!(offset.project(Vec3::ZERO).is_none());
    }

    #[test]
    fn test_bspline_project() {
        let nurbs = crate::step::nurbs::NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        let bspline = SurfaceGeom::BSpline(nurbs);
        let proj = bspline.project(Vec3::new(0.5, 0.5, 0.0));
        assert!(proj.is_some(), "bspline project should now work");
        let (u, v) = proj.unwrap();
        assert!((0.0..=1.0).contains(&u));
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_evaluate_grid_corner_values() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let grid = plane.evaluate_grid((1.0, 2.0), (3.0, 4.0), 1, 1);
        assert_eq!(grid.len(), 2);
        assert_eq!(grid[0].len(), 2);
        // (u=1, v=3)
        assert!((grid[0][0] - Vec3::new(1.0, 3.0, 0.0)).length() < 1e-6);
        // (u=1, v=4)
        assert!((grid[0][1] - Vec3::new(1.0, 4.0, 0.0)).length() < 1e-6);
        // (u=2, v=3)
        assert!((grid[1][0] - Vec3::new(2.0, 3.0, 0.0)).length() < 1e-6);
        // (u=2, v=4)
        assert!((grid[1][1] - Vec3::new(2.0, 4.0, 0.0)).length() < 1e-6);
    }
}
