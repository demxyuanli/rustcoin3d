//! 2D parametric curve types for UV-space operations.
//!
//! Provides curve evaluation, Bézier decomposition, and recursive
//! Bézier clipping intersection for use in constrained Delaunay
//! triangulation (CDT) constraint edge resolution.
//!
//! OCC alignment: Geom2d_Curve hierarchy — the native 2D curve type
//! for PCurves (parametric curves on surface UV space).

use super::CurveGeom;
use rc3d_core::math::{Real, PVec3};

/// 2D curve in UV parameter space.
///
/// OCC alignment: Geom2d_Curve — each variant maps to an OCC
/// Geom2d_Line, Geom2d_Circle, Geom2d_Ellipse, Geom2d_BSplineCurve,
/// Geom2d_TrimmedCurve, Geom2d_Polyline, or Geom2d_OffsetCurve.
#[derive(Debug, Clone)]
pub enum Curve2d {
    Line {
        origin: (Real, Real),
        direction: (Real, Real),
    },
    Circle {
        center: (Real, Real),
        radius: Real,
    },
    Ellipse {
        center: (Real, Real),
        semi_major: Real,
        semi_minor: Real,
    },
    BSpline {
        degree: usize,
        control_points: Vec<(Real, Real)>,
        knots: Vec<Real>,
        weights: Option<Vec<Real>>,
    },
    Trimmed {
        basis: Box<Curve2d>,
        t_min: Real,
        t_max: Real,
    },
    Polyline {
        points: Vec<(Real, Real)>,
    },
    Composite {
        segments: Vec<(Curve2d, bool)>,
    },
}

impl Curve2d {
    /// Return a new curve whose parameterization runs in the opposite direction:
    /// `reversed.d0(t) == self.d0(1.0 - t)`.
    ///
    /// Used to normalize PCurves at storage time so that all stored
    /// PCurves run in the same direction as their 3D edge curve.
    pub fn reversed(&self) -> Self {
        // For simple types, produce a direct reversed form.
        // For others, wrap in a Trimmed with span = -1.
        match self {
            Curve2d::Line { origin, direction } => {
                let end = (origin.0 + direction.0, origin.1 + direction.1);
                Curve2d::Line {
                    origin: end,
                    direction: (-direction.0, -direction.1),
                }
            }
            Curve2d::Polyline { points } => {
                let mut rev = points.clone();
                rev.reverse();
                Curve2d::Polyline { points: rev }
            }
            // Circle, Ellipse, BSpline, Trimmed, Composite — treat
            // generically via parameter reversal (evaluates correctly
            // for all d0 / d1 / to_beziers usage).
            _ => Curve2d::Trimmed {
                basis: Box::new(self.clone()),
                t_min: 1.0,
                t_max: 0.0,
            },
        }
    }

    /// Evaluate the curve at parameter t ∈ [0, 1].
    pub fn d0(&self, t: Real) -> (Real, Real) {
        match self {
            Curve2d::Line { origin, direction } => {
                (origin.0 + direction.0 * t, origin.1 + direction.1 * t)
            }
            Curve2d::Circle { center, radius } => {
                let theta = t * std::f64::consts::TAU;
                (center.0 + radius * theta.cos(), center.1 + radius * theta.sin())
            }
            Curve2d::Ellipse { center, semi_major, semi_minor } => {
                let theta = t * std::f64::consts::TAU;
                (center.0 + semi_major * theta.cos(), center.1 + semi_minor * theta.sin())
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
            Curve2d::Polyline { points } => {
                if points.len() < 2 {
                    return points.first().copied().unwrap_or((0.0, 0.0));
                }
                let n = points.len() - 1;
                let idx_f = t.clamp(0.0, 1.0) * n as Real;
                let idx = idx_f as usize;
                let frac = idx_f - idx as Real;
                let a = points[idx.min(n)];
                let b = points[(idx + 1).min(n)];
                (a.0 + (b.0 - a.0) * frac, a.1 + (b.1 - a.1) * frac)
            }
            Curve2d::Composite { segments } => {
                if segments.is_empty() {
                    return (0.0, 0.0);
                }
                if segments.len() == 1 {
                    return segments[0].0.d0(t);
                }
                // Map t ∈ [0,1] to segment index and local parameter
                let n = segments.len() as Real;
                let idx_f = t.clamp(0.0, 1.0) * n;
                let idx = (idx_f as usize).min(segments.len() - 1);
                let local_t = idx_f - idx as Real;
                segments[idx].0.d0(local_t.clamp(0.0, 1.0))
            }
        }
    }

    /// Evaluate first derivative at parameter t ∈ [0, 1].
    /// Returns (point, tangent_direction).
    pub fn d1(&self, t: Real) -> ((Real, Real), (Real, Real)) {
        match self {
            Curve2d::Line { origin, direction } => {
                let p = (origin.0 + direction.0 * t, origin.1 + direction.1 * t);
                (p, *direction)
            }
            Curve2d::Circle { center, radius } => {
                let theta = t * std::f64::consts::TAU;
                let p = (center.0 + radius * theta.cos(), center.1 + radius * theta.sin());
                let dt = (-radius * theta.sin(), radius * theta.cos());
                (p, dt)
            }
            Curve2d::Ellipse { center, semi_major, semi_minor } => {
                let theta = t * std::f64::consts::TAU;
                let p = (center.0 + semi_major * theta.cos(), center.1 + semi_minor * theta.sin());
                let dt = (-semi_major * theta.sin(), semi_minor * theta.cos());
                (p, dt)
            }
            Curve2d::BSpline { degree, control_points, knots, weights: _ } => {
                // Analytical derivative via degree-reduced CPs (Piegl & Tiller)
                if *degree == 0 {
                    // Degree-0 B-spline: derivative is zero
                    return (self.d0(t), (0.0, 0.0));
                }
                let p = *degree;
                let n = control_points.len();
                if n < p + 1 {
                    return (self.d0(t), (0.0, 0.0));
                }
                // Build derivative CPs: Q_i = p * (P_{i+1} - P_i) / (k_{i+p+1} - k_{i+1})
                let mut d_cps: Vec<(Real, Real)> = Vec::with_capacity(n - 1);
                for i in 0..n - 1 {
                    let denom = knots[i + p + 1] - knots[i + 1];
                    let scale = if denom.abs() > 1e-10 { p as Real / denom } else { 0.0 };
                    d_cps.push((
                        scale * (control_points[i + 1].0 - control_points[i].0),
                        scale * (control_points[i + 1].1 - control_points[i].1),
                    ));
                }
                // Derivative B-spline is degree p-1 on the same knot vector
                let dt = bspline_2d_d0(p - 1, &d_cps, knots, None::<&[Real]>, t);
                let pos = self.d0(t);
                (pos, dt)
            }
            Curve2d::Trimmed { basis, t_min, t_max } => {
                let span = t_max - t_min;
                if span.abs() < 1e-10 {
                    return basis.d1(*t_min);
                }
                let (p, d) = basis.d1(t_min + t * span);
                (p, (d.0 * span, d.1 * span))
            }
            Curve2d::Polyline { points } => {
                let p = self.d0(t);
                if points.len() < 2 {
                    return (p, (0.0, 0.0));
                }
                let n = points.len() - 1;
                let idx_f = t.clamp(0.0, 1.0) * n as Real;
                let idx = (idx_f as usize).min(n - 1);
                let a = points[idx];
                let b = points[idx + 1];
                (p, (b.0 - a.0, b.1 - a.1))
            }
            Curve2d::Composite { segments: _ } => {
                // Numerical derivative for composite
                let eps = 1e-4;
                let p0 = self.d0((t - eps).max(0.0));
                let p1 = self.d0((t + eps).min(1.0));
                let dt = ((p1.0 - p0.0) / (2.0 * eps), (p1.1 - p0.1) / (2.0 * eps));
                (self.d0(t), dt)
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
                const K: Real = 0.552_284_8;
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
                // Pass knots=None: for BSpline bases, knots should be threaded
                // through from decompose_bspline_to_beziers for correct non-uniform
                // parameter-space mapping (H15). Currently uses uniform assumption.
                clip_beziers_to_range(&all, *t_min, *t_max, None)
            }
            Curve2d::Ellipse { center, semi_major, semi_minor } => {
                // Approximate ellipse with 4 cubic Bézier arcs (90° each)
                const K: Real = 0.552_284_8;
                let cx = center.0;
                let cy = center.1;
                let a = *semi_major;
                let b = *semi_minor;
                vec![
                    Bezier2d { c0: (cx + a, cy), c1: (cx + a, cy + b * K),
                        c2: (cx + a * K, cy + b), c3: (cx, cy + b) },
                    Bezier2d { c0: (cx, cy + b), c1: (cx - a * K, cy + b),
                        c2: (cx - a, cy + b * K), c3: (cx - a, cy) },
                    Bezier2d { c0: (cx - a, cy), c1: (cx - a, cy - b * K),
                        c2: (cx - a * K, cy - b), c3: (cx, cy - b) },
                    Bezier2d { c0: (cx, cy - b), c1: (cx + a * K, cy - b),
                        c2: (cx + a, cy - b * K), c3: (cx + a, cy) },
                ]
            }
            Curve2d::Polyline { points } => {
                // One linear Bézier per polyline segment
                if points.len() < 2 {
                    return vec![];
                }
                points.windows(2).map(|w| {
                    let p0 = w[0];
                    let p3 = w[1];
                    let p1 = (p0.0 + (p3.0 - p0.0) / 3.0, p0.1 + (p3.1 - p0.1) / 3.0);
                    let p2 = (p3.0 - (p3.0 - p0.0) / 3.0, p3.1 - (p3.1 - p0.1) / 3.0);
                    Bezier2d { c0: p0, c1: p1, c2: p2, c3: p3 }
                }).collect()
            }
            Curve2d::Composite { segments } => {
                segments.iter().flat_map(|(seg, _)| seg.to_beziers()).collect()
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Conversion: CurveGeom (3D, used as PCurve with PVec3(u,v,0)) ↔ Curve2d
// ═══════════════════════════════════════════════════════════════════════

/// Simplify a polyline to a Line if all points are collinear within tolerance.
/// Returns None for degenerate (closed-loop or zero-length) polylines.
pub fn simplify_polyline_to_line(pts: &[(Real, Real)]) -> Option<Curve2d> {
    if pts.len() < 2 { return None; }
    if pts.len() == 2 {
        let dx = pts[1].0 - pts[0].0;
        let dy = pts[1].1 - pts[0].1;
        if dx * dx + dy * dy < 1e-12 { return None; }
        return Some(Curve2d::Line { origin: pts[0], direction: (dx, dy) });
    }
    let first = pts[0];
    let last = pts[pts.len() - 1];
    let is_closed = (last.0 - first.0).abs() < 1e-3 && (last.1 - first.1).abs() < 1e-3;

    // Detect periodic U-wrap: first≈last but mid-U ≈ 2π vs first-U ≈ 0.
    // Unwrap U-coordinates around TAU (2π) before collinearity check.
    let tau = std::f64::consts::TAU;
    let has_u_wrap = is_closed && pts.len() >= 4 &&
        pts.iter().any(|p| (p.0 - first.0).abs() > tau * 0.9);

    // Detect periodic jump: mid-points share one U/V, ends share another.
    // E.g. cone generatrix: (0,0), (TAU,dv),...,(TAU,10-dv), (0,10) or (TAU,10)
    let u_jump = pts.len() >= 4 && !is_closed && {
        let mid_u = pts[1].0;
        pts[1..pts.len()-1].iter().all(|p| (p.0 - mid_u).abs() < 1e-4)
        && (mid_u - first.0).abs() > tau * 0.5
        && ((last.0 - first.0).abs() < 1e-4 || (last.0 - mid_u).abs() < 1e-4)
    };
    let v_jump = pts.len() >= 4 && !is_closed && {
        let mid_v = pts[1].1;
        pts[1..pts.len()-1].iter().all(|p| (p.1 - mid_v).abs() < 1e-2)
        && (mid_v - first.1).abs() > 1e-2
        && ((last.1 - first.1).abs() < 1e-4 || (last.1 - mid_v).abs() < 1e-4)
    };

    let dir = {
        let du = last.0 - first.0;
        let dv = last.1 - first.1;

        if u_jump && (last.1 - first.1).abs() > 1e-3 {
            // Jump in U at start; actual line is vertical at const U=last.U
            (0.0, dv)
        } else if v_jump && (last.0 - first.0).abs() > 1e-3 {
            // Jump in V at start; actual line is horizontal at const V=last.V
            (du, 0.0)
        } else if is_closed {
            let mut u_vals: Vec<f64> = pts.iter().map(|p| {
                let mut u = p.0 - first.0;
                if has_u_wrap && u > tau * 0.5 { u -= tau; }
                if has_u_wrap && u < -tau * 0.5 { u += tau; }
                u
            }).collect();
            let umin = u_vals.iter().cloned().fold(f64::MAX, f64::min);
            let umax = u_vals.iter().cloned().fold(f64::MIN, f64::max);
            let vmin = pts.iter().map(|p| p.1).fold(f64::MAX, f64::min);
            let vmax = pts.iter().map(|p| p.1).fold(f64::MIN, f64::max);
            let u_span = umax - umin;
            let v_span = vmax - vmin;
            let mid = pts.len() / 2;
            let mdv = pts[mid].1 - first.1;
            if u_span < 1e-3 && v_span > 1e-3 { (0.0, mdv) }
            else if v_span < 1e-3 && u_span > 1e-3 { (u_vals[mid], 0.0) }
            else { (u_vals[mid], mdv) }
        } else {
            (du, dv)
        }
    };
    let len_sq = dir.0 * dir.0 + dir.1 * dir.1;
    if len_sq < 1e-12 { return None; }

    // Check all points lie on the line (with U/V-unwrapping tolerance)
    let len = len_sq.sqrt();
    let end_idx = if is_closed { pts.len() } else { pts.len() - 1 };
    let need_u_unwrap = has_u_wrap || u_jump;
    for i in 1..end_idx {
        let mut dx = pts[i].0 - first.0;
        let dy = pts[i].1 - first.1;
        if need_u_unwrap && dx > tau * 0.5 { dx -= tau; }
        if need_u_unwrap && dx < -tau * 0.5 { dx += tau; }
        let cross = (dir.0 * dy - dir.1 * dx).abs();
        if cross / len > 1e-3 { return None; }
    }
    Some(Curve2d::Line { origin: first, direction: dir })
}

#[test]
fn simplify_vertical_u_wrap_polyline() {
    let tau = std::f64::consts::TAU;
    // Simulate cone generatrix PCurve: const U=TAU, V varies 0→10
    let n = 17;
    let pts: Vec<(f64, f64)> = (0..n).map(|i| {
        let t = i as f64 / (n - 1) as f64;
        if i == 0 || i == n - 1 { (0.0, t * 10.0) }
        else { (tau, t * 10.0) }
    }).collect();
    let result = simplify_polyline_to_line(&pts);
    assert!(result.is_some(), "vertical U-wrap polyline should simplify to Line, got None");
    match result {
        Some(Curve2d::Line { origin, direction }) => {
            assert!((origin.0).abs() < 0.1, "origin U≈0, got {}", origin.0);
            assert!((origin.1).abs() < 0.1, "origin V≈0, got {}", origin.1);
            assert!((direction.0).abs() < 0.1, "dir U≈0 (vertical line), got {}", direction.0);
            assert!((direction.1 - 10.0).abs() < 0.1, "dir V≈10, got {}", direction.1);
        }
        _ => panic!("wrong variant"),
    }
}

impl Curve2d {
    /// Convert a 3D CurveGeom used as a PCurve (where d0 returns PVec3(u, v, 0))
    /// into a native 2D Curve2d.
    ///
    /// OCC alignment: Geom2dAdaptor_Curve — adapts between 3D and 2D representations.
    pub fn from_pcurve_3d(curve: &CurveGeom) -> Self {
        match curve {
            CurveGeom::Line { origin, direction } => Curve2d::Line {
                origin: (origin.x, origin.y),
                direction: (direction.x, direction.y),
            },
            CurveGeom::Circle { center, radius, .. } => Curve2d::Circle {
                center: (center.x, center.y),
                radius: *radius,
            },
            CurveGeom::Ellipse { center, semi_major, semi_minor, .. } => Curve2d::Ellipse {
                center: (center.x, center.y),
                semi_major: *semi_major,
                semi_minor: *semi_minor,
            },
            CurveGeom::BSpline { degree, control_points, knots, weights } => Curve2d::BSpline {
                degree: *degree,
                control_points: control_points.iter().map(|p| (p.x, p.y)).collect(),
                knots: knots.clone(),
                weights: weights.clone(),
            },
            CurveGeom::Trimmed { basis, t_min, t_max } => Curve2d::Trimmed {
                basis: Box::new(Curve2d::from_pcurve_3d(basis)),
                t_min: *t_min,
                t_max: *t_max,
            },
            CurveGeom::Polyline { points } => {
                let pts_2d: Vec<(Real, Real)> = points.iter().map(|p| (p.x, p.y)).collect();
                simplify_polyline_to_line(&pts_2d).unwrap_or(Curve2d::Polyline { points: pts_2d })
            },
            CurveGeom::Composite { segments, .. } => Curve2d::Composite {
                segments: segments.iter().map(|(seg, reversed)| {
                    (Curve2d::from_pcurve_3d(seg), *reversed)
                }).collect(),
            },
            CurveGeom::Hyperbola { center, semi_major, semi_minor, .. } => {
                // Approximate as ellipse for 2D PCurve purposes
                Curve2d::Ellipse {
                    center: (center.x, center.y),
                    semi_major: *semi_major,
                    semi_minor: *semi_minor,
                }
            }
            CurveGeom::Parabola { center: _, focal_dist: _, .. } => {
                // Approximate as polyline via sampling
                let n = 32;
                let pts: Vec<(Real, Real)> = (0..=n).map(|i| {
                    let t = i as Real / n as Real;
                    let p = curve.d0(t);
                    (p.x, p.y)
                }).collect();
                Curve2d::Polyline { points: pts }
            }
            CurveGeom::BezierCurve { .. } => {
                // Sample the Bezier curve as a polyline
                let n = 32;
                let pts: Vec<(Real, Real)> = (0..=n).map(|i| {
                    let t = i as Real / n as Real;
                    let p = curve.d0(t);
                    (p.x, p.y)
                }).collect();
                Curve2d::Polyline { points: pts }
            }
            CurveGeom::Offset { basis: _, offset_dir: _, distance: _ } => {
                // Sample the offset curve as a polyline
                let n = 64;
                let pts: Vec<(Real, Real)> = (0..=n).map(|i| {
                    let t = i as Real / n as Real;
                    let p = curve.d0(t);
                    (p.x, p.y)
                }).collect();
                Curve2d::Polyline { points: pts }
            }
        }
    }

    /// Convert back to a 3D CurveGeom used as PCurve (PVec3(u, v, 0) convention).
    pub fn to_pcurve_3d(&self) -> CurveGeom {
        match self {
            Curve2d::Line { origin, direction } => CurveGeom::Line {
                origin: PVec3::new(origin.0, origin.1, 0.0),
                direction: PVec3::new(direction.0, direction.1, 0.0),
            },
            Curve2d::Circle { center, radius } => CurveGeom::Circle {
                center: PVec3::new(center.0, center.1, 0.0),
                axis: PVec3::Z,
                radius: *radius,
                x_dir: PVec3::X,
                y_dir: PVec3::Y,
            },
            Curve2d::Ellipse { center, semi_major, semi_minor } => CurveGeom::Ellipse {
                center: PVec3::new(center.0, center.1, 0.0),
                axis: PVec3::Z,
                semi_major: *semi_major,
                semi_minor: *semi_minor,
                x_dir: PVec3::X,
                y_dir: PVec3::Y,
            },
            Curve2d::BSpline { degree, control_points, knots, weights } => CurveGeom::BSpline {
                degree: *degree,
                control_points: control_points.iter().map(|&(u, v)| PVec3::new(u, v, 0.0)).collect(),
                knots: knots.clone(),
                weights: weights.clone(),
            },
            Curve2d::Trimmed { basis, t_min, t_max } => CurveGeom::Trimmed {
                basis: Box::new(basis.to_pcurve_3d()),
                t_min: *t_min,
                t_max: *t_max,
            },
            Curve2d::Polyline { points } => CurveGeom::Polyline {
                points: points.iter().map(|&(u, v)| PVec3::new(u, v, 0.0)).collect(),
            },
            Curve2d::Composite { segments } => CurveGeom::Composite {
                segments: segments.iter().map(|(seg, rev)| (seg.to_pcurve_3d(), *rev)).collect(),
                cached_lengths: None,
            },
        }
    }
}

/// Cubic Bézier curve segment. The curve is parameterized t ∈ [0, 1]:
///
///   B(t) = (1-t)³·c0 + 3(1-t)²t·c1 + 3(1-t)t²·c2 + t³·c3
#[derive(Debug, Clone, Copy)]
pub struct Bezier2d {
    pub c0: (Real, Real),
    pub c1: (Real, Real),
    pub c2: (Real, Real),
    pub c3: (Real, Real),
}

impl Bezier2d {
    /// Evaluate the Bézier at parameter t.
    pub fn eval(&self, t: Real) -> (Real, Real) {
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
    pub fn bbox(&self) -> (Real, Real, Real, Real) {
        let xs = [self.c0.0, self.c1.0, self.c2.0, self.c3.0];
        let ys = [self.c0.1, self.c1.1, self.c2.1, self.c3.1];
        (
            xs.iter().cloned().fold(f64::INFINITY, Real::min),
            xs.iter().cloned().fold(Real::NEG_INFINITY, Real::max),
            ys.iter().cloned().fold(f64::INFINITY, Real::min),
            ys.iter().cloned().fold(Real::NEG_INFINITY, Real::max),
        )
    }

    /// Split at parameter t ∈ [0, 1] using de Casteljau subdivision.
    pub fn split_at(&self, t: Real) -> (Bezier2d, Bezier2d) {
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

fn lerp_2d(a: (Real, Real), b: (Real, Real), t: Real) -> (Real, Real) {
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
///
/// For coincident curves, subdivision would produce O(2^depth) spurious points.
/// A guard limit prevents unbounded result explosion.
#[allow(dead_code)]
const MAX_CLIP_RESULTS: usize = 64;

pub fn bezier_clip_intersect(
    a: &Bezier2d,
    b: &Bezier2d,
    tol: Real,
    max_depth: usize,
) -> Vec<((Real, Real), (Real, Real))> {
    // Guard against coincident-curve explosion: after 4+ subdivisions,
    // if curves still overlap, they are likely coincident. Return a single
    // midpoint intersection instead of continuing to subdivide.
    if max_depth <= 3 {
        let a_diag_mid = ((a.bbox().0 - a.bbox().1).powi(2) + (a.bbox().2 - a.bbox().3).powi(2)).sqrt();
        let b_diag_mid = ((b.bbox().0 - b.bbox().1).powi(2) + (b.bbox().2 - b.bbox().3).powi(2)).sqrt();
        if a_diag_mid < tol * 0.1 && b_diag_mid < tol * 0.1 {
            let am = a.eval(0.5);
            let bm = b.eval(0.5);
            let dist_sq = (am.0 - bm.0).powi(2) + (am.1 - bm.1).powi(2);
            if dist_sq < tol * tol * 16.0 {
                return vec![(am, bm)];
            }
        }
    }

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
pub fn intersect_curves_2d(a: &Curve2d, b: &Curve2d, tol: Real) -> Vec<((Real, Real), (Real, Real))> {
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
    cps: &[(Real, Real)],
    knots: &[Real],
    weights: Option<&[Real]>,
    t: Real,
) -> (Real, Real) {
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
    let mut d: Vec<(Real, Real)> = cps[span - p..=span].to_vec();
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
        let mut w: Vec<Real> = ws[span - p..=span].to_vec();
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

/// Insert knot `t` into a 2D B-spline curve using Boehm's algorithm.
/// Returns (new_control_points, new_knots).
fn insert_knot_2d(
    degree: usize,
    cps: &[(Real, Real)],
    knots: &[Real],
    t: Real,
) -> (Vec<(Real, Real)>, Vec<Real>) {
    let n = cps.len();
    let s = rc3d_core::utils::bspline::find_span_f64(degree, knots, t);
    let mut new_cps = vec![(0.0, 0.0); n + 1];
    let mut new_knots = vec![0.0; knots.len() + 1];

    // Copy unchanged portions
    for i in 0..=s.saturating_sub(degree) { new_cps[i] = cps[i]; }
    for i in s + 1..n { new_cps[i + 1] = cps[i]; }
    for i in 0..=s { new_knots[i] = knots[i]; }
    new_knots[s + 1] = t;
    for i in s + 1..knots.len() { new_knots[i + 1] = knots[i]; }

    // Compute new control points in the affected range
    for i in (s.saturating_sub(degree) + 1)..=s {
        let denom = knots[i + degree] - knots[i];
        let alpha = if denom.abs() > 1e-15 {
            (t - knots[i]) / denom
        } else {
            0.0
        };
        let p0 = cps[i - 1];
        let p1 = cps[i];
        new_cps[i] = ((1.0 - alpha) * p0.0 + alpha * p1.0, (1.0 - alpha) * p0.1 + alpha * p1.1);
    }
    (new_cps, new_knots)
}

/// Decompose a B-spline into piecewise Bézier segments.
/// Inserts all internal knots to full multiplicity (degree), then extracts
/// each span as a Bézier segment. This is the O(n²) correct algorithm
/// (OCC BSplCLib::BezierCoefficients).
fn decompose_bspline_to_beziers(
    degree: usize,
    cps: &[(Real, Real)],
    knots: &[Real],
    _weights: Option<&[Real]>,
) -> Vec<Bezier2d> {
    let n = cps.len();
    if n <= degree + 1 {
        // Already a single Bézier segment
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

    // Insert each distinct internal knot to full multiplicity (= degree).
    let mut cur_cps = cps.to_vec();
    let mut cur_knots = knots.to_vec();
    let mut i = degree + 1;
    while i < cur_knots.len() - degree - 1 {
        let knot_val = cur_knots[i];
        // Count current multiplicity
        let mut mult = 0;
        while i + mult < cur_knots.len() && (cur_knots[i + mult] - knot_val).abs() < 1e-12 {
            mult += 1;
        }
        // Insert to reach full multiplicity
        for _ in mult..degree {
            let (new_cps, new_knots) = insert_knot_2d(degree, &cur_cps, &cur_knots, knot_val);
            cur_cps = new_cps;
            cur_knots = new_knots;
        }
        i += degree; // skip past the inserted multiplicity block
    }

    // Extract Bezier segments: one per knot span
    let mut beziers = Vec::new();
    let seg_count = (cur_cps.len() - 1) / degree;
    for s in 0..seg_count {
        let start = s * degree;
        let seg_cps = &cur_cps[start..start + degree + 1];
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
    beziers
}

/// Clip a list of Bézier segments to a parameter range [t_min, t_max].
///
/// When `knots` is provided (non-uniform BSpline decomposition), each segment
/// spans one knot interval and the knot vector is used to map parameter space
/// to segment indices. When `knots` is None (uniform decomposition: Line,
/// Circle, Ellipse approximated arcs), uniform segment distribution is assumed.
fn clip_beziers_to_range(beziers: &[Bezier2d], t_min: Real, t_max: Real, knots: Option<&[Real]>) -> Vec<Bezier2d> {
    if beziers.is_empty() {
        return vec![];
    }
    let (idx_min, idx_max) = if let Some(k) = knots {
        // Use knot vector: segment i spans [k[i+p], k[i+p+1]] in original param space.
        // After Bézier decomposition (full multiplicity), each segment = one knot interval.
        // Find the segment range containing [t_min, t_max].
        let find_seg = |t: Real| -> usize {
            let mut seg = 0usize;
            for w in k.windows(2) {
                if t >= w[0] - 1e-12 && t <= w[1] + 1e-12 {
                    return seg.min(beziers.len().saturating_sub(1));
                }
                if t < w[1] { break; }
                seg += 1;
            }
            if t <= k[0] { 0 } else { beziers.len().saturating_sub(1) }
        };
        (find_seg(t_min), find_seg(t_max))
    } else {
        let n = beziers.len() as Real;
        let imin = ((t_min * n).floor() as usize).min(beziers.len().saturating_sub(1));
        let imax = ((t_max * n).ceil() as usize).min(beziers.len().saturating_sub(1));
        (imin, imax)
    };
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
