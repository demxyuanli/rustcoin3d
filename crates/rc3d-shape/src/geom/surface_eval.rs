//! Surface geometry evaluation (SurfaceGeom enum and methods).

use rc3d_core::math::{Real, PVec3};
use super::curve_eval::{build_ortho_axes, find_param_on_curve, plane_tangent_basis, CurveGeom};
use crate::nurbs::NurbsSurface;

/// Parametric surface geometry (retained, not converted to NURBS).
#[derive(Debug, Clone)]
pub enum SurfaceGeom {
    Plane { origin: PVec3, normal: PVec3, u_dir: PVec3 },
    Cylinder { origin: PVec3, axis: PVec3, radius: Real, x_dir: PVec3, y_dir: PVec3 },
    Cone { apex: PVec3, axis: PVec3, semi_angle: Real, radius_at_apex: Real, x_dir: PVec3, y_dir: PVec3 },
    Sphere { center: PVec3, radius: Real },
    Torus { center: PVec3, axis: PVec3, major_r: Real, minor_r: Real, x_dir: PVec3, y_dir: PVec3 },
    BSpline(NurbsSurface),
    Extrusion { generatrix: Box<CurveGeom>, direction: PVec3 },
    Revolution { generatrix: Box<CurveGeom>, axis_origin: PVec3, axis_dir: PVec3 },
    Offset { basis: Box<SurfaceGeom>, distance: Real },
}

impl SurfaceGeom {
    /// Construct a Cylinder with pre-computed ortho axes.
    pub fn cylinder(origin: PVec3, axis: PVec3, radius: Real) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir }
    }
    /// Construct a Cone with pre-computed ortho axes.
    pub fn cone(apex: PVec3, axis: PVec3, semi_angle: Real, radius_at_apex: Real) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex, x_dir, y_dir }
    }
    /// Construct a Torus with pre-computed ortho axes.
    pub fn torus(center: PVec3, axis: PVec3, major_r: Real, minor_r: Real) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, y_dir }
    }
}

/// Rodrigues rotation: rotate point `p` around the line through `origin` along `axis`
/// by `angle` radians.  `axis` must be unit-length.
fn rotate_around_axis(p: PVec3, origin: PVec3, axis: PVec3, angle: Real) -> PVec3 {
    let rel = p - origin;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let dot = axis.dot(rel);
    let rot = rel * cos_a + axis.cross(rel) * sin_a + axis * dot * (1.0 - cos_a);
    origin + rot
}

/// Map a normalized parameter t ∈ [0, 1] into the B-spline knot-domain interval
/// [knots[degree], knots[count]].
fn map_to_knot_domain(knots: &[Real], degree: usize, count: usize, t: Real) -> Real {
    let t_min = knots[degree];
    let t_max = knots[count];
    t_min + t * (t_max - t_min)
}

/// Width of the B-spline knot-domain interval.
fn knot_domain_width(knots: &[Real], degree: usize, count: usize) -> Real {
    knots[count] - knots[degree]
}

/// Native surface parameter bounds (STEP / `BRepAdaptor_Surface` domain).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceParamRange {
    pub u_min: Real,
    pub u_max: Real,
    pub v_min: Real,
    pub v_max: Real,
}

impl SurfaceParamRange {
    pub fn u_span(&self) -> Real {
        (self.u_max - self.u_min).max(1e-12)
    }

    pub fn v_span(&self) -> Real {
        (self.v_max - self.v_min).max(1e-12)
    }

    pub fn normalize(&self, u: Real, v: Real) -> (Real, Real) {
        (
            (u - self.u_min) / self.u_span(),
            (v - self.v_min) / self.v_span(),
        )
    }

    pub fn denormalize(&self, u_norm: Real, v_norm: Real) -> (Real, Real) {
        (
            self.u_min + u_norm * self.u_span(),
            self.v_min + v_norm * self.v_span(),
        )
    }
}

/// Hierarchical grid search for surface projection.
///
/// Uses a 3-level strategy: coarse 4×4 grid → local 4×4 around top-3 candidates
/// → coordinate-descent refinement (8 iterations).  Total: ~105 evaluations vs
/// the 289+32=321 of a full 16×16 grid with refinement.
pub fn grid_project_2d(
    eval: impl Fn(Real, Real) -> PVec3,
    u_lo: Real, u_hi: Real, v_lo: Real, v_hi: Real,
    point: PVec3,
) -> (Real, Real) {
    /// Evaluate at (u, v), clamping to domain bounds.
    fn eval_at(eval: &impl Fn(Real, Real) -> PVec3, u: Real, v: Real,
               u_lo: Real, u_hi: Real, v_lo: Real, v_hi: Real, point: PVec3) -> (Real, Real, Real) {
        let u = u.clamp(u_lo, u_hi);
        let v = v.clamp(v_lo, v_hi);
        let d2 = (eval(u, v) - point).length_squared();
        (u, v, d2)
    }

    type Candidate = (Real, Real, Real); // (u, v, d2)
    let coarse = 4;
    let mut candidates: Vec<Candidate> = Vec::with_capacity((coarse + 1) * (coarse + 1));

    // Level 1: coarse grid
    for i in 0..=coarse {
        let u = u_lo + (u_hi - u_lo) * i as Real / coarse as Real;
        for j in 0..=coarse {
            let v = v_lo + (v_hi - v_lo) * j as Real / coarse as Real;
            candidates.push((u, v, (eval(u, v) - point).length_squared()));
        }
    }
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let u_seg = (u_hi - u_lo) / coarse as Real;
    let v_seg = (v_hi - v_lo) / coarse as Real;
    let local = 4;
    let mut best = (candidates[0].0, candidates[0].1, candidates[0].2);

    // Level 2: local 4×4 around top-3 coarse candidates
    for cand in candidates.iter().take(3) {
        for i in 0..=local {
            let u = cand.0 - u_seg + 2.0 * u_seg * i as Real / local as Real;
            for j in 0..=local {
                let v = cand.1 - v_seg + 2.0 * v_seg * j as Real / local as Real;
                let (_, _, d2) = eval_at(&eval, u, v, u_lo, u_hi, v_lo, v_hi, point);
                if d2 < best.2 { best = (u.clamp(u_lo, u_hi), v.clamp(v_lo, v_hi), d2); }
            }
        }
    }

    // Level 3: coordinate descent refinement
    let mut step_u = u_seg / local as Real * 0.5;
    let mut step_v = v_seg / local as Real * 0.5;
    for _ in 0..8 {
        for &(du, dv) in &[(step_u, 0.0), (-step_u, 0.0), (0.0, step_v), (0.0, -step_v)] {
            let (_, _, d2) = eval_at(&eval, best.0 + du, best.1 + dv, u_lo, u_hi, v_lo, v_hi, point);
            if d2 < best.2 {
                best = ((best.0 + du).clamp(u_lo, u_hi), (best.1 + dv).clamp(v_lo, v_hi), d2);
            }
        }
        step_u *= 0.5;
        step_v *= 0.5;
    }
    (best.0, best.1)
}

impl SurfaceGeom {
    /// Native parameter bounds in STEP / OCC surface space (not normalized [0,1]^2).
    pub fn param_range(&self) -> SurfaceParamRange {
        match self {
            SurfaceGeom::Plane { .. } => SurfaceParamRange {
                u_min: 0.0, u_max: 1.0, v_min: 0.0, v_max: 1.0,
            },
            SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. } => SurfaceParamRange {
                u_min: 0.0, u_max: std::f64::consts::TAU, v_min: -1.0e6, v_max: 1.0e6,
            },
            SurfaceGeom::Sphere { .. } => SurfaceParamRange {
                u_min: 0.0, u_max: std::f64::consts::TAU, v_min: 0.0, v_max: std::f64::consts::PI,
            },
            SurfaceGeom::Torus { .. } => SurfaceParamRange {
                u_min: 0.0, u_max: std::f64::consts::TAU, v_min: 0.0, v_max: std::f64::consts::TAU,
            },
            SurfaceGeom::BSpline(nurbs) => SurfaceParamRange {
                u_min: nurbs.knots_u[nurbs.degree_u],
                u_max: nurbs.knots_u[nurbs.u_count()],
                v_min: nurbs.knots_v[nurbs.degree_v],
                v_max: nurbs.knots_v[nurbs.v_count()],
            },
            SurfaceGeom::Extrusion { .. } => SurfaceParamRange {
                u_min: 0.0, u_max: 1.0, v_min: 0.0, v_max: 1.0,
            },
            SurfaceGeom::Revolution { .. } => SurfaceParamRange {
                u_min: 0.0, u_max: 1.0, v_min: 0.0, v_max: std::f64::consts::TAU,
            },
            SurfaceGeom::Offset { basis, .. } => basis.param_range(),
        }
    }

    /// Map native STEP / PCurve (u,v) into parameters accepted by `d0` / `d1` / `normal`.
    pub fn native_uv_to_d0(&self, u: Real, v: Real) -> (Real, Real) {
        match self {
            SurfaceGeom::Plane { .. } => (u, v),
            SurfaceGeom::Cylinder { .. } | SurfaceGeom::Cone { .. } => {
                (u / std::f64::consts::TAU, v)
            }
            SurfaceGeom::Sphere { .. } => {
                (u / std::f64::consts::TAU, v / std::f64::consts::PI)
            }
            SurfaceGeom::Torus { .. } => {
                (u / std::f64::consts::TAU, v / std::f64::consts::TAU)
            }
            SurfaceGeom::BSpline(_) => self.param_range().normalize(u, v),
            SurfaceGeom::Extrusion { .. } => (u, v),
            SurfaceGeom::Revolution { .. } => (u, v / std::f64::consts::TAU),
            SurfaceGeom::Offset { basis, .. } => basis.native_uv_to_d0(u, v),
        }
    }

    /// Map `d0` / `project` normalized parameters back to native STEP UV.
    pub fn d0_uv_to_native(&self, u: Real, v: Real) -> (Real, Real) {
        match self {
            SurfaceGeom::Plane { .. } => (u, v),
            SurfaceGeom::Cylinder { .. } | SurfaceGeom::Cone { .. } => {
                (u * std::f64::consts::TAU, v)
            }
            SurfaceGeom::Sphere { .. } => {
                (u * std::f64::consts::TAU, v * std::f64::consts::PI)
            }
            SurfaceGeom::Torus { .. } => {
                (u * std::f64::consts::TAU, v * std::f64::consts::TAU)
            }
            SurfaceGeom::BSpline(_) => self.param_range().denormalize(u, v),
            SurfaceGeom::Extrusion { .. } => (u, v),
            SurfaceGeom::Revolution { .. } => (u, v * std::f64::consts::TAU),
            SurfaceGeom::Offset { basis, .. } => basis.d0_uv_to_native(u, v),
        }
    }

    /// Evaluate surface at native STEP parameters.
    pub fn d0_native(&self, u: Real, v: Real) -> PVec3 {
        let (un, vn) = self.native_uv_to_d0(u, v);
        self.d0(un, vn)
    }

    /// Combined position + first + second derivatives at native parameters.
    /// For BSpline, uses `evaluate_with_hessian` to share basis computation.
    /// For analytical surfaces, calls `d0`, `d1`, `d2` separately (cheap).
    pub fn d0_d1_d2_native(&self, u: Real, v: Real) -> (PVec3, PVec3, PVec3, PVec3, PVec3, PVec3) {
        match self {
            SurfaceGeom::BSpline(nurbs) => {
                let u_k = map_to_knot_domain(
                    &nurbs.knots_u, nurbs.degree_u, nurbs.u_count(), u,
                );
                let v_k = map_to_knot_domain(
                    &nurbs.knots_v, nurbs.degree_v, nurbs.v_count(), v,
                );
                let u_w = knot_domain_width(&nurbs.knots_u, nurbs.degree_u, nurbs.u_count());
                let v_w = knot_domain_width(&nurbs.knots_v, nurbs.degree_v, nurbs.v_count());
                let (pos, du, dv, duu, duv, dvv) =
                    nurbs.evaluate_with_hessian(u_k, v_k);
                (
                    pos,
                    du * u_w, dv * v_w,
                    duu * u_w * u_w, duv * u_w * v_w, dvv * v_w * v_w,
                )
            }
            _ => {
                let s = self.d0_native(u, v);
                let (su, sv) = self.d1_native(u, v);
                let (suu, suv, svv) = self.d2(u, v);
                (s, su, sv, suu, suv, svv)
            }
        }
    }

    /// Native U period for closed/periodic surfaces (None if not periodic).
    pub fn native_u_period(&self) -> Option<Real> {
        match self {
            SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. }
            | SurfaceGeom::Sphere { .. }
            | SurfaceGeom::Torus { .. } => Some(std::f64::consts::TAU),
            SurfaceGeom::Revolution { .. } => None,
            SurfaceGeom::BSpline(nurbs) => nurbs.u_period(),
            SurfaceGeom::Offset { basis, .. } => basis.native_u_period(),
            _ => None,
        }
    }

    /// Native V period for closed/periodic surfaces (None if not periodic).
    pub fn native_v_period(&self) -> Option<Real> {
        match self {
            SurfaceGeom::Sphere { .. } => Some(std::f64::consts::PI),
            SurfaceGeom::Torus { .. } => Some(std::f64::consts::TAU),
            SurfaceGeom::Revolution { .. } => Some(std::f64::consts::TAU),
            SurfaceGeom::BSpline(nurbs) => nurbs.v_period(),
            SurfaceGeom::Offset { basis, .. } => basis.native_v_period(),
            _ => None,
        }
    }

    /// Surface normal at native STEP parameters.
    pub fn normal_native(&self, u: Real, v: Real) -> PVec3 {
        let (un, vn) = self.native_uv_to_d0(u, v);
        self.normal(un, vn)
    }

    /// Evaluate position at parameter (u, v) ∈ [0, 1]^2.
    pub fn d0(&self, u: Real, v: Real) -> PVec3 {
        match self {
            SurfaceGeom::Plane { origin, normal, u_dir } => {
                let (u_axis, v_axis) = plane_tangent_basis(*normal, *u_dir);
                *origin + u_axis * u + v_axis * v
            }
            SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir } => {
                let theta = u * std::f64::consts::TAU;
                let r = *radius;
                *origin
                    + *x_dir * r * theta.cos()
                    + *y_dir * r * theta.sin()
                    + *axis * v
            }
            SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex, x_dir, y_dir } => {
                let theta = u * std::f64::consts::TAU;
                let r = *radius_at_apex + v * semi_angle.tan();
                *apex
                    + *x_dir * r * theta.cos()
                    + *y_dir * r * theta.sin()
                    + *axis * v
            }
            SurfaceGeom::Sphere { center, radius } => {
                let theta = u * std::f64::consts::TAU;
                let phi = v * std::f64::consts::PI;
                let r = *radius;
                *center
                    + r * PVec3::new(
                        phi.sin() * theta.cos(),
                        phi.sin() * theta.sin(),
                        phi.cos(),
                    )
            }
            SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, y_dir } => {
                let theta = u * std::f64::consts::TAU;
                let phi = v * std::f64::consts::TAU;
                let r = *major_r + *minor_r * phi.cos();
                *center
                    + *x_dir * r * theta.cos()
                    + *y_dir * r * theta.sin()
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
                let angle = v * std::f64::consts::TAU;
                rotate_around_axis(generatrix.d0(u), *axis_origin, axis, angle)
            }
            SurfaceGeom::Offset { basis, distance } => {
                let p = basis.d0(u, v);
                let n = basis.normal(u, v);
                if p.is_nan() || n.is_nan() {
                    log::warn!("Offset d0: NaN at ({u},{v}), returning basis point");
                    return p;
                }
                p + n * *distance
            }
        }
    }

    /// Evaluate first-order partial derivatives (∂S/∂u, ∂S/∂v).
    pub fn d1(&self, u: Real, v: Real) -> (PVec3, PVec3) {
        match self {
            SurfaceGeom::Plane { normal, u_dir, .. } => {
                let (u_axis, v_axis) = plane_tangent_basis(*normal, *u_dir);
                (u_axis, v_axis)
            }
            SurfaceGeom::Cylinder { axis, radius, x_dir, y_dir, .. } => {
                let r = *radius;
                let theta = u * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                let du = twopi * r * (-theta.sin() * *x_dir + theta.cos() * *y_dir);
                (du, *axis)
            }
            SurfaceGeom::Cone { axis, semi_angle, radius_at_apex, x_dir, y_dir, .. } => {
                let tan_a = semi_angle.tan();
                let r = *radius_at_apex + v * tan_a;
                let theta = u * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                let du = twopi * r * (-theta.sin() * *x_dir + theta.cos() * *y_dir);
                let dv = tan_a * (theta.cos() * *x_dir + theta.sin() * *y_dir) + *axis;
                (du, dv)
            }
            SurfaceGeom::Sphere { radius, .. } => {
                let r = *radius;
                let theta = u * std::f64::consts::TAU;
                let phi = v * std::f64::consts::PI;
                let twopi = std::f64::consts::TAU;
                let pi = std::f64::consts::PI;
                let du = twopi * r * phi.sin() * PVec3::new(-theta.sin(), theta.cos(), 0.0);
                let dv = pi * r * PVec3::new(
                    phi.cos() * theta.cos(),
                    phi.cos() * theta.sin(),
                    -phi.sin(),
                );
                (du, dv)
            }
            SurfaceGeom::Torus { axis, major_r, minor_r, x_dir, y_dir, .. } => {
                let mr = *major_r;
                let nr = *minor_r;
                let theta = u * std::f64::consts::TAU;
                let phi = v * std::f64::consts::TAU;
                let twopi = std::f64::consts::TAU;
                let r = mr + nr * phi.cos();
                let du = twopi * r * (-theta.sin() * *x_dir + theta.cos() * *y_dir);
                let dv = twopi * (
                    *x_dir * (-nr * phi.sin() * theta.cos())
                    + *y_dir * (-nr * phi.sin() * theta.sin())
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
                let (_, du_k, dv_k) = nurbs.evaluate_with_derivative(u_k, v_k);
                (du_k * u_w, dv_k * v_w)
            }
            SurfaceGeom::Extrusion { generatrix, direction } => {
                (generatrix.d1(u), *direction)
            }
            SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
                let axis = axis_dir.normalize();
                let angle = v * std::f64::consts::TAU;
                let gen_p = generatrix.d0(u);
                let gen_d1 = generatrix.d1(u);
                let du = rotate_around_axis(gen_d1, *axis_origin, axis, angle);
                let rotated_p = rotate_around_axis(gen_p, *axis_origin, axis, angle);
                let dv = axis.cross(rotated_p - *axis_origin) * std::f64::consts::TAU;
                (du, dv)
            }
            SurfaceGeom::Offset { basis, distance } => {
                let (db_du, db_dv) = basis.d1(u, v);
                let (db_uu, db_uv, db_vv) = basis.d2(u, v);

                // First fundamental form
                let e = db_du.dot(db_du);
                let f = db_du.dot(db_dv);
                let g = db_dv.dot(db_dv);
                let det1 = e * g - f * f;

                if det1.abs() < 1e-10 {
                    // Degenerate metric — fall back to numerical differentiation
                    log::warn!("offset d1: degenerate metric (det={det1}), using numerical fallback");
                    let eps = 1e-3_f64;
                    let n_u1 = basis.normal(u + eps, v);
                    let n_u0 = basis.normal(u - eps, v);
                    let n_v1 = basis.normal(u, v + eps);
                    let n_v0 = basis.normal(u, v - eps);
                    let mut dn_du = (n_u1 - n_u0) / (2.0 * eps);
                    let mut dn_dv = (n_v1 - n_v0) / (2.0 * eps);
                    if dn_du.is_nan() { dn_du = PVec3::ZERO; }
                    if dn_dv.is_nan() { dn_dv = PVec3::ZERO; }
                    let d = *distance;
                    return (db_du + dn_du * d, db_dv + dn_dv * d);
                }

                // Surface normal (unnormalized cross, then normalize)
                let n = db_du.cross(db_dv);
                let n_len = n.length();
                let n_hat = n * (1.0 / n_len.max(1e-12));

                // Second fundamental form
                let big_l = db_uu.dot(n_hat);
                let big_m = db_uv.dot(n_hat);
                let big_n = db_vv.dot(n_hat);

                // Weingarten matrix W = I⁻¹ · II
                // W = [W11 W12; W21 W22] where
                //   W11 = (g*L - f*M) / det1,  W12 = (g*M - f*N) / det1
                //   W21 = (e*M - f*L) / det1,  W22 = (e*N - f*M) / det1
                let inv_det = 1.0 / det1;
                let w11 = (g * big_l - f * big_m) * inv_det;
                let w12 = (g * big_m - f * big_n) * inv_det;
                let w21 = (e * big_m - f * big_l) * inv_det;
                let w22 = (e * big_n - f * big_m) * inv_det;

                // dn/du = -(W11 * db_du + W21 * db_dv)
                // dn/dv = -(W12 * db_du + W22 * db_dv)
                let dn_du = (db_du * w11 + db_dv * w21) * -1.0;
                let dn_dv = (db_du * w12 + db_dv * w22) * -1.0;

                let d = *distance;
                (db_du + dn_du * d, db_dv + dn_dv * d)
            }
        }
    }

    /// Surface normal = (∂S/∂u × ∂S/∂v) normalized.
    /// For revolution surfaces we use (∂S/∂v × ∂S/∂u), which matches CAD outward
    /// convention for the native `(u=profile, v=angle)` parameterization.
    /// Handles degeneracy by probing a neighborhood with small epsilon offsets.
    pub fn normal(&self, u: Real, v: Real) -> PVec3 {
        let (du, dv) = self.d1(u, v);
        let n = if matches!(self, SurfaceGeom::Revolution { .. }) {
            dv.cross(du)
        } else {
            du.cross(dv)
        };
        let len = n.length();
        if len > 1e-6 {
            return n * (1.0 / len);
        }
        let eps = 1e-3_f64;
        let probes = [(u + eps, v), (u - eps, v), (u, v + eps), (u, v - eps)];
        let mut best = PVec3::Z;
        let mut best_len = 0.0_f64;
        for (up, vp) in probes {
            if !(0.0..=1.0).contains(&up) || !(0.0..=1.0).contains(&vp) {
                continue;
            }
            let (du2, dv2) = self.d1(up, vp);
            let n2 = if matches!(self, SurfaceGeom::Revolution { .. }) {
                dv2.cross(du2)
            } else {
                du2.cross(dv2)
            };
            let l2 = n2.length();
            if l2 > best_len {
                best_len = l2;
                best = n2 * (1.0 / l2.max(1e-12));
            }
        }
        best
    }

    /// Evaluate surface at native UV coordinates returned by `project` / PCurve.
    pub fn d0_at_native_uv(&self, u: Real, v: Real) -> PVec3 {
        match self {
            SurfaceGeom::BSpline(nurbs) => nurbs.evaluate(u, v),
            SurfaceGeom::Offset { basis, distance } => {
                let (un, vn) = basis.native_uv_to_d0(u, v);
                let p = basis.d0(un, vn);
                let n = basis.normal(un, vn);
                if p.is_nan() || n.is_nan() {
                    log::warn!("Offset d0_at_native: NaN at ({u},{v})→({un},{vn}), returning basis point");
                    return p;
                }
                p + n * *distance
            }
            _ => self.d0_native(u, v),
        }
    }

    /// Returns native STEP UV. `None` for torus, extrusion, offset.
    pub fn project(&self, point: PVec3) -> Option<(Real, Real)> {
        match self {
            SurfaceGeom::Plane { origin, normal, u_dir } => {
                let (u_axis, v_axis) = plane_tangent_basis(*normal, *u_dir);
                let rel = point - *origin;
                Some((rel.dot(u_axis), rel.dot(v_axis)))
            }
            SurfaceGeom::Cylinder { origin, axis, x_dir, y_dir, .. } => {
                let a = axis.normalize();
                let rel = point - *origin;
                let v = rel.dot(a);
                let radial = rel - a * v;
                let u_raw = Real::atan2(radial.dot(*y_dir), radial.dot(*x_dir));
                let u = if u_raw < 0.0 { u_raw / std::f64::consts::TAU + 1.0 } else { u_raw / std::f64::consts::TAU };
                Some(self.d0_uv_to_native(u, v))
            }
            SurfaceGeom::Cone { apex, axis, x_dir, y_dir, .. } => {
                let a = axis.normalize();
                let rel = point - *apex;
                let v = rel.dot(a);
                let radial = rel - a * v;
                let u_raw = Real::atan2(radial.dot(*y_dir), radial.dot(*x_dir));
                let u = if u_raw < 0.0 { u_raw / std::f64::consts::TAU + 1.0 } else { u_raw / std::f64::consts::TAU };
                Some(self.d0_uv_to_native(u, v))
            }
            SurfaceGeom::Sphere { center, .. } => {
                let rel = point - *center;
                let r = rel.length();
                if r < 1e-12 { return Some((0.0, 0.0)); }
                let phi = (rel.z / r).clamp(-1.0, 1.0).acos();
                let v = phi / std::f64::consts::PI;
                let u_raw = Real::atan2(rel.y, rel.x);
                let u = if u_raw < 0.0 { u_raw / std::f64::consts::TAU + 1.0 } else { u_raw / std::f64::consts::TAU };
                Some(self.d0_uv_to_native(u, v))
            }
            SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
                let axis = axis_dir.normalize();
                let rel = point - *axis_origin;
                let along = rel.dot(axis);
                let radial = rel - axis * along;
                let r = radial.length();
                if r < 1e-10 { return Some(self.d0_uv_to_native(0.5, 0.0)); }
                let (x_dir, y_dir) = build_ortho_axes(axis);
                let u_raw = Real::atan2(radial.dot(y_dir), radial.dot(x_dir));
                let v = if u_raw < 0.0 { u_raw / std::f64::consts::TAU + 1.0 } else { u_raw / std::f64::consts::TAU };
                let angle = v * std::f64::consts::TAU;
                let unrotated = rotate_around_axis(point, *axis_origin, axis, -angle);
                let u = find_param_on_curve(generatrix, unrotated);
                Some(self.d0_uv_to_native(u, v))
            }
            SurfaceGeom::BSpline(_) | SurfaceGeom::Torus { .. } => {
                // Use Newton-Raphson; falls back to grid_project_2d internally
                let candidates = super::project::project_point_on_surface(self, point);
                if let Some((u, v, _)) = candidates.first() {
                    return Some((*u, *v));
                }
                None
            }
            SurfaceGeom::Extrusion { generatrix, direction } => {
                let dir = direction.normalize();
                let (u_lo, u_hi) = generatrix.native_param_range();
                let n = 64;
                let mut best_u = u_lo;
                let mut best_v = 0.0_f64;
                let mut best_d2 = f64::MAX;
                for i in 0..=n {
                    let u = u_lo + (u_hi - u_lo) * i as Real / n as Real;
                    let g = generatrix.d0(u);
                    let v = (point - g).dot(dir);
                    let d2 = (g + dir * v - point).length_squared();
                    if d2 < best_d2 { best_d2 = d2; best_u = u; best_v = v; }
                }
                let mut step = (u_hi - u_lo).max(1e-6) / (n as Real * 2.0);
                for _ in 0..8 {
                    for &du in &[-step, step] {
                        let u = (best_u + du).clamp(u_lo, u_hi);
                        let g = generatrix.d0(u);
                        let v = (point - g).dot(dir);
                        let d2 = (g + dir * v - point).length_squared();
                        if d2 < best_d2 { best_d2 = d2; best_u = u; best_v = v; }
                    }
                    step *= 0.5;
                }
                Some((best_u, best_v))
            }
            SurfaceGeom::Offset { basis, distance } => {
                let (u_native, v_native) = basis.project(point)?;
                if matches!(basis.as_ref(), SurfaceGeom::Plane { .. }) {
                    return Some((u_native, v_native));
                }
                let (mut u, mut v) = basis.native_uv_to_d0(u_native, v_native);
                let pr = basis.param_range();
                let mut best_d2 = {
                    let p = basis.d0(u, v) + basis.normal(u, v) * *distance;
                    (p - point).length_squared()
                };
                let mut step = 0.05_f64;
                for _ in 0..8 {
                    for &(du, dv) in &[(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
                        let nu = (u + du).clamp(pr.u_min, pr.u_max);
                        let nv = (v + dv).clamp(pr.v_min, pr.v_max);
                        let p = basis.d0(nu, nv) + basis.normal(nu, nv) * *distance;
                        let d2 = (p - point).length_squared();
                        if d2 < best_d2 { best_d2 = d2; u = nu; v = nv; }
                    }
                    step *= 0.5;
                }
                Some(basis.d0_uv_to_native(u, v))
            }
        }
    }

    /// Map a 3D point to native surface UV; validates `project` and falls back to grid search.
    pub fn inverse_native_uv(&self, point: PVec3, max_dist: Real) -> Option<(Real, Real)> {
        if let SurfaceGeom::Offset { basis, distance } = self {
            return Self::inverse_native_uv_offset(basis, *distance, point, max_dist);
        }
        if let Some(uv) = self.project(point) {
            let err = (self.d0_native(uv.0, uv.1) - point).length();
            if err <= max_dist { return Some(uv); }
        }
        let search_tol = max_dist.max(0.5);
        let uv = self.grid_search_native_uv(point, search_tol, 16, 10)?;
        let err = (self.d0_native(uv.0, uv.1) - point).length();
        if err <= max_dist { Some(uv) } else { None }
    }

    fn inverse_native_uv_offset(basis: &SurfaceGeom, distance: Real, point: PVec3, max_dist: Real) -> Option<(Real, Real)> {
        let search_tol = max_dist.max(0.5);
        let (mut u, mut v) = basis.project(point)?;
        let mut best_u = u;
        let mut best_v = v;
        let mut best_err = (basis.d0_native(u, v) + basis.normal_native(u, v) * distance - point).length();
        for _ in 0..16 {
            if best_err <= max_dist { return Some((best_u, best_v)); }
            let target = point - basis.normal_native(u, v) * distance;
            if let Some((u2, v2)) = basis.inverse_native_uv(target, search_tol) {
                u = u2; v = v2;
                let err = (basis.d0_native(u, v) + basis.normal_native(u, v) * distance - point).length();
                if err < best_err { best_err = err; best_u = u; best_v = v; }
            } else { break; }
        }
        if best_err <= max_dist { Some((best_u, best_v)) } else { None }
    }

    /// Build-time UV inverse (coarser grid) for PCurve synthesis during StepToTopoDS.
    pub fn inverse_native_uv_build(&self, point: PVec3, max_dist: Real) -> Option<(Real, Real)> {
        if let SurfaceGeom::Offset { basis, distance } = self {
            return Self::inverse_native_uv_offset(basis, *distance, point, max_dist);
        }
        if let Some(uv) = self.project(point) {
            let err = (self.d0_native(uv.0, uv.1) - point).length();
            if err <= max_dist { return Some(uv); }
        }
        let search_tol = max_dist.max(0.5);
        let uv = self.grid_search_native_uv(point, search_tol, 8, 5)?;
        let err = (self.d0_native(uv.0, uv.1) - point).length();
        if err <= max_dist { Some(uv) } else { None }
    }

    fn grid_search_native_uv(&self, point: PVec3, max_dist: Real, grid: u32, refine_iters: u32) -> Option<(Real, Real)> {
        let r = self.param_range();
        let (u_lo, u_hi, v_lo, v_hi) = self.native_search_window(&r, point);
        let u_span = (u_hi - u_lo).max(1e-12);
        let v_span = (v_hi - v_lo).max(1e-12);
        if u_span <= 1e-12 && v_span <= 1e-12 { return None; }
        let grid = grid.max(1);
        let mut best_u = u_lo;
        let mut best_v = v_lo;
        let mut best_d2 = f64::MAX;
        for i in 0..=grid {
            let u = u_lo + u_span * i as Real / grid as Real;
            for j in 0..=grid {
                let v = v_lo + v_span * j as Real / grid as Real;
                let d2 = (self.d0_native(u, v) - point).length_squared();
                if d2 < best_d2 { best_d2 = d2; best_u = u; best_v = v; }
            }
        }
        let mut step_u = u_span / grid as Real * 0.5;
        let mut step_v = v_span / grid as Real * 0.5;
        for _ in 0..refine_iters {
            for &(du, dv) in &[(step_u, 0.0), (-step_u, 0.0), (0.0, step_v), (0.0, -step_v)] {
                let nu = (best_u + du).clamp(u_lo, u_hi);
                let nv = (best_v + dv).clamp(v_lo, v_hi);
                let d2 = (self.d0_native(nu, nv) - point).length_squared();
                if d2 < best_d2 { best_d2 = d2; best_u = nu; best_v = nv; }
            }
            step_u *= 0.5;
            step_v *= 0.5;
        }
        if best_d2.sqrt() > max_dist { return None; }
        Some((best_u, best_v))
    }

    fn native_search_window(&self, range: &SurfaceParamRange, point: PVec3) -> (Real, Real, Real, Real) {
        match self {
            SurfaceGeom::Cylinder { .. } | SurfaceGeom::Cone { .. } => {
                let v_seed = self.project(point).map(|(_, v)| v).unwrap_or(0.0);
                let half = 50.0_f64;
                (range.u_min, range.u_max, v_seed - half, v_seed + half)
            }
            SurfaceGeom::Plane { .. } => {
                if let Some((u, v)) = self.project(point) {
                    let half = 10.0_f64;
                    (u - half, u + half, v - half, v + half)
                } else {
                    (range.u_min, range.u_max, range.v_min, range.v_max)
                }
            }
            _ => (range.u_min, range.u_max, range.v_min, range.v_max),
        }
    }

    /// Evaluate a uniform grid of (n_u+1) x (n_v+1) 3D points.
    pub fn evaluate_grid(&self, u_range: (Real, Real), v_range: (Real, Real), n_u: usize, n_v: usize) -> Vec<Vec<PVec3>> {
        let mut grid = Vec::with_capacity(n_u + 1);
        for i in 0..=n_u {
            let u = u_range.0 + (u_range.1 - u_range.0) * i as Real / n_u.max(1) as Real;
            let mut row = Vec::with_capacity(n_v + 1);
            for j in 0..=n_v {
                let v = v_range.0 + (v_range.1 - v_range.0) * j as Real / n_v.max(1) as Real;
                row.push(self.d0(u, v));
            }
            grid.push(row);
        }
        grid
    }

    /// Generatrix curve parameter in [0,1] for a 3D point on a revolution surface.
    pub fn revolution_generatrix_u_at(&self, point: PVec3) -> Option<Real> {
        let SurfaceGeom::Revolution { generatrix, .. } = self else { return None; };
        Some(find_param_on_curve(generatrix, point))
    }

    /// Revolution native (u,v): u=generatrix parameter, v=axis angle in radians [0,TAU].
    pub fn revolution_native_uv_at(&self, point: PVec3) -> Option<(Real, Real)> {
        let SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } = self else { return None; };
        let axis = axis_dir.normalize();
        let rel = point - *axis_origin;
        let radial = rel - axis * rel.dot(axis);
        let (x_dir, y_dir) = build_ortho_axes(axis);
        let angle = if radial.length_squared() < 1e-12 { 0.0_f64 }
        else { let u_raw = Real::atan2(radial.dot(y_dir), radial.dot(x_dir));
            if u_raw < 0.0 { u_raw + std::f64::consts::TAU } else { u_raw } };
        let unrotated = rotate_around_axis(point, *axis_origin, axis, -angle);
        let u = find_param_on_curve(generatrix, unrotated);
        Some((u, angle))
    }

    /// Partial derivatives at native STEP parameters.
    pub fn d1_native(&self, u: Real, v: Real) -> (PVec3, PVec3) {
        let (un, vn) = self.native_uv_to_d0(u, v);
        let (su, sv) = self.d1(un, vn);
        match self {
            SurfaceGeom::Plane { .. } | SurfaceGeom::Extrusion { .. } => (su, sv),
            SurfaceGeom::Cylinder { .. } | SurfaceGeom::Cone { .. } => (su / std::f64::consts::TAU, sv),
            SurfaceGeom::Sphere { .. } => (su / std::f64::consts::TAU, sv / std::f64::consts::PI),
            SurfaceGeom::Torus { .. } => (su / std::f64::consts::TAU, sv / std::f64::consts::TAU),
            SurfaceGeom::BSpline(_) => { let pr = self.param_range(); (su / pr.u_span(), sv / pr.v_span()) }
            SurfaceGeom::Revolution { .. } => (su, sv / std::f64::consts::TAU),
            SurfaceGeom::Offset { basis, .. } => basis.d1_native(u, v),
        }
    }

    /// Adaptive parameter-space subdivision for structured interior grid generation.
    pub fn parameter_division(
        &self,
        range: (Real, Real, Real, Real),
        tol: Real,
        max_depth: usize,
    ) -> (Vec<Real>, Vec<Real>) {
        if let SurfaceGeom::Offset { basis, .. } = self {
            if matches!(basis.as_ref(), SurfaceGeom::Revolution { .. }) {
                return basis.parameter_division(range, tol, max_depth);
            }
        }
        let (u_min, u_max, v_min, v_max) = range;
        let mut u_divs = vec![u_min, u_max];
        let mut v_divs = vec![v_min, v_max];
        let (min_u, min_v) = match self {
            SurfaceGeom::Sphere { .. } | SurfaceGeom::Torus { .. } => (8, 8),
            SurfaceGeom::Cylinder { .. } | SurfaceGeom::Cone { .. } => (8, 4),
            SurfaceGeom::Revolution { .. } => (8, 16),
            SurfaceGeom::BSpline(_) => (4, 4),
            _ => (2, 2),
        };
        for i in 1..min_u { u_divs.push(u_min + (u_max - u_min) * i as Real / min_u as Real); }
        for j in 1..min_v { v_divs.push(v_min + (v_max - v_min) * j as Real / min_v as Real); }
        u_divs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        u_divs.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
        v_divs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v_divs.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
        let max_depth = max_depth.max(1);
        for _ in 0..max_depth {
            let mut new_u = Vec::new(); let mut new_v = Vec::new(); let mut any = false;
            for i in 0..u_divs.len().saturating_sub(1) {
                let u0 = u_divs[i]; let u1 = u_divs[i + 1];
                for j in 0..v_divs.len().saturating_sub(1) {
                    let v0 = v_divs[j]; let v1 = v_divs[j + 1];
                    let p00 = self.d0_native(u0, v0); let p01 = self.d0_native(u0, v1);
                    let p10 = self.d0_native(u1, v0); let p11 = self.d0_native(u1, v1);
                    let pc = self.d0_native((u0 + u1) * 0.5, (v0 + v1) * 0.5);
                    let bilin = (p00 + p01 + p10 + p11) * 0.25;
                    if (pc - bilin).length() > tol {
                        let delu = ((p00 + p01) * 0.5 - self.d0_native(u0, (v0 + v1) * 0.5)).length()
                            + ((p10 + p11) * 0.5 - self.d0_native(u1, (v0 + v1) * 0.5)).length();
                        let delv = ((p00 + p10) * 0.5 - self.d0_native((u0 + u1) * 0.5, v0)).length()
                            + ((p01 + p11) * 0.5 - self.d0_native((u0 + u1) * 0.5, v1)).length();
                        if delu > delv * 2.0 { new_u.push((u0 + u1) * 0.5); any = true; }
                        else if delv > delu * 2.0 { new_v.push((v0 + v1) * 0.5); any = true; }
                        else { new_u.push((u0 + u1) * 0.5); new_v.push((v0 + v1) * 0.5); any = true; }
                    }
                }
            }
            if !any { break; }
            u_divs.append(&mut new_u); v_divs.append(&mut new_v);
            u_divs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            u_divs.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
            v_divs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v_divs.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
        }
        (u_divs, v_divs)
    }

    /// Check if a revolution PCurve UV matches the 3D point within tolerance.
    pub fn revolution_pcurve_matches_3d(&self, pt: PVec3, u: Real, v: Real, tol: Real) -> bool {
        (pt - self.d0_native(u, v)).length() <= tol
    }

    /// Revolution PCurve UV canonicalization for periodic matching.
    pub fn revolution_canonicalize_pcurve_uv(&self, u: Real, v: Real) -> (Real, Real) {
        if !matches!(self, SurfaceGeom::Revolution { .. }) { return (u, v); }
        const TAU: Real = std::f64::consts::TAU;
        let mut uc = u;
        while uc < 0.0 { uc += TAU; }
        while uc >= TAU { uc -= TAU; }
        (uc, v)
    }

    /// Second partial derivatives (d²S/du², d²S/dudv, d²S/dv²).
    ///
    /// Analytical for Plane/Cylinder/Cone/Sphere/Torus; numerical fallback
    /// (central differences on d1) for BSpline/Extrusion/Revolution/Offset.
    pub fn d2(&self, u: Real, v: Real) -> (PVec3, PVec3, PVec3) {
        match self {
            SurfaceGeom::Plane { .. } => (PVec3::ZERO, PVec3::ZERO, PVec3::ZERO),

            SurfaceGeom::Cylinder { radius, x_dir, y_dir, .. } => {
                let r = *radius;
                let theta = u * std::f64::consts::TAU;
                let t2 = std::f64::consts::TAU * std::f64::consts::TAU;
                let duu = -t2 * r * (theta.cos() * *x_dir + theta.sin() * *y_dir);
                (duu, PVec3::ZERO, PVec3::ZERO)
            }

            SurfaceGeom::Cone { semi_angle, radius_at_apex, x_dir, y_dir, .. } => {
                let tan_a = semi_angle.tan();
                let r = *radius_at_apex + v * tan_a;
                let theta = u * std::f64::consts::TAU;
                let t = std::f64::consts::TAU;
                let t2 = t * t;
                let duu = -t2 * r * (theta.cos() * *x_dir + theta.sin() * *y_dir);
                let duv = t * tan_a * (-theta.sin() * *x_dir + theta.cos() * *y_dir);
                (duu, duv, PVec3::ZERO)
            }

            SurfaceGeom::Sphere { radius, .. } => {
                let r = *radius;
                let theta = u * std::f64::consts::TAU;
                let phi = v * std::f64::consts::PI;
                let t = std::f64::consts::TAU;
                let pi = std::f64::consts::PI;
                let duu = -t * t * r * phi.sin()
                    * PVec3::new(theta.cos(), theta.sin(), 0.0);
                let duv = t * pi * r * phi.cos()
                    * PVec3::new(-theta.sin(), theta.cos(), 0.0);
                let dvv = -pi * pi * r
                    * PVec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());
                (duu, duv, dvv)
            }

            SurfaceGeom::Torus { axis, major_r, minor_r, x_dir, y_dir, .. } => {
                let mr = *major_r;
                let nr = *minor_r;
                let theta = u * std::f64::consts::TAU;
                let phi = v * std::f64::consts::TAU;
                let t = std::f64::consts::TAU;
                let t2 = t * t;
                let r = mr + nr * phi.cos();
                let duu = -t2 * r * (theta.cos() * *x_dir + theta.sin() * *y_dir);
                let duv = t2 * nr * (-phi.sin())
                    * (-theta.sin() * *x_dir + theta.cos() * *y_dir);
                let dvv = t2 * (
                    *x_dir * (-nr * phi.cos() * theta.cos())
                    + *y_dir * (-nr * phi.cos() * theta.sin())
                    + *axis * (-nr * phi.sin())
                );
                (duu, duv, dvv)
            }

            // BSpline: use evaluate_with_hessian (shares basis computation).
            SurfaceGeom::BSpline(nurbs) => {
                let u_k = map_to_knot_domain(
                    &nurbs.knots_u, nurbs.degree_u, nurbs.u_count(), u,
                );
                let v_k = map_to_knot_domain(
                    &nurbs.knots_v, nurbs.degree_v, nurbs.v_count(), v,
                );
                let u_w = knot_domain_width(&nurbs.knots_u, nurbs.degree_u, nurbs.u_count());
                let v_w = knot_domain_width(&nurbs.knots_v, nurbs.degree_v, nurbs.v_count());
                let (_pos, _du, _dv, duu, duv, dvv) =
                    nurbs.evaluate_with_hessian(u_k, v_k);
                (duu * u_w * u_w, duv * u_w * v_w, dvv * v_w * v_w)
            }

            // Extrusion: analytical second derivatives
            SurfaceGeom::Extrusion { generatrix, .. } => {
                // S(u,v) = generatrix(u) + direction * v
                // ∂²S/∂u² = generatrix.d2(u)
                // ∂²S/∂u∂v = 0 (direction is constant w.r.t u)
                // ∂²S/∂v² = 0 (linear in v)
                (generatrix.d2(u), PVec3::ZERO, PVec3::ZERO)
            }

            // Revolution: analytical second derivatives
            SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
                let axis = axis_dir.normalize();
                let angle = v * std::f64::consts::TAU;
                let gen_p = generatrix.d0(u);
                let gen_d1 = generatrix.d1(u);
                let gen_d2 = generatrix.d2(u);
                let rotated_p = rotate_around_axis(gen_p, *axis_origin, axis, angle);

                // ∂²S/∂u² = rotate(generatrix.d2(u))
                let duu = rotate_around_axis(gen_d2, *axis_origin, axis, angle);

                // ∂²S/∂u∂v = TAU * axis × (∂S/∂u)
                // ∂S/∂u = rotate(gen_d1)
                let du = rotate_around_axis(gen_d1, *axis_origin, axis, angle);
                let duv = axis.cross(du) * std::f64::consts::TAU;

                // ∂²S/∂v² = -TAU² * radial component
                // radial = (rotated_p - origin) - axis * (axis · (rotated_p - origin))
                let rel = rotated_p - *axis_origin;
                let radial = rel - axis * axis.dot(rel);
                let dvv = radial * (-std::f64::consts::TAU * std::f64::consts::TAU);

                (duu, duv, dvv)
            }

            // Offset: numerical fallback (d2 requires Weingarten + basis d2)
            SurfaceGeom::Offset { .. } => {
                let eps = 1e-4_f64;
                let (du_p, _dv_p) = self.d1(u + eps, v);
                let (du_m, _dv_m) = self.d1(u - eps, v);
                let (du_vp, dv_vp) = self.d1(u, v + eps);
                let (du_vm, dv_vm) = self.d1(u, v - eps);
                let duu = (du_p - du_m) / (2.0 * eps);
                let duv = (du_vp - du_vm) / (2.0 * eps);
                let dvv = (dv_vp - dv_vm) / (2.0 * eps);
                (duu, duv, dvv)
            }
        }
    }

    /// Estimate minimum principal curvature radius at normalized (u, v) ∈ [0,1]².
    /// Returns f64::MAX for flat surfaces (zero curvature).
    pub fn min_curvature_radius(&self, u: Real, v: Real) -> Real {
        let (du, dv) = self.d1(u, v);
        let (duu, duv, dvv) = self.d2(u, v);

        // First fundamental form coefficients
        let e = du.dot(du);
        let f = du.dot(dv);
        let g = dv.dot(dv);

        // Surface normal
        let n = du.cross(dv);
        let n_len = n.length();
        if n_len < 1e-10 { return f64::MAX; }
        let n_hat = n * (1.0 / n_len);

        // Second fundamental form coefficients
        let l = duu.dot(n_hat);
        let m = duv.dot(n_hat);
        let nn = dvv.dot(n_hat);

        // Principal curvatures from shape operator
        let det1 = e * g - f * f;
        if det1.abs() < 1e-10 { return f64::MAX; }

        let trace = (l * g - 2.0 * m * f + nn * e) / det1;
        let det2 = (l * nn - m * m) / det1;

        let disc = (trace * trace - 4.0 * det2).max(0.0);
        let k1 = (trace + disc.sqrt()) / 2.0;
        let k2 = (trace - disc.sqrt()) / 2.0;

        let max_k = k1.abs().max(k2.abs());
        if max_k < 1e-10 { f64::MAX } else { 1.0 / max_k }
    }
}

/// Check if an Offset surface might self-intersect.
/// Returns true if the offset distance exceeds the minimum curvature radius
/// at any sample point on a grid.
pub fn offset_may_self_intersect(
    basis: &SurfaceGeom,
    distance: Real,
    sample_grid: usize,
) -> bool {
    for i in 0..=sample_grid {
        let u = i as Real / sample_grid as Real;
        for j in 0..=sample_grid {
            let v = j as Real / sample_grid as Real;
            let r = basis.min_curvature_radius(u, v);
            if distance.abs() > r && r < f64::MAX {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_d0() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let p = plane.d0(2.0, 3.0);
        assert!((p - PVec3::new(2.0, 3.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_plane_d1_matches_axes() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let (du, dv) = plane.d1(0.0, 0.0);
        assert!((du - PVec3::X).length() < 1e-6);
        assert!((dv - PVec3::Y).length() < 1e-6);
    }

    #[test]
    fn test_plane_normal() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let n = plane.normal(0.0, 0.0);
        assert!((n - PVec3::Z).length() < 1e-6);
    }

    #[test]
    fn test_plane_project() {
        let plane = SurfaceGeom::Plane { origin: PVec3::new(1.0, 0.0, 0.0), normal: PVec3::Z, u_dir: PVec3::X };
        let (u, v) = plane.project(PVec3::new(5.0, 3.0, 0.0)).unwrap();
        assert!((u - 4.0).abs() < 1e-4);
        assert!((v - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_plane_evaluate_grid_shape() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let grid = plane.evaluate_grid((0.0, 1.0), (0.0, 1.0), 3, 2);
        assert_eq!(grid.len(), 4);
        assert_eq!(grid[0].len(), 3);
    }

    #[test]
    fn test_cylinder_d0_on_surface() {
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 2.0);
        let p = cyl.d0(0.0, 5.0);
        assert!((p - PVec3::new(2.0, 0.0, 5.0)).length() < 1e-4);
    }

    #[test]
    fn test_cylinder_d1_du_is_tangent_dv_is_axis() {
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 1.0);
        let (du, dv) = cyl.d1(0.0, 1.0);
        assert!((dv - PVec3::Z).length() < 1e-6);
        assert!(du.dot(PVec3::Z).abs() < 1e-6);
        assert!((du.length() - std::f64::consts::TAU).abs() < 1e-4);
    }

    #[test]
    fn test_cylinder_normal_is_radial() {
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 1.0);
        let n = cyl.normal(0.0, 0.0);
        assert!((n - PVec3::X).length() < 1e-4);
        let n2 = cyl.normal(0.25, 0.0);
        assert!((n2 - PVec3::Y).length() < 1e-4);
    }

    #[test]
    fn test_cylinder_project() {
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 2.0);
        let (u, v) = cyl.project(PVec3::new(2.0, 0.0, 5.0)).unwrap();
        assert!(u.abs() < 1e-4);
        assert!((v - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_cone_d0_at_apex() {
        let cone = SurfaceGeom::cone(PVec3::ZERO, PVec3::Z, std::f64::consts::FRAC_PI_4, 0.0);
        let p = cone.d0(0.0, 0.0);
        assert!(p.length() < 1e-6);
    }

    #[test]
    fn test_cone_d0_radius_grows_with_v() {
        let cone = SurfaceGeom::cone(PVec3::ZERO, PVec3::Z, std::f64::consts::FRAC_PI_4, 0.0);
        let p = cone.d0(0.0, 1.0);
        let r = (p.x * p.x + p.y * p.y).sqrt();
        assert!((r - 1.0).abs() < 1e-4);
        assert!((p.z - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_cone_d1_dv_has_axis_component() {
        let cone = SurfaceGeom::cone(PVec3::ZERO, PVec3::Z, std::f64::consts::FRAC_PI_4, 1.0);
        let (_du, dv) = cone.d1(0.0, 0.5);
        assert!(dv.z > 0.5);
    }

    #[test]
    fn test_sphere_d0_at_equator() {
        let sphere = SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 2.0 };
        let p = sphere.d0(0.0, 0.5);
        assert!((p - PVec3::new(2.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn test_sphere_d0_at_poles() {
        let sphere = SurfaceGeom::Sphere { center: PVec3::new(1.0, 0.0, 0.0), radius: 3.0 };
        let p_north = sphere.d0(0.0, 0.0);
        assert!((p_north - PVec3::new(1.0, 0.0, 3.0)).length() < 1e-4);
        let p_south = sphere.d0(0.0, 1.0);
        assert!((p_south - PVec3::new(1.0, 0.0, -3.0)).length() < 1e-4);
    }

    #[test]
    fn test_sphere_radius_consistency() {
        let sphere = SurfaceGeom::Sphere { center: PVec3::new(10.0, 0.0, 0.0), radius: 5.0 };
        for u in [0.0, 0.25, 0.5, 0.75] {
            for v in [0.1, 0.3, 0.5, 0.7, 0.9] {
                let p = sphere.d0(u, v);
                let dist = (p - PVec3::new(10.0, 0.0, 0.0)).length();
                assert!((dist - 5.0).abs() < 1e-4, "radius error at u={u}, v={v}: dist={dist}");
            }
        }
    }

    #[test]
    fn test_sphere_normal_is_outward() {
        let sphere = SurfaceGeom::Sphere { center: PVec3::new(1.0, 2.0, 3.0), radius: 4.0 };
        for (u, v) in [(0.1, 0.25), (0.5, 0.5), (0.9, 0.75)] {
            let p = sphere.d0(u, v);
            let n = sphere.normal(u, v);
            let radial = (p - PVec3::new(1.0, 2.0, 3.0)).normalize();
            assert!((n.dot(radial).abs() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_sphere_project() {
        let sphere = SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 5.0 };
        let (_u, v) = sphere.project(PVec3::new(0.0, 0.0, 5.0)).unwrap();
        assert!(v.abs() < 1e-4);
    }

    #[test]
    fn test_torus_d0_outer_equator() {
        let torus = SurfaceGeom::torus(PVec3::ZERO, PVec3::Z, 3.0, 1.0);
        let p = torus.d0(0.0, 0.0);
        let r_xy = (p.x * p.x + p.y * p.y).sqrt();
        assert!((r_xy - 4.0).abs() < 1e-4);
        assert!(p.z.abs() < 1e-4);
    }

    #[test]
    fn test_torus_d0_inner_top() {
        let torus = SurfaceGeom::torus(PVec3::ZERO, PVec3::Z, 3.0, 1.0);
        let p = torus.d0(0.0, 0.5);
        let r_xy = (p.x * p.x + p.y * p.y).sqrt();
        assert!((r_xy - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_torus_normal_outward() {
        let torus = SurfaceGeom::torus(PVec3::ZERO, PVec3::Z, 3.0, 1.0);
        let n = torus.normal(0.0, 0.0);
        assert!(n.x > 0.5);
    }

    #[test]
    fn test_torus_project() {
        let torus = SurfaceGeom::torus(PVec3::ZERO, PVec3::Z, 3.0, 1.0);
        let p = torus.d0_native(0.0, 0.0);
        let (u, v) = torus.project(p).expect("torus project");
        let back = torus.d0_native(u, v);
        assert!((back - p).length() < 0.05);
    }

    #[test]
    fn test_extrusion_d0() {
        let generatrix = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let extrusion = SurfaceGeom::Extrusion { generatrix: Box::new(generatrix), direction: PVec3::Z };
        let p = extrusion.d0(2.0, 3.0);
        assert!((p - PVec3::new(2.0, 0.0, 3.0)).length() < 1e-4);
    }

    #[test]
    fn test_extrusion_project() {
        let generatrix = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let extrusion = SurfaceGeom::Extrusion { generatrix: Box::new(generatrix), direction: PVec3::Z };
        let p = extrusion.d0_native(0.5, 2.0);
        let (u, v) = extrusion.project(p).expect("extrusion project");
        let back = extrusion.d0_native(u, v);
        assert!((back - p).length() < 1e-3);
    }

    #[test]
    fn test_extrusion_d1() {
        let generatrix = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let extrusion = SurfaceGeom::Extrusion { generatrix: Box::new(generatrix), direction: PVec3::Z };
        let (du, dv) = extrusion.d1(0.5, 0.5);
        assert!((du - PVec3::X).length() < 1e-6);
        assert!((dv - PVec3::Z).length() < 1e-6);
    }

    #[test]
    fn test_revolution_d0_circle_makes_torus() {
        let circle = CurveGeom::circle(PVec3::new(3.0, 0.0, 0.0), PVec3::Y, 1.0);
        let rev = SurfaceGeom::Revolution { generatrix: Box::new(circle), axis_origin: PVec3::ZERO, axis_dir: PVec3::Z };
        let p = rev.d0(0.0, 0.0);
        assert!((p - PVec3::new(4.0, 0.0, 0.0)).length() < 1e-4);
        let p2 = rev.d0(0.0, 0.25);
        assert!((p2 - PVec3::new(0.0, 4.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn test_revolution_project() {
        let circle = CurveGeom::circle(PVec3::new(3.0, 0.0, 0.0), PVec3::Y, 1.0);
        let rev = SurfaceGeom::Revolution { generatrix: Box::new(circle), axis_origin: PVec3::ZERO, axis_dir: PVec3::Z };
        let proj = rev.project(PVec3::new(4.0, 0.0, 0.0));
        assert!(proj.is_some());
        let (u, v) = proj.unwrap();
        assert!((0.0..=1.0).contains(&u));
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_offset_d0() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let offset = SurfaceGeom::Offset { basis: Box::new(plane), distance: 2.0 };
        let p = offset.d0(1.0, 1.0);
        assert!((p - PVec3::new(1.0, 1.0, 2.0)).length() < 1e-4);
    }

    #[test]
    fn test_offset_project() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let offset = SurfaceGeom::Offset { basis: Box::new(plane), distance: 2.0 };
        let p = offset.d0_native(1.0, 1.0);
        let (u, v) = offset.project(p).expect("offset project");
        let back = offset.d0_native(u, v);
        assert!((back - p).length() < 0.05);
    }

    #[test]
    fn test_bspline_param_range_knot_domain() {
        let nurbs = crate::nurbs::NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        let bspline = SurfaceGeom::BSpline(nurbs);
        let r = bspline.param_range();
        assert!((r.u_min - 0.0).abs() < 1e-6);
        assert!((r.u_max - 1.0).abs() < 1e-6);
        assert!((r.v_min - 0.0).abs() < 1e-6);
        assert!((r.v_max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_native_uv_roundtrip_bspline() {
        let nurbs = crate::nurbs::NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        let bspline = SurfaceGeom::BSpline(nurbs);
        let native = (0.25, 0.75);
        let d0 = bspline.native_uv_to_d0(native.0, native.1);
        let back = bspline.d0_uv_to_native(d0.0, d0.1);
        assert!((back.0 - native.0).abs() < 1e-5);
        assert!((back.1 - native.1).abs() < 1e-5);
        let p_native = bspline.d0_native(native.0, native.1);
        let p_d0 = bspline.d0(d0.0, d0.1);
        assert!((p_native - p_d0).length() < 1e-5);
    }

    #[test]
    fn test_native_uv_roundtrip_cylinder() {
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 1.0);
        let native = (std::f64::consts::PI, 2.0);
        let d0 = cyl.native_uv_to_d0(native.0, native.1);
        assert!((d0.0 - 0.5).abs() < 1e-5);
        assert!((d0.1 - 2.0).abs() < 1e-5);
        let back = cyl.d0_uv_to_native(d0.0, d0.1);
        assert!((back.0 - native.0).abs() < 1e-4);
        assert!((back.1 - native.1).abs() < 1e-5);
    }

    #[test]
    fn test_inverse_native_uv_bspline_off_surface_seed() {
        let nurbs = crate::nurbs::NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        let bspline = SurfaceGeom::BSpline(nurbs);
        let target = PVec3::new(0.5, 0.5, 0.0);
        let uv = bspline.inverse_native_uv(target, 0.01).expect("inverse_native_uv");
        let back = bspline.d0_native(uv.0, uv.1);
        assert!((back - target).length() < 0.01);
    }

    #[test]
    fn test_bspline_project() {
        let nurbs = crate::nurbs::NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        let bspline = SurfaceGeom::BSpline(nurbs);
        let proj = bspline.project(PVec3::new(0.5, 0.5, 0.0));
        assert!(proj.is_some());
        let (u, v) = proj.unwrap();
        assert!((0.0..=1.0).contains(&u));
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_evaluate_grid_corner_values() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let grid = plane.evaluate_grid((1.0, 2.0), (3.0, 4.0), 1, 1);
        assert_eq!(grid.len(), 2);
        assert_eq!(grid[0].len(), 2);
        assert!((grid[0][0] - PVec3::new(1.0, 3.0, 0.0)).length() < 1e-6);
        assert!((grid[0][1] - PVec3::new(1.0, 4.0, 0.0)).length() < 1e-6);
        assert!((grid[1][0] - PVec3::new(2.0, 3.0, 0.0)).length() < 1e-6);
        assert!((grid[1][1] - PVec3::new(2.0, 4.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_offset_plane_d0_normal() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let offset = SurfaceGeom::Offset { basis: Box::new(plane), distance: 2.0 };
        let p = offset.d0(0.5, 0.5);
        assert!((p.z - 2.0).abs() < 1e-4, "offset plane d0 z should be 2.0, got {}", p.z);
        let n = offset.normal(0.5, 0.5);
        assert!((n.z.abs() - 1.0).abs() < 1e-4, "offset normal should be ±Z");
    }

    #[test]
    fn test_offset_cylinder_d0() {
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 1.0);
        let offset = SurfaceGeom::Offset { basis: Box::new(cyl), distance: 0.5 };
        let p = offset.d0(0.0, 0.0);
        assert!((p.x - 1.5).abs() < 1e-3, "offset cylinder x should be ~1.5, got {}", p.x);
    }

    #[test]
    fn test_offset_project_roundtrip() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let offset = SurfaceGeom::Offset { basis: Box::new(plane), distance: 1.0 };
        let target = PVec3::new(0.3, 0.4, 1.0);
        let uv = offset.project(target);
        assert!(uv.is_some(), "offset project should return Some");
        let (u, v) = uv.unwrap();
        let p_back = offset.d0_native(u, v);
        let dist = (p_back - target).length();
        assert!(dist < 0.1, "offset project roundtrip error {} too large", dist);
    }

    #[test]
    fn test_plane_curvature_radius_infinite() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let r = plane.min_curvature_radius(0.5, 0.5);
        assert_eq!(r, f64::MAX, "Plane should have infinite curvature radius");
    }

    #[test]
    fn test_cylinder_curvature_radius_equals_radius() {
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 2.0);
        // At any point, one principal curvature radius = cylinder radius
        let r = cyl.min_curvature_radius(0.25, 0.5);
        assert!(
            (r - 2.0).abs() < 0.5,
            "Cylinder r=2 should have min curvature radius ~2, got {}",
            r
        );
    }

    #[test]
    fn test_sphere_curvature_radius_equals_radius() {
        let sphere = SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 3.0 };
        // Sphere has equal principal curvatures everywhere = 1/radius
        let r = sphere.min_curvature_radius(0.5, 0.5);
        assert!(
            (r - 3.0).abs() < 1.0,
            "Sphere r=3 should have curvature radius ~3, got {}",
            r
        );
    }

    #[test]
    fn test_offset_self_intersection_detection() {
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 1.0);
        // Offset > radius should self-intersect (inward offset)
        assert!(offset_may_self_intersect(&cyl, 2.0, 8),
            "Offset 2.0 > radius 1.0 should detect self-intersection");
        // Offset < radius should be safe
        assert!(!offset_may_self_intersect(&cyl, 0.5, 8),
            "Offset 0.5 < radius 1.0 should not self-intersect");
    }

    #[test]
    fn test_offset_d1_weingarten() {
        // Offset cylinder: r=2.0, offset=0.5 → effective r=2.5.
        // d1 du of the offset surface should match a cylinder with r=2.5.
        let cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 2.0);
        let offset = SurfaceGeom::Offset {
            basis: Box::new(cyl), distance: 0.5,
        };

        let effective_cyl = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 2.5);

        for &(u, v) in &[(0.0, 0.0), (0.25, 0.5), (0.5, 1.0), (0.75, -0.3)] {
            let (du_off, dv_off) = offset.d1(u, v);
            let (du_eff, dv_eff) = effective_cyl.d1(u, v);

            let du_err = (du_off - du_eff).length();
            let dv_err = (dv_off - dv_eff).length();
            assert!(
                du_err < 0.01,
                "offset cylinder du mismatch at ({u},{v}): err={du_err}"
            );
            assert!(
                dv_err < 0.01,
                "offset cylinder dv mismatch at ({u},{v}): err={dv_err}"
            );
        }
    }

    #[test]
    fn test_d2_analytical_vs_numerical() {
        // Cross-validate analytical d2 against numerical (central differences on d1)
        // for Cylinder, Sphere, and Torus at several (u, v) points.
        let surfaces: Vec<(&str, SurfaceGeom)> = vec![
            ("cylinder", SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 2.0)),
            ("sphere", SurfaceGeom::Sphere {
                center: PVec3::ZERO, radius: 3.0,
            }),
            ("torus", SurfaceGeom::torus(PVec3::ZERO, PVec3::Z, 3.0, 1.0)),
        ];

        let uv_points = [
            (0.1, 0.2), (0.25, 0.5), (0.5, 0.25), (0.75, 0.75), (0.9, 0.1),
        ];

        let eps = 1e-4_f64;
        for (name, surf) in &surfaces {
            for &(u, v) in &uv_points {
                // Analytical d2
                let (duu_a, duv_a, dvv_a) = surf.d2(u, v);

                // Numerical d2 (central differences on d1)
                let (du_p, _dv_p) = surf.d1(u + eps, v);
                let (du_m, _dv_m) = surf.d1(u - eps, v);
                let (du_vp, dv_vp) = surf.d1(u, v + eps);
                let (du_vm, dv_vm) = surf.d1(u, v - eps);
                let duu_n = (du_p - du_m) / (2.0 * eps);
                let duv_n = (du_vp - du_vm) / (2.0 * eps);
                let dvv_n = (dv_vp - dv_vm) / (2.0 * eps);

                let tol = 0.5;
                assert!(
                    (duu_a - duu_n).length() < tol,
                    "{name} duu mismatch at ({u},{v}): analytical={duu_a:?}, numerical={duu_n:?}"
                );
                assert!(
                    (duv_a - duv_n).length() < tol,
                    "{name} duv mismatch at ({u},{v}): analytical={duv_a:?}, numerical={duv_n:?}"
                );
                assert!(
                    (dvv_a - dvv_n).length() < tol,
                    "{name} dvv mismatch at ({u},{v}): analytical={dvv_a:?}, numerical={dvv_n:?}"
                );
            }
        }
    }

    #[test]
    fn test_offset_plane_no_self_intersection() {
        let plane = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        // Plane offset never self-intersects (infinite curvature radius)
        assert!(!offset_may_self_intersect(&plane, 100.0, 4),
            "Plane offset should never self-intersect");
    }
}
