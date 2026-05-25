//! Unified curve and surface geometry.
//! T1.1: CurveGeom  T1.2: SurfaceGeom

use rc3d_core::math::Vec3;

// ── CurveGeom ──────────────────────────────────────────────

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
    pub fn d0(&self, _t: f32) -> Vec3 { todo!("T1.1") }
    pub fn d1(&self, _t: f32) -> Vec3 { todo!("T1.1") }
    pub fn d2(&self, _t: f32) -> Vec3 { todo!("T1.1") }
    pub fn arc_length(&self, _t0: f32, _t1: f32) -> f32 { todo!("T1.1") }
    pub fn curvature(&self, _t: f32) -> f32 { todo!("T1.1") }
    pub fn sample_adaptive(&self, _t0: f32, _t1: f32, _tolerance: f32) -> Vec<(f32, Vec3)> { todo!("T1.1") }
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
