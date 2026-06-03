//! Surface-surface intersection for B-rep boolean operations.
//!
//! SSI status:
//! ✅ Plane×Plane → Line
//! ✅ Plane×Cylinder (perp→circle, parallel→2 lines)
//! ✅ Plane×Sphere → Circle
//! ✅ Sphere×Sphere → Circle
//! ✅ Plane×Cone → sampled conic curve
//! ✅ Plane×Torus → sampled 4th-degree curve
//! ✅ Cylinder×Cylinder → sampled space curve
//! ✅ Cylinder×Cone → sampled space curve
//! ⏳ NURBS×NURBS → deferred to marching (P3)

use rc3d_core::math::Vec3;
use crate::store::BRepStore;
use crate::topo::{ShellKey, FaceKey, BRepFace};
use crate::geom::{CurveGeom, SurfaceGeom};

/// Compute all face-face intersections between B-Rep shells.
pub struct FaceIntersectionResult {
    pub face_a: FaceKey,
    pub face_b: FaceKey,
    pub curves_3d: Vec<CurveGeom>,
    pub pcurves_on_a: Vec<CurveGeom>,
    pub pcurves_on_b: Vec<CurveGeom>,
}

/// Compute intersections between two sets of B-Rep shells.
pub fn compute_intersections_brep(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &BRepStore,
) -> Vec<FaceIntersectionResult> {
    let mut results = Vec::new();
    for &sk_a in shells_a {
        let shell_a = match reg.shells.get(sk_a) { Some(s) => s, None => continue };
        for &(face_a_key, _) in &shell_a.faces {
            let face_a = match reg.faces.get(face_a_key) { Some(f) => f, None => continue };
            for &sk_b in shells_b {
                let shell_b = match reg.shells.get(sk_b) { Some(s) => s, None => continue };
                for &(face_b_key, _) in &shell_b.faces {
                    let face_b = match reg.faces.get(face_b_key) { Some(f) => f, None => continue };
                    if let Some(curves) = intersect_surfaces_brep(face_a, face_b, reg) {
                        results.push(FaceIntersectionResult {
                            face_a: face_a_key, face_b: face_b_key,
                            curves_3d: curves, pcurves_on_a: vec![], pcurves_on_b: vec![],
                        });
                    }
                }
            }
        }
    }
    results
}

/// Compute intersection curves between two B-Rep faces.
pub fn intersect_surfaces_brep(
    face_a: &BRepFace,
    face_b: &BRepFace,
    _reg: &BRepStore,
) -> Option<Vec<CurveGeom>> {
    match (&face_a.surface, &face_b.surface) {
        // Pairs using the same order the match arms appear in SurfaceGeom
        (SurfaceGeom::Plane { origin: o1, normal: n1, .. },
         SurfaceGeom::Plane { origin: o2, normal: n2, .. }) => {
            plane_plane(*o1, *n1, *o2, *n2)
        }
        (SurfaceGeom::Plane { origin: o, normal: n, .. },
         SurfaceGeom::Cylinder { origin: co, axis: ca, radius: cr, .. }) => {
            plane_cylinder(*o, *n, *co, *ca, *cr)
        }
        (SurfaceGeom::Cylinder { .. }, SurfaceGeom::Plane { .. }) => {
            intersect_surfaces_brep(face_b, face_a, _reg)
        }
        (SurfaceGeom::Plane { origin: o, normal: n, .. },
         SurfaceGeom::Cone { apex, axis, semi_angle, .. }) => {
            plane_cone(*o, *n, *apex, *axis, *semi_angle)
        }
        (SurfaceGeom::Cone { .. }, SurfaceGeom::Plane { .. }) => {
            intersect_surfaces_brep(face_b, face_a, _reg)
        }
        (SurfaceGeom::Plane { origin: o, normal: n, .. },
         SurfaceGeom::Sphere { center, radius }) => {
            plane_sphere(*o, *n, *center, *radius)
        }
        (SurfaceGeom::Sphere { .. }, SurfaceGeom::Plane { .. }) => {
            intersect_surfaces_brep(face_b, face_a, _reg)
        }
        (SurfaceGeom::Plane { origin: o, normal: n, .. },
         SurfaceGeom::Torus { center, axis, major_r, minor_r, .. }) => {
            plane_torus(*o, *n, *center, *axis, *major_r, *minor_r)
        }
        (SurfaceGeom::Torus { .. }, SurfaceGeom::Plane { .. }) => {
            intersect_surfaces_brep(face_b, face_a, _reg)
        }
        (SurfaceGeom::Cylinder { origin: o1, axis: a1, radius: r1, .. },
         SurfaceGeom::Cylinder { origin: o2, axis: a2, radius: r2, .. }) => {
            cylinder_cylinder(*o1, *a1, *r1, *o2, *a2, *r2)
        }
        (SurfaceGeom::Sphere { center: c1, radius: r1 },
         SurfaceGeom::Sphere { center: c2, radius: r2 }) => {
            sphere_sphere(*c1, *r1, *c2, *r2)
        }
        (SurfaceGeom::Cylinder { origin: o1, axis: a1, radius: r1, .. },
         SurfaceGeom::Cone { apex, axis, semi_angle, .. }) => {
            cylinder_cone(*o1, *a1, *r1, *apex, *axis, *semi_angle)
        }
        (SurfaceGeom::Cone { .. }, SurfaceGeom::Cylinder { .. }) => {
            intersect_surfaces_brep(face_b, face_a, _reg)
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  B-spline construction helper
// ═══════════════════════════════════════════════════════════════════════

/// Convert a sampled polyline to a degree-3 B-spline curve using chord-length parameterization.
fn polyline_to_bspline(points: &[Vec3], degree: usize) -> CurveGeom {
    let n = points.len();
    if n == 0 {
        return CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::ZERO };
    }
    if n < 2 {
        return CurveGeom::Line { origin: points[0], direction: Vec3::ZERO };
    }
    if n <= 3 {
        // Not enough points for cubic — use degree 2
        let cps = points.to_vec();
        let n_cp = cps.len();
        let knot_count = n_cp + 2 + 1; // degree 2
        let knots: Vec<f32> = (0..knot_count).map(|i| i as f32).collect();
        return CurveGeom::BSpline {
            degree: 2,
            control_points: cps,
            knots,
            weights: None,
        };
    }

    // Chord-length parameterization
    let mut chord_lengths = vec![0.0f32];
    for i in 1..n {
        let d = (points[i] - points[i-1]).length();
        chord_lengths.push(chord_lengths[i-1] + d);
    }
    let _total = chord_lengths[n-1];

    // Control points are the sampled points
    let cps = points.to_vec();

    // Uniform knot vector with multiplicity at ends for clamped B-spline
    let deg = degree.min(n - 1);
    let n_knots = cps.len() + deg + 1;
    let mut knots = Vec::with_capacity(n_knots);
    // Clamped: first deg+1 knots = 0, last deg+1 knots = 1
    for _ in 0..=deg { knots.push(0.0); }
    let internal = n_knots.saturating_sub(2 * (deg + 1));
    for i in 1..=internal {
        knots.push(i as f32 / (internal + 1) as f32);
    }
    for _ in 0..=deg { knots.push(1.0); }

    CurveGeom::BSpline {
        degree: deg,
        control_points: cps,
        knots,
        weights: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Analytic SSI functions
// ═══════════════════════════════════════════════════════════════════════

fn build_ortho_axes(axis: Vec3) -> (Vec3, Vec3) {
    let x_dir = if axis.x.abs() < 0.9 {
        axis.cross(Vec3::X).normalize()
    } else {
        axis.cross(Vec3::Y).normalize()
    };
    let y_dir = axis.cross(x_dir).normalize();
    (x_dir, y_dir)
}

// ── Plane × Plane ─────────────────────────────────────────────────────

fn plane_plane(o1: Vec3, n1: Vec3, o2: Vec3, n2: Vec3) -> Option<Vec<CurveGeom>> {
    let cross = n1.cross(n2);
    if cross.length() < 1e-10 { return None; }
    let dir = cross.normalize();
    let d1 = n1.dot(o1);
    let d2 = n2.dot(o2);
    let det = n1.x * n2.y - n1.y * n2.x;
    let origin = if det.abs() > 1e-10 {
        Vec3::new((d1 * n2.y - d2 * n1.y) / det, (n1.x * d2 - n2.x * d1) / det, 0.0)
    } else {
        o1
    };
    Some(vec![CurveGeom::Line { origin, direction: dir }])
}

// ── Plane × Cylinder ──────────────────────────────────────────────────

fn plane_cylinder(
    plane_o: Vec3, plane_n: Vec3,
    cyl_o: Vec3, cyl_axis: Vec3, cyl_r: f32,
) -> Option<Vec<CurveGeom>> {
    let axis = cyl_axis.normalize();
    let cos_angle = plane_n.dot(axis).abs();

    if cos_angle > 0.9999 {
        // Perpendicular → circle
        let d = plane_n.dot(cyl_o - plane_o);
        let center = cyl_o - plane_n * d;
        return Some(vec![CurveGeom::circle(center, plane_n, cyl_r)]);
    }
    if cos_angle < 0.0001 {
        // Parallel → two lines
        let d = plane_n.dot(cyl_o - plane_o);
        let proj_origin = cyl_o - plane_n * d;
        let line_dir = axis;
        let perp = plane_n.cross(line_dir).normalize();
        let p1 = proj_origin + perp * cyl_r;
        let p2 = proj_origin - perp * cyl_r;
        return Some(vec![
            CurveGeom::Line { origin: p1, direction: line_dir },
            CurveGeom::Line { origin: p2, direction: line_dir },
        ]);
    }
    // General case → ellipse (return None for now; P3 marching will handle)
    None
}

// ── Plane × Sphere ────────────────────────────────────────────────────

fn plane_sphere(plane_o: Vec3, plane_n: Vec3, center: Vec3, radius: f32) -> Option<Vec<CurveGeom>> {
    let dist = (plane_n.dot(center - plane_o)).abs();
    if dist > radius { return None; }
    let circle_r = (radius * radius - dist * dist).sqrt();
    if circle_r < 1e-10 { return None; }
    let circle_center = center - plane_n * plane_n.dot(center - plane_o);
    Some(vec![CurveGeom::circle(circle_center, plane_n, circle_r)])
}

// ── Plane × Cone ──────────────────────────────────────────────────────

fn plane_cone(
    plane_o: Vec3, plane_n: Vec3,
    apex: Vec3, axis: Vec3, semi_angle: f32,
) -> Option<Vec<CurveGeom>> {
    let ax = axis.normalize();
    let tan_a = semi_angle.tan();
    let dist = plane_n.dot(apex - plane_o);
    if dist.abs() < 1e-6 { return None; }
    let (u, v) = build_ortho_axes(ax);
    let n = 48;
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let angle = std::f32::consts::TAU * i as f32 / n as f32;
        let dir = u * angle.cos() + v * angle.sin();
        let ray_dir = (ax + dir * tan_a).normalize();
        let denom = plane_n.dot(ray_dir);
        if denom.abs() < 1e-10 { continue; }
        let t = -dist / denom;
        if t > 0.0 && t.is_finite() { pts.push(apex + ray_dir * t); }
    }
    if pts.len() < 4 { return None; }
    Some(vec![polyline_to_bspline(&pts, 3)])
}

// ── Plane × Torus ─────────────────────────────────────────────────────

fn plane_torus(
    plane_o: Vec3, plane_n: Vec3,
    torus_center: Vec3, torus_axis: Vec3, major_r: f32, minor_r: f32,
) -> Option<Vec<CurveGeom>> {
    let axis = torus_axis.normalize();
    let (u, v) = build_ortho_axes(axis);
    let n = 64;
    let mut pts: Vec<Vec3> = Vec::new();
    for i in 0..=n {
        let angle = std::f32::consts::TAU * i as f32 / n as f32;
        let tube_center = torus_center + u * major_r * angle.cos() + v * major_r * angle.sin();
        let d = plane_n.dot(tube_center - plane_o);
        if d.abs() > minor_r + 1e-6 { continue; }
        let h = (minor_r * minor_r - d * d).sqrt();
        if h < 1e-6 { pts.push(tube_center); continue; }
        let perp = plane_n.cross(axis.cross(plane_n));
        let perp = if perp.length() > 1e-6 { perp.normalize() } else { plane_n.cross(u).normalize() };
        pts.push(tube_center - perp * h);
        pts.push(tube_center + perp * h);
    }
    if pts.len() < 4 { return None; }
    Some(vec![polyline_to_bspline(&pts, 3)])
}

// ── Cylinder × Cylinder ───────────────────────────────────────────────

fn cylinder_cylinder(
    o1: Vec3, a1: Vec3, r1: f32, o2: Vec3, a2: Vec3, r2: f32,
) -> Option<Vec<CurveGeom>> {
    let ax1 = a1.normalize();
    let ax2 = a2.normalize();
    if ax1.cross(ax2).length() < 1e-6 {
        let dist = (o1 - o2).cross(ax1).length();
        if dist > r1 + r2 || dist < (r1 - r2).abs() { return None; }
        return None; // parallel cylinders → complex, defer to P3
    }
    let (u, v) = build_ortho_axes(ax1);
    let n = 64;
    let mut pts = Vec::with_capacity(n * 2);
    for i in 0..=n {
        let angle = std::f32::consts::TAU * i as f32 / n as f32;
        let dir = u * angle.cos() + v * angle.sin();
        let surface_pt = o1 + dir * r1;
        let d = surface_pt - o2;
        let a_coeff = (ax1 - ax2 * ax1.dot(ax2)).length_squared();
        if a_coeff < 1e-10 { continue; }
        let b_coeff = 2.0 * (d.dot(ax1) - d.dot(ax2) * ax1.dot(ax2));
        let c_coeff = d.length_squared() - d.dot(ax2).powi(2) - r2 * r2;
        let disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
        if disc >= 0.0 {
            let sqrt_d = disc.sqrt();
            for &t in &[(-b_coeff + sqrt_d) / (2.0 * a_coeff), (-b_coeff - sqrt_d) / (2.0 * a_coeff)] {
                if t.is_finite() { pts.push(surface_pt + ax1 * t); }
            }
        }
    }
    if pts.len() < 4 { return None; }
    Some(vec![polyline_to_bspline(&pts, 3)])
}

// ── Sphere × Sphere ───────────────────────────────────────────────────

fn sphere_sphere(c1: Vec3, r1: f32, c2: Vec3, r2: f32) -> Option<Vec<CurveGeom>> {
    let d_vec = c2 - c1;
    let d = d_vec.length();
    if d < 1e-10 { return None; }
    if d > r1 + r2 || d < (r1 - r2).abs() { return None; }
    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let center = c1 + d_vec * (a / d);
    let h = (r1 * r1 - a * a).sqrt();
    if h < 1e-10 { return None; }
    let normal = d_vec.normalize();
    Some(vec![CurveGeom::circle(center, normal, h)])
}

// ── Cylinder × Cone ───────────────────────────────────────────────────

fn cylinder_cone(
    cyl_o: Vec3, cyl_a: Vec3, cyl_r: f32,
    apex: Vec3, cone_a: Vec3, semi_angle: f32,
) -> Option<Vec<CurveGeom>> {
    let c_ax = cyl_a.normalize();
    let k_ax = cone_a.normalize();
    let tan_a = semi_angle.tan();
    let (u, v) = build_ortho_axes(c_ax);
    let n = 64;
    let mut pts: Vec<Vec3> = Vec::new();
    for i in 0..=n {
        let angle = std::f32::consts::TAU * i as f32 / n as f32;
        let dir = u * angle.cos() + v * angle.sin();
        let cyl_pt = cyl_o + dir * cyl_r;
        let d = cyl_pt - apex;
        let d_along = d.dot(k_ax);
        let d_perp = d - k_ax * d_along;
        let a_along = c_ax.dot(k_ax);
        let a_perp = c_ax - k_ax * a_along;
        let a = a_perp.length_squared() - tan_a * tan_a * a_along * a_along;
        let b = 2.0 * (d_perp.dot(a_perp) - tan_a * tan_a * d_along * a_along);
        let c = d_perp.length_squared() - tan_a * tan_a * d_along * d_along;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sqrt_d = disc.sqrt();
            for &t in &[(-b + sqrt_d) / (2.0 * a), (-b - sqrt_d) / (2.0 * a)] {
                if t.is_finite() { pts.push(cyl_pt + c_ax * t); }
            }
        }
    }
    if pts.len() < 4 { return None; }
    Some(vec![polyline_to_bspline(&pts, 3)])
}

// ═══════════════════════════════════════════════════════════════════════
//  Coplanar / tangent detection (P2)
// ═══════════════════════════════════════════════════════════════════════

pub fn faces_are_coplanar(face_a: &BRepFace, face_b: &BRepFace, tol: f32) -> bool {
    match (&face_a.surface, &face_b.surface) {
        (SurfaceGeom::Plane { origin: o1, normal: n1, .. },
         SurfaceGeom::Plane { origin: o2, normal: n2, .. }) => {
            n1.cross(*n2).length() < tol && (n1.dot(*o2 - *o1)).abs() < tol
        }
        (SurfaceGeom::Cylinder { origin: o1, axis: a1, radius: r1, .. },
         SurfaceGeom::Cylinder { origin: o2, axis: a2, radius: r2, .. }) => {
            a1.cross(*a2).length() < tol
                && (*o1 - *o2).cross(*a1).length() < tol
                && (r1 - r2).abs() < tol
        }
        _ => false,
    }
}

pub fn is_tangent_intersection(face_a: &BRepFace, face_b: &BRepFace, tol: f32) -> bool {
    match (&face_a.surface, &face_b.surface) {
        (SurfaceGeom::Plane { origin, normal, .. }, SurfaceGeom::Sphere { center, radius }) => {
            (normal.dot(*center - *origin).abs() - radius).abs() < tol
        }
        (SurfaceGeom::Sphere { .. }, SurfaceGeom::Plane { .. }) => is_tangent_intersection(face_b, face_a, tol),
        (SurfaceGeom::Sphere { center: c1, radius: r1 },
         SurfaceGeom::Sphere { center: c2, radius: r2 }) => {
            let d = (*c1 - *c2).length();
            (d - (r1 + r2)).abs() < tol || (d - (r1 - r2).abs()).abs() < tol
        }
        _ => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::BRepWire;

    fn make_test_face(surface: SurfaceGeom) -> BRepFace {
        BRepFace {
            surface,
            outer_wire: crate::topo::WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        }
    }

    #[test]
    fn plane_plane_intersecting() {
        let fa = make_test_face(SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X });
        let fb = make_test_face(SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Y, u_dir: Vec3::X });
        let curves = intersect_surfaces_brep(&fa, &fb, &BRepStore::new());
        assert!(curves.is_some());
        assert!(matches!(&curves.unwrap()[0], CurveGeom::Line { .. }));
    }

    #[test]
    fn plane_plane_parallel() {
        let fa = make_test_face(SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X });
        let fb = make_test_face(SurfaceGeom::Plane { origin: Vec3::new(0.0, 0.0, 5.0), normal: Vec3::Z, u_dir: Vec3::X });
        assert!(intersect_surfaces_brep(&fa, &fb, &BRepStore::new()).is_none());
    }

    #[test]
    fn plane_cylinder_perpendicular_circle() {
        let fa = make_test_face(SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X });
        let fb = make_test_face(SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 3.0));
        let curves = intersect_surfaces_brep(&fa, &fb, &BRepStore::new());
        assert!(curves.is_some());
        if let Some(cs) = curves {
            assert!(matches!(&cs[0], CurveGeom::Circle { radius, .. } if (*radius - 3.0).abs() < 0.01));
        }
    }

    #[test]
    fn plane_cylinder_parallel_two_lines() {
        let fa = make_test_face(SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::X, u_dir: Vec3::Y });
        let fb = make_test_face(SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 2.0));
        let curves = intersect_surfaces_brep(&fa, &fb, &BRepStore::new());
        assert!(curves.is_some());
        if let Some(cs) = curves {
            assert!(cs.len() >= 2, "parallel case should produce 2 lines");
        }
    }

    #[test]
    fn plane_sphere_circle() {
        let fa = make_test_face(SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X });
        let fb = make_test_face(SurfaceGeom::Sphere { center: Vec3::new(0.0, 0.0, 1.0), radius: 2.0 });
        let curves = intersect_surfaces_brep(&fa, &fb, &BRepStore::new());
        assert!(curves.is_some());
    }

    #[test]
    fn sphere_sphere_intersect() {
        let fa = make_test_face(SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 2.0 });
        let fb = make_test_face(SurfaceGeom::Sphere { center: Vec3::new(2.0, 0.0, 0.0), radius: 2.0 });
        let curves = intersect_surfaces_brep(&fa, &fb, &BRepStore::new());
        assert!(curves.is_some());
        assert!(matches!(&curves.unwrap()[0], CurveGeom::Circle { .. }));
    }

    #[test]
    fn sphere_sphere_disjoint() {
        let fa = make_test_face(SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 1.0 });
        let fb = make_test_face(SurfaceGeom::Sphere { center: Vec3::new(10.0, 0.0, 0.0), radius: 1.0 });
        assert!(intersect_surfaces_brep(&fa, &fb, &BRepStore::new()).is_none());
    }

    #[test]
    fn coplanar_planes_detected() {
        let fa = make_test_face(SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X });
        let fb = make_test_face(SurfaceGeom::Plane { origin: Vec3::new(0.0, 0.0, 1e-8), normal: Vec3::Z, u_dir: Vec3::X });
        assert!(faces_are_coplanar(&fa, &fb, 1e-4));
    }

    #[test]
    fn tangent_sphere_plane() {
        let fa = make_test_face(SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X });
        let fb = make_test_face(SurfaceGeom::Sphere { center: Vec3::new(0.0, 0.0, 2.0), radius: 2.0 });
        assert!(is_tangent_intersection(&fa, &fb, 1e-4));
    }
}
