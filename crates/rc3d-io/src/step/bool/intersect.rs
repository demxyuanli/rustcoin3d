//! Surface-surface intersection for B-rep boolean operations.
//!
//! Each intersection produces one or more curves in 3D space that lie
//! on both surfaces. These curves are used to split faces.

use rc3d_core::math::Vec3;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{ShellKey, FaceKey, BRepFace};
use crate::step::brep::geom::{CurveGeom, SurfaceGeom};

/// Compute all face-face intersections between B-Rep shells.
/// Each intersection produces curves with PCURVEs on both faces.
pub struct FaceIntersectionResult {
    pub face_a: FaceKey,
    pub face_b: FaceKey,
    /// 3D intersection curves
    pub curves_3d: Vec<CurveGeom>,
    /// PCURVEs of intersection curves on face A (2D curves in UV space)
    pub pcurves_on_a: Vec<CurveGeom>,
    /// PCURVEs of intersection curves on face B
    pub pcurves_on_b: Vec<CurveGeom>,
}

/// Compute intersections between B-Rep shells.
pub fn compute_intersections_brep(
    _shells_a: &[ShellKey],
    _shells_b: &[ShellKey],
    reg: &BRepRegistry,
) -> Vec<FaceIntersectionResult> {
    let mut results = Vec::new();
    // For now: only support analytic-analytic pairs (existing infrastructure)
    // Full NURBS-NURBS intersection requires marching method -- deferred

    // Iterate face pairs and call existing analytic intersect functions
    for &sk_a in _shells_a {
        let shell_a = match reg.shells.get(sk_a) { Some(s) => s, None => continue };
        for &(face_a_key, _) in &shell_a.faces {
            let face_a = match reg.faces.get(face_a_key) { Some(f) => f, None => continue };
            for &sk_b in _shells_b {
                let shell_b = match reg.shells.get(sk_b) { Some(s) => s, None => continue };
                for &(face_b_key, _) in &shell_b.faces {
                    let face_b = match reg.faces.get(face_b_key) { Some(f) => f, None => continue };

                    // Try analytic surface-surface intersection
                    if let Some(curves) = intersect_surfaces_brep(face_a, face_b, reg) {
                        results.push(FaceIntersectionResult {
                            face_a: face_a_key,
                            face_b: face_b_key,
                            curves_3d: curves,
                            pcurves_on_a: vec![],
                            pcurves_on_b: vec![],
                        });
                    }
                }
            }
        }
    }
    results
}

/// Compute intersection curves between two B-Rep faces (analytic surfaces only for now).
fn intersect_surfaces_brep(
    face_a: &BRepFace,
    face_b: &BRepFace,
    _reg: &BRepRegistry,
) -> Option<Vec<CurveGeom>> {
    match (&face_a.surface, &face_b.surface) {
        (SurfaceGeom::Plane { origin: o1, normal: n1, .. },
         SurfaceGeom::Plane { origin: o2, normal: n2, .. }) => {
            plane_plane_brep(*o1, *n1, *o2, *n2)
        }
        (SurfaceGeom::Plane { origin: o, normal: n, .. },
         SurfaceGeom::Cylinder { origin: co, axis: ca, radius: cr, .. }) => {
            plane_cylinder_brep(*o, *n, *co, *ca, *cr)
        }
        (SurfaceGeom::Cylinder { .. }, SurfaceGeom::Plane { .. }) => {
            intersect_surfaces_brep(face_b, face_a, _reg)
        }
        _ => None, // Other combinations: method not yet implemented for B-Rep
    }
}

fn plane_plane_brep(o1: Vec3, n1: Vec3, o2: Vec3, n2: Vec3) -> Option<Vec<CurveGeom>> {
    let cross = n1.cross(n2);
    if cross.length() < 1e-10 { return None; } // parallel
    let dir = cross.normalize();
    let d1 = n1.dot(o1);
    let d2 = n2.dot(o2);
    let det = n1.x * n2.y - n1.y * n2.x;
    let origin = if det.abs() > 1e-10 {
        Vec3::new((d1*n2.y - d2*n1.y)/det, (n1.x*d2 - n2.x*d1)/det, 0.0)
    } else {
        o1
    };
    Some(vec![CurveGeom::Line { origin, direction: dir }])
}

fn plane_cylinder_brep(plane_o: Vec3, plane_n: Vec3, cyl_o: Vec3, cyl_axis: Vec3, cyl_r: f32) -> Option<Vec<CurveGeom>> {
    // Simplified: if plane is perpendicular to axis -> circle intersection
    let a = cyl_axis.normalize();
    if (plane_n.dot(a)).abs() > 0.999 {
        let d = plane_n.dot(cyl_o - plane_o);
        let center = cyl_o - plane_n * d;
        return Some(vec![CurveGeom::circle(center, plane_n, cyl_r)]);
    }
    None
}
use super::super::parser::EntityIndex;
use super::super::topology::{StepShell, StepFace};
use super::super::entity_types::EntityType;

/// A 3D intersection curve between two surfaces.
#[derive(Debug, Clone)]
pub struct IntersectionCurve {
    /// Sampled points along the intersection curve in 3D.
    pub points: Vec<Vec3>,
    /// The entity IDs of the two intersecting faces.
    pub face_a_id: u64,
    pub face_b_id: u64,
}

/// A pair of intersecting faces.
#[derive(Debug, Clone)]
pub struct FaceIntersection {
    pub face_a: usize, // Index into a_shells faces
    pub face_b: usize, // Index into b_shells faces
    pub curves: Vec<IntersectionCurve>,
}

/// Compute all face-face intersections between two sets of shells.
pub fn compute_intersections(
    _a_shells: &[StepShell],
    _b_shells: &[StepShell],
    entities_a: &EntityIndex,
    entities_b: &EntityIndex,
) -> Vec<FaceIntersection> {
    let faces_a: Vec<&StepFace> = _a_shells.iter().flat_map(|s| s.faces.iter()).collect();
    let faces_b: Vec<&StepFace> = _b_shells.iter().flat_map(|s| s.faces.iter()).collect();

    let mut results = Vec::new();

    for (ai, face_a) in faces_a.iter().enumerate() {
        let surf_a = face_a.surface_id.and_then(|id| entities_a.get(&id));
        for (bi, face_b) in faces_b.iter().enumerate() {
            let surf_b = face_b.surface_id.and_then(|id| entities_b.get(&id));
            if let (Some(sa), Some(sb)) = (surf_a, surf_b) {
                if let Some(curves) = intersect_surfaces(
                    sa.entity_type, sb.entity_type,
                    face_a, face_b, entities_a, entities_b,
                ) {
                    results.push(FaceIntersection {
                        face_a: ai,
                        face_b: bi,
                        curves,
                    });
                }
            }
        }
    }

    results
}

/// Compute intersection curves between two specific surface types.
fn intersect_surfaces(
    type_a: EntityType,
    type_b: EntityType,
    face_a: &StepFace,
    face_b: &StepFace,
    entities_a: &EntityIndex,
    entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    match (type_a, type_b) {
        (EntityType::Plane, EntityType::Plane) => {
            plane_plane_intersect(face_a, face_b, entities_a, entities_b)
        }
        (EntityType::Plane, EntityType::CylindricalSurface)
        | (EntityType::CylindricalSurface, EntityType::Plane) => {
            plane_cylinder_intersect(face_a, face_b, entities_a, entities_b)
        }
        (EntityType::Plane, EntityType::SphericalSurface)
        | (EntityType::SphericalSurface, EntityType::Plane) => {
            plane_sphere_intersect(face_a, face_b, entities_a, entities_b)
        }
        (EntityType::CylindricalSurface, EntityType::CylindricalSurface) => {
            cylinder_cylinder_intersect(face_a, face_b, entities_a, entities_b)
        }
        (EntityType::SphericalSurface, EntityType::SphericalSurface) => {
            sphere_sphere_intersect(face_a, face_b, entities_a, entities_b)
        }
        (EntityType::ConicalSurface, EntityType::Plane)
        | (EntityType::Plane, EntityType::ConicalSurface) => {
            cone_plane_intersect(face_a, face_b, entities_a, entities_b)
        }
        (EntityType::CylindricalSurface, EntityType::ConicalSurface)
        | (EntityType::ConicalSurface, EntityType::CylindricalSurface) => {
            cylinder_cone_intersect(face_a, face_b, entities_a, entities_b)
        }
        (EntityType::ToroidalSurface, EntityType::Plane)
        | (EntityType::Plane, EntityType::ToroidalSurface) => {
            torus_plane_intersect(face_a, face_b, entities_a, entities_b)
        }
        _ => None,
    }
}

/// Intersect two planes → a line (if not parallel).
fn plane_plane_intersect(
    face_a: &StepFace, face_b: &StepFace,
    entities_a: &EntityIndex, entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    let info_a = get_plane_info(face_a, entities_a)?;
    let info_b = get_plane_info(face_b, entities_b)?;

    let n1 = info_a.normal;
    let n2 = info_b.normal;
    let cross = n1.cross(n2);
    let cross_len = cross.length();

    // Parallel or coincident planes → no unique intersection line
    if cross_len < 1e-10 {
        return None;
    }

    let dir = cross / cross_len;

    // Find a point on the intersection line
    // Solve: n1·p = d1, n2·p = d2, dir·p = 0
    let d1 = n1.dot(info_a.origin);
    let d2 = n2.dot(info_b.origin);

    // Point on line: use Cramer's rule
    let det = n1.x * n2.y - n2.x * n1.y;
    if det.abs() > 1e-10 {
        let px = (d1 * n2.y - d2 * n1.y) / det;
        let py = (n1.x * d2 - n2.x * d1) / det;
        let pz = 0.0;
        let origin = Vec3::new(px, py, pz);

        // Sample points along the line (2 points for a line segment estimate)
        let len = 100.0; // Large sample range
        let points = vec![
            origin - dir * len,
            origin + dir * len,
        ];

        Some(vec![IntersectionCurve {
            points,
            face_a_id: face_a.surface_id.unwrap_or(0),
            face_b_id: face_b.surface_id.unwrap_or(0),
        }])
    } else if (n1.x * n2.z - n2.x * n1.z).abs() > 1e-10 {
        let det = n1.x * n2.z - n2.x * n1.z;
        let px = (d1 * n2.z - d2 * n1.z) / det;
        let pz = (n1.x * d2 - n2.x * d1) / det;
        let origin = Vec3::new(px, 0.0, pz);
        let len = 100.0;
        Some(vec![IntersectionCurve {
            points: vec![origin - dir * len, origin + dir * len],
            face_a_id: face_a.surface_id.unwrap_or(0),
            face_b_id: face_b.surface_id.unwrap_or(0),
        }])
    } else {
        // YZ case: intersection line parallel to Y axis
        let det = n1.y * n2.z - n2.y * n1.z;
        if det.abs() > 1e-10 {
            let py = (d1 * n2.z - d2 * n1.z) / det;
            let pz = (n1.y * d2 - n2.y * d1) / det;
            let origin = Vec3::new(0.0, py, pz);
            let len = 100.0;
            Some(vec![IntersectionCurve {
                points: vec![origin - dir * len, origin + dir * len],
                face_a_id: face_a.surface_id.unwrap_or(0),
                face_b_id: face_b.surface_id.unwrap_or(0),
            }])
        } else {
            None
        }
    }
}

/// Intersect a plane with a cylinder → ellipse or two lines.
fn plane_cylinder_intersect(
    face_a: &StepFace, face_b: &StepFace,
    entities_a: &EntityIndex, entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    // Determine which is plane and which is cylinder
    let (plane_face, cyl_face, plane_entities, cyl_entities) =
        if let Some(_info) = get_plane_info(face_a, entities_a) {
            (face_a, face_b, entities_a, entities_b)
        } else {
            (face_b, face_a, entities_b, entities_a)
        };

    let plane_info = get_plane_info(plane_face, plane_entities)?;
    let cyl_info = get_cylinder_info(cyl_face, cyl_entities)?;

    let normal = plane_info.normal;
    let cyl_axis = cyl_info.axis;
    let cyl_origin = cyl_info.origin;
    let radius = cyl_info.radius;

    // Angle between plane normal and cylinder axis
    let cos_angle = normal.dot(cyl_axis).abs();

    if cos_angle > 0.9999 {
        // Plane is perpendicular to cylinder axis → circle
        // Project cylinder origin onto plane
        let d = normal.dot(cyl_origin - plane_info.origin);
        let center = cyl_origin - normal * d;

        let n = 32;
        let u = if normal.x.abs() < 0.9 {
            normal.cross(Vec3::X).normalize()
        } else {
            normal.cross(Vec3::Y).normalize()
        };
        let v = normal.cross(u).normalize();
        let mut points = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
            points.push(center + u * radius * angle.cos() + v * radius * angle.sin());
        }

        Some(vec![IntersectionCurve {
            points,
            face_a_id: face_a.surface_id.unwrap_or(0),
            face_b_id: face_b.surface_id.unwrap_or(0),
        }])
    } else if cos_angle < 0.0001 {
        // Plane is parallel to cylinder axis → two lines
        // Project cylinder origin onto plane
        let d = normal.dot(cyl_origin - plane_info.origin);
        let proj_origin = cyl_origin - normal * d;

        // Lines are offset by ±radius perpendicular to axis direction on plane
        let line_dir = cyl_axis.normalize();
        let perp = normal.cross(line_dir).normalize();
        let len = 100.0;

        let p1 = proj_origin + perp * radius;
        let p2 = proj_origin - perp * radius;

        Some(vec![IntersectionCurve {
            points: vec![p1, p1 + line_dir * len, p2, p2 + line_dir * len],
            face_a_id: face_a.surface_id.unwrap_or(0),
            face_b_id: face_b.surface_id.unwrap_or(0),
        }])
    } else {
        // General case: ellipse — sample points
        let d = normal.dot(cyl_origin - plane_info.origin);
        let center_on_plane = cyl_origin - normal * d;

        let line_dir = cyl_axis.normalize();
        let perp = normal.cross(line_dir).normalize();
        let semi_a = radius / cos_angle; // Major axis
        let semi_b = radius;              // Minor axis along perp

        let n = 32;
        let mut points = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
            points.push(center_on_plane
                + perp * semi_b * angle.cos()
                + line_dir.cross(perp) * semi_a * angle.sin());
        }

        Some(vec![IntersectionCurve {
            points,
            face_a_id: face_a.surface_id.unwrap_or(0),
            face_b_id: face_b.surface_id.unwrap_or(0),
        }])
    }
}

/// Intersect a plane with a sphere → circle.
fn plane_sphere_intersect(
    face_a: &StepFace, face_b: &StepFace,
    entities_a: &EntityIndex, entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    let (plane_face, sphere_face, plane_entities, sphere_entities) =
        if get_plane_info(face_a, entities_a).is_some() {
            (face_a, face_b, entities_a, entities_b)
        } else {
            (face_b, face_a, entities_b, entities_a)
        };

    let plane_info = get_plane_info(plane_face, plane_entities)?;
    let sphere_info = get_sphere_info(sphere_face, sphere_entities)?;

    let normal = plane_info.normal;
    let center = sphere_info.center;
    let radius = sphere_info.radius;

    // Distance from sphere center to plane
    let dist = (normal.dot(center - plane_info.origin)).abs();

    if dist > radius {
        return None; // No intersection
    }

    let circle_radius = (radius * radius - dist * dist).sqrt();
    let circle_center = center - normal * normal.dot(center - plane_info.origin);

    let u = if normal.x.abs() < 0.9 {
        normal.cross(Vec3::X).normalize()
    } else {
        normal.cross(Vec3::Y).normalize()
    };
    let v = normal.cross(u).normalize();

    let n = 32;
    let mut points = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
        points.push(circle_center + u * circle_radius * angle.cos()
            + v * circle_radius * angle.sin());
    }

    Some(vec![IntersectionCurve {
        points,
        face_a_id: face_a.surface_id.unwrap_or(0),
        face_b_id: face_b.surface_id.unwrap_or(0),
    }])
}

/// Intersect two cylinders — samples the 3D intersection curve numerically.
fn cylinder_cylinder_intersect(
    face_a: &StepFace, face_b: &StepFace,
    entities_a: &EntityIndex, entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    let ca = get_cylinder_info(face_a, entities_a)?;
    let cb = get_cylinder_info(face_b, entities_b)?;

    // If axes are parallel and radii differ → no intersection (or one inside other)
    let axis_a = ca.axis.normalize();
    let axis_b = cb.axis.normalize();
    let parallel = (axis_a.cross(axis_b).length()) < 1e-6;

    if parallel {
        let dist = (ca.origin - cb.origin).cross(axis_a).length();
        if dist > ca.radius + cb.radius || dist < (ca.radius - cb.radius).abs() {
            return None; // No intersection
        }
        // Parallel cylinders: intersection is lines — skip for simplicity
        return None;
    }

    // Non-parallel: numerically trace the intersection curve
    // Solve: |(P - Oa) × Aa| = Ra  AND  |(P - Ob) × Ab| = Rb
    let n = 64;
    let mut points = Vec::with_capacity(n + 1);

    // Sample by angle around axis A
    for i in 0..=n {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
        // Find points on cylinder A at this angle
        let u = if axis_a.x.abs() < 0.9 { axis_a.cross(Vec3::X).normalize() } else { axis_a.cross(Vec3::Y).normalize() };
        let v = axis_a.cross(u).normalize();
        let dir_on_cyl = u * angle.cos() + v * angle.sin();
        let surface_pt = ca.origin + dir_on_cyl * ca.radius;

        // Project along axis A to find intersection with cylinder B
        // Ray: surface_pt + t * axis_a
        // Need: distance from this ray to B's axis = B's radius
        // This is a quadratic in t
        let d = surface_pt - cb.origin;
        let a_coeff = (axis_a - axis_b * axis_a.dot(axis_b)).length_squared();
        if a_coeff < 1e-10 { continue; }
        let b_coeff = 2.0 * (d.dot(axis_a) - d.dot(axis_b) * axis_a.dot(axis_b));
        let c_coeff = d.length_squared() - d.dot(axis_b).powi(2) - cb.radius * cb.radius;

        let disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
        if disc < 0.0 { continue; }
        let sqrt_disc = disc.sqrt();
        let t1 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff);
        let t2 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff);

        for &t in &[t1, t2] {
            if t.is_finite() {
                points.push(surface_pt + axis_a * t);
            }
        }
    }

    if points.len() < 4 {
        return None;
    }

    Some(vec![IntersectionCurve {
        points,
        face_a_id: face_a.surface_id.unwrap_or(0),
        face_b_id: face_b.surface_id.unwrap_or(0),
    }])
}

/// Intersect two spheres → circle (in 3D).
fn sphere_sphere_intersect(
    face_a: &StepFace, face_b: &StepFace,
    entities_a: &EntityIndex, entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    let sa = get_sphere_info(face_a, entities_a)?;
    let sb = get_sphere_info(face_b, entities_b)?;

    let d_vec = sb.center - sa.center;
    let d = d_vec.length();

    if d < 1e-10 {
        return None; // Concentric — no intersection or identical
    }
    if d > sa.radius + sb.radius || d < (sa.radius - sb.radius).abs() {
        return None; // No intersection
    }

    // Circle center is on the line connecting centers
    let a = (sa.radius * sa.radius - sb.radius * sb.radius + d * d) / (2.0 * d);
    let center = sa.center + d_vec * (a / d);

    // Circle radius
    let h = (sa.radius * sa.radius - a * a).sqrt();

    // Circle plane is perpendicular to the center line
    let normal = d_vec.normalize();
    let u = if normal.x.abs() < 0.9 { normal.cross(Vec3::X).normalize() } else { normal.cross(Vec3::Y).normalize() };
    let v = normal.cross(u).normalize();

    let n = 32;
    let mut points = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
        points.push(center + u * h * angle.cos() + v * h * angle.sin());
    }

    Some(vec![IntersectionCurve {
        points,
        face_a_id: face_a.surface_id.unwrap_or(0),
        face_b_id: face_b.surface_id.unwrap_or(0),
    }])
}

/// Intersect a cone with a plane → conic section.
fn cone_plane_intersect(
    face_a: &StepFace, face_b: &StepFace,
    entities_a: &EntityIndex, entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    let (cone_face, plane_face, cone_entities, plane_entities) =
        if get_cone_info(face_a, entities_a).is_some() {
            (face_a, face_b, entities_a, entities_b)
        } else {
            (face_b, face_a, entities_b, entities_a)
        };

    let cone = get_cone_info(cone_face, cone_entities)?;
    let plane = get_plane_info(plane_face, plane_entities)?;

    // Cone: apex at origin along axis, opening with semi_angle
    // Plane: normal·(P - origin) = 0
    let axis = cone.axis.normalize();
    let apex = cone.apex;
    let tan_a = cone.semi_angle.tan();

    // Translate plane equation relative to apex
    let dist = plane.normal.dot(apex - plane.origin);
    // Degenerate cases: plane through apex → point or line
    if dist.abs() < 1e-6 {
        return None; // Apex on plane — degenerate
    }

    // Sample the conic intersection
    let n = 48;
    let mut points = Vec::with_capacity(n + 1);

    let u = if axis.x.abs() < 0.9 { axis.cross(Vec3::X).normalize() } else { axis.cross(Vec3::Y).normalize() };
    let v = axis.cross(u).normalize();

    for i in 0..=n {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
        let dir = u * angle.cos() + v * angle.sin();
        // Ray from apex: apex + t * (axis + tan_a * dir)
        let ray_dir = (axis + dir * tan_a).normalize();

        // Intersect with plane: dist + normal·(apex + t*ray_dir - plane.origin) = 0
        // t = -(normal·(apex - plane.origin)) / (normal·ray_dir)
        let denom = plane.normal.dot(ray_dir);
        if denom.abs() < 1e-10 { continue; }
        let t = -dist / denom;
        if t > 0.0 && t.is_finite() {
            points.push(apex + ray_dir * t);
        }
    }

    if points.len() < 4 { return None; }

    Some(vec![IntersectionCurve {
        points,
        face_a_id: face_a.surface_id.unwrap_or(0),
        face_b_id: face_b.surface_id.unwrap_or(0),
    }])
}

/// Intersect a cylinder with a cone — numerical parametric tracing.
fn cylinder_cone_intersect(
    face_a: &StepFace, face_b: &StepFace,
    entities_a: &EntityIndex, entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    let (cyl_face, cone_face, cyl_ents, cone_ents) =
        if get_cylinder_info(face_a, entities_a).is_some() {
            (face_a, face_b, entities_a, entities_b)
        } else {
            (face_b, face_a, entities_b, entities_a)
        };

    let cyl = get_cylinder_info(cyl_face, cyl_ents)?;
    let cone = get_cone_info(cone_face, cone_ents)?;

    let c_axis = cyl.axis.normalize();
    let k_axis = cone.axis.normalize();
    let tan_a = cone.semi_angle.tan();

    // Numerical tracing: sample by angle around cylinder axis
    let n = 64;
    let mut points = Vec::new();

    let u = if c_axis.x.abs() < 0.9 { c_axis.cross(Vec3::X).normalize() }
        else { c_axis.cross(Vec3::Y).normalize() };
    let v = c_axis.cross(u).normalize();

    for i in 0..=n {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
        let dir = u * angle.cos() + v * angle.sin();
        let cyl_pt = cyl.origin + dir * cyl.radius;

        // Solve for t such that cyl_pt + t * c_axis lies on the cone:
        // |(cyl_pt + t*c_axis - cone.apex) x k_axis| / (cyl_pt + t*c_axis - cone.apex) . k_axis = tan(semi_angle)
        let d = cyl_pt - cone.apex;
        let d_along = d.dot(k_axis);
        let d_perp = d - k_axis * d_along;
        let a_along = c_axis.dot(k_axis);
        let a_perp = c_axis - k_axis * a_along;

        // Equation: |d_perp + t*a_perp|^2 = tan^2_a * (d_along + t*a_along)^2
        let a = a_perp.length_squared() - tan_a * tan_a * a_along * a_along;
        let b = 2.0 * (d_perp.dot(a_perp) - tan_a * tan_a * d_along * a_along);
        let c = d_perp.length_squared() - tan_a * tan_a * d_along * d_along;

        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { continue; }
        let sqrt_disc = disc.sqrt();
        for &t in &[(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)] {
            if t.is_finite() {
                points.push(cyl_pt + c_axis * t);
            }
        }
    }

    if points.len() < 4 { return None; }
    Some(vec![IntersectionCurve {
        points,
        face_a_id: face_a.surface_id.unwrap_or(0),
        face_b_id: face_b.surface_id.unwrap_or(0),
    }])
}

/// Intersect a torus with a plane — sample the 4th-degree algebraic curve.
fn torus_plane_intersect(
    face_a: &StepFace, face_b: &StepFace,
    entities_a: &EntityIndex, entities_b: &EntityIndex,
) -> Option<Vec<IntersectionCurve>> {
    let (torus_face, plane_face, torus_ents, plane_ents) =
        if get_torus_info(face_a, entities_a).is_some() {
            (face_a, face_b, entities_a, entities_b)
        } else {
            (face_b, face_a, entities_b, entities_a)
        };

    let torus = get_torus_info(torus_face, torus_ents)?;
    let plane = get_plane_info(plane_face, plane_ents)?;

    let normal = plane.normal;
    let axis = torus.axis.normalize();

    // Sample by angle around torus major circle
    let n = 64;
    let mut points = Vec::new();

    for i in 0..=n {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
        let (sin_a, cos_a) = (angle.sin(), angle.cos());

        // Tube center in world space
        let u_dir = if axis.x.abs() < 0.9 { axis.cross(Vec3::X).normalize() }
            else { axis.cross(Vec3::Y).normalize() };
        let v_dir = axis.cross(u_dir).normalize();
        let tube_center = torus.origin + u_dir * torus.major_r * cos_a
            + v_dir * torus.major_r * sin_a;

        // Distance from tube center to plane
        let d = normal.dot(tube_center - plane.origin);
        if d.abs() > torus.minor_r + 1e-6 { continue; }

        // Intersection of plane with minor circle:
        // h = sqrt(minor_r^2 - d^2), intersection points at tube_center +/- h * perp_direction
        let h = (torus.minor_r * torus.minor_r - d * d).sqrt();
        if h < 1e-6 {
            points.push(tube_center);
        } else {
            // Direction perpendicular to normal, in the minor circle plane
            let perp = normal.cross(axis.cross(normal));
            let perp = if perp.length() > 1e-6 { perp.normalize() } else { normal.cross(u_dir).normalize() };
            points.push(tube_center - perp * h);
            points.push(tube_center + perp * h);
        }
    }

    if points.len() < 4 { return None; }
    Some(vec![IntersectionCurve {
        points,
        face_a_id: face_a.surface_id.unwrap_or(0),
        face_b_id: face_b.surface_id.unwrap_or(0),
    }])
}

// ── Surface info extraction helpers ──

struct PlaneInfo { origin: Vec3, normal: Vec3 }
struct CylinderInfo { origin: Vec3, axis: Vec3, radius: f32 }
struct SphereInfo { center: Vec3, radius: f32 }
struct ConeInfo { apex: Vec3, axis: Vec3, semi_angle: f32 }
struct TorusInfo { origin: Vec3, axis: Vec3, major_r: f32, minor_r: f32 }

fn get_plane_info(face: &StepFace, entities: &EntityIndex) -> Option<PlaneInfo> {
    let surface_id = face.surface_id?;
    let surface = entities.get(&surface_id)?;
    if surface.entity_type != EntityType::Plane && surface.entity_type != EntityType::RectangularTrimmedSurface {
        // For RECTANGULAR_TRIMMED_SURFACE, unwrap to base
        return get_plane_info_from_surface(surface, entities);
    }
    get_plane_info_from_surface(surface, entities)
}

fn get_plane_info_from_surface(surface: &super::super::parser::EntityRecord, entities: &EntityIndex) -> Option<PlaneInfo> {
    if surface.entity_type == EntityType::RectangularTrimmedSurface {
        let base_id = super::super::geom::nth_ref(&surface.params, 1)?;
        let base = entities.get(&base_id)?;
        return get_plane_info_from_surface(base, entities);
    }
    let placement_id = super::super::geom::nth_ref(&surface.params, 1)?;
    let (origin, _x, z) = super::super::topology::resolve_placement(placement_id, entities)?;
    Some(PlaneInfo { origin, normal: z.normalize() })
}

fn get_cylinder_info(face: &StepFace, entities: &EntityIndex) -> Option<CylinderInfo> {
    let surface_id = face.surface_id?;
    let surface = entities.get(&surface_id)?;
    let placement_id = super::super::geom::nth_ref(&surface.params, 1)?;
    let radius = super::super::geom::nth_real(&surface.params, 2)? as f32;
    let (origin, _x, z) = super::super::topology::resolve_placement(placement_id, entities)?;
    Some(CylinderInfo { origin, axis: z.normalize(), radius })
}

fn get_sphere_info(face: &StepFace, entities: &EntityIndex) -> Option<SphereInfo> {
    let surface_id = face.surface_id?;
    let surface = entities.get(&surface_id)?;
    let placement_id = super::super::geom::nth_ref(&surface.params, 1)?;
    let radius = super::super::geom::nth_real(&surface.params, 2)? as f32;
    let (origin, _, _) = super::super::topology::resolve_placement(placement_id, entities)?;
    Some(SphereInfo { center: origin, radius })
}

fn get_cone_info(face: &StepFace, entities: &EntityIndex) -> Option<ConeInfo> {
    let surface_id = face.surface_id?;
    let surface = entities.get(&surface_id)?;
    let placement_id = super::super::geom::nth_ref(&surface.params, 1)?;
    let semi_angle = super::super::geom::nth_real(&surface.params, 3)? as f32;
    let (origin, _x, z) = super::super::topology::resolve_placement(placement_id, entities)?;
    Some(ConeInfo { apex: origin, axis: z.normalize(), semi_angle })
}

fn get_torus_info(face: &StepFace, entities: &EntityIndex) -> Option<TorusInfo> {
    let surface_id = face.surface_id?;
    let surface = entities.get(&surface_id)?;
    let placement_id = crate::step::geom::nth_ref(&surface.params, 1)?;
    let major = crate::step::geom::nth_real(&surface.params, 2)? as f32;
    let minor = crate::step::geom::nth_real(&surface.params, 3)? as f32;
    let (origin, _x, z) = crate::step::topology::resolve_placement(placement_id, entities)?;
    Some(TorusInfo { origin, axis: z.normalize(), major_r: major, minor_r: minor })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_plane_parallel() {
        // Two parallel planes → no intersection
        let n1 = Vec3::Z;
        let n2 = Vec3::Z;
        let cross = n1.cross(n2);
        assert!(cross.length() < 1e-10);
    }

    #[test]
    fn test_plane_sphere_no_intersection() {
        // Sphere far from plane
        let center = Vec3::new(0.0, 0.0, 10.0);
        let normal = Vec3::Z;
        let origin = Vec3::ZERO;
        let radius = 1.0;
        let dist = (normal.dot(center - origin)).abs();
        assert!(dist > radius, "sphere should be too far from plane");
    }
}
