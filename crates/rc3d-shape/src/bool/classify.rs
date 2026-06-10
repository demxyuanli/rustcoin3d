//! Point-in-solid classification via ray casting (B-Rep native).

use rc3d_core::math::Vec3;
use crate::store::BRepStore;
use crate::topo::{ShellKey, FaceKey, BRepFace};
use crate::geom::SurfaceGeom;
use super::split::SplitFaceRegion;

/// Classification of a 3D point relative to a B-Rep solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointClassification {
    Inside,
    Outside,
    OnBoundary,
}

/// Classify a 3D point as inside/outside/on a B-Rep solid using ray casting.
/// Odd intersection count = inside, even = outside.
pub fn classify_point_solid(
    point: Vec3,
    shell_key: ShellKey,
    reg: &BRepStore,
    tolerance: f32,
) -> PointClassification {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return PointClassification::Outside,
    };

    let ray_dirs = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
    ];

    let mut confident_votes = Vec::new();

    for &ray_dir in &ray_dirs {
        let mut intersections = 0u32;
        let mut on_boundary = false;

        for &(face_key, _orient) in &shell.faces {
            let face = match reg.faces.get(face_key) {
                Some(f) => f,
                None => continue,
            };

            match ray_surface_intersect(point, ray_dir, face, face_key, reg, tolerance) {
                RayHit::Boundary => { on_boundary = true; break; }
                RayHit::Hit => intersections += 1,
                RayHit::Miss => {}
            }
        }

        if on_boundary { continue; }
        confident_votes.push(intersections % 2 == 1);
    }

    if confident_votes.is_empty() {
        return PointClassification::OnBoundary;
    }

    let inside_count = confident_votes.iter().filter(|&&v| v).count();
    if inside_count > confident_votes.len() / 2 {
        PointClassification::Inside
    } else {
        PointClassification::Outside
    }
}

#[derive(Debug, Clone, Copy)]
enum RayHit { Hit, Miss, Boundary }

fn ray_surface_intersect(
    origin: Vec3, dir: Vec3,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: f32,
) -> RayHit {
    match &face.surface {
        SurfaceGeom::Plane { origin: p0, normal, .. } => {
            let denom = dir.dot(*normal);
            if denom.abs() < 1e-10 { return RayHit::Miss; }
            let t = (*p0 - origin).dot(*normal) / denom;
            if t.abs() < tolerance { return RayHit::Boundary; }
            if t < 0.0 { return RayHit::Miss; }
            let hit = origin + dir * t;
            if point_in_face_uv(hit, face, face_key, reg) { RayHit::Hit } else { RayHit::Miss }
        }
        SurfaceGeom::Cylinder { origin: c0, axis, radius, .. } => {
            ray_cylinder_intersect(origin, dir, *c0, *axis, *radius, face, face_key, reg, tolerance)
        }
        SurfaceGeom::Sphere { center, radius } => {
            ray_sphere_intersect(origin, dir, *center, *radius, face, face_key, reg, tolerance)
        }
        SurfaceGeom::Cone { apex, axis, semi_angle, .. } => {
            ray_cone_intersect(origin, dir, *apex, *axis, *semi_angle, face, face_key, reg, tolerance)
        }
        SurfaceGeom::Torus { center, axis, major_r, minor_r, .. } => {
            ray_torus_intersect(origin, dir, *center, *axis, *major_r, *minor_r, face, face_key, reg, tolerance)
        }
        SurfaceGeom::BSpline(_)
        | SurfaceGeom::Revolution { .. }
        | SurfaceGeom::Extrusion { .. }
        | SurfaceGeom::Offset { .. } => {
            ray_general_intersect(origin, dir, face, face_key, reg, tolerance)
        }
    }
}

// Helper: given quadratic A*t^2 + B*t + C = 0, find ray hits
fn solve_quadratic_ray(
    a: f32, b: f32, c: f32,
    origin: Vec3, dir: Vec3,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: f32,
) -> RayHit {
    if a.abs() < 1e-10 {
        if b.abs() < 1e-10 { return RayHit::Miss; }
        let t = -c / b;
        if t.abs() < tolerance { return RayHit::Boundary; }
        if t > 0.0 {
            let hit = origin + dir * t;
            if point_in_face_uv(hit, face, face_key, reg) { return RayHit::Hit; }
        }
        return RayHit::Miss;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return RayHit::Miss; }
    let sqrt_disc = disc.sqrt();
    for &t in &[(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
        if t.abs() < tolerance { return RayHit::Boundary; }
        if t > 0.0 {
            let hit = origin + dir * t;
            if point_in_face_uv(hit, face, face_key, reg) { return RayHit::Hit; }
        }
    }
    RayHit::Miss
}

fn ray_cone_intersect(
    origin: Vec3, dir: Vec3,
    apex: Vec3, axis: Vec3, semi_angle: f32,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: f32,
) -> RayHit {
    let a = axis.normalize();
    let tan_a = semi_angle.tan();

    let delta = origin - apex;
    let d_par = dir.dot(a);
    let d_perp = dir - a * d_par;
    let o_par = delta.dot(a);
    let o_perp = delta - a * o_par;

    // Cone equation: |P_perp|^2 = (P_par * tan_a)^2
    // Quadratic in t: A*t^2 + B*t + C = 0
    let cap_a = d_perp.length_squared() - d_par * d_par * tan_a * tan_a;
    let cap_b = 2.0 * (d_perp.dot(o_perp) - d_par * o_par * tan_a * tan_a);
    let cap_c = o_perp.length_squared() - o_par * o_par * tan_a * tan_a;

    solve_quadratic_ray(cap_a, cap_b, cap_c, origin, dir, face, face_key, reg, tolerance)
}

fn ray_torus_intersect(
    origin: Vec3, dir: Vec3,
    center: Vec3, axis: Vec3, major_r: f32, minor_r: f32,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: f32,
) -> RayHit {
    // Numerical approach: sample along ray, project to surface
    let a = axis.normalize();
    let ray_len = dir.length();
    if ray_len < 1e-10 { return RayHit::Miss; }
    let step = major_r.min(minor_r) * 0.1;
    let max_t = (major_r + minor_r) * 4.0;
    let mut t = step;
    while t < max_t {
        let pt = origin + dir * (t / ray_len);
        // Project onto torus: find closest surface point
        let rel = pt - center;
        let axial = a * rel.dot(a);
        let radial = rel - axial;
        let radial_dist = radial.length();
        if radial_dist < 1e-10 { t += step; continue; }
        let radial_dir = radial / radial_dist;
        // Project to tube center circle
        let tube_center = center + radial_dir * major_r;
        let to_pt = pt - tube_center;
        let dist = to_pt.length();
        if dist <= minor_r + tolerance {
            let surf_pt = tube_center + to_pt.normalize() * minor_r;
            if (surf_pt - pt).length() < tolerance * 2.0
                && point_in_face_uv(surf_pt, face, face_key, reg) { return RayHit::Hit; }
        }
        t += step;
    }
    RayHit::Miss
}

fn ray_general_intersect(
    origin: Vec3, dir: Vec3,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: f32,
) -> RayHit {
    let ray_len = dir.length();
    if ray_len < 1e-10 { return RayHit::Miss; }
    let d = dir / ray_len;
    let surface = &face.surface;

    // Phase 1: Coarse proximity search — find t-intervals where the
    // ray is close to the surface (< search_tol).
    let search_tol = tolerance.max(1e-3) * 100.0;
    let coarse_steps = 32;
    let max_t = 200.0;
    let mut candidates: Vec<f32> = Vec::new();
    let mut prev_dist = f32::MAX;

    for i in 0..=coarse_steps {
        let t = max_t * i as f32 / coarse_steps as f32;
        let pt = origin + d * t;
        if let Some((u, v)) = surface.project(pt) {
            let surf_pt = surface.d0_native(u, v);
            let dist = (surf_pt - pt).length();
            // Detect sign change in (dist - search_tol): crossing from far to near
            if dist < search_tol && prev_dist >= search_tol {
                candidates.push(t);
            }
            prev_dist = dist;
        }
    }

    // Phase 2: Refine each candidate with binary search
    for &t_seed in &candidates {
        let refined = refine_ray_hit(origin, d, t_seed, surface, tolerance, 8);
        if let Some(t) = refined {
            if t.abs() < tolerance { return RayHit::Boundary; }
            if t > 0.0 {
                let hit = origin + d * t;
                if point_in_face_uv(hit, face, face_key, reg) {
                    return RayHit::Hit;
                }
            }
        }
    }

    RayHit::Miss
}

/// Refine a ray-surface intersection using binary search on the distance.
///
/// Given an initial t where distance < search_tol, narrows down to the
/// exact t where distance ≈ 0 using bisection.
fn refine_ray_hit(
    origin: Vec3, dir: Vec3, t_seed: f32,
    surface: &SurfaceGeom, tolerance: f32, max_iter: usize,
) -> Option<f32> {
    let search_radius = tolerance * 50.0;
    let mut lo = (t_seed - search_radius).max(0.0);
    let mut hi = t_seed + search_radius;

    // Ensure lo has distance > 0 (ray is above surface) and hi has
    // a valid projection close to the surface.
    let mut best_t = t_seed;
    let mut best_dist = f32::MAX;

    for _ in 0..max_iter {
        let mid = (lo + hi) * 0.5;
        let pt = origin + dir * mid;
        if let Some((u, v)) = surface.project(pt) {
            let surf_pt = surface.d0_native(u, v);
            let dist = (surf_pt - pt).length();
            if dist < best_dist {
                best_dist = dist;
                best_t = mid;
            }
            if dist < tolerance {
                return Some(mid);
            }
            // Shrink interval toward the minimum
            if dist < tolerance * 10.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        } else {
            // Can't project — ray is far from surface, move toward seed
            lo = mid;
        }
    }

    if best_dist < tolerance * 10.0 {
        Some(best_t)
    } else {
        None
    }
}

fn ray_cylinder_intersect(
    origin: Vec3, dir: Vec3,
    cyl_origin: Vec3, cyl_axis: Vec3, cyl_radius: f32,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: f32,
) -> RayHit {
    let a = cyl_axis.normalize();
    let d_perp = dir - a * dir.dot(a);
    let oc = origin - cyl_origin;
    let oc_perp = oc - a * oc.dot(a);
    let qa = d_perp.dot(d_perp);
    let qb = 2.0 * d_perp.dot(oc_perp);
    let qc = oc_perp.dot(oc_perp) - cyl_radius * cyl_radius;
    if qa.abs() < 1e-10 { return RayHit::Miss; }
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 { return RayHit::Miss; }
    let sqrt_disc = disc.sqrt();
    for &t in &[(-qb - sqrt_disc) / (2.0 * qa), (-qb + sqrt_disc) / (2.0 * qa)] {
        if t.abs() < tolerance { return RayHit::Boundary; }
        if t > 0.0 {
            let hit = origin + dir * t;
            if point_in_face_uv(hit, face, face_key, reg) { return RayHit::Hit; }
        }
    }
    RayHit::Miss
}

fn ray_sphere_intersect(
    origin: Vec3, dir: Vec3,
    center: Vec3, radius: f32,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: f32,
) -> RayHit {
    let oc = origin - center;
    let a = dir.dot(dir);
    let b = 2.0 * oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return RayHit::Miss; }
    let sqrt_disc = disc.sqrt();
    for &t in &[(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
        if t.abs() < tolerance { return RayHit::Boundary; }
        if t > 0.0 {
            let hit = origin + dir * t;
            if point_in_face_uv(hit, face, face_key, reg) { return RayHit::Hit; }
        }
    }
    RayHit::Miss
}

fn point_in_face_uv(point: Vec3, face: &BRepFace, face_key: FaceKey, reg: &BRepStore) -> bool {
    let uv = match face.surface.project(point) {
        Some(uv) => uv,
        None => return false,
    };
    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return true,
    };
    if wire.edges.is_empty() { return true; }
    let mut polygon_uv: Vec<(f32, f32)> = Vec::new();
    for &(edge_key, _orient) in &wire.edges {
        let edge = match reg.edges.get(edge_key) {
            Some(e) => e,
            None => continue,
        };
        if let Some((pcurve, _same_sense)) = edge.pcurves.get(&face_key) {
            let start_uv = pcurve.d0(0.0);
            polygon_uv.push((start_uv.0, start_uv.1));
        } else {
            let v_pos = reg.vertices.get(edge.v_low).map(|v| v.position).unwrap_or(Vec3::ZERO);
            if let Some(uv) = face.surface.project(v_pos) {
                polygon_uv.push(uv);
            }
        }
    }
    if polygon_uv.len() < 3 { return true; }
    point_in_polygon_2d(uv, &polygon_uv)
}

fn point_in_polygon_2d(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let n = polygon.len();
    if n < 3 { return false; }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if ((yi > point.1) != (yj > point.1))
            && (point.0 < (xj - xi) * (point.1 - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub fn classify_brep_regions(
    regions: &[SplitFaceRegion],
    other_shells: &[ShellKey],
    reg: &BRepStore,
) -> Vec<(usize, Vec<PointClassification>)> {
    regions.iter().enumerate().map(|(i, region)| {
        let classes: Vec<PointClassification> = region.sub_faces.iter()
            .map(|sub| {
                // A point is inside if ANY of the other shells contains it
                for &sk in other_shells {
                    if classify_point_solid(sub.interior_point_3d, sk, reg, 1e-4)
                        == PointClassification::Inside
                    {
                        return PointClassification::Inside;
                    }
                }
                PointClassification::Outside
            })
            .collect();
        (i, classes)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_returns_outside_for_default_shell() {
        let reg = BRepStore::new();
        let result = classify_point_solid(Vec3::ZERO, ShellKey::default(), &reg, 1e-4);
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_classification_values_distinct() {
        assert_ne!(PointClassification::Inside as u8, PointClassification::Outside as u8);
        assert_ne!(PointClassification::Inside as u8, PointClassification::OnBoundary as u8);
    }
}
