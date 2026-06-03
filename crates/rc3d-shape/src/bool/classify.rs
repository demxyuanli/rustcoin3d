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
        _ => RayHit::Miss,
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
        if let Some(pcurve) = edge.pcurves.get(&face_key) {
            let start_uv = pcurve.d0(0.0);
            polygon_uv.push((start_uv.x, start_uv.y));
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
    other_shell: ShellKey,
    reg: &BRepStore,
) -> Vec<(usize, Vec<PointClassification>)> {
    regions.iter().enumerate().map(|(i, region)| {
        let classes: Vec<PointClassification> = region.sub_faces.iter()
            .map(|sub| classify_point_solid(sub.interior_point_3d, other_shell, reg, 1e-4))
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
