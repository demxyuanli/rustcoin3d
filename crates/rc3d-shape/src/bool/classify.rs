//! Point-in-solid classification via ray casting (B-Rep native).

use rc3d_core::math::{Real, PVec3};
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
    point: PVec3,
    shell_key: ShellKey,
    reg: &BRepStore,
    tolerance: Real,
) -> PointClassification {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return PointClassification::Outside,
    };

    let ray_dirs = [
        PVec3::new(0.0, 0.0, 1.0),
        PVec3::new(0.0, 1.0, 0.0),
        PVec3::new(1.0, 0.0, 0.0),
        PVec3::new(0.0, 0.0, -1.0),
        PVec3::new(0.0, -1.0, 0.0),
        PVec3::new(-1.0, 0.0, 0.0),
    ];

    let mut confident_votes = Vec::new();

    // Phase 1: try 6 cardinal axes
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

    // Phase 2: if cardinal axes gave no confident results (all hit boundaries),
    // try random directions (OCC BRepClass3d_SClassifier strategy).
    if confident_votes.is_empty() {
        // Use fixed off-axis directions to avoid RNG non-determinism
        let extra_dirs = [
            PVec3::new(0.577350269, 0.577350269, 0.577350269),
            PVec3::new(-0.577350269, 0.577350269, 0.577350269),
            PVec3::new(0.577350269, -0.577350269, 0.577350269),
            PVec3::new(0.577350269, 0.577350269, -0.577350269),
        ];
        for &ray_dir in &extra_dirs {
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
    }

    if confident_votes.is_empty() {
        // OCC BRepClass3d_SolidClassifier: points within tolerance of boundary
        // are considered Inside (PonU classification).
        return PointClassification::Inside;
    }

    let inside_count = confident_votes.iter().filter(|&&v| v).count();
    let outside_count = confident_votes.len() - inside_count;

    // When vote is close (ambiguous), use winding number as tiebreaker.
    // OCC: BRepClass3d_SClassifier uses solid angle for degenerate cases.
    if inside_count.abs_diff(outside_count) <= 1 && !confident_votes.is_empty() {
        let wn = winding_number_approximate(point, shell_key, reg, tolerance);
        if wn > 0.5 {
            return PointClassification::Inside;
        } else if wn < -0.5 {
            return PointClassification::Outside;
        }
        // Winding number ambiguous too — fall through to majority vote
    }

    if inside_count > outside_count {
        PointClassification::Inside
    } else {
        PointClassification::Outside
    }
}

/// Approximate winding number via solid angle subtended by shell faces.
/// OCC: BRepClass3d_SClassifier — solid angle integral over visible faces.
fn winding_number_approximate(
    point: PVec3, shell_key: ShellKey, reg: &BRepStore, _tolerance: Real,
) -> Real {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return 0.0,
    };
    let mut wn = 0.0_f64;
    for &(face_key, _orient) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        // Use face center + normal as proxy for solid angle contribution
        let pts = match face_vertex_positions(face_key, face, reg) {
            Some(p) => p,
            None => continue,
        };
        if pts.len() < 3 { continue; }
        // Solid angle of triangle fan from point
        for i in 1..pts.len() - 1 {
            let a = pts[0] - point;
            let b = pts[i] - point;
            let c = pts[i + 1] - point;
            let la = a.length();
            let lb = b.length();
            let lc = c.length();
            if la < 1e-10 || lb < 1e-10 || lc < 1e-10 { continue; }
            let triple = a.dot(b.cross(c));
            let denom = la * lb * lc + a.dot(b) * lc + a.dot(c) * lb + b.dot(c) * la;
            if denom.abs() < 1e-10 { continue; }
            wn += 2.0 * triple.atan2(denom);
        }
    }
    wn / (4.0 * std::f64::consts::PI)
}

fn face_vertex_positions(
    _face_key: FaceKey, face: &BRepFace, reg: &BRepStore,
) -> Option<Vec<PVec3>> {
    let outer = reg.wires.get(face.outer_wire)?;
    let mut pts = Vec::new();
    for wire in std::iter::once(outer).chain(face.inner_wires.iter().filter_map(|&wk| reg.wires.get(wk))) {
        for &(ek, _) in &wire.edges {
            let edge = reg.edges.get(ek)?;
            let v = reg.vertices.get(edge.v_low)?;
            if pts.last().map(|p: &PVec3| (*p - v.position).length() > 1e-10).unwrap_or(true) {
                pts.push(v.position);
            }
        }
    }
    if pts.len() < 3 { None } else { Some(pts) }
}

#[derive(Debug, Clone, Copy)]
enum RayHit { Hit, Miss, Boundary }

fn ray_surface_intersect(
    origin: PVec3, dir: PVec3,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: Real,
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
    a: Real, b: Real, c: Real,
    origin: PVec3, dir: PVec3,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: Real,
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
    origin: PVec3, dir: PVec3,
    apex: PVec3, axis: PVec3, semi_angle: Real,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: Real,
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
    origin: PVec3, dir: PVec3,
    center: PVec3, axis: PVec3, major_r: Real, minor_r: Real,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: Real,
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
    origin: PVec3, dir: PVec3,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: Real,
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
    let mut candidates: Vec<Real> = Vec::new();
    let mut prev_dist = f64::MAX;

    for i in 0..=coarse_steps {
        let t = max_t * i as Real / coarse_steps as Real;
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
    origin: PVec3, dir: PVec3, t_seed: Real,
    surface: &SurfaceGeom, tolerance: Real, max_iter: usize,
) -> Option<Real> {
    let search_radius = tolerance * 50.0;
    let mut lo = (t_seed - search_radius).max(0.0);
    let mut hi = t_seed + search_radius;

    // Ensure lo has distance > 0 (ray is above surface) and hi has
    // a valid projection close to the surface.
    let mut best_t = t_seed;
    let mut best_dist = f64::MAX;

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
    origin: PVec3, dir: PVec3,
    cyl_origin: PVec3, cyl_axis: PVec3, cyl_radius: Real,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: Real,
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
    origin: PVec3, dir: PVec3,
    center: PVec3, radius: Real,
    face: &BRepFace, face_key: FaceKey, reg: &BRepStore, tolerance: Real,
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

fn point_in_face_uv(point: PVec3, face: &BRepFace, face_key: FaceKey, reg: &BRepStore) -> bool {
    let uv = match face.surface.project(point) {
        Some(uv) => uv,
        None => return false,
    };
    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return true,
    };
    if wire.edges.is_empty() { return true; }
    let mut polygon_uv: Vec<(Real, Real)> = Vec::new();
    for &(edge_key, _orient) in &wire.edges {
        let edge = match reg.edges.get(edge_key) {
            Some(e) => e,
            None => continue,
        };
        if let Some(pcurve) = edge.pcurves.get(&face_key) {
            let start_uv = pcurve.d0(0.0);
            polygon_uv.push((start_uv.0, start_uv.1));
        } else {
            let v_pos = reg.vertices.get(edge.v_low).map(|v| v.position).unwrap_or(PVec3::ZERO);
            if let Some(uv) = face.surface.project(v_pos) {
                polygon_uv.push(uv);
            }
        }
    }
    if polygon_uv.len() < 3 { return true; }
    point_in_polygon_2d(uv, &polygon_uv)
}

fn point_in_polygon_2d(point: (Real, Real), polygon: &[(Real, Real)]) -> bool {
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

/// Classify sub-face regions against the opposite solid's shells.
///
/// NOTE: Currently classifies a point as Inside if ANY of `other_shells`
/// contains it. For multi-shell arguments (e.g., a solid with multiple
/// disconnected shells), this overestimates Inside for the Difference
/// operation. Single-shell usage (the current boolean pipeline) is correct.
/// Multi-shell support would require per-shell classification with
/// per-operation combination semantics.
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
        let result = classify_point_solid(PVec3::ZERO, ShellKey::default(), &reg, 1e-4);
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_classification_values_distinct() {
        assert_ne!(PointClassification::Inside as u8, PointClassification::Outside as u8);
        assert_ne!(PointClassification::Inside as u8, PointClassification::OnBoundary as u8);
    }
}
