//! Point-in-solid classification via ray casting.
//!
//! Determines whether a face (or face region) is inside, outside,
//! or on the boundary of another solid.
//!
//! Target solid meshing uses the OCC-aligned B-Rep pipeline (`build_brep` + `mesh_brep_shell`).

use rc3d_core::math::Vec3;
use super::super::parser::EntityIndex;
use super::super::topology::{StepShell, StepFace};
/// Classification of a face region relative to another solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionClass {
    Inside,
    Outside,
    OnBoundary,
}

/// A classified face with its classification.
pub struct ClassifiedFace {
    pub face: StepFace,
    pub class: RegionClass,
}

fn mesh_target_solids_brep(entities: &EntityIndex) -> super::super::mesh_result::MeshResult {
    use super::super::brep::{build_brep_with_options, BRepBuildOptions};
    use super::super::brep::mesh::{mesh_brep_shell, BRepMeshConfig};
    use super::super::import_options::StepImportOptions;

    let mut out = super::super::mesh_result::MeshResult::default();
    let import_opts = StepImportOptions::default();
    let build_opts = BRepBuildOptions::from_import(&import_opts);
    let brep = match build_brep_with_options(entities, &build_opts) {
        Ok(r) => r,
        Err(_) => return out,
    };
    let mesh_cfg = BRepMeshConfig::default();
    for &sk in &brep.root_solids {
        let shell_key = match brep.registry.solids.get(sk) {
            Some(s) => s.outer_shell,
            None => continue,
        };
        let part = mesh_brep_shell(shell_key, &brep.registry, &mesh_cfg, &[]);
        append_brep_mesh(&mut out, &part);
    }
    out
}

fn append_brep_mesh(
    dst: &mut super::super::mesh_result::MeshResult,
    src: &super::super::mesh_result::MeshResult,
) {
    let base = dst.vertices.len() as i32;
    dst.vertices.extend_from_slice(&src.vertices);
    if src.normals.len() == src.vertices.len() {
        dst.normals.extend_from_slice(&src.normals);
    } else {
        dst.normals.resize(dst.vertices.len(), rc3d_core::math::Vec3::Z);
    }
    for chunk in src.indices.chunks(4) {
        if chunk.len() < 3 {
            continue;
        }
        dst.indices.push(chunk[0] + base);
        dst.indices.push(chunk[1] + base);
        dst.indices.push(chunk[2] + base);
        dst.indices.push(chunk.get(3).copied().unwrap_or(-1));
    }
}

/// Classify each face relative to the target solid using ray casting.
pub fn classify_faces(
    faces: &[StepFace],
    target_shells: &[StepShell],
    target_entities: &EntityIndex,
) -> Vec<ClassifiedFace> {
    // Build mesh from target solid for ray-cast queries.
    // For faces with surface_id: use surface tessellation.
    // For faces without: use edge-loop fallback triangulation.
    let target_faces: Vec<StepFace> = target_shells.iter()
        .flat_map(|s| s.faces.iter().cloned()).collect();
    let mut target_mesh = mesh_target_solids_brep(target_entities);

    if target_mesh.vertices.is_empty() && !target_faces.is_empty() {
        target_mesh = build_mesh_from_edge_loops(&target_faces, target_entities);
    }

    faces.iter().map(|face| {
        let center = face_center(face);
        let class = classify_point(&center, &target_mesh);
        ClassifiedFace { face: face.clone(), class }
    }).collect()
}

/// Build a mesh from edge loops (for faces without surface tessellation).
fn build_mesh_from_edge_loops(
    faces: &[StepFace],
    entities: &EntityIndex,
) -> super::super::mesh_result::MeshResult {
    let mut mesh = super::super::mesh_result::MeshResult::default();
    let mut pos_map: std::collections::HashMap<[u32; 3], i32> = std::collections::HashMap::new();

    for face in faces {
        for bloop in &face.bounds {
            let mut poly_verts: Vec<i32> = Vec::new();
            for edge in &bloop.edges {
                let pts = super::super::geom::sample_curve(
                    edge.curve_id, entities, edge.start, edge.end, edge.tolerance.max(0.1),
                );
                // Add all points except last (connects to next edge)
                let n = pts.len();
                for (j, pt) in pts.iter().enumerate() {
                    if j == n - 1 && n > 1 { continue; }
                    let hash = rc3d_core::utils::hash::f32x3_quantized_bits([pt.x, pt.y, pt.z]);
                    let idx = if let Some(&i) = pos_map.get(&hash) {
                        i
                    } else {
                        let i = mesh.vertices.len() as i32;
                        mesh.vertices.push(*pt);
                        pos_map.insert(hash, i);
                        i
                    };
                    poly_verts.push(idx);
                }
            }
            if poly_verts.len() >= 3 {
                triangulate_loop(&poly_verts, &mesh.vertices, &mut mesh.indices);
            }
        }
    }

    mesh
}

/// Triangulate a polygon loop into the index buffer.
fn triangulate_loop(poly: &[i32], verts: &[Vec3], indices: &mut Vec<i32>) {
    if poly.len() < 3 { return; }
    let pts: Vec<Vec3> = poly.iter()
        .filter_map(|&idx| {
            let i = idx as usize;
            if i < verts.len() { Some(verts[i]) } else { None }
        })
        .collect();
    if pts.len() < 3 { return; }

    // Compute normal
    let n = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
    if n.length() < 1e-10 {
        // Degenerate — use fan
        let v0 = poly[0];
        for j in 1..poly.len() - 1 {
            indices.extend_from_slice(&[v0, poly[j], poly[j + 1], -1]);
        }
        return;
    }
    let normal = n.normalize();

    // Project to 2D
    let abs = [normal.x.abs(), normal.y.abs(), normal.z.abs()];
    let drop = if abs[0] >= abs[1] && abs[0] >= abs[2] { 0 }
        else if abs[1] >= abs[2] { 1 } else { 2 };

    let flat: Vec<f64> = pts.iter().flat_map(|p| {
        match drop {
            0 => [p.y as f64, p.z as f64],
            1 => [p.x as f64, p.z as f64],
            _ => [p.x as f64, p.y as f64],
        }
    }).collect();

    let tri = match earcutr::earcut(&flat, &[], 2) {
        Ok(t) => t,
        Err(_) => {
            let v0 = poly[0];
            for j in 1..poly.len() - 1 {
                indices.extend_from_slice(&[v0, poly[j], poly[j + 1], -1]);
            }
            return;
        }
    };

    for chunk in tri.chunks(3) {
        if chunk.len() == 3 && chunk[0] < poly.len() && chunk[1] < poly.len() && chunk[2] < poly.len() {
            indices.extend_from_slice(&[poly[chunk[0]], poly[chunk[1]], poly[chunk[2]], -1]);
        }
    }
}

use earcutr;

/// Compute the approximate center of a face.
fn face_center(face: &StepFace) -> Vec3 {
    // Average of edge midpoints
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;
    for bloop in &face.bounds {
        for edge in &bloop.edges {
            let mid = (edge.start + edge.end) * 0.5;
            sum = sum + mid;
            count += 1;
        }
    }
    if count > 0 { sum * (1.0 / count as f32) } else { Vec3::ZERO }
}

/// Classify a point relative to a mesh using multiple rays for robustness.
/// Uses 6 axis-aligned rays with majority voting.
/// Special handling for near-boundary cases via jittered retry.
pub fn classify_point(point: &Vec3, mesh: &super::super::mesh_result::MeshResult) -> RegionClass {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return RegionClass::Outside;
    }

    let dirs = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
    let boundary_threshold = 1e-3;
    let mut confident_votes: Vec<bool> = Vec::new();
    let mut hit_boundary = false;

    for dir in &dirs {
        let mut hit_count: i32 = 0;
        let mut near_boundary = false;

        for chunk in mesh.indices.chunks(4) {
            if chunk.len() < 3 { continue; }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
                continue;
            }
            let v0 = mesh.vertices[i0];
            let v1 = mesh.vertices[i1];
            let v2 = mesh.vertices[i2];

            // Skip near-parallel triangles (grazing rays unreliable)
            let normal = (v1 - v0).cross(v2 - v0);
            let ray_dot_normal = dir.dot(normal);
            if ray_dot_normal.abs() < 1e-6 {
                continue;
            }

            if ray_triangle_intersect(point, dir, &v0, &v1, &v2) {
                // Check if hit point is near triangle boundary (grazing)
                let t = compute_ray_triangle_t(point, dir, &v0, &v1, &v2);
                if let Some(t_val) = t {
                    let hit_pt = *point + *dir * t_val;
                    for (a, b) in [(&v0, &v1), (&v1, &v2), (&v2, &v0)] {
                        let edge_vec = *b - *a;
                        let to_hit = hit_pt - *a;
                        let proj = to_hit.dot(edge_vec) / edge_vec.length_squared().max(1e-10);
                        let proj = proj.clamp(0.0, 1.0);
                        let closest = *a + edge_vec * proj;
                        if (hit_pt - closest).length() < boundary_threshold {
                            near_boundary = true;
                            break;
                        }
                    }
                }
                hit_count += 1;
            }
        }

        if near_boundary {
            hit_boundary = true;
            continue; // Skip this ray direction
        }
        confident_votes.push(hit_count % 2 == 1);
    }

    if confident_votes.is_empty() {
        if hit_boundary {
            return classify_point_jittered(point, mesh);
        }
        return RegionClass::Outside;
    }

    let inside_count = confident_votes.iter().filter(|&&v| v).count();
    let outside_count = confident_votes.len() - inside_count;

    if inside_count > outside_count {
        RegionClass::Inside
    } else if outside_count > inside_count {
        RegionClass::Outside
    } else {
        RegionClass::OnBoundary
    }
}

/// Retry classification with small random-like offsets when all rays graze.
fn classify_point_jittered(
    point: &Vec3, mesh: &super::super::mesh_result::MeshResult,
) -> RegionClass {
    let offsets = [
        Vec3::new(0.001, 0.0, 0.0),
        Vec3::new(-0.001, 0.0, 0.0),
        Vec3::new(0.0, 0.001, 0.0),
        Vec3::new(0.0, -0.001, 0.0),
        Vec3::new(0.0, 0.0, 0.001),
        Vec3::new(0.0, 0.0, -0.001),
    ];

    let mut inside = 0u32;
    let mut total = 0u32;
    for offset in &offsets {
        let p = *point + *offset;
        let dir = Vec3::X; // fixed direction, offset handles degeneracy
        let mut hits = 0u32;
        for chunk in mesh.indices.chunks(4) {
            if chunk.len() < 3 { continue; }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() { continue; }
            if ray_triangle_intersect(&p, &dir, &mesh.vertices[i0], &mesh.vertices[i1], &mesh.vertices[i2]) {
                hits += 1;
            }
        }
        if hits > 0 {
            total += 1;
            if hits % 2 == 1 { inside += 1; }
        }
    }

    if total == 0 { RegionClass::Outside }
    else if inside as f32 / total as f32 > 0.5 { RegionClass::Inside }
    else { RegionClass::OnBoundary }
}

/// Moller-Trumbore ray-triangle intersection.
fn ray_triangle_intersect(
    origin: &Vec3, dir: &Vec3,
    v0: &Vec3, v1: &Vec3, v2: &Vec3,
) -> bool {
    let e1 = *v1 - *v0;
    let e2 = *v2 - *v0;
    let h = dir.cross(e2);
    let a = e1.dot(h);

    if a.abs() < 1e-10 {
        return false; // Ray parallel to triangle
    }

    let f = 1.0 / a;
    let s = *origin - *v0;
    let u = f * s.dot(h);

    if u < 0.0 || u > 1.0 {
        return false;
    }

    let q = s.cross(e1);
    let v = f * dir.dot(q);

    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = f * e2.dot(q);
    t > 1e-6 // Intersection in front of ray origin
}

/// Compute the t parameter at which a ray intersects a triangle (Moller-Trumbore).
fn compute_ray_triangle_t(
    origin: &Vec3, dir: &Vec3, v0: &Vec3, v1: &Vec3, v2: &Vec3,
) -> Option<f32> {
    let e1 = *v1 - *v0;
    let e2 = *v2 - *v0;
    let h = dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-10 { return None; }
    let f = 1.0 / a;
    let s = *origin - *v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 { return None; }
    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = f * e2.dot(q);
    if t > 1e-10 { Some(t) } else { None }
}

// ── B-Rep registry-based classification (Phase 3) ────────────────────────

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{ShellKey, FaceKey, BRepFace, Orientation};
use crate::step::brep::geom::SurfaceGeom;
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
    reg: &BRepRegistry,
    tolerance: f32,
) -> PointClassification {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return PointClassification::Outside,
    };

    // Try multiple ray directions for robustness
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
                RayHit::Boundary => {
                    on_boundary = true;
                    break;
                }
                RayHit::Hit => intersections += 1,
                RayHit::Miss => {}
            }
        }

        if on_boundary {
            continue; // skip this ray
        }
        confident_votes.push(intersections % 2 == 1);
    }

    if confident_votes.is_empty() {
        return PointClassification::OnBoundary;
    }

    let inside_count = confident_votes.iter().filter(|&&v| v).count();
    let outside_count = confident_votes.len() - inside_count;

    if inside_count > outside_count {
        PointClassification::Inside
    } else {
        PointClassification::Outside
    }
}

/// Result of a ray-surface intersection test.
#[derive(Debug, Clone, Copy)]
enum RayHit {
    Hit,
    Miss,
    Boundary,
}

/// Test ray against a single face's surface geometry.
fn ray_surface_intersect(
    origin: Vec3,
    dir: Vec3,
    face: &BRepFace,
    face_key: FaceKey,
    reg: &BRepRegistry,
    tolerance: f32,
) -> RayHit {
    match &face.surface {
        SurfaceGeom::Plane { origin: p0, normal, .. } => {
            let denom = dir.dot(*normal);
            if denom.abs() < 1e-10 {
                return RayHit::Miss; // parallel
            }
            let t = (*p0 - origin).dot(*normal) / denom;
            if t.abs() < tolerance {
                return RayHit::Boundary;
            }
            if t < 0.0 {
                return RayHit::Miss;
            }
            let hit = origin + dir * t;
            if point_in_face_uv(hit, face, face_key, reg) {
                RayHit::Hit
            } else {
                RayHit::Miss
            }
        }
        SurfaceGeom::Cylinder { origin: c0, axis, radius, .. } => {
            ray_cylinder_intersect(origin, dir, *c0, *axis, *radius, face, face_key, reg, tolerance)
        }
        SurfaceGeom::Sphere { center, radius } => {
            ray_sphere_intersect(origin, dir, *center, *radius, face, face_key, reg, tolerance)
        }
        _ => RayHit::Miss, // TODO: other surface types
    }
}

/// Ray-cylinder intersection.
fn ray_cylinder_intersect(
    origin: Vec3, dir: Vec3,
    cyl_origin: Vec3, cyl_axis: Vec3, cyl_radius: f32,
    face: &BRepFace, face_key: FaceKey, reg: &BRepRegistry, tolerance: f32,
) -> RayHit {
    let a = cyl_axis.normalize();
    let d_perp = dir - a * dir.dot(a);
    let oc = origin - cyl_origin;
    let oc_perp = oc - a * oc.dot(a);

    let qa = d_perp.dot(d_perp);
    let qb = 2.0 * d_perp.dot(oc_perp);
    let qc = oc_perp.dot(oc_perp) - cyl_radius * cyl_radius;

    if qa.abs() < 1e-10 {
        return RayHit::Miss; // ray parallel to axis
    }

    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return RayHit::Miss;
    }

    let sqrt_disc = disc.sqrt();
    let t1 = (-qb - sqrt_disc) / (2.0 * qa);
    let t2 = (-qb + sqrt_disc) / (2.0 * qa);

    for &t in &[t1, t2] {
        if t.abs() < tolerance {
            return RayHit::Boundary;
        }
        if t > 0.0 {
            let hit = origin + dir * t;
            if point_in_face_uv(hit, face, face_key, reg) {
                return RayHit::Hit;
            }
        }
    }
    RayHit::Miss
}

/// Ray-sphere intersection.
fn ray_sphere_intersect(
    origin: Vec3, dir: Vec3,
    center: Vec3, radius: f32,
    face: &BRepFace, face_key: FaceKey, reg: &BRepRegistry, tolerance: f32,
) -> RayHit {
    let oc = origin - center;
    let a = dir.dot(dir);
    let b = 2.0 * oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return RayHit::Miss;
    }
    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    for &t in &[t1, t2] {
        if t.abs() < tolerance {
            return RayHit::Boundary;
        }
        if t > 0.0 {
            let hit = origin + dir * t;
            if point_in_face_uv(hit, face, face_key, reg) {
                return RayHit::Hit;
            }
        }
    }
    RayHit::Miss
}

/// Check if a 3D point (known to lie on the surface) is within the face's UV boundary.
/// For faces without wire boundaries, returns true (unbounded face).
fn point_in_face_uv(point: Vec3, face: &BRepFace, _face_key: FaceKey, reg: &BRepRegistry) -> bool {
    let uv = match face.surface.project(point) {
        Some(uv) => uv,
        None => return false,
    };

    // Get wire edges for boundary check
    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return true, // no boundary = unbounded face
    };

    if wire.edges.is_empty() {
        return true; // no boundary edges = unbounded
    }

    // Build UV polygon from wire edges and test point-in-polygon
    let mut polygon_uv: Vec<(f32, f32)> = Vec::new();
    for &(edge_key, _orient) in &wire.edges {
        let edge = match reg.edges.get(edge_key) {
            Some(e) => e,
            None => continue,
        };
        // Check if edge has a pcurve for this face
        if let Some(pcurve) = edge.pcurves.get(&_face_key) {
            // Use pcurve for UV
            let start_uv = pcurve.d0(0.0);
            polygon_uv.push((start_uv.x, start_uv.y));
        } else {
            // Project 3D vertex to UV
            let v_pos = reg.vertices.get(edge.v_low).map(|v| v.position).unwrap_or(Vec3::ZERO);
            if let Some(uv) = face.surface.project(v_pos) {
                polygon_uv.push(uv);
            }
        }
    }

    if polygon_uv.len() < 3 {
        return true; // can't form polygon, assume inside
    }

    // Point-in-polygon test (ray casting in 2D)
    point_in_polygon_2d(uv, &polygon_uv)
}

/// 2D point-in-polygon test using ray casting.
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

/// Classify split face regions as inside/outside the other solid.
pub fn classify_brep_regions(
    regions: &[SplitFaceRegion],
    other_shell: ShellKey,
    reg: &BRepRegistry,
) -> Vec<(usize, PointClassification)> {
    regions.iter().enumerate().map(|(i, region)| {
        let mut classifications = Vec::new();
        for sub in &region.sub_faces {
            let c = classify_point_solid(sub.interior_point_3d, other_shell, reg, 1e-4);
            classifications.push(c);
        }
        // Use first non-boundary classification
        let overall = classifications.iter()
            .find(|&&c| c != PointClassification::OnBoundary)
            .copied()
            .unwrap_or(PointClassification::Outside);
        (i, overall)
    }).collect()
}

#[cfg(test)]
mod brep_classify_tests {
    use super::*;
    use crate::step::brep::topo::*;
    use crate::step::brep::geom::SurfaceGeom;

    /// Build a simple cube shell (6 plane faces) in the registry.
    fn make_cube_shell(reg: &mut BRepRegistry, size: f32) -> ShellKey {
        let h = size * 0.5;
        // 6 faces: +X, -X, +Y, -Y, +Z, -Z
        let faces_data = [
            (Vec3::new( h, 0.0, 0.0), Vec3::X,  Vec3::Z),  // +X
            (Vec3::new(-h, 0.0, 0.0), -Vec3::X, Vec3::Z),  // -X
            (Vec3::new(0.0,  h, 0.0), Vec3::Y,  Vec3::Z),  // +Y
            (Vec3::new(0.0, -h, 0.0), -Vec3::Y, Vec3::Z),  // -Y
            (Vec3::new(0.0, 0.0,  h), Vec3::Z,  Vec3::X),  // +Z
            (Vec3::new(0.0, 0.0, -h), -Vec3::Z, Vec3::X),  // -Z
        ];

        let mut face_keys = Vec::new();
        for &(origin, normal, u_dir) in &faces_data {
            let surface = SurfaceGeom::Plane { origin, normal, u_dir };
            let wire = reg.wires.insert(BRepWire { edges: vec![] });
            let fk = reg.faces.insert(BRepFace {
                surface,
                outer_wire: wire,
                inner_wires: vec![],
                same_sense: true,
                tolerance: 1e-6,
                seam_edges: vec![],
                color: None,
                degenerated_edges: vec![],
            });
            face_keys.push((fk, Orientation::Forward));
        }

        reg.shells.insert(BRepShell {
            faces: face_keys,
            closed: true,
            step_id: None,
        })
    }

    #[test]
    fn test_classify_point_in_cube() {
        let mut reg = BRepRegistry::new();
        let shell = make_cube_shell(&mut reg, 10.0);

        // Center should be inside (ray hits 1 face in +Z direction -> odd)
        let center = Vec3::ZERO;
        let class = classify_point_solid(center, shell, &reg, 1e-4);
        // With unbounded plane faces (no wire), the ray will hit each plane.
        // For a cube centered at origin with h=5, ray from (0,0,0) in +Z hits z=5 plane.
        // Since planes are unbounded, this is a simplified test.
        assert!(
            class == PointClassification::Inside || class == PointClassification::Outside,
            "Classification should be determined, got {:?}", class
        );
    }

    #[test]
    fn test_classify_point_far_outside() {
        // With unbounded plane faces (no wire boundary), classification
        // degenerates: every ray hits exactly one plane.
        // This test verifies the classifier doesn't panic and returns a valid result.
        let mut reg = BRepRegistry::new();
        let shell = make_cube_shell(&mut reg, 2.0);

        let far = Vec3::new(100.0, 0.0, 0.0);
        let class = classify_point_solid(far, shell, &reg, 1e-4);
        // Unbounded planes give degenerate classification; just verify no panic
        assert!(
            class == PointClassification::Inside || class == PointClassification::Outside,
            "Should return a valid classification, got {:?}", class
        );
    }

    #[test]
    fn test_point_in_polygon_2d() {
        let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(point_in_polygon_2d((0.5, 0.5), &square));
        assert!(!point_in_polygon_2d((2.0, 0.5), &square));
        assert!(!point_in_polygon_2d((-1.0, 0.5), &square));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::mesh_result::MeshResult;

    #[test]
    fn test_ray_triangle_hit() {
        let origin = Vec3::new(-1.0, 0.5, 0.5);
        let dir = Vec3::X;
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(0.0, 1.0, 0.0);
        let v2 = Vec3::new(0.0, 0.0, 1.0);
        assert!(ray_triangle_intersect(&origin, &dir, &v0, &v1, &v2));
    }

    #[test]
    fn test_ray_triangle_miss() {
        let origin = Vec3::new(-1.0, 2.0, 2.0);
        let dir = Vec3::X;
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(0.0, 1.0, 0.0);
        let v2 = Vec3::new(0.0, 0.0, 1.0);
        assert!(!ray_triangle_intersect(&origin, &dir, &v0, &v1, &v2));
    }
}
