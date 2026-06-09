//! Plane fan triangulation fallback for faces that lie on or near a plane.

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;

use crate::geom::{plane_tangent_basis, SurfaceGeom};
use crate::topo::{BRepFace, EdgeKey, FaceKey};

use super::edge_disc::EdgePolygon;
use super::face_fill::{
    emit_filtered_triangles, measure_face_chord_error, FaceFillConfig, FaceMeshRange,
};
use super::face_uv::FaceUvLoops;

/// Planar face fill: sort boundary in plane frame, center fan (reuses welded edge verts).
pub fn project_point_to_plane(pt: Vec3, origin: Vec3, normal: Vec3) -> Vec3 {
    let n = normal.normalize();
    pt - n * (pt - origin).dot(n)
}

pub fn filter_plane_ring_coplanar_cluster(
    ring: &mut Vec<(usize, Vec3)>,
    origin: Vec3,
    normal: Vec3,
) {
    if ring.len() < 4 {
        return;
    }
    let n = normal.normalize();
    let dists: Vec<f32> = ring.iter().map(|(_, p)| (p - origin).dot(n)).collect();
    let mut best_anchor = dists[0];
    let mut best_count = 0usize;
    for &anchor in &dists {
        let band = (dists.iter().copied().fold(0.0f32, f32::max)
            - dists.iter().copied().fold(0.0f32, f32::min))
            .max(1e-3)
            * 0.15
            + normal.length().max(1e-6) * 1e-4;
        let count = dists.iter().filter(|d| (**d - anchor).abs() <= band).count();
        if count > best_count {
            best_count = count;
            best_anchor = anchor;
        }
    }
    let band = (dists.iter().copied().fold(0.0f32, f32::max)
        - dists.iter().copied().fold(0.0f32, f32::min))
        .max(1e-3)
        * 0.15
        + 1e-3;
    ring.retain(|(_, p)| ((p - origin).dot(n) - best_anchor).abs() <= band);
    for (_, p) in ring.iter_mut() {
        *p = project_point_to_plane(*p, origin, n);
    }
}

fn plane_ring_position_tol(pos_to_idx: &super::boundary::BoundaryPosIndex) -> f32 {
    pos_to_idx.cell_size().max(1e-6)
}

/// Build a sorted CCW boundary ring; merges samples that coincide in 3D within tolerance.
pub fn prepare_plane_boundary_ring(
    face: &BRepFace,
    boundary: &[usize],
    global_vertices: &[Vec3],
    pos_to_idx: &super::boundary::BoundaryPosIndex,
) -> Option<Vec<(usize, Vec3)>> {
    let SurfaceGeom::Plane {
        origin,
        normal,
        u_dir,
    } = &face.surface
    else {
        return None;
    };
    let n = (*normal).normalize();
    let (u_axis, v_axis) = plane_tangent_basis(n, *u_dir);
    let tol_sq = plane_ring_position_tol(pos_to_idx).powi(2);

    let mut ring: Vec<(usize, Vec3)> = Vec::new();
    for &gi in boundary {
        let Some(&pt) = global_vertices.get(gi) else {
            continue;
        };
        if ring
            .iter()
            .any(|(_, p)| (*p - pt).length_squared() <= tol_sq)
        {
            continue;
        }
        ring.push((gi, pt));
    }
    if ring.len() >= 2 && (ring[0].1 - ring[ring.len() - 1].1).length_squared() <= tol_sq {
        ring.pop();
    }
    if ring.len() < 3 {
        return None;
    }
    filter_plane_ring_coplanar_cluster(&mut ring, *origin, n);
    if ring.len() < 3 {
        return None;
    }

    let centroid = ring.iter().map(|(_, p)| *p).sum::<Vec3>() / ring.len() as f32;
    let centroid = project_point_to_plane(centroid, *origin, n);
    let mut radii: Vec<f32> = ring
        .iter()
        .map(|(_, p)| (*p - centroid).length())
        .collect();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_r = radii.first().copied().unwrap_or(0.0);
    let max_r = radii.last().copied().unwrap_or(0.0);
    let range = (max_r - min_r).max(min_r * 0.5).max(1e-3);
    let tol = range * 0.5;
    ring.retain(|(_, p)| {
        let r = (*p - centroid).length();
        r >= min_r - tol && r <= max_r + tol
    });
    if ring.len() < 3 {
        return None;
    }

    let center = ring.iter().map(|(_, p)| *p).sum::<Vec3>() / ring.len() as f32;
    let center = project_point_to_plane(center, *origin, n);
    sort_plane_ring_by_angle(&mut ring, center, u_axis, v_axis);
    orient_plane_ring_ccw(&mut ring, u_axis, v_axis);
    let tol = plane_ring_position_tol(pos_to_idx);
    simplify_collinear_plane_ring(&mut ring, tol);
    if ring.len() > 4 {
        ring = convex_hull_plane_ring(&ring, u_axis, v_axis, tol);
        let center = ring.iter().map(|(_, p)| *p).sum::<Vec3>() / ring.len() as f32;
        let center = project_point_to_plane(center, *origin, n);
        sort_plane_ring_by_angle(&mut ring, center, u_axis, v_axis);
        orient_plane_ring_ccw(&mut ring, u_axis, v_axis);
    }
    if ring.len() < 3 {
        return None;
    }
    Some(ring)
}

/// 2D convex hull in the plane frame; keeps extreme corners for rectangular faces.
fn convex_hull_plane_ring(
    ring: &[(usize, Vec3)],
    u_axis: Vec3,
    v_axis: Vec3,
    tol: f32,
) -> Vec<(usize, Vec3)> {
    if ring.len() <= 4 {
        return ring.to_vec();
    }
    let mut pts: Vec<(usize, Vec3, f32, f32)> = ring
        .iter()
        .map(|&(gi, p)| (gi, p, p.dot(u_axis), p.dot(v_axis)))
        .collect();
    pts.sort_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| {
        (a.2 - b.2).abs() <= tol && (a.3 - b.3).abs() <= tol
    });
    if pts.len() <= 4 {
        return pts.into_iter().map(|(gi, p, _, _)| (gi, p)).collect();
    }

    let cross2 = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut lower: Vec<usize> = Vec::new();
    for (i, p) in pts.iter().enumerate() {
        while lower.len() >= 2 {
            let a = lower[lower.len() - 2];
            let b = lower[lower.len() - 1];
            if cross2(
                (pts[a].2, pts[a].3),
                (pts[b].2, pts[b].3),
                (p.2, p.3),
            ) <= tol
            {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(i);
    }

    let mut upper: Vec<usize> = Vec::new();
    for (i, p) in pts.iter().enumerate().rev() {
        while upper.len() >= 2 {
            let a = upper[upper.len() - 2];
            let b = upper[upper.len() - 1];
            if cross2(
                (pts[a].2, pts[a].3),
                (pts[b].2, pts[b].3),
                (p.2, p.3),
            ) <= tol
            {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(i);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
        .into_iter()
        .map(|i| (pts[i].0, pts[i].1))
        .collect()
}

/// Drop interior samples on straight boundary segments (cube face: 8 edge pts → 4 corners).
fn simplify_collinear_plane_ring(ring: &mut Vec<(usize, Vec3)>, tol: f32) {
    if ring.len() <= 4 {
        return;
    }
    loop {
        if ring.len() <= 3 {
            break;
        }
        let mut removed = false;
        for i in 0..ring.len() {
            let prev = ring[(i + ring.len() - 1) % ring.len()].1;
            let curr = ring[i].1;
            let next = ring[(i + 1) % ring.len()].1;
            let edge = next - prev;
            let len = edge.length();
            if len <= 1e-12 {
                ring.remove(i);
                removed = true;
                break;
            }
            let dist = (curr - prev).cross(edge).length() / len;
            if dist <= tol {
                ring.remove(i);
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }
}

/// Fan from the first boundary vertex — (n-2) triangles, no Steiner points (quad → 2 tris).
pub fn fan_triangulate_sorted_ring(ring: &[(usize, Vec3)]) -> Vec<(i32, i32, i32)> {
    if ring.len() < 3 {
        return Vec::new();
    }
    let g0 = ring[0].0 as i32;
    let mut tris = Vec::new();
    for i in 1..ring.len().saturating_sub(1) {
        let g1 = ring[i].0 as i32;
        let g2 = ring[i + 1].0 as i32;
        if g0 != g1 && g1 != g2 && g2 != g0 {
            tris.push((g0, g1, g2));
        }
    }
    tris
}

fn apply_plane_face_normals_to_ring(
    face: &BRepFace,
    ring: &[(usize, Vec3)],
    origin: Vec3,
    normal: Vec3,
    global_normals: &mut Vec<Vec3>,
) -> Vec3 {
    let n = normal.normalize();
    let center = ring.iter().map(|(_, p)| *p).sum::<Vec3>() / ring.len() as f32;
    let center = project_point_to_plane(center, origin, n);
    let mut normal_vec = n;
    if ring.len() >= 2 {
        let e0 = ring[0].1 - center;
        let e1 = ring[1].1 - center;
        if e0.cross(e1).dot(n) < 0.0 {
            normal_vec = -n;
        }
    }
    if !face.same_sense {
        normal_vec = -normal_vec;
    }
    for (gi, _pt) in ring {
        if *gi >= global_normals.len() {
            global_normals.resize(*gi + 1, normal_vec);
        }
        let prev = global_normals[*gi];
        let blended = (prev + normal_vec).normalize();
        if blended.length_squared() > 0.5 {
            global_normals[*gi] = blended;
        } else {
            global_normals[*gi] = normal_vec;
        }
    }
    normal_vec
}

/// Planar fill without interior vertices: 4-gon → 2 triangles (cube → 12 total).
pub fn triangulate_plane_boundary_fan(
    face: &BRepFace,
    loops: &FaceUvLoops,
    global_vertices: &[Vec3],
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &super::boundary::BoundaryPosIndex,
) -> Vec<(i32, i32, i32)> {
    let boundary: Vec<usize> = loops.outer.boundary.iter().map(|v| v.global_idx).collect();
    let SurfaceGeom::Plane {
        origin,
        normal,
        ..
    } = &face.surface
    else {
        return Vec::new();
    };
    let Some(ring) = prepare_plane_boundary_ring(face, &boundary, global_vertices, pos_to_idx)
    else {
        return Vec::new();
    };
    apply_plane_face_normals_to_ring(face, &ring, *origin, *normal, global_normals);
    fan_triangulate_sorted_ring(&ring)
}

pub fn triangulate_plane_center_fan(
    face: &BRepFace,
    loops: &FaceUvLoops,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut super::boundary::BoundaryPosIndex,
    config: &FaceFillConfig,
    shared_boundary: Option<&super::boundary::SharedBoundaryPool>,
) -> Vec<(i32, i32, i32)> {
    let boundary: Vec<usize> = loops.outer.boundary.iter().map(|v| v.global_idx).collect();
    triangulate_plane_center_fan_from_boundary(
        face,
        &boundary,
        global_vertices,
        global_normals,
        pos_to_idx,
        plane_fan_max_edge(config),
        shared_boundary,
        config.skip_interior_edge_split,
    )
}

pub fn plane_angle_at(center: Vec3, pt: Vec3, u_axis: Vec3, v_axis: Vec3) -> f32 {
    let d = pt - center;
    f32::atan2(d.dot(v_axis), d.dot(u_axis))
}

pub fn plane_ring_circular_mean_angle(angles: &[f32]) -> f32 {
    let (mut sin_s, mut cos_s) = (0.0f32, 0.0f32);
    for &a in angles {
        sin_s += a.sin();
        cos_s += a.cos();
    }
    if sin_s * sin_s + cos_s * cos_s < 1e-16 {
        angles.first().copied().unwrap_or(0.0)
    } else {
        f32::atan2(sin_s, cos_s)
    }
}

pub fn plane_unwrap_angle_near(a: f32, ref_angle: f32) -> f32 {
    const PI: f32 = std::f32::consts::PI;
    const TAU: f32 = std::f32::consts::TAU;
    let mut x = a;
    while x - ref_angle > PI {
        x -= TAU;
    }
    while x - ref_angle < -PI {
        x += TAU;
    }
    x
}

pub fn sort_plane_ring_by_angle(
    ring: &mut Vec<(usize, Vec3)>,
    center: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
) {
    let raw: Vec<f32> = ring
        .iter()
        .map(|(_, p)| plane_angle_at(center, *p, u_axis, v_axis))
        .collect();
    let ref_angle = plane_ring_circular_mean_angle(&raw);
    ring.sort_by(|a, b| {
        let aa = plane_unwrap_angle_near(plane_angle_at(center, a.1, u_axis, v_axis), ref_angle);
        let ab = plane_unwrap_angle_near(plane_angle_at(center, b.1, u_axis, v_axis), ref_angle);
        aa.partial_cmp(&ab)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub fn plane_ring_signed_area(ring: &[(usize, Vec3)], u_axis: Vec3, v_axis: Vec3) -> f32 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..ring.len() {
        let (_, p0) = ring[i];
        let (_, p1) = ring[(i + 1) % ring.len()];
        let u0 = p0.dot(u_axis);
        let v0 = p0.dot(v_axis);
        let u1 = p1.dot(u_axis);
        let v1 = p1.dot(v_axis);
        area += u0 * v1 - u1 * v0;
    }
    area * 0.5
}

pub fn orient_plane_ring_ccw(ring: &mut Vec<(usize, Vec3)>, u_axis: Vec3, v_axis: Vec3) {
    if plane_ring_signed_area(ring, u_axis, v_axis) < 0.0 {
        ring.reverse();
    }
}

/// Fan triangulation along the outer boundary wire order (3D topology, not UV).
pub fn fan_triangulate_outer(loops: &FaceUvLoops) -> Vec<(i32, i32, i32)> {
    let b = &loops.outer.boundary;
    if b.len() < 3 {
        return Vec::new();
    }
    let g0 = b[0].global_idx as i32;
    let mut tris = Vec::new();
    for i in 1..b.len().saturating_sub(1) {
        let g1 = b[i].global_idx as i32;
        let g2 = b[i + 1].global_idx as i32;
        if g0 != g1 && g1 != g2 && g2 != g0 {
            tris.push((g0, g1, g2));
        }
    }
    tris
}

/// Plane cap fan from wire edge polygon samples (distinct 3D ring).
pub fn mesh_plane_fan_wire_polygons(
    face_key: FaceKey,
    face: &BRepFace,
    wire_edges: &[(EdgeKey, crate::topo::Orientation, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
) -> FaceMeshRange {
    let SurfaceGeom::Plane { normal, u_dir, .. } = &face.surface else {
        return FaceMeshRange {
            face_key,
            first_tri: all_indices.len() / 4,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
        };
    };
    let first_tri = all_indices.len() / 4;
    let n = normal.normalize();
    let mut ring: Vec<Vec3> = Vec::new();
    for &(ek, _orient, ref pis) in wire_edges {
        let Some(poly) = edge_polygons.get(&ek) else {
            continue;
        };
        for &pi in pis {
            let Some(&(_, pt)) = poly.params_3d.get(pi) else {
                continue;
            };
            if ring
                .last()
                .map(|p| (*p - pt).length_squared() > 1e-12)
                .unwrap_or(true)
            {
                ring.push(pt);
            }
        }
    }
    if ring.len() >= 2 && (ring[0] - ring[ring.len() - 1]).length_squared() < 1e-12 {
        ring.pop();
    }
    if ring.len() < 3 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
        };
    }
    let mut center = Vec3::ZERO;
    for p in &ring {
        center += *p;
    }
    center /= ring.len() as f32;
    let (u_axis, v_axis) = plane_tangent_basis(n, *u_dir);
    let mut order: Vec<usize> = (0..ring.len()).collect();
    let mut ring_pairs: Vec<(usize, Vec3)> = order.iter().map(|&i| (i, ring[i])).collect();
    orient_plane_ring_ccw(&mut ring_pairs, u_axis, v_axis);
    order = ring_pairs.into_iter().map(|(i, _)| i).collect();
    let mut idx_ring: Vec<usize> = Vec::new();
    for &i in &order {
        let pt = ring[i];
        let gi = global_vertices.len();
        global_vertices.push(pt);
        idx_ring.push(gi);
    }
    let center_idx = global_vertices.len();
    global_vertices.push(center);
    let mut normal_vec = n;
    if idx_ring.len() >= 2 {
        let e0 = global_vertices[idx_ring[0]] - center;
        let e1 = global_vertices[idx_ring[1]] - center;
        if e0.cross(e1).dot(n) < 0.0 {
            normal_vec = -n;
        }
    }
    if global_normals.len() < global_vertices.len() {
        global_normals.resize(global_vertices.len(), normal_vec);
    }
    global_normals[center_idx] = normal_vec;
    for i in 0..idx_ring.len() {
        let i0 = idx_ring[i];
        let i1 = idx_ring[(i + 1) % idx_ring.len()];
        if face.same_sense {
            all_indices.extend_from_slice(&[center_idx as i32, i0 as i32, i1 as i32, -1]);
        } else {
            all_indices.extend_from_slice(&[center_idx as i32, i1 as i32, i0 as i32, -1]);
        }
    }
    FaceMeshRange {
        face_key,
        first_tri,
        tri_count: all_indices.len() / 4 - first_tri,
        boundary_global: idx_ring.iter().copied().collect(),
        max_chord_error: 0.0,
        cdt_constraint_failures: 0,
    }
}

/// Fan triangulation in 3D when plane UV trim loops collapse (duplicate/coincident UV).
pub fn mesh_plane_fan_3d(
    face_key: FaceKey,
    boundary_ordered: &[usize],
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut super::boundary::BoundaryPosIndex,
    config: &FaceFillConfig,
    shared_boundary: Option<&super::boundary::SharedBoundaryPool>,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let max_edge = plane_fan_max_edge(config);
    let tris = triangulate_plane_center_fan_from_boundary(
        face,
        boundary_ordered,
        global_vertices,
        global_normals,
        pos_to_idx,
        max_edge,
        shared_boundary,
        config.skip_interior_edge_split,
    );
    let emitted = emit_filtered_triangles(
        tris,
        false,
        f32::MAX,
        face,
        global_vertices,
        global_normals,
        all_indices,
    );
    let boundary_global: HashSet<usize> = boundary_ordered.iter().copied().collect();
    let mut range = FaceMeshRange {
        face_key,
        first_tri,
        tri_count: emitted,
        boundary_global,
        max_chord_error: 0.0,
        cdt_constraint_failures: 0,
    };
    if emitted > 0 {
        range.max_chord_error =
            measure_face_chord_error(face, global_vertices, all_indices, &range);
    }
    range
}

pub fn plane_fan_max_edge(config: &FaceFillConfig) -> f32 {
    let def_floor = (config.deflection_interior * 2.0).max(1e-4);
    let mut max_edge = config.min_size.max(def_floor);
    if config.shell_min_size > 0.0 {
        max_edge = max_edge.max(config.shell_min_size);
    }
    max_edge
}

pub fn triangulate_plane_center_fan_from_boundary(
    face: &BRepFace,
    boundary: &[usize],
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut super::boundary::BoundaryPosIndex,
    max_edge: f32,
    shared_boundary: Option<&super::boundary::SharedBoundaryPool>,
    force_single_fan: bool,
) -> Vec<(i32, i32, i32)> {
    let SurfaceGeom::Plane {
        origin,
        normal,
        ..
    } = &face.surface
    else {
        return Vec::new();
    };
    let Some(ring) = prepare_plane_boundary_ring(face, boundary, global_vertices, pos_to_idx) else {
        return Vec::new();
    };
    let n = (*normal).normalize();
    let normal_vec = apply_plane_face_normals_to_ring(face, &ring, *origin, *normal, global_normals);
    let center = ring.iter().map(|(_, p)| *p).sum::<Vec3>() / ring.len() as f32;
    let center = project_point_to_plane(center, *origin, n);

    let center_gi = super::boundary::register_boundary_point_with_normal_indexed_shared(
        center,
        global_vertices,
        global_normals,
        pos_to_idx,
        shared_boundary,
        || normal_vec,
    );
    if center_gi < global_vertices.len() {
        global_vertices[center_gi] = center;
        if center_gi < global_normals.len() {
            global_normals[center_gi] = normal_vec;
        }
    }

    let mut max_boundary_edge = 0.0f32;
    for i in 0..ring.len() {
        let a = ring[i].1;
        let b = ring[(i + 1) % ring.len()].1;
        max_boundary_edge = max_boundary_edge.max((b - a).length());
    }
    let n_rings = if force_single_fan {
        1
    } else {
        ((max_boundary_edge / max_edge.max(1e-6)).ceil() as u32).clamp(1, 32) as usize
    };

    if n_rings <= 1 {
        let mut tris = Vec::new();
        for i in 0..ring.len() {
            let i0 = ring[i].0 as i32;
            let i1 = ring[(i + 1) % ring.len()].0 as i32;
            let c = center_gi as i32;
            if c != i0 && i0 != i1 && i1 != c {
                tris.push((c, i0, i1));
            }
        }
        return tris;
    }

    let mut levels: Vec<Vec<usize>> = Vec::with_capacity(n_rings);
    levels.push(ring.iter().map(|(gi, _)| *gi).collect());
    for k in 1..n_rings {
        let t = k as f32 / n_rings as f32;
        let mut level = Vec::with_capacity(ring.len());
        for (_, p) in &ring {
            let pt = project_point_to_plane(*p * (1.0 - t) + center * t, *origin, n);
            let gi = super::boundary::register_boundary_point_with_normal_indexed_shared(
                pt,
                global_vertices,
                global_normals,
                pos_to_idx,
                shared_boundary,
                || normal_vec,
            );
            level.push(gi);
        }
        levels.push(level);
    }

    let mut tris = Vec::new();
    for level in 0..(levels.len() - 1) {
        let outer = &levels[level];
        let inner = &levels[level + 1];
        let seg_n = outer.len();
        for i in 0..seg_n {
            let o0 = outer[i] as i32;
            let o1 = outer[(i + 1) % seg_n] as i32;
            let i0 = inner[i] as i32;
            let i1 = inner[(i + 1) % seg_n] as i32;
            if o0 != o1 && i0 != i1 && o0 != i0 {
                tris.push((o0, o1, i1));
            }
            if o0 != i0 && i0 != i1 && o0 != i1 {
                tris.push((o0, i1, i0));
            }
        }
    }
    let inner = levels.last().expect("n_rings >= 1");
    let c = center_gi as i32;
    for i in 0..inner.len() {
        let i0 = inner[i] as i32;
        let i1 = inner[(i + 1) % inner.len()] as i32;
        if c != i0 && i0 != i1 && i1 != c {
            tris.push((c, i0, i1));
        }
    }
    tris
}
