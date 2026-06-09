use rc3d_core::math::Vec3;
use super::boundary::{
    register_boundary_point_with_normal_indexed_shared, BoundaryPosIndex, SharedBoundaryPool,
};
use crate::geom::{SurfaceGeom, SurfaceParamRange};
use super::config::MESH_CLOSED_SURFACE_SEGS;
use super::face_fill::{fix_tri_winding, FaceFillConfig};
use super::face_uv::{point_in_trim, FaceUvLoops};

/// Grid resolution from DeflectionInterior (OCC BRepMesh-style, not a fixed 64x64).
pub fn parametric_grid_segs(face: &crate::topo::BRepFace, config: &FaceFillConfig) -> u32 {
    let defl = config.deflection_interior.max(1e-6);
    match &face.surface {
        SurfaceGeom::Plane { .. } => 2,
        SurfaceGeom::Sphere { radius, .. } => {
            let circ = 2.0 * std::f32::consts::PI * radius;
            ((circ / defl).ceil() as u32).clamp(8, 64)
        }
        SurfaceGeom::Torus { major_r, minor_r, .. } => {
            let circ = 2.0 * std::f32::consts::PI * (major_r + minor_r);
            ((circ / defl).ceil() as u32).clamp(8, 64)
        }
        SurfaceGeom::Cylinder { radius, .. } => {
            let circ = 2.0 * std::f32::consts::PI * radius;
            ((circ / defl).ceil() as u32).clamp(8, 64)
        }
        SurfaceGeom::Cone { radius_at_apex, .. } => {
            let circ = 2.0 * std::f32::consts::PI * radius_at_apex.max(1e-6);
            ((circ / defl).ceil() as u32).clamp(8, 64)
        }
        SurfaceGeom::Revolution { generatrix, .. } => {
            let max_r = generatrix_max_radius(generatrix);
            let circ = 2.0 * std::f32::consts::PI * max_r;
            ((circ / defl).ceil() as u32).clamp(8, 48)
        }
        SurfaceGeom::BSpline(_) | SurfaceGeom::Offset { .. } => 24,
        _ => 24,
    }
}

/// Estimate max radius of a revolution generatrix curve.
fn generatrix_max_radius(curve: &crate::geom::CurveGeom) -> f32 {
    let n = 16;
    let mut max_r2 = 0.0f32;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let p = curve.d0(t);
        max_r2 = max_r2.max(p.x * p.x + p.y * p.y);
    }
    max_r2.sqrt().max(1.0)
}

/// Parametric grid over the boundary UV bounding box (no trim test; for 2-edge revolution patches).
pub fn mesh_uv_bbox_grid(
    face: &crate::topo::BRepFace,
    uv_bounds: (f32, f32, f32, f32),
    fill_config: Option<&FaceFillConfig>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut BoundaryPosIndex,
    all_indices: &mut Vec<i32>,
) {
    mesh_parametric_grid(
        face,
        Some(uv_bounds),
        fill_config,
        global_vertices,
        global_normals,
        pos_to_idx,
        all_indices,
        None,
    );
}

/// Parametric grid clipped to the face trim (outer/holes in UV).

// -- Sutherland-Hodgman polygon clipping helpers for partial trim cells --

/// Check if a 2D polygon is convex by verifying all cross products have the same sign.
fn is_polygon_convex(poly: &[(f32, f32)]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let n = poly.len();
    let mut positive = false;
    let mut negative = false;
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        let p2 = poly[(i + 2) % n];
        let cross = (p1.0 - p0.0) * (p2.1 - p1.1) - (p1.1 - p0.1) * (p2.0 - p1.0);
        if cross > 1e-8 {
            positive = true;
        } else if cross < -1e-8 {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }
    true
}

/// Find the intersection point of two 2D line segments (p1->p2) and (p3->p4).
/// Returns the intersection point if the segments are not parallel and intersect.
fn line_segment_intersection(
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    p4: (f32, f32),
) -> Option<(f32, f32)> {
    let dx1 = p2.0 - p1.0;
    let dy1 = p2.1 - p1.1;
    let dx2 = p4.0 - p3.0;
    let dy2 = p4.1 - p3.1;
    let denom = dx1 * dy2 - dy1 * dx2;
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = ((p3.0 - p1.0) * dy2 - (p3.1 - p1.1) * dx2) / denom;
    let s = ((p3.0 - p1.0) * dy1 - (p3.1 - p1.1) * dx1) / denom;
    if t >= -1e-6 && t <= 1.0 + 1e-6 && s >= -1e-6 && s <= 1.0 + 1e-6 {
        Some((p1.0 + t * dx1, p1.1 + t * dy1))
    } else {
        None
    }
}

/// Clip a convex polygon (subject) against each edge of a clip polygon using
/// the Sutherland-Hodgman algorithm. Returns the resulting polygon vertices.
fn sutherland_hodgman_clip(
    subject: &[(f32, f32)],
    clip: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    if subject.is_empty() || clip.len() < 3 {
        return subject.to_vec();
    }
    let mut output = subject.to_vec();
    let n = clip.len();
    for i in 0..n {
        if output.is_empty() {
            return Vec::new();
        }
        let a = clip[i];
        let b = clip[(i + 1) % n];
        let edge_dx = b.0 - a.0;
        let edge_dy = b.1 - a.1;
        let input = std::mem::take(&mut output);
        let m = input.len();
        for j in 0..m {
            let current = input[j];
            let prev = input[(j + m - 1) % m];
            let cur_inside = edge_dx * (current.1 - a.1) - edge_dy * (current.0 - a.0) >= -1e-8;
            let prev_inside = edge_dx * (prev.1 - a.1) - edge_dy * (prev.0 - a.0) >= -1e-8;
            match (prev_inside, cur_inside) {
                (true, true) => { output.push(current); }
                (true, false) => {
                    if let Some(pt) = line_segment_intersection(prev, current, a, b) {
                        output.push(pt);
                    }
                }
                (false, true) => {
                    if let Some(pt) = line_segment_intersection(prev, current, a, b) {
                        output.push(pt);
                    }
                    output.push(current);
                }
                (false, false) => {}
            }
        }
    }
    output
}

pub fn mesh_trimmed_uv_grid(
    face: &crate::topo::BRepFace,
    loops: &FaceUvLoops,
    uv_bounds: (f32, f32, f32, f32),
    fill_config: Option<&FaceFillConfig>,
    grid_segs: Option<u32>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut BoundaryPosIndex,
    all_indices: &mut Vec<i32>,
    shared_boundary: Option<&SharedBoundaryPool>,
) {
    let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
    if outer_uv.len() < 3 {
        return;
    }
    let holes: Vec<Vec<(f32, f32)>> = loops
        .inners
        .iter()
        .map(|l| l.boundary.iter().map(|v| v.uv).collect())
        .collect();

    let segs = grid_segs
        .unwrap_or_else(|| {
            fill_config
                .map(|c| parametric_grid_segs(face, c))
                .unwrap_or(16)
        })
        .clamp(4, 64);
    let (u_min, u_max, v_min, v_max) = uv_bounds;

    for iu in 0..segs {
        for iv in 0..segs {
            let u0 = u_min + (u_max - u_min) * iu as f32 / segs as f32;
            let u1 = u_min + (u_max - u_min) * (iu + 1) as f32 / segs as f32;
            let v0 = v_min + (v_max - v_min) * iv as f32 / segs as f32;
            let v1 = v_min + (v_max - v_min) * (iv + 1) as f32 / segs as f32;
            let corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)];
            let n_inside = corners
                .iter()
                .filter(|&&(u, v)| point_in_trim(u, v, &outer_uv, &holes))
                .count();

            if n_inside == 0 {
                continue;
            }

            if n_inside == 4 {
                // Fully inside: emit quad as two triangles
                let mut idx = [0i32; 4];
                for (k, &(u, v)) in corners.iter().enumerate() {
                    let pt = face.surface.d0_native(u, v);
                    let mut n = face.surface.normal_native(u, v);
                    if !face.same_sense {
                        n = -n;
                    }
                    let gi = register_boundary_point_with_normal_indexed_shared(
                        pt,
                        global_vertices,
                        global_normals,
                        pos_to_idx,
                        shared_boundary,
                        || n,
                    );
                    idx[k] = gi as i32;
                }
                for (mut i0, mut i1, mut i2) in [(idx[0], idx[1], idx[2]), (idx[0], idx[2], idx[3])] {
                    if i0 == i1 || i1 == i2 || i2 == i0 {
                        continue;
                    }
                    fix_tri_winding(
                        &mut i0,
                        &mut i1,
                        &mut i2,
                        global_vertices,
                        &face.surface,
                        face.same_sense,
                    );
                    all_indices.extend_from_slice(&[i0, i1, i2, -1]);
                }
            } else {
                // Partially inside: clip cell quad against outer trim boundary.
                // NOTE: Sutherland-Hodgman requires a convex clip polygon. For non-convex
                // UV boundaries, fall back to using only the corners that are inside the trim.
                let cell_poly: Vec<(f32, f32)> = corners.to_vec();
                let clipped = if is_polygon_convex(&outer_uv) {
                    sutherland_hodgman_clip(&cell_poly, &outer_uv)
                } else {
                    // Non-convex fallback: keep only corners that pass point_in_trim
                    corners
                        .iter()
                        .copied()
                        .filter(|&(u, v)| point_in_trim(u, v, &outer_uv, &holes))
                        .collect()
                };
                if clipped.len() < 3 {
                    continue;
                }
                // Filter out points that fall inside a hole
                let clipped: Vec<(f32, f32)> = clipped
                    .into_iter()
                    .filter(|&(u, v)| point_in_trim(u, v, &outer_uv, &holes))
                    .collect();
                if clipped.len() < 3 {
                    continue;
                }
                // Fan triangulation from first vertex
                let mut tri_idx = Vec::new();
                for &(u, v) in &clipped {
                    let pt = face.surface.d0_native(u, v);
                    let mut n = face.surface.normal_native(u, v);
                    if !face.same_sense {
                        n = -n;
                    }
                    let gi = register_boundary_point_with_normal_indexed_shared(
                        pt,
                        global_vertices,
                        global_normals,
                        pos_to_idx,
                        shared_boundary,
                        || n,
                    );
                    tri_idx.push(gi as i32);
                }
                for k in 1..tri_idx.len().saturating_sub(1) {
                    let mut i0 = tri_idx[0];
                    let mut i1 = tri_idx[k];
                    let mut i2 = tri_idx[k + 1];
                    if i0 == i1 || i1 == i2 || i2 == i0 {
                        continue;
                    }
                    fix_tri_winding(
                        &mut i0,
                        &mut i1,
                        &mut i2,
                        global_vertices,
                        &face.surface,
                        face.same_sense,
                    );
                    all_indices.extend_from_slice(&[i0, i1, i2, -1]);
                }
            }
        }
    }
}

/// UV parametric grid tessellation for any surface type (last-resort / closed faces).
pub fn mesh_parametric_grid(
    face: &crate::topo::BRepFace,
    uv_bounds: Option<(f32, f32, f32, f32)>,
    fill_config: Option<&FaceFillConfig>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut BoundaryPosIndex,
    all_indices: &mut Vec<i32>,
    shared_boundary: Option<&SharedBoundaryPool>,
) {
    let segs = fill_config
        .map(|c| parametric_grid_segs(face, c))
        .unwrap_or_else(|| match &face.surface {
            SurfaceGeom::Plane { .. } => 2,
            _ => MESH_CLOSED_SURFACE_SEGS,
        });
    let pr = if let Some((u_min, u_max, v_min, v_max)) = uv_bounds {
        SurfaceParamRange {
            u_min,
            u_max,
            v_min,
            v_max,
        }
    } else {
        face.surface.param_range()
    };

    for iu in 0..segs {
        for iv in 0..segs {
            let u0 = pr.u_min + (pr.u_max - pr.u_min) * iu as f32 / segs as f32;
            let u1 = pr.u_min + (pr.u_max - pr.u_min) * (iu + 1) as f32 / segs as f32;
            let v0 = pr.v_min + (pr.v_max - pr.v_min) * iv as f32 / segs as f32;
            let v1 = pr.v_min + (pr.v_max - pr.v_min) * (iv + 1) as f32 / segs as f32;
            let corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)];
            let mut idx = [0i32; 4];
            for (k, &(u, v)) in corners.iter().enumerate() {
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if !face.same_sense {
                    n = -n;
                }
                let gi = register_boundary_point_with_normal_indexed_shared(
                    pt,
                    global_vertices,
                    global_normals,
                    pos_to_idx,
                    shared_boundary,
                    || n,
                );
                idx[k] = gi as i32;
            }
            for (mut i0, mut i1, mut i2) in [(idx[0], idx[1], idx[2]), (idx[0], idx[2], idx[3])] {
                if i0 == i1 || i1 == i2 || i2 == i0 {
                    continue;
                }
                fix_tri_winding(
                    &mut i0,
                    &mut i1,
                    &mut i2,
                    global_vertices,
                    &face.surface,
                    face.same_sense,
                );
                all_indices.extend_from_slice(&[i0, i1, i2, -1]);
                let p0 = global_vertices[i0 as usize];
                let p1 = global_vertices[i1 as usize];
                let p2 = global_vertices[i2 as usize];
                let tri_n = (p1 - p0).cross(p2 - p0);
                if tri_n.length() > 1e-10 {
                    let n = tri_n.normalize();
                    global_normals[i0 as usize] = global_normals[i0 as usize] + n;
                    global_normals[i1 as usize] = global_normals[i1 as usize] + n;
                    global_normals[i2 as usize] = global_normals[i2 as usize] + n;
                }
            }
        }
    }
}

/// Tessellate a closed analytic surface face (VERTEX_LOOP fallback when CDT cannot run).
pub fn mesh_closed_surface(
    face: &crate::topo::BRepFace,
    fill_config: &FaceFillConfig,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut BoundaryPosIndex,
    all_indices: &mut Vec<i32>,
    shared_boundary: Option<&SharedBoundaryPool>,
) {
    mesh_parametric_grid(
        face,
        None,
        Some(fill_config),
        global_vertices,
        global_normals,
        pos_to_idx,
        all_indices,
        shared_boundary,
    );
}
