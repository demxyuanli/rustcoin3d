//! BRepMesh_Face param-domain fill — OCC BRepMesh_Delaun stand-in (earcut P0).
//! Maps trim loops in UV -> Geom_Surface::Value(u,v).

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;

use super::edge_disc::EdgePolygon;
use super::face_cdt::triangulate_uv_cdt_with_steiner;
use super::face_uv::{
    assign_revolution_native_uv_along_wire, ensure_loop_orientation,
    rebuild_loop_uv_local_frame, revolution_boundary_v_collapsed, revolution_u_span_collapsed,
    split_boundary_chains_at_3d_jumps, unwrap_periodic_uv_loops, FaceUvLoops, UvLoop, UvVertex,
    loops_native_surface_uv, UvSource, uv_loop_is_degenerate,
    cylinder_loop_needs_uv_rebuild, repair_cylinder_uv_loops,
};
use crate::geom::{plane_tangent_basis, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::{BRepFace, EdgeKey, FaceKey};

#[derive(Debug, Clone)]
pub struct FaceFillConfig {
    pub enable_interior: bool,
    /// IMeshTools_Parameters::DeflectionInterior — max chord-to-surface gap (3D).
    pub deflection_interior: f32,
    /// IMeshTools_Parameters::MinSize — absolute minimum triangle edge length (3D).
    pub min_size: f32,
    /// Fraction of shell bbox diagonal used as MinSize floor when `relative_deflection > 0`.
    pub min_size_relative: f32,
    /// Set by shell mesher: shell_diag * min_size_relative when relative mode active.
    pub shell_min_size: f32,
    /// Max deflection-driven split iterations during fill (OCC node-insertion loop).
    pub max_adapt_iterations: usize,
    /// IMeshTools_Parameters::Angle — max angular deflection in radians for Steiner splits.
    pub angular_deflection: f32,
    /// Adaptive UV subdivision depth for structured interior grid (default 4).
    pub parameter_division_max_depth: usize,
    /// Skip expensive surface.project edge splits after structured grid insert.
    pub skip_interior_edge_split: bool,
}

impl Default for FaceFillConfig {
    fn default() -> Self {
        Self {
            enable_interior: true,
            deflection_interior: 0.01,
            min_size: 1e-3,
            min_size_relative: 0.01,
            shell_min_size: 0.0,
            max_adapt_iterations: 2,
            angular_deflection: 0.2,
            parameter_division_max_depth: 1,
            skip_interior_edge_split: true,
        }
    }
}

#[derive(Clone)]
pub struct FaceMeshRange {
    pub face_key: FaceKey,
    pub first_tri: usize,
    pub tri_count: usize,
    pub boundary_global: HashSet<usize>,
    pub max_chord_error: f32,
}

pub fn default_grid_for_surface(surface: &SurfaceGeom) -> (u32, u32) {
    match surface {
        SurfaceGeom::Plane { .. } => (1, 1),
        _ => (4, 4),
    }
}

/// Effective MinSize for a face (absolute floor + relative to boundary bbox diagonal).
pub fn effective_min_size(config: &FaceFillConfig) -> f32 {
    // MinSize floor tied to DeflectionInterior (OCC: min edge ~ few × deflection).
    let def_floor = config.deflection_interior * 2.0;
    let mut min = config.min_size.max(def_floor);
    if config.shell_min_size > 0.0 {
        min = min.max(config.shell_min_size);
    }
    min
}

pub fn face_boundary_is_mixed(loops: &FaceUvLoops, verts: &[Vec3]) -> bool {
    super::face_uv::boundary_is_mixed(&loops.outer.boundary, verts)
}

fn max_allowed_triangle_edge(loops: &FaceUvLoops, verts: &[Vec3]) -> f32 {
    let boundary = &loops.outer.boundary;
    let n = boundary.len();
    if n < 2 {
        return 1.0;
    }
    let mut lens = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let len = if boundary[i].global_idx < verts.len() && boundary[j].global_idx < verts.len() {
            (verts[boundary[j].global_idx] - verts[boundary[i].global_idx]).length()
        } else {
            0.0
        };
        lens.push(len);
    }
    for inner in &loops.inners {
        let m = inner.boundary.len();
        for i in 0..m {
            let j = (i + 1) % m;
            let len = if inner.boundary[i].global_idx < verts.len()
                && inner.boundary[j].global_idx < verts.len()
            {
                (verts[inner.boundary[j].global_idx] - verts[inner.boundary[i].global_idx]).length()
            } else {
                0.0
            };
            lens.push(len);
        }
    }
    let mut sorted = lens.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[sorted.len() / 2];
    let max_b = lens.iter().copied().fold(0.0f32, f32::max);
    (med * 3.0).max(1e-4).min(max_b * 1.5)
}

fn tri_max_edge_len(i0: i32, i1: i32, i2: i32, verts: &[Vec3]) -> f32 {
    let (i0, i1, i2) = (i0 as usize, i1 as usize, i2 as usize);
    if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
        return f32::MAX;
    }
    let (p0, p1, p2) = (verts[i0], verts[i1], verts[i2]);
    (p1 - p0)
        .length()
        .max((p2 - p1).length())
        .max((p0 - p2).length())
}

fn triangle_passes_quality(
    i0: i32,
    i1: i32,
    i2: i32,
    verts: &[Vec3],
    max_edge: f32,
    surface: &SurfaceGeom,
    same_sense: bool,
) -> bool {
    if tri_max_edge_len(i0, i1, i2, verts) > max_edge {
        return false;
    }
    let (p0, p1, p2) = (verts[i0 as usize], verts[i1 as usize], verts[i2 as usize]);
    let tri_n = (p1 - p0).cross(p2 - p0);
    if tri_n.length() <= 1e-10 {
        return false;
    }
    let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
    let face_n = if let Some((u, v)) = surface.project(centroid) {
        let mut sn = surface.normal_native(u, v);
        if !same_sense {
            sn = -sn;
        }
        sn
    } else {
        // Fallback: use vertex-averaged normal from surface projection
        compute_vertex_averaged_normal(surface, &[p0, p1, p2], same_sense)
    };
    tri_n.normalize().dot(face_n) > -0.05
}

/// Compute an approximate face normal by averaging surface normals at projected vertices.
/// Used when centroid projection fails.
fn compute_vertex_averaged_normal(
    surface: &SurfaceGeom,
    vertices: &[Vec3],
    same_sense: bool,
) -> Vec3 {
    let mut avg = Vec3::ZERO;
    let mut count = 0usize;
    for &pt in vertices {
        if let Some((u, v)) = surface.project(pt) {
            let mut n = surface.normal_native(u, v);
            if !same_sense {
                n = -n;
            }
            if n.length_squared() > 1e-12 {
                avg += n.normalize();
                count += 1;
            }
        }
    }
    if count > 0 {
        let n = avg / count as f32;
        let len = n.length();
        if len > 1e-10 {
            n / len
        } else {
            Vec3::Z
        }
    } else {
        Vec3::Z
    }
}

fn chain_arc_length(chain: &[super::face_uv::UvVertex], verts: &[Vec3]) -> f32 {
    if chain.len() < 2 {
        return 0.0;
    }
    let mut arc = 0.0f32;
    for w in chain.windows(2) {
        let a = w[0].global_idx;
        let b = w[1].global_idx;
        if a < verts.len() && b < verts.len() {
            arc += (verts[b] - verts[a]).length();
        }
    }
    arc
}

fn triangulate_open_chain_strip(chain: &[super::face_uv::UvVertex]) -> Vec<(i32, i32, i32)> {
    if chain.len() < 3 {
        return Vec::new();
    }
    let mut tris = Vec::new();
    for i in 0..chain.len().saturating_sub(2) {
        let (g0, g1, g2) = (
            chain[i].global_idx as i32,
            chain[i + 1].global_idx as i32,
            chain[i + 2].global_idx as i32,
        );
        if g0 != g1 && g1 != g2 && g2 != g0 {
            tris.push((g0, g1, g2));
        }
    }
    tris
}

fn chain_planarity_deviation(chain: &[super::face_uv::UvVertex], verts: &[Vec3]) -> f32 {
    if chain.len() < 3 {
        return f32::MAX;
    }
    let mut normal = Vec3::ZERO;
    for i in 0..chain.len() {
        let j = (i + 1) % chain.len();
        let Some(pi) = verts.get(chain[i].global_idx) else { continue; };
        let Some(pj) = verts.get(chain[j].global_idx) else { continue; };
        normal.x += (pi.y - pj.y) * (pi.z + pj.z);
        normal.y += (pi.z - pj.z) * (pi.x + pj.x);
        normal.z += (pi.x - pj.x) * (pi.y + pj.y);
    }
    if normal.length_squared() < 1e-20 {
        return f32::MAX;
    }
    normal = normal.normalize();
    let Some(origin) = verts.get(chain[0].global_idx) else {
        return f32::MAX;
    };
    chain
        .iter()
        .filter_map(|v| verts.get(v.global_idx))
        .map(|p| (*p - *origin).cross(normal).length())
        .fold(0.0f32, f32::max)
}

fn chain_should_earcut(chain: &[super::face_uv::UvVertex], verts: &[Vec3]) -> bool {
    let arc = chain_arc_length(chain, verts);
    if arc < 1e-6 {
        return false;
    }
    chain_planarity_deviation(chain, verts) < arc * 0.08
}

fn triangulate_open_chain_uv(chain: &[super::face_uv::UvVertex], verts: &[Vec3]) -> Vec<(i32, i32, i32)> {
    if chain.len() < 3 {
        return Vec::new();
    }
    if !chain_should_earcut(chain, verts) {
        return triangulate_open_chain_strip(chain);
    }
    let uv_pairs: Vec<(f64, f64)> = chain
        .iter()
        .map(|v| (v.uv.0 as f64, v.uv.1 as f64))
        .collect();
    let ear_tris = super::delaunay2d::earcut_uv_polygon(&uv_pairs, &[]);
    if ear_tris.is_empty() {
        return triangulate_open_chain_strip(chain);
    }
    let mut tris = Vec::new();
    for tri in &ear_tris {
        let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
        if i0 >= chain.len() || i1 >= chain.len() || i2 >= chain.len() {
            continue;
        }
        let (g0, g1, g2) = (
            chain[i0].global_idx as i32,
            chain[i1].global_idx as i32,
            chain[i2].global_idx as i32,
        );
        if g0 != g1 && g1 != g2 && g2 != g0 {
            tris.push((g0, g1, g2));
        }
    }
    if tris.is_empty() {
        triangulate_open_chain_strip(chain)
    } else {
        tris
    }
}

#[allow(clippy::too_many_arguments)]
fn triangulate_loops_cdt(
    work_loops: &FaceUvLoops,
    face: &BRepFace,
    face_key: FaceKey,
    reg: &BRepStore,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    fill_cfg: &FaceFillConfig,
) -> (Vec<(i32, i32, i32)>, f32) {
    let (tris_flat, max_chord_error) = triangulate_uv_cdt_with_steiner(
        work_loops,
        face,
        Some(face_key),
        global_vertices,
        global_normals,
        pos_to_idx,
        fill_cfg,
        Some(reg),
    );
    let mut tris = Vec::new();
    for chunk in tris_flat.chunks(3) {
        if chunk.len() != 3 {
            continue;
        }
        tris.push((chunk[0] as i32, chunk[1] as i32, chunk[2] as i32));
    }
    (tris, max_chord_error)
}

#[allow(clippy::too_many_arguments)]
fn try_cdt_or_earcut(
    work_loops: &FaceUvLoops,
    face: &BRepFace,
    face_key: FaceKey,
    reg: &BRepStore,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    fill_cfg: &FaceFillConfig,
) -> (Vec<(i32, i32, i32)>, f32) {
    triangulate_loops_cdt(
        work_loops,
        face,
        face_key,
        reg,
        global_vertices,
        global_normals,
        pos_to_idx,
        fill_cfg,
    )
}

fn triangulate_loops_earcut_fallback(work_loops: &FaceUvLoops) -> Vec<(i32, i32, i32)> {
    let mut tris: Vec<(i32, i32, i32)> = Vec::new();
    let mut earcut_verts: Vec<(usize, (f32, f32))> = Vec::new();
    for v in &work_loops.outer.boundary {
        earcut_verts.push((v.global_idx, v.uv));
    }
    let mut hole_indices = Vec::new();
    for inner in &work_loops.inners {
        hole_indices.push(earcut_verts.len());
        for v in &inner.boundary {
            earcut_verts.push((v.global_idx, v.uv));
        }
    }
    if earcut_verts.len() < 3 {
        return tris;
    }
    // For polygons with holes, go directly to fan triangulation.
    if !hole_indices.is_empty() {
        let g0 = earcut_verts[0].0 as i32;
        for i in 1..earcut_verts.len() - 1 {
            tris.push((g0, earcut_verts[i].0 as i32, earcut_verts[i + 1].0 as i32));
        }
        return tris;
    }
    let uv_pairs: Vec<(f64, f64)> = earcut_verts
        .iter()
        .map(|&(_, (u, v))| (u as f64, v as f64))
        .collect();
    let ear_tris = super::delaunay2d::earcut_uv_polygon(&uv_pairs, &[]);
    if ear_tris.is_empty() {
        let g0 = earcut_verts[0].0 as i32;
        for i in 1..earcut_verts.len() - 1 {
            tris.push((g0, earcut_verts[i].0 as i32, earcut_verts[i + 1].0 as i32));
        }
        return tris;
    }
    for tri in &ear_tris {
        let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
        if i0 >= earcut_verts.len() || i1 >= earcut_verts.len() || i2 >= earcut_verts.len() {
            continue;
        }
        tris.push((
            earcut_verts[i0].0 as i32,
            earcut_verts[i1].0 as i32,
            earcut_verts[i2].0 as i32,
        ));
    }
    tris
}

#[allow(clippy::too_many_arguments)]
fn emit_filtered_triangles(
    tris: Vec<(i32, i32, i32)>,
    apply_quality: bool,
    max_edge: f32,
    face: &BRepFace,
    global_vertices: &mut [Vec3],
    global_normals: &mut [Vec3],
    all_indices: &mut Vec<i32>,
) -> usize {
    let mut emitted = 0usize;
    for (mut i0, mut i1, mut i2) in tris.iter().copied() {
        if i0 == i1 || i1 == i2 || i2 == i0 {
            continue;
        }
        if apply_quality
            && !triangle_passes_quality(
                i0,
                i1,
                i2,
                global_vertices,
                max_edge,
                &face.surface,
                face.same_sense,
            )
        {
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
        accumulate_normals(i0, i1, i2, global_vertices, global_normals);
        emitted += 1;
    }

    if emitted == 0 && apply_quality {
        for (mut i0, mut i1, mut i2) in tris {
            if i0 == i1 || i1 == i2 || i2 == i0 {
                continue;
            }
            if tri_max_edge_len(i0, i1, i2, global_vertices) > max_edge * 1.25 {
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
            accumulate_normals(i0, i1, i2, global_vertices, global_normals);
            emitted += 1;
        }
    }

    emitted
}

#[allow(clippy::too_many_arguments)]
fn fill_mixed_boundary_segmented(
    face_key: FaceKey,
    work_loops: &FaceUvLoops,
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    first_tri: usize,
    boundary_global: HashSet<usize>,
) -> FaceMeshRange {
    let (chains, _) = split_boundary_chains_at_3d_jumps(
        &work_loops.outer.boundary,
        global_vertices,
        8.0,
    );
    let mut tris = Vec::new();
    for chain in &chains {
        tris.extend(triangulate_open_chain_uv(chain, global_vertices));
    }
    let max_edge = max_allowed_triangle_edge(work_loops, global_vertices);
    let emitted = emit_filtered_triangles(
        tris,
        false,
        max_edge,
        face,
        global_vertices,
        global_normals,
        all_indices,
    );
    let _ = first_tri;
    FaceMeshRange {
        face_key,
        first_tri,
        tri_count: emitted,
        boundary_global,
        max_chord_error: 0.0,
    }
}

/// Planar face fill: sort boundary in plane frame, center fan (reuses welded edge verts).
fn project_point_to_plane(pt: Vec3, origin: Vec3, normal: Vec3) -> Vec3 {
    let n = normal.normalize();
    pt - n * (pt - origin).dot(n)
}

fn filter_plane_ring_coplanar_cluster(
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

fn triangulate_plane_center_fan(
    face: &BRepFace,
    loops: &FaceUvLoops,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
) -> Vec<(i32, i32, i32)> {
    let boundary: Vec<usize> = loops.outer.boundary.iter().map(|v| v.global_idx).collect();
    triangulate_plane_center_fan_from_boundary(
        face,
        &boundary,
        global_vertices,
        global_normals,
        pos_to_idx,
        plane_fan_max_edge(config),
    )
}

fn plane_angle_at(center: Vec3, pt: Vec3, u_axis: Vec3, v_axis: Vec3) -> f32 {
    let d = pt - center;
    f32::atan2(d.dot(v_axis), d.dot(u_axis))
}

fn plane_ring_circular_mean_angle(angles: &[f32]) -> f32 {
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

fn plane_unwrap_angle_near(a: f32, ref_angle: f32) -> f32 {
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

fn sort_plane_ring_by_angle(
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

fn plane_ring_signed_area(ring: &[(usize, Vec3)], u_axis: Vec3, v_axis: Vec3) -> f32 {
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

fn orient_plane_ring_ccw(ring: &mut Vec<(usize, Vec3)>, u_axis: Vec3, v_axis: Vec3) {
    if plane_ring_signed_area(ring, u_axis, v_axis) < 0.0 {
        ring.reverse();
    }
}

pub(crate) fn surface_is_revolution_like(surface: &SurfaceGeom) -> bool {
    match surface {
        SurfaceGeom::Revolution { .. } => true,
        SurfaceGeom::Offset { basis, .. } => {
            matches!(basis.as_ref(), SurfaceGeom::Revolution { .. })
        }
        _ => false,
    }
}

pub(crate) fn surface_uv_basis(surface: &SurfaceGeom) -> &SurfaceGeom {
    match surface {
        SurfaceGeom::Offset { basis, .. } => basis.as_ref(),
        other => other,
    }
}

pub(crate) fn wire_pair_is_same_edge_seam(wire_edges: &[(EdgeKey, Vec<usize>)]) -> bool {
    wire_edges.len() == 2 && wire_edges[0].0 == wire_edges[1].0
}

pub(crate) fn prefers_native_uv_ruled(
    surface: &SurfaceGeom,
    wire_edges: &[(EdgeKey, Vec<usize>)],
) -> bool {
    surface_is_revolution_like(surface) || wire_pair_is_same_edge_seam(wire_edges)
}

fn wire_native_v_span(samples: &[(f32, f32)]) -> (f32, f32) {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for &(_, v) in samples {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

fn nearest_periodic(value: f32, anchor: f32, period: f32) -> f32 {
    if period <= 0.0 {
        return value;
    }
    let k = ((anchor - value) / period).round();
    value + k * period
}

fn normalized_v_bounds_for_period(uv0: &[(f32, f32)], uv1: &[(f32, f32)], period: f32) -> Option<(f32, f32)> {
    if period <= 0.0 {
        return None;
    }
    let mut values: Vec<f32> = uv0.iter().chain(uv1.iter()).map(|&(_, v)| v).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let anchor = values[values.len() / 2];
    let mut lo = f32::INFINITY;
    let mut hi = f32::MIN;
    for &v in &values {
        let vn = nearest_periodic(v, anchor, period);
        lo = lo.min(vn);
        hi = hi.max(vn);
    }
    Some((lo, hi))
}

fn seam_face_v_bounds(
    face: &BRepFace,
    uv0: &[(f32, f32)],
    uv1: &[(f32, f32)],
    profile_sweep: bool,
) -> (f32, f32) {
    let pr = face.surface.param_range();
    if profile_sweep && surface_is_revolution_like(&face.surface) {
        return (pr.v_min, pr.v_max);
    }
    let (v0_lo, v0_hi) = wire_native_v_span(uv0);
    let (v1_lo, v1_hi) = wire_native_v_span(uv1);
    let mut v_lo = v0_lo.min(v1_lo);
    let mut v_hi = v0_hi.max(v1_hi);
    if surface_is_revolution_like(&face.surface) {
        if let Some(period) = surface_uv_basis(&face.surface).native_v_period() {
            if let Some((n_lo, n_hi)) = normalized_v_bounds_for_period(uv0, uv1, period) {
                v_lo = n_lo;
                v_hi = n_hi;
            }
            // Guardrail: avoid pathological multi-turn spans (e.g. 18*TAU) that explode grid cost.
            let max_span = period * 1.25;
            if v_hi - v_lo > max_span {
                v_lo = pr.v_min;
                v_hi = pr.v_max;
            }
        }
    }
    if v_hi - v_lo < 1e-5 {
        v_lo = pr.v_min;
        v_hi = pr.v_max;
    }
    (v_lo, v_hi)
}

/// Fan triangulation along the outer boundary wire order (3D topology, not UV).
fn fan_triangulate_outer(loops: &FaceUvLoops) -> Vec<(i32, i32, i32)> {
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

/// CDT-first triangulation with Steiner refinement, earcut as ultimate fallback.
#[allow(clippy::too_many_arguments)]
fn polyline_surface_uv(surface: &SurfaceGeom, indices: &[usize], verts: &[Vec3]) -> Vec<(f32, f32)> {
    let uv_surface = surface_uv_basis(surface);
    if matches!(uv_surface, SurfaceGeom::Revolution { .. }) && indices.len() >= 2 {
        let mut loop_data = UvLoop {
            boundary: indices
                .iter()
                .map(|&gi| UvVertex {
                    global_idx: gi,
                    uv: (0.0, 0.0),
                })
                .collect(),
        };
        assign_revolution_native_uv_along_wire(&mut loop_data, uv_surface, verts);
        return loop_data.boundary.iter().map(|v| v.uv).collect();
    }
    let tol = 1e-3;
    indices
        .iter()
        .filter_map(|&gi| verts.get(gi))
        .filter_map(|pt| {
            surface
                .project(*pt)
                .or_else(|| surface.inverse_native_uv(*pt, tol))
        })
        .collect()
}

/// Interpolate native U at a fixed revolution angle V along one wire's projected samples.
fn u_at_v_on_wire(samples: &[(f32, f32)], v: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    if samples.len() == 1 {
        return samples[0].0;
    }
    let mut sorted: Vec<(f32, f32)> = samples.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    if v <= sorted[0].1 {
        return sorted[0].0;
    }
    if v >= sorted[sorted.len() - 1].1 {
        return sorted[sorted.len() - 1].0;
    }
    for w in sorted.windows(2) {
        let (u0, v0) = w[0];
        let (u1, v1) = w[1];
        if v >= v0 && v <= v1 {
            let t = if (v1 - v0).abs() > 1e-12 {
                (v - v0) / (v1 - v0)
            } else {
                0.0
            };
            return u0 * (1.0 - t) + u1 * t;
        }
    }
    sorted[0].0
}

/// Interpolate native V at a fixed generatrix U along one wire's projected samples.
fn v_at_u_on_wire(samples: &[(f32, f32)], u: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    if samples.len() == 1 {
        return samples[0].1;
    }
    let mut sorted: Vec<(f32, f32)> = samples.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if u <= sorted[0].0 {
        return sorted[0].1;
    }
    if u >= sorted[sorted.len() - 1].0 {
        return sorted[sorted.len() - 1].1;
    }
    for w in sorted.windows(2) {
        let (u0, v0) = w[0];
        let (u1, v1) = w[1];
        if u >= u0 && u <= u1 {
            let t = if (u1 - u0).abs() > 1e-12 {
                (u - u0) / (u1 - u0)
            } else {
                0.0
            };
            return v0 * (1.0 - t) + v1 * t;
        }
    }
    sorted[0].1
}

fn wire_native_uv_spans(samples: &[(f32, f32)]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut u_min = f32::INFINITY;
    let mut u_max = f32::MIN;
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::MIN;
    for &(u, v) in samples {
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    (u_max - u_min, v_max - v_min)
}

/// True when wire samples run mainly along generatrix U with near-constant V (profile edge).
fn revolution_wire_is_generatrix_profile(samples: &[(f32, f32)]) -> bool {
    let (du, dv) = wire_native_uv_spans(samples);
    du > dv * 2.0 && du > 1e-4
}

fn uv_mindiff(u: f32, u0: f32, period: f32) -> f32 {
    (-4..=4)
        .map(|i| u + i as f32 * period)
        .min_by(|a, b| (a - u0).abs().partial_cmp(&(b - u0).abs()).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(u)
}

/// Revolution ruled strip: prefer PCurve UV on the face (OCC CurveOnSurface), then 3D fallback.
fn revolution_wire_uv_polyline(
    face_key: FaceKey,
    surface: &SurfaceGeom,
    ek: EdgeKey,
    pis: &[usize],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    global_vertices: &[Vec3],
    curve_gi: &[usize],
) -> Vec<(f32, f32)> {
    if let Some(poly) = edge_polygons.get(&ek) {
        if let Some(pcurve) = poly.params_2d.get(&face_key) {
            let mut uv = Vec::with_capacity(pis.len());
            for &pi in pis {
                if let Some(&(_, uvi)) = pcurve.get(pi) {
                    uv.push(uvi);
                }
            }
            if uv.len() >= 2 {
                let v_period = surface_uv_basis(surface).native_v_period();
                for i in 1..uv.len() {
                    let prev = uv[i - 1];
                    let mut cur = uv[i];
                    if let Some(pv) = v_period {
                        cur.1 = uv_mindiff(cur.1, prev.1, pv);
                    }
                    uv[i] = cur;
                }
                return uv;
            }
        }
    }
    polyline_surface_uv(surface, curve_gi, global_vertices)
}

fn revolution_ruled_grid_from_uv(
    face: &BRepFace,
    uv0: &[(f32, f32)],
    uv1: &[(f32, f32)],
    nu: usize,
    nv: usize,
) -> Option<Vec<Vec<(Vec3, Vec3)>>> {
    if uv0.len() < 2 || uv1.len() < 2 {
        return None;
    }
    let u_min = uv0
        .iter()
        .chain(uv1.iter())
        .map(|p| p.0)
        .fold(f32::INFINITY, f32::min);
    let u_max = uv0
        .iter()
        .chain(uv1.iter())
        .map(|p| p.0)
        .fold(f32::MIN, f32::max);
    let v_min = uv0
        .iter()
        .chain(uv1.iter())
        .map(|p| p.1)
        .fold(f32::INFINITY, f32::min);
    let v_max = uv0
        .iter()
        .chain(uv1.iter())
        .map(|p| p.1)
        .fold(f32::MIN, f32::max);
    let du = u_max - u_min;
    let dv = v_max - v_min;
    let profile_sweep = revolution_wire_is_generatrix_profile(&uv0)
        && revolution_wire_is_generatrix_profile(&uv1);
    if std::env::var("SHAPE_FACE_DIAG").is_ok() {
        let (du0, dv0) = wire_native_uv_spans(uv0);
        let (du1, dv1) = wire_native_uv_spans(uv1);
        eprintln!(
            "[rev ruled] du={:.4} dv={:.4} profile={} uv0 du/dv={:.4}/{:.4} uv1 du/dv={:.4}/{:.4}",
            du, dv, profile_sweep, du0, dv0, du1, dv1
        );
    }
    let mut grid = vec![vec![(Vec3::ZERO, Vec3::Y); nv + 1]; nu + 1];
    // Same-edge seam (F+R): wires share UV so dv=0. Fill native (u,v) rectangle, not wire-to-wire blend.
    if dv < 1e-4 && du > 1e-6 {
        let r0 = resample_uv_polyline(&uv0, nu);
        let (v_lo, v_hi) = seam_face_v_bounds(face, uv0, uv1, profile_sweep);
        if std::env::var("SHAPE_FACE_DIAG").is_ok() {
            eprintln!(
                "[rev ruled] seam structured u=[{:.4},{:.4}] v=[{:.4},{:.4}] profile={}",
                u_min, u_max, v_lo, v_hi, profile_sweep
            );
        }
        for i in 0..=nu {
            let u = r0[i].0;
            for j in 0..=nv {
                let v = v_lo + (v_hi - v_lo) * j as f32 / nv as f32;
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if n.length_squared() < 1e-12 {
                    n = Vec3::Y;
                }
                if !face.same_sense {
                    n = -n;
                }
                grid[i][j] = (pt, n);
            }
        }
        return Some(grid);
    }
    // Collapsed U: interpolate between the two wire polylines in UV.
    if du < 1e-6 {
        let r0 = resample_uv_polyline(&uv0, nu);
        let r1 = resample_uv_polyline(&uv1, nu);
        for i in 0..=nu {
            for j in 0..=nv {
                let s = j as f32 / nv as f32;
                let (u0, v0) = r0[i];
                let (u1, v1) = r1[i];
                let u = u0 * (1.0 - s) + u1 * s;
                let v = v0 * (1.0 - s) + v1 * s;
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if n.length_squared() < 1e-12 {
                    n = Vec3::Y;
                }
                if !face.same_sense {
                    n = -n;
                }
                grid[i][j] = (pt, n);
            }
        }
        return Some(grid);
    }
    if profile_sweep || dv <= du * 1.5 {
        // Two generatrix/profile wires: sweep V between wires at each native U.
        for i in 0..=nu {
            let u = u_min + du * i as f32 / nu as f32;
            let v0 = v_at_u_on_wire(&uv0, u);
            let v1 = v_at_u_on_wire(&uv1, u);
            for j in 0..=nv {
                let s = j as f32 / nv as f32;
                let v = v0 * (1.0 - s) + v1 * s;
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if !face.same_sense {
                    n = -n;
                }
                grid[i][j] = (pt, n);
            }
        }
    } else {
        // Two circular wires: sweep U between wires at each native V.
        for j in 0..=nv {
            let v = v_min + dv * j as f32 / nv as f32;
            let u0 = u_at_v_on_wire(&uv0, v);
            let u1 = u_at_v_on_wire(&uv1, v);
            for i in 0..=nu {
                let t = i as f32 / nu as f32;
                let u = u0 * (1.0 - t) + u1 * t;
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if !face.same_sense {
                    n = -n;
                }
                grid[i][j] = (pt, n);
            }
        }
    }
    Some(grid)
}

fn resample_uv_polyline(uvs: &[(f32, f32)], samples: usize) -> Vec<(f32, f32)> {
    if uvs.is_empty() || samples == 0 {
        return Vec::new();
    }
    if uvs.len() == 1 {
        return vec![uvs[0]; samples + 1];
    }
    let mut out = Vec::with_capacity(samples + 1);
    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let f = t * (uvs.len() - 1) as f32;
        let k = f.floor() as usize;
        let j = (k + 1).min(uvs.len() - 1);
        let u = f - k as f32;
        let (a, b) = (uvs[k], uvs[j]);
        out.push((a.0 * (1.0 - u) + b.0 * u, a.1 * (1.0 - u) + b.1 * u));
    }
    out
}

/// Ruled quad strip in surface (u,v): OCC-style ruled patch, not 3D linear blend.
pub fn mesh_ruled_two_wire_edges(
    face_key: FaceKey,
    face: &BRepFace,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &super::edge_pool::FaceEdgeBoundaryIdx,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    segs_u: u32,
    segs_v: u32,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let mut boundary_global = HashSet::new();
    if wire_edges.len() != 2 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global,
            max_chord_error: 0.0,
        };
    }

    let mut curves: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (side, &(ek, ref pis)) in wire_edges.iter().enumerate() {
        for &pi in pis {
            if let Some(gi) = edge_boundary_idx.get(&(face_key, ek, pi)).copied() {
                if curves[side].last() != Some(&gi) {
                    curves[side].push(gi);
                }
                boundary_global.insert(gi);
            }
        }
    }
    if curves[0].len() < 2 || curves[1].len() < 2 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global,
            max_chord_error: 0.0,
        };
    }

    let nu = segs_u.max(2) as usize;
    let nv = segs_v.max(2) as usize;

    let mut grid: Vec<Vec<usize>> = vec![vec![0; nv + 1]; nu + 1];
    let uv_surface = surface_uv_basis(&face.surface);
    let rev_grid = if surface_is_revolution_like(&face.surface)
        || wire_pair_is_same_edge_seam(wire_edges)
    {
        let uv0 = revolution_wire_uv_polyline(
            face_key,
            uv_surface,
            wire_edges[0].0,
            &wire_edges[0].1,
            edge_polygons,
            global_vertices,
            &curves[0],
        );
        let uv1 = revolution_wire_uv_polyline(
            face_key,
            uv_surface,
            wire_edges[1].0,
            &wire_edges[1].1,
            edge_polygons,
            global_vertices,
            &curves[1],
        );
        revolution_ruled_grid_from_uv(face, &uv0, &uv1, nu, nv)
    } else {
        None
    };

    if let Some(ref rg) = rev_grid {
        for (i, row) in rg.iter().enumerate() {
            for (j, &(pt, n)) in row.iter().enumerate() {
                let idx = global_vertices.len();
                global_vertices.push(pt);
                global_normals.push(n);
                grid[i][j] = idx;
            }
        }
    } else {
        let uv_surface = surface_uv_basis(&face.surface);
        let uv0 = polyline_surface_uv(uv_surface, &curves[0], global_vertices);
        let uv1 = polyline_surface_uv(uv_surface, &curves[1], global_vertices);
        let use_native_uv = uv0.len() >= 2 && uv1.len() >= 2;
        let resampled = if use_native_uv {
            Some((
                resample_uv_polyline(&uv0, nu),
                resample_uv_polyline(&uv1, nu),
            ))
        } else {
            None
        };
        for i in 0..=nu {
            for j in 0..=nv {
                let s = j as f32 / nv as f32;
                let (pt, n) = if let Some((ref r0, ref r1)) = resampled {
                    let (u0, v0) = r0[i];
                    let (u1, v1) = r1[i];
                    let u = u0 * (1.0 - s) + u1 * s;
                    let v = v0 * (1.0 - s) + v1 * s;
                    let pt = face.surface.d0_native(u, v);
                    let mut n = face.surface.normal_native(u, v);
                    if !face.same_sense {
                        n = -n;
                    }
                    (pt, n)
                } else {
                    let t = i as f32 / nu as f32;
                    let p0 = sample_polyline(&curves[0], global_vertices, t);
                    let p1 = sample_polyline(&curves[1], global_vertices, t);
                    let pt = p0 * (1.0 - s) + p1 * s;
                    (pt, Vec3::Z)
                };
                let gi = if use_native_uv {
                    let idx = global_vertices.len();
                    global_vertices.push(pt);
                    global_normals.push(n);
                    idx
                } else {
                    let hash = rc3d_core::utils::hash::f32x3_quantized_bits([pt.x, pt.y, pt.z]);
                    *pos_to_idx.entry(hash).or_insert_with(|| {
                        let idx = global_vertices.len();
                        global_vertices.push(pt);
                        global_normals.push(n);
                        idx
                    })
                };
                grid[i][j] = gi;
            }
        }
    }

    for i in 0..nu {
        for j in 0..nv {
            let i00 = grid[i][j] as i32;
            let i10 = grid[i + 1][j] as i32;
            let i11 = grid[i + 1][j + 1] as i32;
            let i01 = grid[i][j + 1] as i32;
            for (mut a, mut b, mut c) in [(i00, i10, i11), (i00, i11, i01)] {
                if a == b || b == c || c == a {
                    continue;
                }
                fix_tri_winding(
                    &mut a,
                    &mut b,
                    &mut c,
                    global_vertices,
                    &face.surface,
                    face.same_sense,
                );
                all_indices.extend_from_slice(&[a, b, c, -1]);
                accumulate_normals(a, b, c, global_vertices, global_normals);
            }
        }
    }

    let tri_count = all_indices.len() / 4 - first_tri;
    FaceMeshRange {
        face_key,
        first_tri,
        tri_count,
        boundary_global,
        max_chord_error: 0.0,
    }
}

/// Ruled strip using raw edge polygon 3D samples (avoids boundary pool dedup collapsing wires).
pub fn mesh_ruled_wire_polygons_3d(
    face_key: FaceKey,
    face: &BRepFace,
    wire_edges: &[(EdgeKey, crate::topo::Orientation, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    segs_u: u32,
    segs_v: u32,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    if wire_edges.len() != 2 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
        };
    }
    let mut curves: [Vec<Vec3>; 2] = [Vec::new(), Vec::new()];
    for (side, &(ek, _orient, ref pis)) in wire_edges.iter().enumerate() {
        let Some(poly) = edge_polygons.get(&ek) else {
            return FaceMeshRange {
                face_key,
                first_tri,
                tri_count: 0,
                boundary_global: HashSet::new(),
                max_chord_error: 0.0,
            };
        };
        for &pi in pis {
            let Some(&(_, pt)) = poly.params_3d.get(pi) else {
                continue;
            };
            if curves[side]
                .last()
                .map(|p| (*p - pt).length_squared() > 1e-14)
                .unwrap_or(true)
            {
                curves[side].push(pt);
            }
        }
    }
    if curves[0].len() < 2 || curves[1].len() < 2 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
        };
    }
    let sample_curve = |c: &[Vec3], t: f32| -> Vec3 {
        if c.len() == 1 {
            return c[0];
        }
        let f = t * (c.len() - 1) as f32;
        let i = f.floor() as usize;
        let j = (i + 1).min(c.len() - 1);
        let u = f - i as f32;
        c[i].lerp(c[j], u)
    };
    // If the two curves are nearly coincident, the ruled strip has zero width -> all degenerate.
    let n_samples = curves[0].len().max(curves[1].len()).max(2);
    let max_separation = (0..n_samples)
        .map(|i| {
            let t = i as f32 / (n_samples - 1) as f32;
            let a = sample_curve(&curves[0], t);
            let b = sample_curve(&curves[1], t);
            (a - b).length_squared()
        })
        .fold(0.0f32, f32::max);
    if max_separation < 1e-6 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
        };
    }
    let nu = segs_u.max(2) as usize;
    let nv = segs_v.max(2) as usize;
    let mut grid: Vec<Vec<usize>> = vec![vec![0; nv + 1]; nu + 1];
    for i in 0..=nu {
        let t = i as f32 / nu as f32;
        let p0 = sample_curve(&curves[0], t);
        let p1 = sample_curve(&curves[1], t);
        for j in 0..=nv {
            let s = j as f32 / nv as f32;
            let pm = p0 * (1.0 - s) + p1 * s;
            let (pt, mut n) = match &face.surface {
                SurfaceGeom::Revolution { .. } => {
                    let n = face
                        .surface
                        .revolution_native_uv_at(pm)
                        .map(|(u, v)| face.surface.normal_native(u, v))
                        .unwrap_or_else(|| {
                            let e0 = p1 - p0;
                            let e1 = pm - p0;
                            e0.cross(e1)
                        });
                    (pm, n)
                }
                SurfaceGeom::Offset { .. } | SurfaceGeom::BSpline(_) => {
                    if let Some(uv) = face.surface.project(pm) {
                        let pt = face.surface.d0_native(uv.0, uv.1);
                        let n = face.surface.normal_native(uv.0, uv.1);
                        (pt, n)
                    } else {
                        let e0 = p1 - p0;
                        let e1 = pm - p0;
                        (pm, e0.cross(e1))
                    }
                }
                _ => {
                    let e0 = p1 - p0;
                    let e1 = pm - p0;
                    (pm, e0.cross(e1))
                }
            };
            if n.length_squared() < 1e-12 {
                n = Vec3::Y;
            } else {
                n = n.normalize();
            }
            if !face.same_sense {
                n = -n;
            }
            let idx = global_vertices.len();
            global_vertices.push(pt);
            if global_normals.len() < global_vertices.len() {
                global_normals.resize(global_vertices.len(), n);
            }
            global_normals[idx] = n;
            grid[i][j] = idx;
        }
    }
    for i in 0..nu {
        for j in 0..nv {
            let i00 = grid[i][j] as i32;
            let i10 = grid[i + 1][j] as i32;
            let i11 = grid[i + 1][j + 1] as i32;
            let i01 = grid[i][j + 1] as i32;
            for (mut a, mut b, mut c) in [(i00, i10, i11), (i00, i11, i01)] {
                if a == b || b == c || c == a {
                    continue;
                }
                fix_tri_winding(
                    &mut a,
                    &mut b,
                    &mut c,
                    global_vertices,
                    &face.surface,
                    face.same_sense,
                );
                all_indices.extend_from_slice(&[a, b, c, -1]);
                accumulate_normals(a, b, c, global_vertices, global_normals);
            }
        }
    }
    let tri_count = all_indices.len() / 4 - first_tri;
    let mut range = FaceMeshRange {
        face_key,
        first_tri,
        tri_count,
        boundary_global: HashSet::new(),
        max_chord_error: 0.0,
    };
    if tri_count > 0 {
        range.max_chord_error =
            measure_face_chord_error(face, global_vertices, all_indices, &range);
    }
    range
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
    }
}

/// Fill a revolution face in native (u,v) on the analytic surface (vertices stay on-surface).
/// Prefer `mesh_trimmed_uv_grid` + CDT; kept for tests and future closed patches.
#[allow(clippy::too_many_arguments)]
pub fn mesh_revolution_native_grid(
    face_key: FaceKey,
    face: &BRepFace,
    uv_bounds: (f32, f32, f32, f32),
    segs_u: u32,
    segs_v: u32,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let (u_min, u_max, v_min, v_max) = uv_bounds;
    if (u_max - u_min).abs() < 1e-6 || (v_max - v_min).abs() < 1e-5 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
        };
    }
    // Avoid ring-only mesh when native U is degenerate but V spans (use ruled path instead).
    if (u_max - u_min).abs() < 1e-4 && (v_max - v_min).abs() > 1e-3 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
        };
    }
    let nu = segs_u.max(2) as usize;
    let nv = segs_v.max(2) as usize;
    let mut grid: Vec<Vec<usize>> = vec![vec![0; nv + 1]; nu + 1];

    for i in 0..=nu {
        let u = u_min + (u_max - u_min) * i as f32 / nu as f32;
        for j in 0..=nv {
            let v = v_min + (v_max - v_min) * j as f32 / nv as f32;
            let pt = face.surface.d0_native(u, v);
            let mut n = face.surface.normal_native(u, v);
            if !face.same_sense {
                n = -n;
            }
            let gi = global_vertices.len();
            global_vertices.push(pt);
            global_normals.push(n);
            grid[i][j] = gi;
        }
    }

    for i in 0..nu {
        for j in 0..nv {
            let i00 = grid[i][j] as i32;
            let i10 = grid[i + 1][j] as i32;
            let i11 = grid[i + 1][j + 1] as i32;
            let i01 = grid[i][j + 1] as i32;
            for (mut a, mut b, mut c) in [(i00, i10, i11), (i00, i11, i01)] {
                if a == b || b == c || c == a {
                    continue;
                }
                fix_tri_winding(
                    &mut a,
                    &mut b,
                    &mut c,
                    global_vertices,
                    &face.surface,
                    face.same_sense,
                );
                all_indices.extend_from_slice(&[a, b, c, -1]);
                accumulate_normals(a, b, c, global_vertices, global_normals);
            }
        }
    }

    FaceMeshRange {
        face_key,
        first_tri,
        tri_count: all_indices.len() / 4 - first_tri,
        boundary_global: HashSet::new(),
        max_chord_error: 0.0,
    }
}

fn sample_polyline(indices: &[usize], verts: &[Vec3], t: f32) -> Vec3 {
    if indices.is_empty() {
        return Vec3::ZERO;
    }
    if indices.len() == 1 {
        return verts[indices[0]];
    }
    let f = t * (indices.len() - 1) as f32;
    let i = f.floor() as usize;
    let j = (i + 1).min(indices.len() - 1);
    let u = f - i as f32;
    verts[indices[i]].lerp(verts[indices[j]], u)
}

/// Fan triangulation in 3D when plane UV trim loops collapse (duplicate/coincident UV).
pub fn mesh_plane_fan_3d(
    face_key: FaceKey,
    boundary_ordered: &[usize],
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
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
    };
    if emitted > 0 {
        range.max_chord_error =
            measure_face_chord_error(face, global_vertices, all_indices, &range);
    }
    range
}

fn plane_fan_max_edge(config: &FaceFillConfig) -> f32 {
    let def_floor = (config.deflection_interior * 2.0).max(1e-4);
    let mut max_edge = config.min_size.max(def_floor);
    if config.shell_min_size > 0.0 {
        max_edge = max_edge.max(config.shell_min_size);
    }
    max_edge
}

fn triangulate_plane_center_fan_from_boundary(
    face: &BRepFace,
    boundary: &[usize],
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    max_edge: f32,
) -> Vec<(i32, i32, i32)> {
    let SurfaceGeom::Plane {
        origin,
        normal,
        u_dir,
    } = &face.surface
    else {
        return Vec::new();
    };
    let n = (*normal).normalize();
    let (u_axis, v_axis) = plane_tangent_basis(n, *u_dir);

    let mut ring: Vec<(usize, Vec3)> = Vec::new();
    for &gi in boundary {
        let Some(&pt) = global_vertices.get(gi) else {
            continue;
        };
        if ring
            .last()
            .map(|(_, p)| (*p - pt).length_squared() < 1e-12)
            .unwrap_or(false)
        {
            continue;
        }
        ring.push((gi, pt));
    }
    if ring.len() >= 2 && (ring[0].1 - ring[ring.len() - 1].1).length_squared() < 1e-12 {
        ring.pop();
    }
    if ring.len() < 3 {
        return Vec::new();
    }
    filter_plane_ring_coplanar_cluster(&mut ring, *origin, n);
    if ring.len() < 3 {
        return Vec::new();
    }

    let centroid = ring.iter().map(|(_, p)| *p).sum::<Vec3>() / ring.len() as f32;
    let centroid = project_point_to_plane(centroid, *origin, n);
    let mut radii: Vec<f32> = ring
        .iter()
        .map(|(_, p)| (*p - centroid).length())
        .collect();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Use the full radius range with padding to filter only extreme outliers.
    // The previous median ± 20% filter was too aggressive for non-circular
    // faces (e.g. squares: edge midpoints are closer to center than corners).
    let min_r = radii.first().copied().unwrap_or(0.0);
    let max_r = radii.last().copied().unwrap_or(0.0);
    let range = (max_r - min_r).max(min_r * 0.5).max(1e-3);
    let tol = range * 0.5;
    ring.retain(|(_, p)| {
        let r = (*p - centroid).length();
        r >= min_r - tol && r <= max_r + tol
    });
    if ring.len() < 3 {
        return Vec::new();
    }

    let center = ring.iter().map(|(_, p)| *p).sum::<Vec3>() / ring.len() as f32;
    let center = project_point_to_plane(center, *origin, n);
    sort_plane_ring_by_angle(&mut ring, center, u_axis, v_axis);

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

    for (gi, _pt) in &ring {
        if *gi < global_vertices.len() {
            // Do NOT overwrite shared vertex positions — boundary vertices are
            // shared across adjacent faces and modifying them breaks watertightness.
            if global_normals.len() <= *gi {
                global_normals.resize(*gi + 1, normal_vec);
            }
            // Accumulate face normal for smooth shading at shared vertices.
            let prev = global_normals[*gi];
            let blended = (prev + normal_vec).normalize();
            if blended.length_squared() > 0.5 {
                global_normals[*gi] = blended;
            } else {
                global_normals[*gi] = normal_vec;
            }
        }
    }

    let center_hash = rc3d_core::utils::hash::f32x3_quantized_bits([center.x, center.y, center.z]);
    let center_gi = *pos_to_idx.entry(center_hash).or_insert_with(|| {
        let i = global_vertices.len();
        global_vertices.push(center);
        if global_normals.len() < global_vertices.len() {
            global_normals.resize(global_vertices.len(), normal_vec);
        }
        global_normals[i] = normal_vec;
        i
    });
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
    let n_rings = ((max_boundary_edge / max_edge.max(1e-6)).ceil() as u32).clamp(1, 32) as usize;

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
            let hash = rc3d_core::utils::hash::f32x3_quantized_bits([pt.x, pt.y, pt.z]);
            let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                let idx = global_vertices.len();
                global_vertices.push(pt);
                if global_normals.len() < global_vertices.len() {
                    global_normals.resize(global_vertices.len(), normal_vec);
                }
                global_normals[idx] = normal_vec;
                idx
            });
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

pub fn fill_trimmed(
    face_key: FaceKey,
    loops: &FaceUvLoops,
    face: &BRepFace,
    reg: &BRepStore,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    wire_edge_count: usize,
    config: &FaceFillConfig,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let mut boundary_global = HashSet::new();
    for v in &loops.outer.boundary {
        boundary_global.insert(v.global_idx);
    }
    for inner in &loops.inners {
        for v in &inner.boundary {
            boundary_global.insert(v.global_idx);
        }
    }

    let is_plane = matches!(face.surface, SurfaceGeom::Plane { .. });
    let is_revolution = matches!(face.surface, SurfaceGeom::Revolution { .. });
    let prefer_closed_cdt = matches!(
        &face.surface,
        SurfaceGeom::Revolution { .. }
            | SurfaceGeom::BSpline(_)
            | SurfaceGeom::Offset { .. }
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. }
    );
    let revolution_uv_ok = is_revolution
        && loops
            .native_uv_bounds()
            .map(|(u0, u1, _, _)| !revolution_u_span_collapsed(u1 - u0))
            .unwrap_or(false);
    let mut work_loops = if is_plane {
        loops.clone()
    } else if is_revolution {
        let native = loops_native_surface_uv(loops, face, global_vertices);
        if !uv_loop_is_degenerate(&native) && !revolution_boundary_v_collapsed(&native) {
            native
        } else if !uv_loop_is_degenerate(loops) {
            loops.clone()
        } else {
            native
        }
    } else if matches!(face.surface, SurfaceGeom::BSpline(_))
        && (uv_loop_is_degenerate(loops) || loops.uv_source != UvSource::Pcurve)
    {
        let native = loops_native_surface_uv(loops, face, global_vertices);
        if !uv_loop_is_degenerate(&native) {
            native
        } else {
            loops.clone()
        }
    } else if matches!(face.surface, SurfaceGeom::Cylinder { .. })
        && cylinder_loop_needs_uv_rebuild(loops, &face.surface)
    {
        let native = repair_cylinder_uv_loops(loops, face, global_vertices);
        if !uv_loop_is_degenerate(&native) {
            native
        } else {
            loops.clone()
        }
    } else if revolution_uv_ok {
        loops.clone()
    } else {
        loops_native_surface_uv(loops, face, global_vertices)
    };

    // Always rebuild plane loops from 3D boundary (STEP PCURVE UV can be skewed, e.g. Cube.step).
    if is_plane {
        rebuild_loop_uv_local_frame(
            &mut work_loops.outer,
            global_vertices,
            Some(&face.surface),
        );
        for inner in &mut work_loops.inners {
            rebuild_loop_uv_local_frame(inner, global_vertices, Some(&face.surface));
        }
        ensure_loop_orientation(&mut work_loops.outer.boundary, false);
        for inner in &mut work_loops.inners {
            ensure_loop_orientation(&mut inner.boundary, true);
        }
        work_loops.uv_source = UvSource::Synthetic;

        // Planar faces: 3D sorted center fan (wire-order/UV earcut leaves holes on Cube.step).
        if work_loops.inners.is_empty() {
            let tris = triangulate_plane_center_fan(
                face,
                &work_loops,
                global_vertices,
                global_normals,
                pos_to_idx,
                config,
            );
            if !tris.is_empty() {
                let emitted = emit_filtered_triangles(
                    tris,
                    false,
                    f32::MAX,
                    face,
                    global_vertices,
                    global_normals,
                    all_indices,
                );
                return FaceMeshRange {
                    face_key,
                    first_tri,
                    tri_count: emitted,
                    boundary_global,
                    max_chord_error: 0.0,
                };
            }
        }
    }

    let mut fill_cfg = config.clone();
    if is_revolution && wire_edge_count >= 3 {
        fill_cfg.deflection_interior *= 0.25;
    }

    let (chains, long_count) = split_boundary_chains_at_3d_jumps(
        &work_loops.outer.boundary,
        global_vertices,
        8.0,
    );
    let use_segmentation =
        !is_plane && !prefer_closed_cdt && chains.len() >= 2 && long_count >= 2;

    let (mut tris, mut max_chord_error) = if use_segmentation {
        let mut seg_tris = Vec::new();
        for chain in &chains {
            seg_tris.extend(triangulate_open_chain_uv(chain, global_vertices));
        }
        (seg_tris, 0.0f32)
    } else {
        try_cdt_or_earcut(
            &work_loops,
            face,
            face_key,
            reg,
            global_vertices,
            global_normals,
            pos_to_idx,
            &fill_cfg,
        )
    };

    if tris.is_empty() {
        tris = triangulate_loops_earcut_fallback(&work_loops);
    }

    if tris.is_empty() && use_segmentation {
        let (cdt_tris, chord) = try_cdt_or_earcut(
            &work_loops,
            face,
            face_key,
            reg,
            global_vertices,
            global_normals,
            pos_to_idx,
            &fill_cfg,
        );
        tris = cdt_tris;
        max_chord_error = chord;
    }

    if tris.is_empty() && is_plane {
        tris = fan_triangulate_outer(&work_loops);
    }

    if tris.is_empty() {
        log::debug!("[BRep mesh] face fill: triangulation failed, skipping");
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global,
            max_chord_error: 0.0,
        };
    }

    let apply_quality = !is_plane && !use_segmentation && !prefer_closed_cdt;
    let max_edge = if apply_quality {
        max_allowed_triangle_edge(&work_loops, global_vertices)
    } else {
        f32::MAX
    };

    let emitted = emit_filtered_triangles(
        tris,
        apply_quality,
        max_edge,
        face,
        global_vertices,
        global_normals,
        all_indices,
    );

    FaceMeshRange {
        face_key,
        first_tri,
        tri_count: emitted,
        boundary_global,
        max_chord_error,
    }
}

/// Surface-aware fill: project boundary to surface, CDT + Steiner refinement.
/// OCC BRepMesh_Face surface-only degraded path (when PCURVEs unavailable).
///
/// Two strategies depending on boundary->surface projection success rate:
/// - >70%: CDT in projected UV space with insert-time Steiner refinement.
/// - ≤70%: fit a plane for 2D coords, CDT in plane space, Steiner points
///   snapped to surface via `surface.project()` + `surface.d0_native()`.
///
/// All interior points are evaluated via `surface.d0_native()` — none remain
/// on a fitted plane.
#[allow(clippy::too_many_arguments)]
pub fn surface_fill_3d(
    face_key: FaceKey,
    boundary_global: &[usize],
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let boundary_set: HashSet<usize> = boundary_global.iter().copied().collect();

    if boundary_global.len() < 3 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: boundary_set,
            max_chord_error: 0.0,
        };
    }

    log::debug!(
        "[BRep mesh] face {:?}: surface fill 3D ({} boundary verts)",
        face_key,
        boundary_global.len()
    );

    let pseudo_loops = super::face_uv::FaceUvLoops {
        outer: super::face_uv::UvLoop {
            boundary: boundary_global
                .iter()
                .map(|&gi| super::face_uv::UvVertex {
                    global_idx: gi,
                    uv: (0.0, 0.0),
                })
                .collect(),
        },
        inners: vec![],
        uv_source: super::face_uv::UvSource::SurfaceFill,
    };
    if face_boundary_is_mixed(&pseudo_loops, global_vertices) {
        log::debug!(
            "[BRep mesh] face {:?}: mixed boundary, segmented chain fill",
            face_key
        );
        let work_loops = loops_native_surface_uv(&pseudo_loops, face, global_vertices);
        return fill_mixed_boundary_segmented(
            face_key,
            &work_loops,
            face,
            global_vertices,
            global_normals,
            all_indices,
            first_tri,
            boundary_set,
        );
    }

    // ── Phase 1: project boundary points to surface ──
    let inv_tol = face.tolerance.max(1e-3);
    let mut projected: Vec<Option<(f32, f32)>> = Vec::with_capacity(boundary_global.len());
    let mut success_count = 0usize;
    for &gi in boundary_global {
        let pt = global_vertices[gi];
        let uv = if matches!(face.surface, SurfaceGeom::Revolution { .. }) {
            face.surface
                .revolution_native_uv_at(pt)
                .or_else(|| face.surface.project(pt))
                .or_else(|| face.surface.inverse_native_uv(pt, inv_tol))
        } else {
            face.surface
                .project(pt)
                .or_else(|| face.surface.inverse_native_uv(pt, inv_tol))
        };
        if uv.is_some() {
            success_count += 1;
        }
        projected.push(uv);
    }
    let success_ratio = success_count as f32 / boundary_global.len() as f32;

    log::debug!(
        "[BRep mesh] face {:?}: {}/{} boundary pts projected ({:.0}%)",
        face_key, success_count, boundary_global.len(), success_ratio * 100.0
    );

    // ── Phase 2a: >70% → UV-based CDT + Steiner (reuse triangulate_uv_cdt_with_steiner) ──
    if success_ratio > 0.7 {
        let boundary_uvs: Vec<super::face_uv::UvVertex> = boundary_global
            .iter()
            .zip(projected.iter())
            .filter_map(|(&gi, uv)| uv.map(|uv| super::face_uv::UvVertex { global_idx: gi, uv }))
            .collect();

        if boundary_uvs.len() >= 3 {
            let mut loops = super::face_uv::FaceUvLoops {
                outer: super::face_uv::UvLoop {
                    boundary: boundary_uvs,
                },
                inners: vec![],
                uv_source: super::face_uv::UvSource::SurfaceFill,
            };
            unwrap_periodic_uv_loops(&mut loops, &face.surface);
            if matches!(face.surface, SurfaceGeom::Revolution { .. }) {
                assign_revolution_native_uv_along_wire(&mut loops.outer, &face.surface, global_vertices);
            }

            log::debug!(
                "[BRep mesh] face {:?}: UV CDT + Steiner ({} boundary UVs)",
                face_key,
                loops.outer.boundary.len()
            );

            if uv_loop_is_degenerate(&loops) {
                log::debug!(
                    "[BRep mesh] face {:?}: native UV loop degenerate after rebuild, planar fallback",
                    face_key
                );
            } else {
                let (tris_flat, max_chord_error) = triangulate_uv_cdt_with_steiner(
                    &loops, face, Some(face_key), global_vertices, global_normals, pos_to_idx, config, None,
                );

                if !tris_flat.is_empty() {
                    let mut tris: Vec<(i32, i32, i32)> = Vec::new();
                    for chunk in tris_flat.chunks(3) {
                        if chunk.len() == 3 {
                            tris.push((chunk[0] as i32, chunk[1] as i32, chunk[2] as i32));
                        }
                    }
                    for (mut i0, mut i1, mut i2) in tris {
                        if i0 == i1 || i1 == i2 || i2 == i0 {
                            continue;
                        }
                        fix_tri_winding(
                            &mut i0, &mut i1, &mut i2,
                            global_vertices,
                            &face.surface,
                            face.same_sense,
                        );
                        all_indices.extend_from_slice(&[i0, i1, i2, -1]);
                        accumulate_normals(i0, i1, i2, global_vertices, global_normals);
                    }
                    let tri_count = all_indices.len() / 4 - first_tri;
                    return FaceMeshRange {
                        face_key,
                        first_tri,
                        tri_count,
                        boundary_global: boundary_set,
                        max_chord_error,
                    };
                }
                log::debug!(
                    "[BRep mesh] face {:?}: UV CDT returned no triangles, falling back to planar",
                    face_key
                );
            }
        }
    }

    // ── Phase 2b: planar parameterization + surface-projected Steiner ──
    surface_fill_3d_planar(
        face_key,
        boundary_global,
        face,
        global_vertices,
        global_normals,
        all_indices,
        pos_to_idx,
        config,
        first_tri,
        boundary_set,
    )
}

/// Planar fallback: fit a plane to boundary, CDT in 2D plane coords,
/// Steiner points snapped to surface via `surface.project()` + `surface.d0_native()`.
fn surface_fill_3d_planar(
    face_key: FaceKey,
    boundary_global: &[usize],
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
    first_tri: usize,
    boundary_set: HashSet<usize>,
) -> FaceMeshRange {
    use rc3d_core::utils::hash::f32x3_quantized_bits;

    use super::delaunay2d::{CdtVertHandle, NativeCdt};

    log::debug!(
        "[BRep mesh] face {:?}: planar CDT + surface-projected Steiner ({} boundary verts)",
        face_key,
        boundary_global.len()
    );

    // Fit plane
    let points: Vec<Vec3> = boundary_global
        .iter()
        .map(|&gi| global_vertices[gi])
        .collect();

    let mut normal = Vec3::ZERO;
    for i in 0..points.len() {
        let p0 = points[i];
        let p1 = points[(i + 1) % points.len()];
        normal.x += (p0.y - p1.y) * (p0.z + p1.z);
        normal.y += (p0.z - p1.z) * (p0.x + p1.x);
        normal.z += (p0.x - p1.x) * (p0.y + p1.y);
    }
    if normal.length_squared() < 1e-20 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: boundary_set,
            max_chord_error: 0.0,
        };
    }
    normal = normal.normalize();

    let u_axis = if normal.dot(Vec3::Z).abs() < 0.9 {
        normal.cross(Vec3::Z).normalize()
    } else {
        normal.cross(Vec3::Y).normalize()
    };
    let v_axis = normal.cross(u_axis).normalize();
    let origin = points[0];

    // 2D coords on plane
    let plane_uvs: Vec<(f64, f64)> = points
        .iter()
        .map(|p| {
            let rel = *p - origin;
            (rel.dot(u_axis) as f64, rel.dot(v_axis) as f64)
        })
        .collect();

    let mut u_min = f64::MAX;
    let mut u_max = f64::MIN;
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for &(pu, pv) in &plane_uvs {
        u_min = u_min.min(pu);
        u_max = u_max.max(pu);
        v_min = v_min.min(pv);
        v_max = v_max.max(pv);
    }
    let mut cdt = NativeCdt::from_uv_bbox(
        u_min as f32,
        v_min as f32,
        u_max as f32,
        v_max as f32,
    );
    let mut handles: Vec<CdtVertHandle> = Vec::new();

    for (i, &gi) in boundary_global.iter().enumerate() {
        let (pu, pv) = plane_uvs[i];
        let Some(h) = cdt.insert(pu, pv, gi) else {
            return FaceMeshRange {
                face_key,
                first_tri,
                tri_count: 0,
                boundary_global: boundary_set,
                max_chord_error: 0.0,
            };
        };
        handles.push(h);
    }

    let n = boundary_global.len();
    for i in 0..n {
        let a = handles[i];
        let b = handles[(i + 1) % n];
        let _ = cdt.try_add_constraint(a, b);
    }

    // Helper: compute 3D point on fitted plane from 2D coords
    let plane_to_3d = |pu: f64, pv: f64| -> Vec3 {
        origin + u_axis * (pu as f32) + v_axis * (pv as f32)
    };

    // ── Steiner refinement (surface-projected) ──
    let mut max_chord = 0.0f32;
    if config.enable_interior && config.deflection_interior > 0.0 {
        let min_sz = effective_min_size(config);
        for _iter in 0..config.max_adapt_iterations {
            let mut splits: Vec<(f64, f64)> = Vec::new();
            for (gids, uvs) in cdt.inner_faces_detail() {
                let uv0 = (uvs[0].0 as f64, uvs[0].1 as f64);
                let uv1 = (uvs[1].0 as f64, uvs[1].1 as f64);
                let uv2 = (uvs[2].0 as f64, uvs[2].1 as f64);
                let p0 = global_vertices[gids[0]];
                let p1 = global_vertices[gids[1]];
                let p2 = global_vertices[gids[2]];

                let mut tri_split = false;
                for (a, b) in [(&p0, &p1), (&p1, &p2), (&p2, &p0)] {
                    let edge_len = (*a - *b).length();
                    let mid_3d = (*a + *b) * 0.5;
                    if let Some((su, sv)) = face.surface.project(mid_3d) {
                        let on_surf = face.surface.d0_native(su, sv);
                        let dev = (mid_3d - on_surf).length();
                        max_chord = max_chord.max(dev);
                        if dev > config.deflection_interior && edge_len > min_sz {
                            tri_split = true;
                        }
                    }
                    // Angular deflection (project endpoints to surface for normals)
                    if let (Some((su_a, sv_a)), Some((su_b, sv_b))) =
                        (face.surface.project(*a), face.surface.project(*b))
                    {
                        let na = face.surface.normal_native(su_a, sv_a);
                        let nb = face.surface.normal_native(su_b, sv_b);
                        let angle = na.dot(nb).max(-1.0).min(1.0).acos();
                        if angle > config.angular_deflection && edge_len > min_sz {
                            tri_split = true;
                        }
                    }
                }
                if tri_split {
                    // Insert Steiner at triangle centroid in plane 2D space
                    let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
                    let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
                    splits.push((cu, cv));
                }
            }
            if splits.is_empty() {
                break;
            }
            // Dedup and insert Steiner points
            let mut dedup = HashSet::new();
            for (pu, pv) in splits {
                let key = ((pu * 1e4) as u64, (pv * 1e4) as u64);
                if dedup.contains(&key) {
                    continue;
                }
                dedup.insert(key);

                // Snap plane 3D point to surface
                let mid_3d_plane = plane_to_3d(pu, pv);
                let Some((su, sv)) = face.surface.project(mid_3d_plane) else {
                    continue;
                };
                let pt_3d = face.surface.d0_native(su, sv);
                let hash = f32x3_quantized_bits([pt_3d.x, pt_3d.y, pt_3d.z]);
                let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                    let i = global_vertices.len();
                    global_vertices.push(pt_3d);
                    let mut n = face.surface.normal_native(su, sv);
                    if !face.same_sense {
                        n = -n;
                    }
                    global_normals.push(n);
                    i
                });

                if cdt.insert(pu, pv, gi).is_some() {
                    // Steiner vertex registered in NativeCdt global-index map.
                }
            }
        }
    }

    cdt.finalize();

    let mut tris: Vec<(i32, i32, i32)> = Vec::new();
    for (gids, _uvs) in cdt.inner_faces_detail() {
        tris.push((gids[0] as i32, gids[1] as i32, gids[2] as i32));
    }

    if tris.is_empty() && boundary_global.len() >= 3 {
        let g0 = boundary_global[0] as i32;
        for i in 1..boundary_global.len().saturating_sub(1) {
            let g1 = boundary_global[i] as i32;
            let g2 = boundary_global[i + 1] as i32;
            if g0 != g1 && g1 != g2 && g2 != g0 {
                tris.push((g0, g1, g2));
            }
        }
    }

    if tris.is_empty() {
        log::debug!(
            "[BRep mesh] face {:?}: planar CDT produced no triangles",
            face_key
        );
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: boundary_set,
            max_chord_error: 0.0,
        };
    }

    let pseudo_loops = FaceUvLoops {
        outer: super::face_uv::UvLoop {
            boundary: boundary_global
                .iter()
                .map(|&gi| super::face_uv::UvVertex {
                    global_idx: gi,
                    uv: (0.0, 0.0),
                })
                .collect(),
        },
        inners: vec![],
        uv_source: super::face_uv::UvSource::SurfaceFill,
    };
    let max_edge = max_allowed_triangle_edge(&pseudo_loops, global_vertices);
    let tri_count = emit_filtered_triangles(
        tris,
        true,
        max_edge,
        face,
        global_vertices,
        global_normals,
        all_indices,
    );

    FaceMeshRange {
        face_key,
        first_tri,
        tri_count,
        boundary_global: boundary_set,
        max_chord_error: max_chord,
    }
}

pub fn measure_face_chord_error(
    face: &BRepFace,
    global_vertices: &[Vec3],
    all_indices: &[i32],
    range: &FaceMeshRange,
) -> f32 {
    let inv_tol = face.tolerance.max(1e-3);
    let start = range.first_tri * 4;
    let end = start + range.tri_count * 4;
    let mut max_chord = 0.0f32;
    let slice = all_indices.get(start..end.min(all_indices.len()));
    let Some(tris) = slice else { return max_chord; };
    // For large faces, sample at most MAX_CHORD_SAMPLES evenly-spaced
    // triangles to avoid O(tri_count) surface_point_deviation calls.
    const MAX_CHORD_SAMPLES: usize = 256;
    let tri_count = tris.len() / 4;
    let step = if tri_count > MAX_CHORD_SAMPLES {
        (tri_count / MAX_CHORD_SAMPLES).max(1)
    } else {
        1
    };
    // Per-measurement projection cache: the same boundary vertices and edge
    // midpoints are sampled across multiple triangles; cache avoids redundant
    // Newton-Raphson surface projections (9 seeds × 20 iters each).
    let mut proj_cache: ProjectionCache = HashMap::with_capacity(MAX_CHORD_SAMPLES * 4);
    for ti in (0..tri_count).step_by(step) {
        let ci = ti * 4;
        let chunk = &tris[ci..(ci + 4).min(tris.len())];
        if chunk.len() < 3 {
            continue;
        }
        max_chord = max_chord.max(tri_max_chord_error(
            chunk[0],
            chunk[1],
            chunk[2],
            global_vertices,
            &face.surface,
            inv_tol,
            &mut proj_cache,
        ));
    }
    max_chord
}

/// Projection cache keyed by quantized 3D position.
/// Eliminates redundant `surface.project()` calls when the same 3D point
/// is sampled across multiple triangles during chord error measurement.
type ProjectionCache = HashMap<[u32; 3], Option<(f32, f32)>>;

fn surface_point_deviation_cached(
    p: Vec3,
    surface: &SurfaceGeom,
    inv_tol: f32,
    cache: &mut ProjectionCache,
) -> f32 {
    let key = rc3d_core::utils::hash::f32x3_quantized_bits([p.x, p.y, p.z]);
    if let Some(cached) = cache.get(&key) {
        return match cached {
            Some((u, v)) => (p - surface.d0_at_native_uv(*u, *v)).length(),
            None => 0.0,
        };
    }
    // Fast path: project() alone is sufficient for most surface types.
    let result = if let Some((u, v)) = surface.project(p) {
        let dev = (p - surface.d0_at_native_uv(u, v)).length();
        // Sub-micron accuracy is good enough — skip grid-search fallback.
        if dev < 1e-6 {
            cache.insert(key, Some((u, v)));
            return dev;
        }
        cache.insert(key, Some((u, v)));
        dev
    } else {
        // Fallback: grid-search inverse (handles edge cases where Newton diverges).
        let search_tol = inv_tol.max(0.01).min(1.0);
        if let Some((u, v)) = surface.inverse_native_uv(p, search_tol.max(0.5)) {
            let dev = (p - surface.d0_at_native_uv(u, v)).length();
            cache.insert(key, Some((u, v)));
            dev
        } else {
            cache.insert(key, None);
            0.0
        }
    };
    result
}

fn tri_max_chord_error(
    i0: i32,
    i1: i32,
    i2: i32,
    verts: &[Vec3],
    surface: &SurfaceGeom,
    inv_tol: f32,
    cache: &mut ProjectionCache,
) -> f32 {
    let p0 = verts[i0 as usize];
    let p1 = verts[i1 as usize];
    let p2 = verts[i2 as usize];
    let mut max_dev = 0.0f32;
    for p in [p0, p1, p2] {
        max_dev = max_dev.max(surface_point_deviation_cached(p, surface, inv_tol, cache));
    }
    for (a, b) in [(p0, p1), (p1, p2), (p2, p0)] {
        let mid = (a + b) * 0.5;
        max_dev = max_dev.max(surface_point_deviation_cached(mid, surface, inv_tol, cache));
    }
    let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
    max_dev = max_dev.max(surface_point_deviation_cached(centroid, surface, inv_tol, cache));
    max_dev
}

pub(crate) fn fix_tri_winding(
    i0: &mut i32,
    i1: &mut i32,
    i2: &mut i32,
    global_vertices: &[Vec3],
    surface: &SurfaceGeom,
    same_sense: bool,
) {
    let p0 = global_vertices[*i0 as usize];
    let p1 = global_vertices[*i1 as usize];
    let p2 = global_vertices[*i2 as usize];
    let tri_n = (p1 - p0).cross(p2 - p0);
    if tri_n.length() <= 1e-10 {
        return;
    }
    // For freeform surfaces (BSpline/Offset), surface.project() is expensive
    // (Newton-Raphson multi-start). Use the triangle's own vertex normals
    // (pre-computed from the surface) as a cheap proxy. For analytic surfaces
    // (Plane, Cylinder, etc.), project() is O(1) trigonometric.
    let face_n = if matches!(surface, SurfaceGeom::BSpline(_) | SurfaceGeom::Offset { .. }) {
        // Skip per-triangle project() — the CDT already produces consistently-
        // oriented triangles for parametric surfaces. Only flip if the triangle
        // normal is grossly misaligned with same_sense.
        if !same_sense { -tri_n } else { tri_n }
    } else {
        let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
        match surface.project(centroid) {
            Some((uc, vc)) => {
                let mut n = surface.normal_native(uc, vc);
                if !same_sense { n = -n; }
                n
            }
            None => return,
        }
    };
    if tri_n.dot(face_n) < 0.0 {
        std::mem::swap(i1, i2);
    }
}

fn accumulate_normals(
    i0: i32,
    i1: i32,
    i2: i32,
    global_vertices: &[Vec3],
    global_normals: &mut [Vec3],
) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::face_uv::{point_in_trim, signed_area_2d, UvLoop, UvSource, UvVertex};
    use crate::store::BRepStore;
    use crate::topo::{BRepFace, FaceKey};

    #[test]
    fn fill_trimmed_plane_no_tris_inside_hole() {
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 0.0) },
                UvVertex { global_idx: 1, uv: (4.0, 0.0) },
                UvVertex { global_idx: 2, uv: (4.0, 4.0) },
                UvVertex { global_idx: 3, uv: (0.0, 4.0) },
            ],
        };
        let inner = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 4, uv: (1.0, 1.0) },
                UvVertex { global_idx: 5, uv: (3.0, 1.0) },
                UvVertex { global_idx: 6, uv: (3.0, 3.0) },
                UvVertex { global_idx: 7, uv: (1.0, 3.0) },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![inner],
            uv_source: UvSource::Pcurve,
        };
        let face = BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: Default::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        };
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 4.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::new(3.0, 3.0, 0.0),
            Vec3::new(1.0, 3.0, 0.0),
        ];
        let mut norms = vec![Vec3::Z; 8];
        let mut indices = Vec::new();
        let mut pos_map = HashMap::new();
        let reg = BRepStore::new();
        let fk = FaceKey::default();
        let range = fill_trimmed(
            fk,
            &loops,
            &face,
            &reg,
            &mut verts,
            &mut norms,
            &mut indices,
            &mut pos_map,
            1,
            &FaceFillConfig {
                enable_interior: false,
                ..Default::default()
            },
        );
        assert!(range.tri_count > 0);
        let hole = vec![(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)];
        let outer_poly = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let uv_map: HashMap<usize, (f32, f32)> = loops
            .outer
            .boundary
            .iter()
            .chain(loops.inners.iter().flat_map(|l| l.boundary.iter()))
            .map(|v| (v.global_idx, v.uv))
            .collect();
        for chunk in indices.chunks(4) {
            if chunk.len() < 4 {
                continue;
            }
            let uv = |gi: i32| uv_map[&(gi as usize)];
            let cu = (uv(chunk[0]).0 + uv(chunk[1]).0 + uv(chunk[2]).0) / 3.0;
            let cv = (uv(chunk[0]).1 + uv(chunk[1]).1 + uv(chunk[2]).1) / 3.0;
            assert!(
                point_in_trim(cu, cv, &outer_poly, &[hole.clone()]),
                "triangle centroid ({cu},{cv}) inside hole"
            );
        }
    }

    #[test]
    fn signed_area_used_for_orientation() {
        let cw = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        assert!(signed_area_2d(&cw) < 0.0);
    }
}
