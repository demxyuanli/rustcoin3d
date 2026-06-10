//! BRepMesh_Face param-domain fill — OCC BRepMesh_Delaun stand-in (earcut P0).
//! Maps trim loops in UV -> Geom_Surface::Value(u,v).

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;

use super::face_cdt::{triangulate_uv_cdt_with_steiner, CdtConstraintReport};
use super::grid::mesh_trimmed_uv_grid;
use super::face_uv::{
    ensure_loop_orientation, point_in_trim,
    rebuild_loop_uv_local_frame, revolution_boundary_v_collapsed, revolution_u_span_collapsed,
    split_boundary_chains_at_3d_jumps, FaceUvLoops,
    loops_native_surface_uv, UvSource, uv_loop_is_degenerate,
    cylinder_loop_needs_uv_rebuild, repair_cylinder_uv_loops,
};
use crate::geom::SurfaceGeom;
use crate::store::BRepStore;
use crate::topo::{BRepFace, EdgeKey, FaceKey};

// Re-exports from fill_surface (extracted to keep face_fill.rs focused)
pub use super::fill_surface::{
    default_grid_for_surface, effective_min_size, face_boundary_is_mixed,
    surface_fill_3d,
};
pub(crate) use super::fill_surface::{
    max_allowed_triangle_edge, polyline_surface_uv,
};

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
    /// Maximum CDT vertices per face (OCC IMeshTools_Parameters::MaxNodes).
    /// Prevents runaway memory on large faces with tight deflection.
    pub max_cdt_vertices: usize,
    /// Delaunay triangulation backend for CDT face meshing.
    pub delaunay_backend: super::delaunay2d::DelaunayBackend,
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
            max_cdt_vertices: 4096,
            delaunay_backend: super::delaunay2d::DelaunayBackend::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FaceMeshRange {
    pub face_key: FaceKey,
    pub first_tri: usize,
    pub tri_count: usize,
    pub boundary_global: HashSet<usize>,
    pub max_chord_error: f32,
    /// CDT boundary constraints that failed enforcement (0 = success).
    pub cdt_constraint_failures: usize,
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

pub(crate) fn triangulate_open_chain_uv(chain: &[super::face_uv::UvVertex], verts: &[Vec3]) -> Vec<(i32, i32, i32)> {
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
    pos_to_idx: &mut super::boundary::BoundaryPosIndex,
    fill_cfg: &FaceFillConfig,
    shared_boundary: Option<&super::boundary::SharedBoundaryPool>,
) -> (Vec<(i32, i32, i32)>, f32, CdtConstraintReport) {
    let (tris_flat, max_chord_error, constraint_report) = triangulate_uv_cdt_with_steiner(
        work_loops,
        face,
        Some(face_key),
        global_vertices,
        global_normals,
        pos_to_idx,
        fill_cfg,
        Some(reg),
        shared_boundary,
    );
    let mut tris = Vec::new();
    for chunk in tris_flat.chunks(3) {
        if chunk.len() != 3 {
            continue;
        }
        tris.push((chunk[0] as i32, chunk[1] as i32, chunk[2] as i32));
    }
    (tris, max_chord_error, constraint_report)
}

fn uv_bounds_from_loops(loops: &FaceUvLoops) -> (f32, f32, f32, f32) {
    loops.outer.boundary.iter().fold(
        (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
        |(u0, u1, v0, v1), v| (u0.min(v.uv.0), u1.max(v.uv.0), v0.min(v.uv.1), v1.max(v.uv.1)),
    )
}


fn filter_tris_outside_uv_holes(
    tris: Vec<(i32, i32, i32)>,
    work_loops: &FaceUvLoops,
) -> Vec<(i32, i32, i32)> {
    if work_loops.inners.is_empty() {
        return tris;
    }
    let outer_uv: Vec<(f32, f32)> = work_loops
        .outer
        .boundary
        .iter()
        .map(|v| v.uv)
        .collect();
    let holes: Vec<Vec<(f32, f32)>> = work_loops
        .inners
        .iter()
        .map(|l| l.boundary.iter().map(|v| v.uv).collect())
        .collect();
    let mut uv_by_gi: HashMap<usize, (f32, f32)> = HashMap::new();
    for v in work_loops
        .outer
        .boundary
        .iter()
        .chain(work_loops.inners.iter().flat_map(|l| l.boundary.iter()))
    {
        uv_by_gi.insert(v.global_idx, v.uv);
    }
    tris.into_iter()
        .filter(|&(i0, i1, i2)| {
            let Some(uv0) = uv_by_gi.get(&(i0 as usize)) else {
                return false;
            };
            let Some(uv1) = uv_by_gi.get(&(i1 as usize)) else {
                return false;
            };
            let Some(uv2) = uv_by_gi.get(&(i2 as usize)) else {
                return false;
            };
            let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
            let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
            point_in_trim(cu, cv, &outer_uv, &holes)
        })
        .collect()
}

fn loop_uv_centroid(boundary: &[super::face_uv::UvVertex]) -> (f32, f32) {
    let n = boundary.len() as f32;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let (su, sv) = boundary
        .iter()
        .fold((0.0f32, 0.0f32), |(u, v), vtx| (u + vtx.uv.0, v + vtx.uv.1));
    (su / n, sv / n)
}

fn match_inner_indices_by_quadrant(
    outer: &[super::face_uv::UvVertex],
    inner: &[super::face_uv::UvVertex],
) -> Vec<usize> {
    let oc = loop_uv_centroid(outer);
    let ic = loop_uv_centroid(inner);
    outer
        .iter()
        .map(|ov| {
            let o_hi_u = ov.uv.0 >= oc.0;
            let o_hi_v = ov.uv.1 >= oc.1;
            let mut best = 0usize;
            let mut best_d2 = f32::MAX;
            for (ii, iv) in inner.iter().enumerate() {
                if (iv.uv.0 >= ic.0) != o_hi_u || (iv.uv.1 >= ic.1) != o_hi_v {
                    continue;
                }
                let du = ov.uv.0 - iv.uv.0;
                let dv = ov.uv.1 - iv.uv.1;
                let d2 = du * du + dv * dv;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = ii;
                }
            }
            best
        })
        .collect()
}

fn inner_corner_indices_adjacent(a: usize, b: usize, n: usize) -> bool {
    n >= 3 && (a + 1) % n == b || (b + 1) % n == a
}

/// Equal-vertex loops: pair corners by quadrant and emit corridor quads (2 tris each).
fn triangulate_holed_uv_corridor_quads(work_loops: &FaceUvLoops) -> Vec<(i32, i32, i32)> {
    if work_loops.inners.len() != 1 {
        return Vec::new();
    }
    let outer = &work_loops.outer.boundary;
    let inner = &work_loops.inners[0].boundary;
    let n = outer.len();
    if n != inner.len() || n < 3 {
        return Vec::new();
    }
    let matches = match_inner_indices_by_quadrant(outer, inner);
    for i in 0..n {
        if !inner_corner_indices_adjacent(matches[i], matches[(i + 1) % n], n) {
            return Vec::new();
        }
    }
    let mut tris = Vec::with_capacity(2 * n);
    for i in 0..n {
        let o0 = &outer[i];
        let o1 = &outer[(i + 1) % n];
        let i0 = &inner[matches[i]];
        let i1 = &inner[matches[(i + 1) % n]];
        tris.push((
            o0.global_idx as i32,
            o1.global_idx as i32,
            i1.global_idx as i32,
        ));
        tris.push((
            o0.global_idx as i32,
            i1.global_idx as i32,
            i0.global_idx as i32,
        ));
    }
    filter_tris_outside_uv_holes(tris, work_loops)
}

fn find_closest_bridge_pair(
    poly: &[(usize, (f32, f32))],
    inner: &[super::face_uv::UvVertex],
) -> (usize, usize) {
    let mut best_pi = 0usize;
    let mut best_ii = 0usize;
    let mut best_d2 = f32::MAX;
    for (pi, &(_, puv)) in poly.iter().enumerate() {
        for (ii, iv) in inner.iter().enumerate() {
            let du = puv.0 - iv.uv.0;
            let dv = puv.1 - iv.uv.1;
            let d2 = du * du + dv * dv;
            if d2 < best_d2 {
                best_d2 = d2;
                best_pi = pi;
                best_ii = ii;
            }
        }
    }
    (best_pi, best_ii)
}

fn insert_hole_loop_into_polygon(
    poly: &mut Vec<(usize, (f32, f32))>,
    insert_after: usize,
    inner: &[super::face_uv::UvVertex],
    inner_start: usize,
) {
    let n = inner.len();
    if n < 3 {
        return;
    }
    let bridge = &inner[inner_start];
    let mut hole_verts: Vec<(usize, (f32, f32))> = vec![(bridge.global_idx, bridge.uv)];
    for i in 1..n {
        let v = &inner[(inner_start + n - i) % n];
        hole_verts.push((v.global_idx, v.uv));
    }
    // Duplicate bridge vertex so the slit returns to the outer ring.
    hole_verts.push((bridge.global_idx, bridge.uv));

    let tail = poly.split_off(insert_after + 1);
    poly.extend(hole_verts);
    poly.extend(tail);
}

/// Multi-hole earcut: connect each inner loop to the outer via a bridge slit.
fn triangulate_holed_uv_multi_bridge_earcut(work_loops: &FaceUvLoops) -> Vec<(i32, i32, i32)> {
    if work_loops.inners.is_empty() {
        return Vec::new();
    }
    let outer = &work_loops.outer.boundary;
    if outer.len() < 3 {
        return Vec::new();
    }
    let mut poly: Vec<(usize, (f32, f32))> = outer
        .iter()
        .map(|v| (v.global_idx, v.uv))
        .collect();

    for inner_loop in &work_loops.inners {
        if inner_loop.boundary.len() < 3 {
            continue;
        }
        let (insert_after, inner_start) = find_closest_bridge_pair(&poly, &inner_loop.boundary);
        insert_hole_loop_into_polygon(&mut poly, insert_after, &inner_loop.boundary, inner_start);
    }
    if poly.len() < 3 {
        return Vec::new();
    }

    let uv_pairs: Vec<(f64, f64)> = poly
        .iter()
        .map(|&(_, (u, v))| (u as f64, v as f64))
        .collect();
    let ear_tris = super::delaunay2d::earcut_uv_polygon(&uv_pairs, &[]);
    if ear_tris.is_empty() {
        return Vec::new();
    }
    let mut tris = Vec::with_capacity(ear_tris.len());
    for tri in &ear_tris {
        let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
        if i0 >= poly.len() || i1 >= poly.len() || i2 >= poly.len() {
            continue;
        }
        tris.push((
            poly[i0].0 as i32,
            poly[i1].0 as i32,
            poly[i2].0 as i32,
        ));
    }
    filter_tris_outside_uv_holes(tris, work_loops)
}

fn triangulate_planar_holed_face(work_loops: &FaceUvLoops) -> Vec<(i32, i32, i32)> {
    let corridor = triangulate_holed_uv_corridor_quads(work_loops);
    if !corridor.is_empty() {
        return corridor;
    }
    if work_loops.inners.len() == 1 {
        let single = triangulate_holed_uv_bridge_earcut(work_loops);
        if !single.is_empty() {
            return single;
        }
    }
    triangulate_holed_uv_multi_bridge_earcut(work_loops)
}

/// Single-hole earcut via a bridge edge (outer CCW + inner reversed).
fn triangulate_holed_uv_bridge_earcut(work_loops: &FaceUvLoops) -> Vec<(i32, i32, i32)> {
    if work_loops.inners.len() != 1 {
        return Vec::new();
    }
    let outer = &work_loops.outer.boundary;
    let inner = &work_loops.inners[0].boundary;
    if outer.len() < 3 || inner.len() < 3 {
        return Vec::new();
    }

    let mut best_oi = 0usize;
    let mut best_ii = 0usize;
    let mut best_d2 = f32::MAX;
    for (oi, ov) in outer.iter().enumerate() {
        for (ii, iv) in inner.iter().enumerate() {
            let du = ov.uv.0 - iv.uv.0;
            let dv = ov.uv.1 - iv.uv.1;
            let d2 = du * du + dv * dv;
            if d2 < best_d2 {
                best_d2 = d2;
                best_oi = oi;
                best_ii = ii;
            }
        }
    }

    let mut earcut_verts: Vec<(usize, (f32, f32))> = Vec::with_capacity(outer.len() + inner.len() + 1);
    for i in 0..outer.len() {
        let v = &outer[(best_oi + i) % outer.len()];
        earcut_verts.push((v.global_idx, v.uv));
    }
    earcut_verts.push((inner[best_ii].global_idx, inner[best_ii].uv));
    for i in 1..inner.len() {
        let v = &inner[(best_ii + inner.len() - i) % inner.len()];
        earcut_verts.push((v.global_idx, v.uv));
    }

    let uv_pairs: Vec<(f64, f64)> = earcut_verts
        .iter()
        .map(|&(_, (u, v))| (u as f64, v as f64))
        .collect();
    let ear_tris = super::delaunay2d::earcut_uv_polygon(&uv_pairs, &[]);
    if ear_tris.is_empty() {
        return Vec::new();
    }
    let mut tris = Vec::with_capacity(ear_tris.len());
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
    filter_tris_outside_uv_holes(tris, work_loops)
}

fn triangulate_loops_earcut_fallback(work_loops: &FaceUvLoops) -> Vec<(i32, i32, i32)> {
    let mut tris: Vec<(i32, i32, i32)> = Vec::new();
    if !work_loops.inners.is_empty() {
        return triangulate_planar_holed_face(work_loops);
    }
    let mut earcut_verts: Vec<(usize, (f32, f32))> = Vec::new();
    for v in &work_loops.outer.boundary {
        earcut_verts.push((v.global_idx, v.uv));
    }
    if earcut_verts.len() < 3 {
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
pub(crate) fn emit_filtered_triangles(
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


pub(crate) fn surface_is_revolution_like(surface: &SurfaceGeom) -> bool {
    match surface {
        SurfaceGeom::Revolution { .. } | SurfaceGeom::Extrusion { .. } => true,
        SurfaceGeom::Offset { basis, .. } => {
            matches!(basis.as_ref(), SurfaceGeom::Revolution { .. } | SurfaceGeom::Extrusion { .. })
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

/// Plane or offset-of-plane: trim loops share the basis plane UV frame.
pub(crate) fn surface_is_planar_trim(surface: &SurfaceGeom) -> bool {
    matches!(surface, SurfaceGeom::Plane { .. })
        || matches!(
            surface,
            SurfaceGeom::Offset { basis, .. }
                if matches!(basis.as_ref(), SurfaceGeom::Plane { .. })
        )
}

fn planar_uv_basis(surface: &SurfaceGeom) -> Option<&SurfaceGeom> {
    match surface {
        SurfaceGeom::Plane { .. } => Some(surface),
        SurfaceGeom::Offset { basis, .. }
            if matches!(basis.as_ref(), SurfaceGeom::Plane { .. }) =>
        {
            Some(basis.as_ref())
        }
        _ => None,
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

// Re-exports from ruled (moved here to keep ruled functions together)
pub use super::ruled::{
    mesh_ruled_two_wire_edges, mesh_ruled_wire_polygons_3d, sample_polyline,
};

// Re-exports from fill_plane
pub use super::fill_plane::{
    fan_triangulate_outer, filter_plane_ring_coplanar_cluster, mesh_plane_fan_3d,
    mesh_plane_fan_wire_polygons, orient_plane_ring_ccw, plane_angle_at,
    plane_fan_max_edge, plane_ring_circular_mean_angle, plane_ring_signed_area,
    plane_unwrap_angle_near, project_point_to_plane, sort_plane_ring_by_angle,
    triangulate_plane_boundary_fan, triangulate_plane_center_fan,
    triangulate_plane_center_fan_from_boundary,
};


pub fn fill_trimmed(
    face_key: FaceKey,
    loops: &FaceUvLoops,
    face: &BRepFace,
    reg: &BRepStore,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut super::boundary::BoundaryPosIndex,
    wire_edge_count: usize,
    config: &FaceFillConfig,
    shared_boundary: Option<&super::boundary::SharedBoundaryPool>,
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

    let is_planar_trim = surface_is_planar_trim(&face.surface);
    let is_revolution = matches!(face.surface, SurfaceGeom::Revolution { .. });
    let prefer_closed_cdt = matches!(
        &face.surface,
        SurfaceGeom::Revolution { .. }
            | SurfaceGeom::BSpline(_)
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. }
    ) || matches!(
        &face.surface,
        SurfaceGeom::Offset { basis, .. }
            if !matches!(basis.as_ref(), SurfaceGeom::Plane { .. })
    );
    let revolution_uv_ok = is_revolution
        && loops
            .native_uv_bounds()
            .map(|(u0, u1, _, _)| !revolution_u_span_collapsed(u1 - u0))
            .unwrap_or(false);
    let mut work_loops = if is_planar_trim {
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

    // Rebuild plane / offset-of-plane loops from 3D boundary (STEP PCURVE UV can be skewed).
    if let Some(plane_basis) = planar_uv_basis(&face.surface) {
        rebuild_loop_uv_local_frame(
            &mut work_loops.outer,
            global_vertices,
            Some(plane_basis),
        );
        for inner in &mut work_loops.inners {
            rebuild_loop_uv_local_frame(inner, global_vertices, Some(plane_basis));
        }
        ensure_loop_orientation(&mut work_loops.outer.boundary, false);
        for inner in &mut work_loops.inners {
            ensure_loop_orientation(&mut inner.boundary, true);
        }
        work_loops.uv_source = UvSource::Synthetic;

        // Planar faces: sorted boundary fan (no Steiner verts; cube quad → 2 tris/face).
        if work_loops.inners.is_empty() {
            let mut tris = triangulate_plane_boundary_fan(
                face,
                &work_loops,
                global_vertices,
                global_normals,
                pos_to_idx,
            );
            if tris.is_empty() {
                tris = triangulate_plane_center_fan(
                    face,
                    &work_loops,
                    global_vertices,
                    global_normals,
                    pos_to_idx,
                    config,
                    shared_boundary,
                );
            }
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
                    cdt_constraint_failures: 0,
                };
            }
        }

        if !work_loops.inners.is_empty() {
            let holed_tris = triangulate_planar_holed_face(&work_loops);
            if !holed_tris.is_empty() {
                let emitted = emit_filtered_triangles(
                    holed_tris,
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
                    cdt_constraint_failures: 0,
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
        !is_planar_trim && !prefer_closed_cdt && chains.len() >= 2 && long_count >= 2;

    let (mut tris, mut max_chord_error, mut cdt_report) = if use_segmentation {
        let mut seg_tris = Vec::new();
        for chain in &chains {
            seg_tris.extend(triangulate_open_chain_uv(chain, global_vertices));
        }
        (seg_tris, 0.0f32, CdtConstraintReport::default())
    } else {
        triangulate_loops_cdt(
            &work_loops,
            face,
            face_key,
            reg,
            global_vertices,
            global_normals,
            pos_to_idx,
            &fill_cfg,
            shared_boundary,
        )
    };

    if cdt_report.has_failures() {
        log::debug!(
            "[BRep mesh] face {:?}: {} CDT constraint failures, trimmed UV grid fallback",
            face_key,
            cdt_report.failed
        );
        if is_planar_trim && !work_loops.inners.is_empty() {
            let bridge_tris = triangulate_planar_holed_face(&work_loops);
            if !bridge_tris.is_empty() {
                let emitted = emit_filtered_triangles(
                    bridge_tris,
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
                    cdt_constraint_failures: cdt_report.failed,
                };
            }
            tris = triangulate_loops_earcut_fallback(&work_loops);
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
                    cdt_constraint_failures: cdt_report.failed,
                };
            }
        }
        tris.clear();
        let grid_first = all_indices.len() / 4;
        mesh_trimmed_uv_grid(
            face,
            &work_loops,
            uv_bounds_from_loops(&work_loops),
            Some(&fill_cfg),
            None,
            global_vertices,
            global_normals,
            pos_to_idx,
            all_indices,
            shared_boundary,
        );
        let grid_count = all_indices.len() / 4 - grid_first;
        if grid_count > 0 {
            return FaceMeshRange {
                face_key,
                first_tri,
                tri_count: grid_count,
                boundary_global,
                max_chord_error: 0.0,
                cdt_constraint_failures: cdt_report.failed,
            };
        }
    }

    if tris.is_empty() {
        tris = triangulate_loops_earcut_fallback(&work_loops);
    }

    if tris.is_empty() && use_segmentation {
        let (cdt_tris, chord, seg_report) = triangulate_loops_cdt(
            &work_loops,
            face,
            face_key,
            reg,
            global_vertices,
            global_normals,
            pos_to_idx,
            &fill_cfg,
            shared_boundary,
        );
        tris = cdt_tris;
        max_chord_error = chord;
        cdt_report = seg_report;
    }

    if tris.is_empty() && is_planar_trim && work_loops.inners.is_empty() {
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
            cdt_constraint_failures: cdt_report.failed,
        };
    }

    let apply_quality = !is_planar_trim && !use_segmentation && !prefer_closed_cdt;
    let max_edge = if apply_quality {
        max_allowed_triangle_edge(&work_loops, global_vertices)
    } else {
        f32::MAX
    };

    if is_planar_trim && !work_loops.inners.is_empty() {
        tris = filter_tris_outside_uv_holes(tris, &work_loops);
    }

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
        cdt_constraint_failures: cdt_report.failed,
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
    // Always use the surface normal at the triangle centroid for correct winding.
    // OCC: BRepMesh_FastDiscretFace — evaluates surface normal at face barycenter.
    let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
    let face_n = match surface.project(centroid) {
        Some((uc, vc)) => {
            let mut n = surface.normal_native(uc, vc);
            if !same_sense { n = -n; }
            n
        }
        None => {
            // Fallback for project() failure: use triangle normal with same_sense
            if !same_sense { -tri_n } else { tri_n }
        }
    };
    if tri_n.dot(face_n) < 0.0 {
        std::mem::swap(i1, i2);
    }
}

pub(crate) fn accumulate_normals(
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
        let mut pos_map = crate::mesh::boundary::BoundaryPosIndex::with_cell_size(
            crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE,
        );
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(i, [v.x, v.y, v.z]);
        }
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
            None,
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
