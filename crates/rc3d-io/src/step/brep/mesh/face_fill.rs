//! BRepMesh_Face param-domain fill — OCC BRepMesh_Delaun stand-in (earcut P0).
//! Maps trim loops in UV -> Geom_Surface::Value(u,v).

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;

use super::face_cdt::triangulate_uv_cdt_with_steiner;
use super::face_uv::{split_boundary_chains_at_3d_jumps, FaceUvLoops, loops_native_surface_uv};
use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::topo::{BRepFace, FaceKey};

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
}

impl Default for FaceFillConfig {
    fn default() -> Self {
        Self {
            enable_interior: true,
            deflection_interior: 0.01,
            min_size: 1e-3,
            min_size_relative: 0.01,
            shell_min_size: 0.0,
            max_adapt_iterations: 8,
            angular_deflection: 0.2,
        }
    }
}

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
    let Some((u, v)) = surface.project(centroid) else {
        return true;
    };
    let mut sn = surface.normal_native(u, v);
    if !same_sense {
        sn = -sn;
    }
    tri_n.normalize().dot(sn) > -0.05
}

fn chain_closing_3d_len(chain: &[super::face_uv::UvVertex], verts: &[Vec3]) -> f32 {
    if chain.len() < 2 {
        return f32::MAX;
    }
    let a = chain[0].global_idx;
    let b = chain[chain.len() - 1].global_idx;
    if a >= verts.len() || b >= verts.len() {
        return f32::MAX;
    }
    (verts[b] - verts[a]).length()
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
    let flat: Vec<f64> = chain
        .iter()
        .flat_map(|v| [v.uv.0 as f64, v.uv.1 as f64])
        .collect();
    let Ok(indices) = earcutr::earcut(&flat, &[], 2) else {
        return triangulate_open_chain_strip(chain);
    };
    let mut tris = Vec::new();
    for chunk in indices.chunks(3) {
        if chunk.len() != 3 {
            continue;
        }
        let (i0, i1, i2) = (chunk[0], chunk[1], chunk[2]);
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
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    fill_cfg: &FaceFillConfig,
) -> (Vec<(i32, i32, i32)>, f32) {
    let (tris_flat, max_chord_error) = triangulate_uv_cdt_with_steiner(
        work_loops,
        face,
        global_vertices,
        global_normals,
        pos_to_idx,
        fill_cfg,
        None,
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
    let flat: Vec<f64> = earcut_verts
        .iter()
        .flat_map(|&(_, (u, v))| [u as f64, v as f64])
        .collect();
    let ear_indices = match earcutr::earcut(&flat, &hole_indices, 2) {
        Ok(indices) if !indices.is_empty() => indices,
        _ => {
            let g0 = earcut_verts[0].0 as i32;
            for i in 1..earcut_verts.len() - 1 {
                tris.push((g0, earcut_verts[i].0 as i32, earcut_verts[i + 1].0 as i32));
            }
            return tris;
        }
    };
    for chunk in ear_indices.chunks(3) {
        if chunk.len() != 3 {
            continue;
        }
        let (i0, i1, i2) = (chunk[0], chunk[1], chunk[2]);
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
        fix_winding(
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
            fix_winding(
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
pub fn fill_trimmed(
    face_key: FaceKey,
    loops: &FaceUvLoops,
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
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
    let work_loops = if is_plane {
        loops.clone()
    } else {
        loops_native_surface_uv(loops, face, global_vertices)
    };

    let mut tris: Vec<(i32, i32, i32)> = Vec::new();
    let mut max_chord_error = 0.0f32;

    let fill_cfg = if is_plane {
        config.clone()
    } else {
        FaceFillConfig {
            enable_interior: false,
            max_adapt_iterations: 0,
            ..config.clone()
        }
    };

    let (chains, long_count) = split_boundary_chains_at_3d_jumps(
        &work_loops.outer.boundary,
        global_vertices,
        8.0,
    );
    let use_segmentation = !is_plane && chains.len() >= 2 && long_count >= 2;

    if use_segmentation {
        for chain in &chains {
            tris.extend(triangulate_open_chain_uv(chain, global_vertices));
        }
    } else {
        let (cdt_tris, chord) = triangulate_loops_cdt(
            &work_loops,
            face,
            global_vertices,
            global_normals,
            pos_to_idx,
            &fill_cfg,
        );
        tris = cdt_tris;
        max_chord_error = chord;
    }

    if tris.is_empty() {
        tris = triangulate_loops_earcut_fallback(&work_loops);
    }

    if tris.is_empty() && use_segmentation {
        let (cdt_tris, chord) = triangulate_loops_cdt(
            &work_loops,
            face,
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

    let apply_quality = !is_plane && !use_segmentation;
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
        let uv = face
            .surface
            .project(pt)
            .or_else(|| face.surface.inverse_native_uv(pt, inv_tol));
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
            let loops = super::face_uv::FaceUvLoops {
                outer: super::face_uv::UvLoop {
                    boundary: boundary_uvs,
                },
                inners: vec![],
                uv_source: super::face_uv::UvSource::SurfaceFill,
            };

            log::debug!(
                "[BRep mesh] face {:?}: UV CDT + Steiner ({} boundary UVs)",
                face_key,
                loops.outer.boundary.len()
            );

            let (tris_flat, max_chord_error) = triangulate_uv_cdt_with_steiner(
                &loops, face, global_vertices, global_normals, pos_to_idx, config, None,
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
                    fix_winding(
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
    use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

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

    // Build CDT in plane 2D space
    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::new();
    let mut handles: Vec<spade::handles::FixedVertexHandle> = Vec::new();
    let mut uv_to_gi: HashMap<(u64, u64), usize> = HashMap::new();

    for (i, &gi) in boundary_global.iter().enumerate() {
        let (pu, pv) = plane_uvs[i];
        let pt = Point2::new(pu, pv);
        let Ok(h) = cdt.insert(pt) else {
            return FaceMeshRange {
                face_key, first_tri, tri_count: 0,
                boundary_global: boundary_set, max_chord_error: 0.0,
            };
        };
        let key = ((pu * 1e6) as u64, (pv * 1e6) as u64);
        uv_to_gi.insert(key, gi);
        handles.push(h);
    }

    // Add outer boundary constraints
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
            for face_h in cdt.inner_faces() {
                let verts: Vec<_> = face_h
                    .vertices()
                    .iter()
                    .map(|v| v.fix().index())
                    .collect();
                if verts.len() != 3 {
                    continue;
                }
                let i0 = verts[0];
                let i1 = verts[1];
                let i2 = verts[2];

                let uv0 = (
                    cdt.vertex(handles[i0]).position().x,
                    cdt.vertex(handles[i0]).position().y,
                );
                let uv1 = (
                    cdt.vertex(handles[i1]).position().x,
                    cdt.vertex(handles[i1]).position().y,
                );
                let uv2 = (
                    cdt.vertex(handles[i2]).position().x,
                    cdt.vertex(handles[i2]).position().y,
                );

                // Look up 3D positions (always from surface via d0_native)
                let key0 = ((uv0.0 * 1e6) as u64, (uv0.1 * 1e6) as u64);
                let key1 = ((uv1.0 * 1e6) as u64, (uv1.1 * 1e6) as u64);
                let key2 = ((uv2.0 * 1e6) as u64, (uv2.1 * 1e6) as u64);
                let Some(&gi0) = uv_to_gi.get(&key0) else { continue; };
                let Some(&gi1) = uv_to_gi.get(&key1) else { continue; };
                let Some(&gi2) = uv_to_gi.get(&key2) else { continue; };
                let p0 = global_vertices[gi0];
                let p1 = global_vertices[gi1];
                let p2 = global_vertices[gi2];

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

                let pt = Point2::new(pu, pv);
                if let Ok(h) = cdt.insert(pt) {
                    let map_key = ((pu * 1e6) as u64, (pv * 1e6) as u64);
                    uv_to_gi.insert(map_key, gi);
                    handles.push(h);
                }
            }
        }
    }

    // ── Extract all triangles (no trim filtering — no trim domain) ──
    let mut tris: Vec<(i32, i32, i32)> = Vec::new();
    for face_h in cdt.inner_faces() {
        let verts: Vec<_> = face_h
            .vertices()
            .iter()
            .map(|v| v.fix().index())
            .collect();
        if verts.len() != 3 {
            continue;
        }
        let uv0 = (
            cdt.vertex(handles[verts[0]]).position().x,
            cdt.vertex(handles[verts[0]]).position().y,
        );
        let uv1 = (
            cdt.vertex(handles[verts[1]]).position().x,
            cdt.vertex(handles[verts[1]]).position().y,
        );
        let uv2 = (
            cdt.vertex(handles[verts[2]]).position().x,
            cdt.vertex(handles[verts[2]]).position().y,
        );
        let key0 = ((uv0.0 * 1e6) as u64, (uv0.1 * 1e6) as u64);
        let key1 = ((uv1.0 * 1e6) as u64, (uv1.1 * 1e6) as u64);
        let key2 = ((uv2.0 * 1e6) as u64, (uv2.1 * 1e6) as u64);
        if let (Some(&gi0), Some(&gi1), Some(&gi2)) =
            (uv_to_gi.get(&key0), uv_to_gi.get(&key1), uv_to_gi.get(&key2))
        {
            tris.push((gi0 as i32, gi1 as i32, gi2 as i32));
        }
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
    if let Some(tris) = slice {
        for chunk in tris.chunks(4) {
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
            ));
        }
    }
    max_chord
}

fn surface_point_deviation(p: Vec3, surface: &SurfaceGeom, inv_tol: f32) -> f32 {
    let snap_tol = inv_tol.max(0.1);
    if let Some((u, v)) = surface.inverse_native_uv(p, snap_tol) {
        let d = (p - surface.d0_at_native_uv(u, v)).length();
        if d <= snap_tol {
            return d;
        }
    }
    if let Some((u, v)) = surface.project(p) {
        let d = (p - surface.d0_at_native_uv(u, v)).length();
        if d <= snap_tol * 4.0 {
            return d;
        }
    }
    if let Some((u, v)) = surface.inverse_native_uv_build(p, snap_tol * 4.0) {
        let d = (p - surface.d0_at_native_uv(u, v)).length();
        if d <= snap_tol * 4.0 {
            return d;
        }
    }
    0.0
}

fn tri_max_chord_error(
    i0: i32,
    i1: i32,
    i2: i32,
    verts: &[Vec3],
    surface: &SurfaceGeom,
    inv_tol: f32,
) -> f32 {
    let p0 = verts[i0 as usize];
    let p1 = verts[i1 as usize];
    let p2 = verts[i2 as usize];
    let mut max_dev = 0.0f32;
    for p in [p0, p1, p2] {
        max_dev = max_dev.max(surface_point_deviation(p, surface, inv_tol));
    }
    for (a, b) in [(p0, p1), (p1, p2), (p2, p0)] {
        let mid = (a + b) * 0.5;
        max_dev = max_dev.max(surface_point_deviation(mid, surface, inv_tol));
    }
    let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
    max_dev = max_dev.max(surface_point_deviation(centroid, surface, inv_tol));
    max_dev
}

fn fix_winding(
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
    let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
    if let Some((uc, vc)) = surface.project(centroid) {
        let mut face_n = surface.normal_native(uc, vc);
        if !same_sense {
            face_n = -face_n;
        }
        if tri_n.dot(face_n) < 0.0 {
            std::mem::swap(i1, i2);
        }
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
    use crate::step::brep::mesh::face_uv::{point_in_trim, signed_area_2d, UvLoop, UvSource, UvVertex};
    use crate::step::brep::topo::{BRepFace, FaceKey};

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
        let fk = FaceKey::default();
        let range = fill_trimmed(
            fk,
            &loops,
            &face,
            &mut verts,
            &mut norms,
            &mut indices,
            &mut pos_map,
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
