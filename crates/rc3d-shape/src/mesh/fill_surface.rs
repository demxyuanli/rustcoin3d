//! Surface fill when PCURVEs are unavailable (project boundary to surface).

use std::collections::HashSet;

use rc3d_core::math::Vec3;
use super::boundary::{
    register_boundary_point_with_normal_indexed_shared, BoundaryPosIndex, SharedBoundaryPool,
};

use super::delaunay2d::{CdtVertHandle, NativeCdt};
use super::face_cdt::triangulate_uv_cdt_with_steiner;
use super::face_fill::{
    accumulate_normals, emit_filtered_triangles, fix_tri_winding, surface_uv_basis,
    triangulate_open_chain_uv, FaceFillConfig, FaceMeshRange,
};
use super::face_uv::{
    assign_revolution_native_uv_along_wire, boundary_is_mixed, loops_native_surface_uv,
    split_boundary_chains_at_3d_jumps, unwrap_periodic_uv_loops, uv_loop_is_degenerate,
    FaceUvLoops, UvLoop, UvSource, UvVertex,
};
use crate::geom::SurfaceGeom;
use crate::topo::{BRepFace, FaceKey};

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
    boundary_is_mixed(&loops.outer.boundary, verts)
}

pub(crate) fn max_allowed_triangle_edge(loops: &FaceUvLoops, verts: &[Vec3]) -> f32 {
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_mixed_boundary_segmented(
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
        cdt_constraint_failures: 0,
    }
}

/// CDT-first triangulation with Steiner refinement, earcut as ultimate fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn polyline_surface_uv(surface: &SurfaceGeom, indices: &[usize], verts: &[Vec3]) -> Vec<(f32, f32)> {
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
    inner_boundaries: &[Vec<usize>],
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut BoundaryPosIndex,
    config: &FaceFillConfig,
    shared_boundary: Option<&SharedBoundaryPool>,
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
            cdt_constraint_failures: 0,
        };
    }

    log::debug!(
        "[BRep mesh] face {:?}: surface fill 3D ({} boundary verts, {} inner loops)",
        face_key,
        boundary_global.len(),
        inner_boundaries.len()
    );

    let inners: Vec<UvLoop> = inner_boundaries.iter().map(|ib| UvLoop {
        boundary: ib.iter().map(|&gi| UvVertex { global_idx: gi, uv: (0.0, 0.0) }).collect(),
    }).collect();

    let pseudo_loops = FaceUvLoops {
        outer: UvLoop {
            boundary: boundary_global
                .iter()
                .map(|&gi| UvVertex {
                    global_idx: gi,
                    uv: (0.0, 0.0),
                })
                .collect(),
        },
        inners,
        uv_source: UvSource::SurfaceFill,
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
        let boundary_uvs: Vec<UvVertex> = boundary_global
            .iter()
            .zip(projected.iter())
            .filter_map(|(&gi, uv)| uv.map(|uv| UvVertex { global_idx: gi, uv }))
            .collect();

        if boundary_uvs.len() >= 3 {
            let mut loops = FaceUvLoops {
                outer: UvLoop {
                    boundary: boundary_uvs,
                },
                inners: vec![],
                uv_source: UvSource::SurfaceFill,
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
                let (tris_flat, max_chord_error, cdt_report) = triangulate_uv_cdt_with_steiner(
                    &loops,
                    face,
                    Some(face_key),
                    global_vertices,
                    global_normals,
                    pos_to_idx,
                    config,
                    None,
                    shared_boundary,
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
                        cdt_constraint_failures: cdt_report.failed,
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
        inner_boundaries,
        face,
        global_vertices,
        global_normals,
        all_indices,
        pos_to_idx,
        config,
        first_tri,
        boundary_set,
        shared_boundary,
    )
}

/// Planar fallback: fit a plane to boundary, CDT in 2D plane coords,
/// Steiner points snapped to surface via `surface.project()` + `surface.d0_native()`.
pub(crate) fn surface_fill_3d_planar(
    face_key: FaceKey,
    boundary_global: &[usize],
    inner_boundaries: &[Vec<usize>],
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut BoundaryPosIndex,
    config: &FaceFillConfig,
    first_tri: usize,
    boundary_set: HashSet<usize>,
    shared_boundary: Option<&SharedBoundaryPool>,
) -> FaceMeshRange {
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
            cdt_constraint_failures: 0,
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
                cdt_constraint_failures: 0,
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

    // Insert inner boundary loops as hole constraints
    for inner_bdy in inner_boundaries {
        if inner_bdy.len() < 3 {
            continue;
        }
        let inner_pts: Vec<Vec3> = inner_bdy.iter().map(|&gi| global_vertices[gi]).collect();
        let mut inner_handles: Vec<CdtVertHandle> = Vec::new();
        for pt in &inner_pts {
            let rel = *pt - origin;
            let pu = rel.dot(u_axis) as f64;
            let pv = rel.dot(v_axis) as f64;
            let Some(h) = cdt.insert(pu, pv, 0) else { continue; };
            inner_handles.push(h);
        }
        let m = inner_handles.len();
        for i in 0..m {
            let a = inner_handles[i];
            let b = inner_handles[(i + 1) % m];
            let _ = cdt.try_add_constraint(a, b);
        }
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
                let gi = register_boundary_point_with_normal_indexed_shared(
                    pt_3d,
                    global_vertices,
                    global_normals,
                    pos_to_idx,
                    shared_boundary,
                    || {
                        let mut n = face.surface.normal_native(su, sv);
                        if !face.same_sense {
                            n = -n;
                        }
                        n
                    },
                );

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
            cdt_constraint_failures: 0,
        };
    }

    let inners_loops: Vec<UvLoop> = inner_boundaries.iter().map(|ib| UvLoop {
        boundary: ib.iter().map(|&gi| UvVertex { global_idx: gi, uv: (0.0, 0.0) }).collect(),
    }).collect();
    let pseudo_loops = FaceUvLoops {
        outer: UvLoop {
            boundary: boundary_global
                .iter()
                .map(|&gi| UvVertex {
                    global_idx: gi,
                    uv: (0.0, 0.0),
                })
                .collect(),
        },
        inners: inners_loops,
        uv_source: UvSource::SurfaceFill,
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
        cdt_constraint_failures: 0,
    }
}
