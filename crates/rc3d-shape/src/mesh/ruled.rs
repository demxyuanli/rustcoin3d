use std::collections::{HashMap, HashSet};

use rc3d_core::math::{Real, PVec3};
use super::boundary::{
    register_boundary_point_with_normal_indexed_shared, BoundaryPosIndex, SharedBoundaryPool,
};
use crate::geom::SurfaceGeom;
use crate::topo::{BRepFace, EdgeKey, FaceKey, Orientation};
use super::edge_disc::EdgePolygon;
use super::edge_pool::FaceEdgeBoundaryIdx;
use super::face_fill::{
    accumulate_normals, fix_tri_winding, measure_face_chord_error, FaceFillConfig,
    FaceMeshRange, polyline_surface_uv, prefers_native_uv_ruled,
    surface_is_revolution_like, surface_uv_basis, wire_pair_is_same_edge_seam,
};
use super::fill_revolution::{
    resample_uv_polyline, revolution_ruled_grid_from_uv, revolution_wire_uv_polyline,
};
use super::grid::parametric_grid_segs;
use super::config::min_adequate_trim_tris;

const MAX_ADAPTIVE_RULED_PASSES: u32 = 3;

fn needs_adaptive_ruled(surface: &SurfaceGeom, _wire_edges: &[(EdgeKey, Vec<usize>)]) -> bool {
    matches!(
        surface,
        SurfaceGeom::BSpline(_) | SurfaceGeom::Offset { .. }
    )
}

pub struct RuledMeshBuffers<'a> {
    pub global_vertices: &'a mut Vec<PVec3>,
    pub global_normals: &'a mut Vec<PVec3>,
    pub all_indices: &'a mut Vec<i32>,
    pub pos_to_idx: &'a mut BoundaryPosIndex,
    pub shared_boundary: Option<&'a SharedBoundaryPool>,
}

/// Ruled quad strip between exactly two boundary wires (no untrimmed native UV sheet).
pub fn try_ruled_two_wire_mesh(
    face_key: FaceKey,
    face: &crate::topo::BRepFace,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &FaceEdgeBoundaryIdx,
    buffers: RuledMeshBuffers<'_>,
    fill_config: &FaceFillConfig,
) -> FaceMeshRange {
    if needs_adaptive_ruled(&face.surface, wire_edges) {
        return try_ruled_two_wire_mesh_adaptive(
            face_key,
            face,
            wire_edges,
            edge_polygons,
            edge_boundary_idx,
            buffers,
            fill_config,
        );
    }
    ruled_two_wire_once(
        face_key,
        face,
        wire_edges,
        edge_polygons,
        edge_boundary_idx,
        buffers,
        fill_config,
        None,
    )
}

/// Freeform two-wire patches: double ruled grid until chord error meets deflection goal.
fn try_ruled_two_wire_mesh_adaptive(
    face_key: FaceKey,
    face: &crate::topo::BRepFace,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &FaceEdgeBoundaryIdx,
    buffers: RuledMeshBuffers<'_>,
    fill_config: &FaceFillConfig,
) -> FaceMeshRange {
    let chord_goal = (fill_config.deflection_interior * 2.0).max(0.02);
    let min_tris_goal = if matches!(face.surface, SurfaceGeom::Offset { .. }) {
        min_adequate_trim_tris(&face.surface, wire_edges.len()).saturating_mul(8)
    } else {
        0
    };
    let (mut segs_u, mut segs_v) = initial_ruled_segs(face, wire_edges, edge_polygons, fill_config);
    let base_tri = buffers.all_indices.len() / 4;
    let mut last = FaceMeshRange {
        face_key,
        first_tri: base_tri,
        tri_count: 0,
        boundary_global: HashSet::new(),
        max_chord_error: 0.0,
        cdt_constraint_failures: 0,
    };

    for pass in 0..MAX_ADAPTIVE_RULED_PASSES {
        if pass > 0 {
            buffers.all_indices.truncate(base_tri * 4);
        }
        last = ruled_two_wire_once(
            face_key,
            face,
            wire_edges,
            edge_polygons,
            edge_boundary_idx,
            RuledMeshBuffers {
                global_vertices: buffers.global_vertices,
                global_normals: buffers.global_normals,
                all_indices: buffers.all_indices,
                pos_to_idx: buffers.pos_to_idx,
                shared_boundary: buffers.shared_boundary,
            },
            fill_config,
            Some((segs_u, segs_v)),
        );
        if last.tri_count == 0 {
            break;
        }
        last.max_chord_error = measure_face_chord_error(
            face,
            buffers.global_vertices,
            buffers.all_indices,
            &last,
        );
        log::debug!(
            "[BRep mesh] face {:?}: adaptive ruled pass {} segs={}/{} tris={} chord={:.4}",
            face_key,
            pass + 1,
            segs_u,
            segs_v,
            last.tri_count,
            last.max_chord_error,
        );
        let chord_ok = last.max_chord_error <= chord_goal;
        let tris_ok = min_tris_goal == 0 || last.tri_count >= min_tris_goal;
        let done = match &face.surface {
            // Offset+Revolution: chord reports are unreliable; refine until tri budget met.
            SurfaceGeom::Offset { .. } => tris_ok,
            _ => chord_ok && tris_ok,
        };
        if done || pass + 1 == MAX_ADAPTIVE_RULED_PASSES {
            break;
        }
        segs_u = (segs_u * 2).min(128);
        segs_v = (segs_v * 2).min(64);
    }

    last
}

fn initial_ruled_segs(
    face: &crate::topo::BRepFace,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    fill_config: &FaceFillConfig,
) -> (u32, u32) {
    let (mut segs_u, mut segs_v) = ruled_strip_uv_segs(wire_edges, edge_polygons, fill_config);
    let base = parametric_grid_segs(face, fill_config);
    segs_u = segs_u.max(base).max(16).min(64);
    segs_v = segs_v.max(8).min(32);
    if matches!(face.surface, SurfaceGeom::Offset { .. }) {
        segs_u = segs_u.max(32);
        segs_v = segs_v.max(16);
    }
    (segs_u, segs_v)
}

fn ruled_two_wire_once(
    face_key: FaceKey,
    face: &crate::topo::BRepFace,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &FaceEdgeBoundaryIdx,
    buffers: RuledMeshBuffers<'_>,
    fill_config: &FaceFillConfig,
    segs_override: Option<(u32, u32)>,
) -> FaceMeshRange {
    let RuledMeshBuffers {
        global_vertices,
        global_normals,
        all_indices,
        pos_to_idx,
        shared_boundary,
    } = buffers;
    let first_tri = all_indices.len() / 4;
    if wire_edges.len() != 2 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
        };
    }

    let (segs_u, segs_v) = segs_override.unwrap_or_else(|| {
        initial_ruled_segs(face, wire_edges, edge_polygons, fill_config)
    });

    if prefers_native_uv_ruled(&face.surface, wire_edges) {
        return mesh_ruled_two_wire_edges(
            face_key,
            face,
            wire_edges,
            edge_polygons,
            edge_boundary_idx,
            global_vertices,
            global_normals,
            all_indices,
            pos_to_idx,
            segs_u,
            segs_v,
            shared_boundary,
        );
    }

    let we_orient: Vec<_> = wire_edges
        .iter()
        .map(|&(ek, ref pis)| (ek, Orientation::Forward, pis.clone()))
        .collect();

    let ruled = mesh_ruled_wire_polygons_3d(
        face_key,
        face,
        &we_orient,
        edge_polygons,
        global_vertices,
        global_normals,
        all_indices,
        segs_u,
        segs_v,
    );
    if ruled.tri_count > 0 {
        ruled
    } else {
        mesh_ruled_two_wire_edges(
            face_key,
            face,
            wire_edges,
            edge_polygons,
            edge_boundary_idx,
            global_vertices,
            global_normals,
            all_indices,
            pos_to_idx,
            segs_u,
            segs_v,
            shared_boundary,
        )
    }
}

fn sample_ruled_polyline(curve: &[PVec3], t: Real) -> PVec3 {
    if curve.is_empty() {
        return PVec3::ZERO;
    }
    if curve.len() == 1 {
        return curve[0];
    }
    let f = t * (curve.len() - 1) as Real;
    let i = f.floor() as usize;
    let j = (i + 1).min(curve.len() - 1);
    let u = f - i as Real;
    curve[i].lerp(curve[j], u)
}

fn ruled_strip_uv_segs(
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    config: &FaceFillConfig,
) -> (u32, u32) {
    let defl = config.deflection_interior.max(1e-6);
    let mut max_arc = 0.0_f64;
    let mut curves: [Vec<PVec3>; 2] = [Vec::new(), Vec::new()];
    for (side, &(ek, ref pis)) in wire_edges.iter().enumerate() {
        let Some(poly) = edge_polygons.get(&ek) else {
            continue;
        };
        let mut chain_wire = 0.0_f64;
        let mut prev: Option<PVec3> = None;
        for &pi in pis {
            let Some(&(_, pt)) = poly.params_3d.get(pi) else {
                continue;
            };
            if let Some(p) = prev {
                chain_wire += (pt - p).length();
            }
            prev = Some(pt);
            if curves[side]
                .last()
                .map(|q| (*q - pt).length_squared() > 1e-14)
                .unwrap_or(true)
            {
                curves[side].push(pt);
            }
        }
        let mut chain_full = 0.0_f64;
        for w in poly.params_3d.windows(2) {
            chain_full += (w[1].1 - w[0].1).length();
        }
        if curves[side].len() < 2 && poly.params_3d.len() >= 2 {
            curves[side].clear();
            for &(_, pt) in &poly.params_3d {
                if curves[side]
                    .last()
                    .map(|q| (*q - pt).length_squared() > 1e-14)
                    .unwrap_or(true)
                {
                    curves[side].push(pt);
                }
            }
        }
        max_arc = max_arc.max(chain_wire.max(chain_full));
    }
    let segs_u = ((max_arc / (defl * 0.85)).ceil() as u32).clamp(8, 96);
    let mut max_width = 0.0_f64;
    if curves[0].len() >= 2 && curves[1].len() >= 2 {
        for i in 0..=16 {
            let t = i as Real / 16.0;
            let p0 = sample_ruled_polyline(&curves[0], t);
            let p1 = sample_ruled_polyline(&curves[1], t);
            max_width = max_width.max((p1 - p0).length());
        }
    }
    let segs_v = ((max_width / (defl * 0.85)).ceil() as u32).clamp(4, 48);
    (segs_u, segs_v)
}

pub fn sample_polyline(indices: &[usize], verts: &[PVec3], t: Real) -> PVec3 {
    if indices.is_empty() {
        return PVec3::ZERO;
    }
    if indices.len() == 1 {
        return verts[indices[0]];
    }
    let f = t * (indices.len() - 1) as Real;
    let i = f.floor() as usize;
    let j = (i + 1).min(indices.len() - 1);
    let u = f - i as Real;
    verts[indices[i]].lerp(verts[indices[j]], u)
}

/// Ruled quad strip in surface (u,v): OCC-style ruled patch, not 3D linear blend.
pub fn mesh_ruled_two_wire_edges(
    face_key: FaceKey,
    face: &BRepFace,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &FaceEdgeBoundaryIdx,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut BoundaryPosIndex,
    segs_u: u32,
    segs_v: u32,
    shared_boundary: Option<&SharedBoundaryPool>,
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
            cdt_constraint_failures: 0,
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
            cdt_constraint_failures: 0,
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
                let s = j as Real / nv as Real;
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
                    let t = i as Real / nu as Real;
                    let p0 = sample_polyline(&curves[0], global_vertices, t);
                    let p1 = sample_polyline(&curves[1], global_vertices, t);
                    let pt = p0 * (1.0 - s) + p1 * s;
                    (pt, PVec3::Z)
                };
                let gi = if use_native_uv {
                    let idx = global_vertices.len();
                    global_vertices.push(pt);
                    global_normals.push(n);
                    idx
                } else {
                    register_boundary_point_with_normal_indexed_shared(
                        pt,
                        global_vertices,
                        global_normals,
                        pos_to_idx,
                        shared_boundary,
                        || n,
                    )
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
        cdt_constraint_failures: 0,
    }
}

/// Ruled strip using raw edge polygon 3D samples (avoids boundary pool dedup collapsing wires).
pub fn mesh_ruled_wire_polygons_3d(
    face_key: FaceKey,
    face: &BRepFace,
    wire_edges: &[(EdgeKey, Orientation, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
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
            cdt_constraint_failures: 0,
        };
    }
    let mut curves: [Vec<PVec3>; 2] = [Vec::new(), Vec::new()];
    for (side, &(ek, _orient, ref pis)) in wire_edges.iter().enumerate() {
        let Some(poly) = edge_polygons.get(&ek) else {
            return FaceMeshRange {
                face_key,
                first_tri,
                tri_count: 0,
                boundary_global: HashSet::new(),
                max_chord_error: 0.0,
                cdt_constraint_failures: 0,
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
            cdt_constraint_failures: 0,
        };
    }
    let sample_curve = |c: &[PVec3], t: Real| -> PVec3 {
        if c.len() == 1 {
            return c[0];
        }
        let f = t * (c.len() - 1) as Real;
        let i = f.floor() as usize;
        let j = (i + 1).min(c.len() - 1);
        let u = f - i as Real;
        c[i].lerp(c[j], u)
    };
    // If the two curves are nearly coincident, the ruled strip has zero width -> all degenerate.
    let n_samples = curves[0].len().max(curves[1].len()).max(2);
    let max_separation = (0..n_samples)
        .map(|i| {
            let t = i as Real / (n_samples - 1) as Real;
            let a = sample_curve(&curves[0], t);
            let b = sample_curve(&curves[1], t);
            (a - b).length_squared()
        })
        .fold(0.0_f64, Real::max);
    if max_separation < 1e-6 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
        };
    }
    let nu = segs_u.max(2) as usize;
    let nv = segs_v.max(2) as usize;
    let mut grid: Vec<Vec<usize>> = vec![vec![0; nv + 1]; nu + 1];
    for i in 0..=nu {
        let t = i as Real / nu as Real;
        let p0 = sample_curve(&curves[0], t);
        let p1 = sample_curve(&curves[1], t);
        for j in 0..=nv {
            let s = j as Real / nv as Real;
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
                n = PVec3::Y;
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
        cdt_constraint_failures: 0,
    };
    if tri_count > 0 {
        range.max_chord_error =
            measure_face_chord_error(face, global_vertices, all_indices, &range);
    }
    range
}
