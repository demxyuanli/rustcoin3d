use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;
use crate::geom::SurfaceGeom;
use crate::topo::{EdgeKey, FaceKey, Orientation};
use super::edge_disc::EdgePolygon;
use super::edge_pool::FaceEdgeBoundaryIdx;
use super::face_fill::{
    measure_face_chord_error, FaceFillConfig, FaceMeshRange, mesh_ruled_two_wire_edges,
    mesh_ruled_wire_polygons_3d, prefers_native_uv_ruled,
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
    pub global_vertices: &'a mut Vec<Vec3>,
    pub global_normals: &'a mut Vec<Vec3>,
    pub all_indices: &'a mut Vec<i32>,
    pub pos_to_idx: &'a mut HashMap<[u32; 3], usize>,
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
    } = buffers;
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
        )
    }
}

fn sample_ruled_polyline(curve: &[Vec3], t: f32) -> Vec3 {
    if curve.is_empty() {
        return Vec3::ZERO;
    }
    if curve.len() == 1 {
        return curve[0];
    }
    let f = t * (curve.len() - 1) as f32;
    let i = f.floor() as usize;
    let j = (i + 1).min(curve.len() - 1);
    let u = f - i as f32;
    curve[i].lerp(curve[j], u)
}

fn ruled_strip_uv_segs(
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    config: &FaceFillConfig,
) -> (u32, u32) {
    let defl = config.deflection_interior.max(1e-6);
    let mut max_arc = 0.0f32;
    let mut curves: [Vec<Vec3>; 2] = [Vec::new(), Vec::new()];
    for (side, &(ek, ref pis)) in wire_edges.iter().enumerate() {
        let Some(poly) = edge_polygons.get(&ek) else {
            continue;
        };
        let mut chain_wire = 0.0f32;
        let mut prev: Option<Vec3> = None;
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
        let mut chain_full = 0.0f32;
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
    let mut max_width = 0.0f32;
    if curves[0].len() >= 2 && curves[1].len() >= 2 {
        for i in 0..=16 {
            let t = i as f32 / 16.0;
            let p0 = sample_ruled_polyline(&curves[0], t);
            let p1 = sample_ruled_polyline(&curves[1], t);
            max_width = max_width.max((p1 - p0).length());
        }
    }
    let segs_v = ((max_width / (defl * 0.85)).ceil() as u32).clamp(4, 48);
    (segs_u, segs_v)
}
