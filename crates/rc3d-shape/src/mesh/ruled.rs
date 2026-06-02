use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;
use crate::topo::{EdgeKey, FaceKey, Orientation};
use super::edge_disc::{EdgePolygon};
use super::face_fill::{
    FaceFillConfig, FaceMeshRange, mesh_ruled_two_wire_edges, mesh_ruled_wire_polygons_3d,
    prefers_native_uv_ruled,
};
use super::grid::parametric_grid_segs;

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
    edge_boundary_idx: &HashMap<(EdgeKey, usize), usize>,
    buffers: RuledMeshBuffers<'_>,
    fill_config: &FaceFillConfig,
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

    let (mut segs_u, mut segs_v) = ruled_strip_uv_segs(wire_edges, edge_polygons, fill_config);
    let base = parametric_grid_segs(face, fill_config);
    segs_u = segs_u.max(base).min(64);
    segs_v = segs_v.max(2).min(32);

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
