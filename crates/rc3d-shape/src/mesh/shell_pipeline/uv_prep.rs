use std::collections::HashMap;

use rc3d_core::math::{Real, PVec3};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey};
use crate::mesh::edge_disc::EdgePolygon;
use crate::mesh::face_uv::collect_face_loops;
use super::types::{FaceLoopData, FaceWireInfo};
use rayon::prelude::*;

pub(crate) fn precompute_face_uv_loops(
    face_infos: &[FaceWireInfo],
    reg: &BRepStore,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &HashMap<(FaceKey, EdgeKey, usize), usize>,
    global_vertices: &[PVec3],
) -> Vec<FaceLoopData> {
    face_infos
        .par_iter()
        .filter_map(|finfo| {
            let face = reg.faces.get(finfo.face_key)?;
            if finfo.wire_edges.is_empty() {
                return Some(FaceLoopData {
                    face_key: finfo.face_key,
                    wire_edges: finfo.wire_edges.clone(),
                    loops: None,
                });
            }
            let loops = collect_face_loops(
                finfo.face_key,
                face,
                reg,
                edge_polygons,
                edge_boundary_idx,
                global_vertices,
            );
            Some(FaceLoopData {
                face_key: finfo.face_key,
                wire_edges: finfo.wire_edges.clone(),
                loops: Some(loops),
            })
        })
        .collect()
}
