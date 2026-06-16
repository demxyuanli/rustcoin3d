use std::collections::HashMap;

use rc3d_core::math::{Real, PVec3};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, ShellKey, VertexKey};
use crate::mesh::boundary::{register_boundary_point, BoundaryPosIndex};
use crate::mesh::edge_disc::EdgePolygon;
use crate::mesh::edge_pool::build_face_boundary_pool;
use super::types::{BoundaryPoolResult, FaceWireInfo};

pub(crate) fn build_shell_boundary_pool(
    shell_key: ShellKey,
    reg: &BRepStore,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    skip_face_keys: &[FaceKey],
    dedup_tolerance: Real,
) -> BoundaryPoolResult {
    let mut global_vertices: Vec<PVec3> = Vec::new();
    let mut global_normals: Vec<PVec3> = Vec::new();
    let mut pos_to_idx = BoundaryPosIndex::with_cell_size(dedup_tolerance);
    let mut vertex_mesh_idx: HashMap<VertexKey, usize> = HashMap::new();

    for (vk, v) in reg.vertices.iter() {
        let idx = register_boundary_point(
            v.position,
            dedup_tolerance,
            &mut global_vertices,
            &mut global_normals,
            &mut pos_to_idx,
        );
        vertex_mesh_idx.insert(vk, idx);
    }

    let total_edge_pts: usize = edge_polygons.values().map(|p| p.params_3d.len()).sum();
    let edge_boundary_idx = build_face_boundary_pool(
        reg,
        shell_key,
        edge_polygons,
        &mut global_vertices,
        &mut global_normals,
        &mut pos_to_idx,
        &vertex_mesh_idx,
    );
    log::debug!(
        "[BRep mesh] {} edge polygons, {} total edge pts -> {} unique boundary verts",
        edge_polygons.len(),
        total_edge_pts,
        global_vertices.len()
    );

    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => {
            return BoundaryPoolResult {
                boundary_vertices: global_vertices.clone(),
                boundary_normals: global_normals.clone(),
                boundary_pos_to_idx: pos_to_idx.clone(),
                boundary_vertex_count: global_vertices.len(),
                edge_boundary_idx,
                face_infos: Vec::new(),
                face_orient: HashMap::new(),
                heal_skipped_faces: Vec::new(),
                global_vertices,
            };
        }
    };

    let mut face_infos: Vec<FaceWireInfo> = Vec::new();
    let mut heal_skipped_faces: Vec<FaceKey> = Vec::new();

    let face_orient: HashMap<FaceKey, bool> = shell
        .faces
        .iter()
        .filter_map(|&(fk, orient)| {
            reg.faces
                .get(fk)
                .map(|f| (fk, crate::mesh::orient::effective_sense_raw(f.same_sense, orient)))
        })
        .collect();

    for &(face_key, _orient) in &shell.faces {
        if skip_face_keys.contains(&face_key) {
            heal_skipped_faces.push(face_key);
            continue;
        }
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wire = match reg.wires.get(face.outer_wire) {
            Some(w) => w,
            None => continue,
        };

        let mut wire_edges = Vec::new();
        for &(ek, orient) in &wire.edges {
            if let Some(poly) = edge_polygons.get(&ek) {
                let n_pts = poly.params_3d.len();
                if n_pts == 0 {
                    continue;
                }
                let mut indices: Vec<usize> = (0..n_pts).collect();
                if orient == Orientation::Reversed {
                    indices.reverse();
                }
                wire_edges.push((ek, indices));
            }
        }

        if !wire_edges.is_empty() || wire.edges.is_empty() {
            face_infos.push(FaceWireInfo {
                face_key,
                wire_edges,
            });
        }
    }

    let boundary_vertices: Vec<PVec3> = global_vertices.clone();
    let boundary_normals: Vec<PVec3> = global_normals.clone();
    let boundary_pos_to_idx = pos_to_idx.clone();
    let boundary_vertex_count = boundary_vertices.len();

    BoundaryPoolResult {
        boundary_vertices,
        boundary_normals,
        boundary_pos_to_idx,
        boundary_vertex_count,
        edge_boundary_idx,
        face_infos,
        face_orient,
        heal_skipped_faces,
        global_vertices,
    }
}
