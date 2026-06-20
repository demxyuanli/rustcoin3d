use crate::mesh_result::MeshResult;
use crate::store::BRepStore;
use crate::topo::{FaceKey, ShellKey};
use crate::mesh::boundary::SharedBoundaryPool;
use crate::mesh::config::BRepMeshConfig;
use crate::mesh::diagnostic::diag_enabled;
use crate::mesh::model_preprocessor::preprocess_shell_wires;
use crate::mesh::report::{shell_bbox_diagonal, ShellMeshReport};
use crate::mesh::shell_mesh::ShellMeshOutput;
use super::boundary_pool::build_shell_boundary_pool;
use super::face_chunk::mesh_faces_in_chunk;
use super::finalize::finalize_shell_mesh;
use super::init::init_shell_mesh;
use super::merge::merge_chunk_outputs;
use super::types::{BoundaryPoolResult, ChunkOutput, FaceLoopData};
use super::uv_prep::precompute_face_uv_loops;
use rayon::prelude::*;

pub(crate) fn mesh_brep_shell_with_report_impl(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    config: &BRepMeshConfig,
    skip_face_keys: &[FaceKey],
) -> ShellMeshOutput {
    let Some((scaled_config, shell_diag, _report, edge_polygons)) =
        init_shell_mesh(shell_key, reg, config)
    else {
        return ShellMeshOutput {
            mesh: MeshResult::default(),
            report: ShellMeshReport {
                shell_diag: shell_bbox_diagonal(shell_key, reg),
                ..Default::default()
            },
        };
    };

    let BoundaryPoolResult {
        boundary_vertices,
        boundary_normals,
        boundary_pos_to_idx,
        boundary_vertex_count,
        edge_boundary_idx,
        face_infos,
        face_orient,
        heal_skipped_faces,
        global_vertices,
    } = build_shell_boundary_pool(
        shell_key,
        reg,
        &edge_polygons,
        skip_face_keys,
        reg.tolerance.boundary_dedup(&scaled_config),
    );

    let face_loop_data = precompute_face_uv_loops(
        &face_infos,
        reg,
        &edge_polygons,
        &edge_boundary_idx,
        &global_vertices,
    );

    // OCC BRepMesh_ModelPreProcessor: detect self-intersecting / open wires.
    // Warn only — never filter. Both flags can be false positives from UV
    // projection on curved surfaces. Filtering belongs at the app level.
    let _ = preprocess_shell_wires(shell_key, reg); // logs warnings, app decides on filtering

    let shared_boundary = SharedBoundaryPool::new(
        boundary_vertices,
        boundary_normals,
        boundary_pos_to_idx,
    );

    let collect_diag = diag_enabled();

    let _tp2 = std::time::Instant::now();
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = face_loop_data.len().div_ceil(num_threads);

    let chunk_outputs: Vec<ChunkOutput> = if face_loop_data.is_empty() {
        Vec::new()
    } else {
        face_loop_data
            .par_chunks(chunk_size)
            .map(|chunk: &[FaceLoopData]| {
                mesh_faces_in_chunk(
                    chunk,
                    reg,
                    &scaled_config,
                    shell_diag,
                    &edge_polygons,
                    &edge_boundary_idx,
                    &face_orient,
                    collect_diag,
                    &shared_boundary,
                )
            })
            .collect()
    };

    log::info!(
        "[mesh] face meshing ({} faces): {:.1}s",
        face_loop_data.len(),
        _tp2.elapsed().as_secs_f32()
    );

    let (global_vertices, global_normals, all_indices, face_ranges, report, wire_diag) =
        merge_chunk_outputs(chunk_outputs, boundary_vertex_count, collect_diag);

    let face_infos_len = face_infos.len();

    finalize_shell_mesh(
        shell_key,
        reg,
        &face_ranges,
        all_indices,
        global_vertices,
        global_normals,
        report,
        &heal_skipped_faces,
        &scaled_config,
        shell_diag,
        face_infos_len,
        boundary_vertex_count,
        &wire_diag,
        collect_diag,
    )
}
