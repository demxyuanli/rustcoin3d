use rc3d_core::math::PVec3;
use crate::mesh::face_fill::FaceMeshRange;
use crate::mesh::report::ShellMeshReport;
use super::types::ChunkOutput;

pub(crate) fn merge_chunk_outputs(
    chunk_outputs: Vec<ChunkOutput>,
    boundary_vertex_count: usize,
    collect_diag: bool,
) -> (Vec<PVec3>, Vec<PVec3>, Vec<i32>, Vec<FaceMeshRange>, ShellMeshReport, Vec<String>) {
    if chunk_outputs.is_empty() {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            ShellMeshReport::default(),
            Vec::new(),
        );
    }

    let base = &chunk_outputs[0];
    let mut global_vertices = base.vertices.clone();
    let mut global_normals = base.normals.clone();
    let mut all_indices = base.indices.clone();
    let mut face_ranges = base.face_ranges.clone();
    let mut report = base.report.clone();
    let mut wire_diag = base.wire_diag.clone();

    for chunk in &chunk_outputs[1..] {
        let interior_offset = global_vertices.len() as i32;
        let chunk_interior_start = boundary_vertex_count;

        global_vertices.extend_from_slice(&chunk.vertices[chunk_interior_start..]);
        global_normals.extend_from_slice(&chunk.normals[chunk_interior_start..]);

        let index_base = all_indices.len() / 4;
        for &idx in &chunk.indices {
            if idx < 0 {
                all_indices.push(idx);
            } else if (idx as usize) < boundary_vertex_count {
                all_indices.push(idx);
            } else {
                all_indices.push(idx + interior_offset - boundary_vertex_count as i32);
            }
        }

        for mut fr in chunk.face_ranges.clone() {
            fr.first_tri += index_base;
            face_ranges.push(fr);
        }

        report.face_count += chunk.report.face_count;
        report.meshed_faces += chunk.report.meshed_faces;
        report.grid_fallback_count += chunk.report.grid_fallback_count;
        report.cdt_constraint_failure_count += chunk.report.cdt_constraint_failure_count;
        report.total_tris += chunk.report.total_tris;
        report.max_equiv_edge_weld_gap = report
            .max_equiv_edge_weld_gap
            .max(chunk.report.max_equiv_edge_weld_gap);
        report.faces.extend_from_slice(&chunk.report.faces);

        if collect_diag {
            wire_diag.extend_from_slice(&chunk.wire_diag);
        }
    }

    (
        global_vertices,
        global_normals,
        all_indices,
        face_ranges,
        report,
        wire_diag,
    )
}
