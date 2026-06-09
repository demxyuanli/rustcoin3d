use std::collections::HashSet;

use rc3d_core::math::Vec3;
use crate::geom::SurfaceGeom;
use crate::mesh_result::MeshResult;
use crate::store::BRepStore;
use crate::topo::{FaceKey, ShellKey};
use crate::mesh::config::BRepMeshConfig;
use crate::mesh::diagnostic::log_mesh_coordinates_if_requested;
use crate::mesh::face_fill::{measure_face_chord_error, FaceMeshRange};
use crate::mesh::face_uv::UvSource;
use crate::mesh::optimize::optimize_mesh;
use crate::mesh::post_process::{
    compact_mesh_vertices, cull_degenerate_tris, recompute_normals_from_tris_preserving,
};
use crate::mesh::refiner::{extract_face_mesh_with_map, merge_refined_face, refine_mesh_interior};
use crate::mesh::report::{FaceMeshStats, ShellMeshReport};
use crate::mesh::shell_mesh::ShellMeshOutput;

pub(crate) fn finalize_shell_mesh(
    shell_key: ShellKey,
    reg: &BRepStore,
    face_ranges: &[FaceMeshRange],
    mut all_indices: Vec<i32>,
    mut global_vertices: Vec<Vec3>,
    mut global_normals: Vec<Vec3>,
    mut report: ShellMeshReport,
    heal_skipped_faces: &[FaceKey],
    scaled_config: &BRepMeshConfig,
    shell_diag: f32,
    face_infos_len: usize,
    boundary_vertex_count: usize,
    wire_diag: &[String],
    collect_diag: bool,
) -> ShellMeshOutput {
    for face_key in heal_skipped_faces {
        report.faces.push(FaceMeshStats {
            face_key: *face_key,
            tri_count: 0,
            first_tri: 0,
            uv_source: UvSource::SurfaceFill,
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
            grid_fallback: false,
        });
        log::debug!("[BRep mesh] face {:?} skipped by heal (no grid fallback)", face_key);
    }

    if scaled_config.refine.enable_post_refine && scaled_config.refine.max_iterations > 0 {
        let mut tri_offset: isize = 0;
        for range in face_ranges {
            let face = match reg.faces.get(range.face_key) {
                Some(f) => f,
                None => continue,
            };
            if surface_chord_error_is_trivial(&face.surface) {
                continue;
            }
            let adj_start_raw = (range.first_tri as isize + tri_offset) * 4;
            if adj_start_raw < 0 {
                tri_offset -= adj_start_raw / 4;
                continue;
            }
            let adj_start = adj_start_raw as usize;
            let adj_end = adj_start + range.tri_count * 4;
            let (local_mesh, local_to_global) = extract_face_mesh_with_map(
                &global_vertices,
                &global_normals,
                &all_indices,
                adj_start,
                adj_end,
            );
            let local_boundary: HashSet<usize> = local_to_global
                .iter()
                .filter(|(_, &g)| range.boundary_global.contains(&g))
                .map(|(&l, _)| l)
                .collect();
            let refined = refine_mesh_interior(
                &local_mesh,
                face,
                &local_boundary,
                &scaled_config.refine,
            );
            let new_tri_count = merge_refined_face(
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                adj_start / 4,
                adj_end / 4,
                &refined,
                &local_to_global,
            );
            tri_offset += new_tri_count as isize - range.tri_count as isize;
        }
    }

    report.total_tris = all_indices.len() / 4;
    if !scaled_config.fast_export {
        for stats in &mut report.faces {
            if stats.grid_fallback || stats.tri_count == 0 {
                continue;
            }
            let Some(range) = face_ranges.iter().find(|r| r.face_key == stats.face_key) else {
                continue;
            };
            let Some(face) = reg.faces.get(stats.face_key) else {
                continue;
            };
            if surface_chord_error_is_trivial(&face.surface) {
                continue;
            }
            stats.max_chord_error =
                measure_face_chord_error(face, &global_vertices, &all_indices, range);
        }
    }
    report.log_summary(shell_key);

    log::debug!(
        "[BRep mesh] shell {:?}: {} boundary verts, {} faces, {} tris total",
        shell_key,
        global_vertices.len(),
        face_infos_len,
        report.total_tris
    );
    let mut mesh = MeshResult {
        vertices: global_vertices,
        indices: all_indices,
        normals: global_normals,
    };
    let culled = {
        let _c = cull_degenerate_tris(&mut mesh.indices, &mesh.vertices);
        log::info!(
            "[BRep mesh] culled {} degenerate tris, {} remain",
            _c,
            mesh.indices.len() / 4
        );
        _c
    };
    if culled > 0 {
        log::debug!("[BRep mesh] culled {culled} degenerate triangle(s)");
        report.total_tris = mesh.indices.len() / 4;
    }
    compact_mesh_vertices(&mut mesh);
    report.total_tris = mesh.indices.len() / 4;
    let weld_tol = scaled_config
        .weld_tolerance
        .max(shell_diag * 1e-5)
        .max(1e-6);

    let mut protected: HashSet<usize> = HashSet::new();
    for chunk in mesh.indices.chunks(4) {
        if chunk.len() < 3 {
            continue;
        }
        for &idx in &chunk[0..3] {
            if idx >= 0 {
                let i = idx as usize;
                if i < boundary_vertex_count {
                    protected.insert(i);
                }
            }
        }
    }
    for i in 0..boundary_vertex_count.min(mesh.vertices.len()) {
        protected.insert(i);
    }

    let protected_positions: HashSet<[u32; 3]> = protected
        .iter()
        .filter_map(|&i| mesh.vertices.get(i))
        .map(|v| rc3d_core::utils::hash::f32x3_quantized_bits([v.x, v.y, v.z]))
        .collect();

    let welded = mesh.weld_vertices_protected(weld_tol, &protected);
    log::debug!(
        "[BRep mesh] welded {} interior vertices ({} boundary protected)",
        welded,
        protected.len()
    );

    if welded > 0 {
        let preserve: Vec<usize> = mesh
            .vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| {
                let key = rc3d_core::utils::hash::f32x3_quantized_bits([v.x, v.y, v.z]);
                protected_positions.contains(&key)
            })
            .map(|(i, _)| i)
            .collect();
        recompute_normals_from_tris_preserving(
            &mesh.vertices,
            &mesh.indices,
            &mut mesh.normals,
            &preserve,
        );
    }
    optimize_mesh(&mut mesh, &scaled_config.optimize);
    if collect_diag {
        log_mesh_coordinates_if_requested(
            shell_key,
            reg,
            &mesh.vertices,
            &mesh.indices,
            face_ranges,
            &report,
            wire_diag,
        );
    }
    ShellMeshOutput { mesh, report }
}

fn surface_chord_error_is_trivial(surface: &SurfaceGeom) -> bool {
    matches!(
        surface,
        SurfaceGeom::Plane { .. }
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. }
            | SurfaceGeom::Sphere { .. }
            | SurfaceGeom::Torus { .. }
            | SurfaceGeom::Revolution { .. }
            | SurfaceGeom::Extrusion { .. }
    )
}
