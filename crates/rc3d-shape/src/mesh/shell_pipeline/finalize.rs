use std::collections::{HashMap, HashSet};

use rc3d_core::math::{Real, PVec3};
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
    mut global_vertices: Vec<PVec3>,
    mut global_normals: Vec<PVec3>,
    mut report: ShellMeshReport,
    heal_skipped_faces: &[FaceKey],
    scaled_config: &BRepMeshConfig,
    shell_diag: Real,
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
                range.face_key,
                reg,
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

    let protected_positions: HashSet<[u64; 3]> = protected
        .iter()
        .filter_map(|&i| mesh.vertices.get(i))
        .map(|v| rc3d_core::utils::hash::f64x3_quantized_bits([v.x, v.y, v.z]))
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
                let key = rc3d_core::utils::hash::f64x3_quantized_bits([v.x, v.y, v.z]);
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

    // ── ModelHealer: repair gaps between adjacent face meshes ──
    let weld_tol_for_heal = scaled_config
        .weld_tolerance
        .max(shell_diag * 1e-5)
        .max(1e-6);
    // Build per-face sub-meshes from face_ranges
    let mut face_sub_meshes: Vec<(MeshResult, usize)> = Vec::new();
    for (fi, range) in face_ranges.iter().enumerate() {
        let tri_start = range.first_tri;
        let tri_end = tri_start + range.tri_count;
        if tri_start >= mesh.indices.len() / 4 || tri_end > mesh.indices.len() / 4 {
            continue;
        }
        // Collect unique vertex indices used by this face
        let mut global_to_local: HashMap<usize, usize> = HashMap::new();
        let mut local_verts: Vec<PVec3> = Vec::new();
        let mut local_norms: Vec<PVec3> = Vec::new();
        for tri_idx in tri_start..tri_end {
            let base = tri_idx * 4;
            for j in 0..3 {
                let gi = mesh.indices[base + j] as usize;
                if gi < mesh.vertices.len() && !global_to_local.contains_key(&gi) {
                    global_to_local.insert(gi, local_verts.len());
                    local_verts.push(mesh.vertices[gi]);
                    local_norms.push(mesh.normals[gi]);
                }
            }
        }
        let mut local_indices: Vec<i32> = Vec::new();
        for tri_idx in tri_start..tri_end {
            let base = tri_idx * 4;
            for j in 0..3 {
                let gi = mesh.indices[base + j] as usize;
                if let Some(&li) = global_to_local.get(&gi) {
                    local_indices.push(li as i32);
                }
            }
            local_indices.push(-1);
        }
        face_sub_meshes.push((
            MeshResult {
                vertices: local_verts,
                indices: local_indices,
                normals: local_norms,
            },
            fi,
        ));
    }

    let gap_welded = if face_sub_meshes.len() >= 2 {
        let mut meshes_with_ids: Vec<(&mut MeshResult, usize)> = face_sub_meshes
            .iter_mut()
            .map(|(m, id)| (m, *id))
            .collect();
        crate::mesh::post_process::heal_mesh_gaps(&mut meshes_with_ids, weld_tol_for_heal)
    } else {
        0
    };

    // Rebuild combined mesh from healed per-face meshes.
    // After heal_mesh_gaps welds boundary vertices across sub-meshes, the
    // same 3D position may appear in multiple sub-mesh vertex buffers.
    // Deduplicate by spatial position during reassembly to avoid seam duplicates.
    if gap_welded > 0 {
        let mut new_vertices: Vec<PVec3> = Vec::new();
        let mut new_indices: Vec<i32> = Vec::new();
        let mut new_normals: Vec<PVec3> = Vec::new();
        // Spatial dedup map: quantized position → vertex index in combined mesh
        let weld_tol = weld_tol_for_heal.max(1e-6);
        let inv_tol = 1.0 / weld_tol;
        let mut pos_to_idx: std::collections::HashMap<(i32, i32, i32), usize> =
            std::collections::HashMap::new();

        for (face_mesh, _) in &face_sub_meshes {
            // Build local→global index mapping with deduplication
            let mut local_to_global: Vec<i32> = Vec::with_capacity(face_mesh.vertices.len());
            for (vi, &v) in face_mesh.vertices.iter().enumerate() {
                let key = (
                    (v.x * inv_tol).round() as i32,
                    (v.y * inv_tol).round() as i32,
                    (v.z * inv_tol).round() as i32,
                );
                if let Some(&gi) = pos_to_idx.get(&key) {
                    local_to_global.push(gi as i32);
                } else {
                    let gi = new_vertices.len();
                    pos_to_idx.insert(key, gi);
                    new_vertices.push(v);
                    if vi < face_mesh.normals.len() {
                        new_normals.push(face_mesh.normals[vi]);
                    } else {
                        new_normals.push(PVec3::Z);
                    }
                    local_to_global.push(gi as i32);
                }
            }
            // Remap indices
            for chunk in face_mesh.indices.chunks(4) {
                for &idx in chunk {
                    new_indices.push(if idx >= 0 {
                        *local_to_global.get(idx as usize).unwrap_or(&0) as i32
                    } else {
                        idx
                    });
                }
            }
        }
        mesh.vertices = new_vertices;
        mesh.indices = new_indices;
        mesh.normals = new_normals;
        report.total_tris = mesh.indices.len() / 4;
    }

    // OCC BRepMesh_ModelHealer: fix T-junctions on boundary edges
    let t_junctions = crate::mesh::post_process::fix_t_junctions(&mut mesh, weld_tol_for_heal);
    report.gap_welded = gap_welded;
    report.t_junctions_fixed = t_junctions;

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
    // Only Plane is exactly representable by triangles; all curved surfaces
    // (Cylinder, Cone, Sphere, Torus, Revolution, Extrusion, BSpline)
    // need chord error measurement to enforce deflection quality.
    matches!(surface, SurfaceGeom::Plane { .. })
}
