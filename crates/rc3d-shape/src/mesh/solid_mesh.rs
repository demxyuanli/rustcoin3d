//! Solid-level mesh assembly (OCC `BREP_WITH_VOIDS` semantics).
//!
//! Tessellates outer shell (Forward) and appends void shells with Reversed
//! winding. No mesh-level boolean subtraction.

use std::collections::HashMap;

use crate::mesh_result::MeshResult;
use crate::mesh_split::FaceTriRange;
use crate::store::BRepStore;
use crate::topo::{FaceKey, SolidKey};

use super::config::BRepMeshConfig;
use super::report::ShellMeshReport;
use super::shell_mesh::mesh_brep_shell_with_report;

#[derive(Debug)]
pub struct SolidMeshOutput {
    pub mesh: MeshResult,
    pub face_tri_ranges: HashMap<FaceKey, FaceTriRange>,
    pub report: ShellMeshReport,
}

fn face_ranges_from_report(
    report: &ShellMeshReport,
    tri_offset: usize,
) -> HashMap<FaceKey, FaceTriRange> {
    report
        .faces
        .iter()
        .filter(|f| f.tri_count > 0)
        .map(|f| {
            (
                f.face_key,
                FaceTriRange {
                    first_tri: f.first_tri + tri_offset,
                    tri_count: f.tri_count,
                },
            )
        })
        .collect()
}

fn merge_shell_report(acc: &mut ShellMeshReport, shell: &ShellMeshReport) {
    acc.face_count += shell.face_count;
    acc.meshed_faces += shell.meshed_faces;
    acc.grid_fallback_count += shell.grid_fallback_count;
    acc.total_tris += shell.total_tris;
    acc.shell_diag = acc.shell_diag.max(shell.shell_diag);
    acc.max_equiv_edge_weld_gap = acc
        .max_equiv_edge_weld_gap
        .max(shell.max_equiv_edge_weld_gap);
    acc.faces.extend_from_slice(&shell.faces);
}

/// Tessellate a solid: outer shell + reversed void shells merged into one mesh.
pub fn mesh_solid_with_voids(
    store: &BRepStore,
    sk: SolidKey,
    config: &BRepMeshConfig,
    skip_faces: &[FaceKey],
) -> Option<SolidMeshOutput> {
    let solid = store.solids.get(sk)?;
    let outer_out = mesh_brep_shell_with_report(solid.outer_shell, store, config, skip_faces);
    if outer_out.mesh.vertices.is_empty() || outer_out.mesh.indices.is_empty() {
        return None;
    }

    let mut mesh = outer_out.mesh;
    let mut face_tri_ranges = face_ranges_from_report(&outer_out.report, 0);
    let mut report = outer_out.report;

    for &void_sk in &solid.void_shells {
        let void_out = mesh_brep_shell_with_report(void_sk, store, config, skip_faces);
        if void_out.mesh.vertices.is_empty() || void_out.mesh.indices.is_empty() {
            continue;
        }

        let tri_offset = mesh.indices.len() / 4;
        face_tri_ranges.extend(face_ranges_from_report(&void_out.report, tri_offset));

        let mut void_mesh = void_out.mesh;
        void_mesh.reverse_winding();
        mesh.append_from(&void_mesh);
        merge_shell_report(&mut report, &void_out.report);
    }
    // Weld once after all void shells are merged (avoids O(V²) per void)
    if !solid.void_shells.is_empty() {
        mesh.weld_vertices(config.weld_tolerance);
    }

    Some(SolidMeshOutput {
        mesh,
        face_tri_ranges,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;

    #[test]
    fn merge_appends_void_tris_with_reversed_winding() {
        let outer_mesh = MeshResult {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            indices: vec![0, 1, 2, -1],
        };
        let void_mesh = MeshResult {
            vertices: vec![
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
            ],
            normals: vec![Vec3::Z; 3],
            indices: vec![0, 1, 2, -1],
        };

        let mut combined = outer_mesh;
        let mut void_copy = void_mesh;
        void_copy.reverse_winding();
        combined.append_from(&void_copy);

        assert_eq!(combined.indices.len(), 8);
        assert_eq!(combined.indices[4..7], [3, 5, 4], "void tri winding reversed");
    }
}
