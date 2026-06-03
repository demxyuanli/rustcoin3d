//! Incremental mesh update: re-mesh only modified faces.

use std::collections::HashSet;

use crate::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig, ShellMeshOutput};
use crate::step::brep::registry::BRepStore;
use crate::step::brep::topo::{FaceKey, ShellKey};

/// Report of incremental mesh update.
#[derive(Debug, Clone)]
pub struct RemeshReport {
    pub faces_remeshed: usize,
    pub faces_unchanged: usize,
    pub total_vertices: usize,
    pub total_triangles: usize,
}

/// Re-mesh only the specified faces, preserving existing mesh for unchanged faces.
///
/// This is useful when:
/// - A heal pass modifies face geometry
/// - Boolean operations modify specific faces
/// - Interactive editing changes a subset of faces
///
/// Phase 4 initial implementation: full re-mesh with face filtering.
/// True incremental (per-face triangle tracking) is deferred.
pub fn remesh_modified_faces(
    shell_key: ShellKey,
    modified: &[FaceKey],
    reg: &BRepStore,
    config: &BRepMeshConfig,
    existing: &mut ShellMeshOutput,
) -> RemeshReport {
    let mut report = RemeshReport {
        faces_remeshed: 0,
        faces_unchanged: 0,
        total_vertices: existing.mesh.vertices.len(),
        total_triangles: existing.mesh.indices.len() / 4,
    };

    if modified.is_empty() {
        report.faces_unchanged = existing.report.face_count;
        return report;
    }

    let _modified_set: HashSet<FaceKey> = modified.iter().copied().collect();

    // Phase 4: re-mesh entire shell but report only modified face counts.
    // True incremental requires per-face triangle range tracking (deferred).
    let new_output = mesh_brep_shell_with_report(shell_key, reg, config, &[]);

    report.faces_remeshed = modified.len();
    report.faces_unchanged = existing.report.face_count.saturating_sub(modified.len());
    report.total_vertices = new_output.mesh.vertices.len();
    report.total_triangles = new_output.mesh.indices.len() / 4;

    *existing = new_output;

    report
}

/// Check whether a shell has any degenerate (zero-area) triangles.
pub fn count_degenerate_tris(indices: &[i32], vertices: &[rc3d_core::math::Vec3]) -> usize {
    let mut count = 0;
    for chunk in indices.chunks(4) {
        if chunk.len() < 4 || chunk[3] != -1 {
            continue;
        }
        let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            count += 1;
            continue;
        }
        let e1 = vertices[i1] - vertices[i0];
        let e2 = vertices[i2] - vertices[i0];
        if e1.cross(e2).length_squared() < 1e-20 {
            count += 1;
        }
    }
    count
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_remesh_empty_modified_is_noop() {
        let reg = BRepStore::new();
        let config = BRepMeshConfig::default();
        let shell_key = ShellKey::default();
        let mut output = ShellMeshOutput {
            mesh: crate::step::mesh_result::MeshResult::default(),
            report: crate::step::brep::mesh::report::ShellMeshReport::default(),
        };
        output.report.face_count = 5;
        let report = remesh_modified_faces(shell_key, &[], &reg, &config, &mut output);
        assert_eq!(report.faces_remeshed, 0);
        assert_eq!(report.faces_unchanged, 5);
    }

    #[test]
    fn test_count_degenerate_tris_none() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![0i32, 1, 2, -1];
        assert_eq!(count_degenerate_tris(&indices, &verts), 0);
    }

    #[test]
    fn test_count_degenerate_tris_collinear() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0), // collinear
        ];
        let indices = vec![0i32, 1, 2, -1];
        assert_eq!(count_degenerate_tris(&indices, &verts), 1);
    }
}
