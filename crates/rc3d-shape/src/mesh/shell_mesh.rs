//! Shell-level B-Rep tessellation entry points.

use super::shell_impl::mesh_brep_shell_with_report_impl;
use super::config::BRepMeshConfig;
use super::report::ShellMeshReport;
use crate::store::BRepStore;
use crate::topo::{FaceKey, ShellKey};
use crate::mesh_result::MeshResult;

#[derive(Debug)]
pub struct ShellMeshOutput {
    pub mesh: MeshResult,
    pub report: ShellMeshReport,
}

/// Mesh a B-Rep shell (OCC BRepMesh_IncrementalMesh equivalent).
pub fn mesh_brep_shell(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    config: &BRepMeshConfig,
    skip_face_keys: &[FaceKey],
) -> MeshResult {
    mesh_brep_shell_with_report(shell_key, reg, config, skip_face_keys).mesh
}

pub fn mesh_brep_shell_with_report(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    config: &BRepMeshConfig,
    skip_face_keys: &[FaceKey],
) -> ShellMeshOutput {
    mesh_brep_shell_with_report_impl(shell_key, reg, config, skip_face_keys)
}
