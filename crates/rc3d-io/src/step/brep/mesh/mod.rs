pub mod edge_disc;
pub mod face_tri;
pub mod refiner;
pub mod optimize;

use super::topo::ShellKey;
use super::topo::SolidKey;
use super::registry::BRepRegistry;
use crate::step::tessellate::MeshResult;

#[derive(Debug, Clone)]
pub struct BRepMeshConfig {
    pub edge_deflection: f32,
    pub edge_angle_deflection: f32,
    pub face_deflection: f32,
    pub min_angle_degrees: f32,
    pub refine_iterations: usize,
    pub optimize_iterations: usize,
}

impl Default for BRepMeshConfig {
    fn default() -> Self {
        Self {
            edge_deflection: 0.1,
            edge_angle_deflection: 0.1,
            face_deflection: 0.05,
            min_angle_degrees: 15.0,
            refine_iterations: 5,
            optimize_iterations: 3,
        }
    }
}

pub fn mesh_brep_shell(_shell_key: ShellKey, _reg: &BRepRegistry, _config: &BRepMeshConfig) -> MeshResult {
    MeshResult::default() // TODO: implement after T2.x
}

pub fn mesh_brep_solid(_solid_key: SolidKey, _reg: &BRepRegistry, _config: &BRepMeshConfig) -> MeshResult {
    MeshResult::default()
}
