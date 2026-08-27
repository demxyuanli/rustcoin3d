//! Multi-geometry GPU batch (three.js BatchedMesh analog).

use rc3d_core::math::Mat4;
use rc3d_mesh::TriangleMesh;
use rc3d_scene::node_data::{BatchedMeshNode, NodeData};

/// Builder that packs distinct meshes into one [`BatchedMeshNode`].
#[derive(Clone, Debug, Default)]
pub struct BatchedMesh {
    node: BatchedMeshNode,
}

impl BatchedMesh {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pack a triangle mesh and return its geometry id.
    pub fn add_mesh(&mut self, mesh: &TriangleMesh) -> u32 {
        let mut mesh = mesh.clone();
        mesh.compute_tangents();
        let (phong, indices) = mesh.phong_buffers();
        let mut positions = Vec::with_capacity(phong.len());
        let mut normals = Vec::with_capacity(phong.len());
        let mut texcoords = Vec::with_capacity(phong.len());
        let mut tangents = Vec::with_capacity(phong.len());
        for v in &phong {
            positions.push([v[0], v[1], v[2]]);
            normals.push([v[3], v[4], v[5]]);
            texcoords.push([v[6], v[7]]);
            tangents.push([v[8], v[9], v[10], v[11]]);
        }
        self.node
            .add_geometry(&positions, &normals, &texcoords, &tangents, &indices)
    }

    pub fn add_cube(&mut self, width: f32, height: f32, depth: f32) -> u32 {
        self.add_mesh(&rc3d_mesh::tessellate_cube(width, height, depth))
    }

    pub fn add_sphere(&mut self, radius: f32) -> u32 {
        self.add_mesh(&rc3d_mesh::tessellate_sphere(radius, 24, 16))
    }

    /// Instance `geometry` at `transform`. Returns the instance id.
    pub fn add_instance(&mut self, geometry: u32, transform: Mat4) -> u32 {
        self.node
            .add_instance(geometry, transform.to_cols_array_2d())
    }

    pub fn set_instance_color(&mut self, instance: u32, color: [f32; 4]) {
        if let Some(inst) = self.node.instances.get_mut(instance as usize) {
            inst.color = color;
        }
    }

    pub fn set_instance_visible(&mut self, instance: u32, visible: bool) {
        if let Some(inst) = self.node.instances.get_mut(instance as usize) {
            inst.visible = visible;
        }
    }

    pub fn into_node(self) -> NodeData {
        NodeData::BatchedMesh(self.node)
    }
}
