//! Geometry compilation layer.

use rc3d_core::Aabb;
use rc3d_mesh::topology::{EdgeId, TriangleMesh};

use crate::shape::Shape;

/// Result of geometry validation.
#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub vertex_count: usize,
    pub face_count: usize,
    pub edge_count: usize,
    pub boundary_edge_count: usize,
    /// Indices of degenerate triangles (area near zero).
    pub degenerate_faces: Vec<usize>,
    /// Edges with more than 2 adjacent faces (non-manifold).
    pub non_manifold_edges: Vec<EdgeId>,
    /// True if no issues found.
    pub is_valid: bool,
}

/// Compiled geometry wrapping a `TriangleMesh`.
///
/// Created via `Geometry::compile(shape)` or `Geometry::from_mesh(triangle_mesh)`.
#[derive(Clone, Debug)]
pub struct Geometry {
    mesh: TriangleMesh,
}

impl Geometry {
    /// Compile a shape into a topologically-rich mesh.
    pub fn compile(shape: &dyn Shape) -> Self {
        Self { mesh: shape.compile() }
    }

    /// Wrap an existing TriangleMesh.
    pub fn from_mesh(mesh: TriangleMesh) -> Self {
        Self { mesh }
    }

    /// Validate the mesh geometry.
    pub fn validate(&self) -> ValidationResult {
        let boundary_count = self.mesh.boundary_edges().len();
        let degenerate_faces: Vec<usize> = self
            .mesh
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                let v = f.vertices;
                let p0 = self.mesh.positions[v[0] as usize];
                let p1 = self.mesh.positions[v[1] as usize];
                let p2 = self.mesh.positions[v[2] as usize];
                (p1 - p0).cross(p2 - p0).length() < 1e-10
            })
            .map(|(i, _)| i)
            .collect();

        // Count edges with more than 2 faces
        let mut face_count_per_edge: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for face in &self.mesh.faces {
            for &eid in &face.edges {
                *face_count_per_edge.entry(eid.0).or_insert(0) += 1;
            }
        }
        let non_manifold: Vec<EdgeId> = face_count_per_edge
            .iter()
            .filter(|(_, &c)| c > 2)
            .map(|(&k, _)| EdgeId(k))
            .collect();

        let is_valid = degenerate_faces.is_empty() && non_manifold.is_empty();

        ValidationResult {
            vertex_count: self.mesh.positions.len(),
            face_count: self.mesh.faces.len(),
            edge_count: self.mesh.edges.len(),
            boundary_edge_count: boundary_count,
            degenerate_faces,
            non_manifold_edges: non_manifold,
            is_valid,
        }
    }

    /// Bounding box in local space.
    pub fn aabb(&self) -> Aabb {
        self.mesh.bounding_box()
    }

    /// Number of faces.
    pub fn face_count(&self) -> usize {
        self.mesh.faces.len()
    }

    /// List of boundary edge IDs.
    pub fn boundary_edges(&self) -> Vec<EdgeId> {
        self.mesh.boundary_edges()
    }

    /// Raw triangle buffers for GPU: `(positions, indices)`.
    pub fn triangle_buffers(&self) -> (Vec<[f32; 3]>, Vec<u32>) {
        self.mesh.triangle_buffers()
    }

    /// Interleaved PBR buffers for GPU: `(interleaved_vertices, indices)`.
    pub fn phong_buffers(&self) -> (Vec<[f32; 12]>, Vec<u32>) {
        self.mesh.phong_buffers()
    }

    /// Access the inner TriangleMesh.
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{Cube, Sphere};

    #[test]
    fn test_compile_cube_geometry() {
        let g = Geometry::compile(&Cube::default());
        let v = g.validate();
        assert_eq!(v.face_count, 12); // 6 faces × 2 triangles
        assert!(v.degenerate_faces.is_empty());
        assert!(v.is_valid);
    }

    #[test]
    fn test_compile_sphere_geometry() {
        let g = Geometry::compile(&Sphere::default());
        assert!(g.face_count() > 0);
    }

    #[test]
    fn test_cube_aabb() {
        let g = Geometry::compile(&Cube::default());
        let aabb = g.aabb();
        assert_eq!(aabb.min, (-0.5, -0.5, -0.5).into());
        assert_eq!(aabb.max, (0.5, 0.5, 0.5).into());
    }

    #[test]
    fn test_triangle_buffers() {
        let g = Geometry::compile(&Cube::default());
        let (positions, indices) = g.triangle_buffers();
        assert!(!positions.is_empty());
        assert!(!indices.is_empty());
    }
}
