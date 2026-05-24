use std::collections::HashMap;
use rc3d_core::math::Vec3;

/// A unique topological vertex shared across edges and faces.
/// Multiple STEP CARTESIAN_POINT entities at the same position
/// map to a single TopoVertex via spatial hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexId(pub u32);

#[derive(Debug, Clone)]
pub struct TopoVertex {
    pub id: VertexId,
    pub position: Vec3,
}

/// Registry for unique topological vertices.
/// Ensures one vertex per spatial position (within tolerance).
#[derive(Debug, Default)]
pub struct VertexRegistry {
    vertices: Vec<TopoVertex>,
    /// Spatial hash → vertex index
    index: HashMap<[u32; 3], u32>,
}

impl VertexRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a vertex position, returning its VertexId.
    /// Deduplicates: same position returns existing VertexId.
    pub fn insert(&mut self, position: Vec3) -> VertexId {
        let hash = rc3d_core::utils::hash::f32x3_quantized_bits([
            position.x, position.y, position.z,
        ]);
        if let Some(&idx) = self.index.get(&hash) {
            return VertexId(idx);
        }
        let id = VertexId(self.vertices.len() as u32);
        self.vertices.push(TopoVertex { id, position });
        self.index.insert(hash, id.0);
        id
    }

    pub fn get(&self, id: VertexId) -> Option<&TopoVertex> {
        self.vertices.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &TopoVertex> {
        self.vertices.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_deduplication() {
        let mut reg = VertexRegistry::new();
        let a = reg.insert(Vec3::new(1.0, 2.0, 3.0));
        let b = reg.insert(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a, b);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_vertex_distinct_positions() {
        let mut reg = VertexRegistry::new();
        let a = reg.insert(Vec3::new(0.0, 0.0, 0.0));
        let b = reg.insert(Vec3::new(1.0, 0.0, 0.0));
        assert_ne!(a, b);
        assert_eq!(reg.len(), 2);
    }
}
