//! Axis-aligned bounding box utilities for boolean operation acceleration.

use rc3d_core::math::Vec3;
use crate::store::BRepStore;
use crate::topo::FaceKey;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    /// Empty AABB (inverted bounds).
    pub fn empty() -> Self {
        AABB {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }

    /// Expand to include a point.
    pub fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Check if two AABBs overlap (including touching).
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

/// Compute the AABB of a face from its edge vertex positions.
pub fn face_vertex_bbox(face: FaceKey, reg: &BRepStore) -> Option<AABB> {
    let mut bbox = AABB::empty();
    let mut any = false;

    for wire_key in crate::topo_iter::iter_wires_of_face(face, reg) {
        let wire = reg.wires.get(wire_key)?;
        for &(ek, _) in &wire.edges {
            let edge = reg.edges.get(ek)?;
            for vk in [edge.v_low, edge.v_high] {
                if let Some(v) = reg.vertices.get(vk) {
                    bbox.expand(v.position);
                    any = true;
                }
            }
        }
    }

    if any { Some(bbox) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_overlap_touching() {
        let a = AABB { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let b = AABB { min: Vec3::new(1.0, 0.0, 0.0), max: Vec3::new(2.0, 1.0, 1.0) };
        assert!(a.overlaps(&b), "touching faces should overlap");
    }

    #[test]
    fn test_aabb_no_overlap() {
        let a = AABB { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let b = AABB { min: Vec3::new(2.0, 2.0, 2.0), max: Vec3::new(3.0, 3.0, 3.0) };
        assert!(!a.overlaps(&b), "separated faces should not overlap");
    }

    #[test]
    fn test_aabb_overlap_partial() {
        let a = AABB { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(2.0, 2.0, 2.0) };
        let b = AABB { min: Vec3::new(1.0, 1.0, 1.0), max: Vec3::new(3.0, 3.0, 3.0) };
        assert!(a.overlaps(&b), "partially overlapping faces should overlap");
    }

    #[test]
    fn test_aabb_overlap_contained() {
        let a = AABB { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(4.0, 4.0, 4.0) };
        let b = AABB { min: Vec3::new(1.0, 1.0, 1.0), max: Vec3::new(2.0, 2.0, 2.0) };
        assert!(a.overlaps(&b), "contained face should overlap");
    }
}
