//! Fillet and chamfer operations on B-rep edges.
//!
//! Planned: constant-radius fillet via offset surface computation,
//! face-face intersection for center curve, and circular arc sweep.

use crate::step::parser::EntityIndex;
use crate::step::topology::{StepEdge, StepFace};

/// Apply a constant-radius fillet to the specified edges.
/// Planned algorithm:
/// 1. Compute offset surfaces on both adjacent faces
/// 2. Intersect offset surfaces to get fillet center curve
/// 3. Sweep circular arc along center curve
/// 4. Trim original faces against fillet surface
pub fn fillet_edges(
    _edges: &[StepEdge],
    _radius: f32,
    _entities: &EntityIndex,
) -> Result<Vec<StepFace>, String> {
    Err("fillet not yet implemented — requires offset surface computation and analytic surface-surface intersection".into())
}

/// Apply a chamfer to the specified edges with given distances.
pub fn chamfer_edges(
    _edges: &[StepEdge],
    _distance1: f32,
    _distance2: f32,
    _entities: &EntityIndex,
) -> Result<Vec<StepFace>, String> {
    Err("chamfer not yet implemented".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fillet_returns_not_implemented() {
        assert!(fillet_edges(&[], 1.0, &EntityIndex::new()).is_err());
    }

    #[test]
    fn test_chamfer_returns_not_implemented() {
        assert!(chamfer_edges(&[], 1.0, 1.0, &EntityIndex::new()).is_err());
    }
}
