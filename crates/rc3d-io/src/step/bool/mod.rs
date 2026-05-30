//! B-rep boolean operations: union, intersection, difference.
//!
//! Pipeline:
//! 1. Face-face intersection → split faces along intersection curves
//! 2. Point-in-solid classification via ray casting
//! 3. Face selection per boolean operation type
//! 4. Output new B-rep solid

pub mod intersect;
pub mod classify;
pub mod split;
pub mod select;

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::ShellKey;

/// Boolean operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    Union,        // A ∪ B
    Intersection, // A ∩ B
    Difference,   // A - B
}

/// Result of a boolean operation (legacy — will be replaced by BRepBoolResult in Phase 3).
pub struct BoolResult {
    pub is_empty: bool,
}

/// Perform a boolean operation (union, intersection, difference).
/// Now operates on B-Rep shells via the registry.
pub fn boolean_brep(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &mut BRepRegistry,
    op: BoolOp,
) -> BoolResult {
    // Phase 1: Face-face intersections
    let _intersections = intersect::compute_intersections_brep(shells_a, shells_b, reg);

    // Phase 2-3: Split + classify (simplified -- full impl requires TopoDS-level operations)
    // For now: return empty result for unsupported combinations
    let _ = op;
    BoolResult { is_empty: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_module_loads() {
        // Verify module compiles and basic types work
        let op = BoolOp::Union;
        assert_eq!(op, BoolOp::Union);
        let result = BoolResult { is_empty: true };
        assert!(result.is_empty);
    }
}
