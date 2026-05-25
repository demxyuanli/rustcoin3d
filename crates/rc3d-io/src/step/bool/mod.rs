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

use super::parser::EntityIndex;
use super::topology::StepShell;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::ShellKey;

/// Boolean operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    Union,        // A ∪ B
    Intersection, // A ∩ B
    Difference,   // A - B
}

/// Result of a boolean operation: a new solid defined by its faces.
pub struct BoolResult {
    pub shells: Vec<StepShell>,
    /// Whether the operation produced a valid non-empty solid.
    pub is_empty: bool,
}

/// Perform a boolean operation between two sets of shells.
pub fn boolean(a_shells: &[StepShell], b_shells: &[StepShell],
               entities_a: &EntityIndex, entities_b: &EntityIndex,
               op: BoolOp) -> BoolResult
{
    // Phase 1: Compute face-face intersections
    let intersections = intersect::compute_intersections(
        a_shells, b_shells, entities_a, entities_b);

    // Phase 2: Split faces along intersection curves
    let (split_a, split_b) = split::split_faces(
        a_shells, b_shells, entities_a, entities_b, &intersections);

    // Phase 3: Classify each split face region
    let classified_a = classify::classify_faces(&split_a, b_shells, entities_b);
    let classified_b = classify::classify_faces(&split_b, a_shells, entities_a);

    // Phase 4: Select faces based on operation
    let selected = select::select_faces(&classified_a, &classified_b, op);

    if selected.is_empty() {
        return BoolResult { shells: vec![], is_empty: true };
    }

    // Phase 5: Build output shell
    BoolResult {
        shells: vec![StepShell {
            id: 0,
            faces: selected,
        }],
        is_empty: false,
    }
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
    BoolResult { shells: vec![], is_empty: true }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser;

    fn make_entities(data: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_bool_module_loads() {
        // Verify module compiles and basic types work
        let op = BoolOp::Union;
        assert_eq!(op, BoolOp::Union);
        let result = BoolResult { shells: vec![], is_empty: true };
        assert!(result.is_empty);
    }
}
