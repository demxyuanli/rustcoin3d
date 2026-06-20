//! Integration tests for B-rep boolean operations.
//! The old `boolean()` StepShell-based API has been removed in Phase 1 cleanup.
//! B-Rep boolean operations will be implemented in Phase 3.

use rc3d_io::step::bool::{BoolOp, BRepBoolOptions, boolean_brep};

#[test]
fn test_bool_op_enum() {
    let op = BoolOp::Union;
    assert_eq!(op, BoolOp::Union);
    assert_ne!(op, BoolOp::Intersection);
    assert_ne!(op, BoolOp::Difference);
}

#[test]
fn test_boolean_brep_stub() {
    // boolean_brep is a stub that returns empty result
    // Full implementation comes in Phase 3
    let reg = &mut rc3d_shape::BRepStore::new();
    let result = boolean_brep(&[], &[], reg, BoolOp::Union, &BRepBoolOptions::default());
    assert!(result.is_empty);
}
