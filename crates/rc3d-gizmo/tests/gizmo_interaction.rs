use rc3d_gizmo::GizmoMode;

#[test]
fn gizmo_modes_are_distinct() {
    assert_ne!(
        std::mem::discriminant(&GizmoMode::Translate),
        std::mem::discriminant(&GizmoMode::Rotate)
    );
    assert_ne!(
        std::mem::discriminant(&GizmoMode::Translate),
        std::mem::discriminant(&GizmoMode::Scale)
    );
}
