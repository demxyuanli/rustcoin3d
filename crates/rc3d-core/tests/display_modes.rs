use rc3d_core::DisplayMode;

#[test]
fn display_modes_are_distinct() {
    assert_ne!(
        std::mem::discriminant(&DisplayMode::Shaded),
        std::mem::discriminant(&DisplayMode::Wireframe)
    );
    assert_ne!(
        std::mem::discriminant(&DisplayMode::Shaded),
        std::mem::discriminant(&DisplayMode::Flat)
    );
}
