use rc3d_core::DisplayMode;

pub fn display_mode_sort_key(mode: DisplayMode) -> u8 {
    match mode {
        DisplayMode::Shaded => 0,
        DisplayMode::ShadedWithEdges => 1,
        DisplayMode::Wireframe => 2,
        DisplayMode::HiddenLine => 3,
        DisplayMode::Flat => 4,
        DisplayMode::FlatWithEdge => 5,
    }
}

pub fn color_sort_key(color: [f32; 4]) -> [u32; 4] {
    rc3d_core::utils::hash::f32x4_to_bits(color)
}

