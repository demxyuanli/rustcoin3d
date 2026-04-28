use crate::render_action::DrawCall;
use rc3d_core::DisplayMode;

pub fn display_mode_sort_key(mode: DisplayMode) -> u8 {
    match mode {
        DisplayMode::Shaded => 0,
        DisplayMode::ShadedWithEdges => 1,
        DisplayMode::Wireframe => 2,
        DisplayMode::HiddenLine => 3,
    }
}

pub fn color_sort_key(color: [f32; 4]) -> [u32; 4] {
    [
        color[0].to_bits(),
        color[1].to_bits(),
        color[2].to_bits(),
        color[3].to_bits(),
    ]
}

pub fn vec4_array_sort_key(arr: [[f32; 4]; 4]) -> [[u32; 4]; 4] {
    arr.map(color_sort_key)
}

/// Composite key used to batch draw calls sharing the same light parameters.
pub fn light_sort_key(dc: &DrawCall) -> (
    [[u32; 4]; 4], [[u32; 4]; 4], [[u32; 4]; 4], [[u32; 4]; 4], [[u32; 4]; 4],
    usize, [u32; 4], [u32; 4], [u32; 4], u32,
) {
    (
        vec4_array_sort_key(dc.light_dirs),
        vec4_array_sort_key(dc.light_colors),
        vec4_array_sort_key(dc.light_types),
        vec4_array_sort_key(dc.light_positions),
    vec4_array_sort_key(dc.spot_params),
    dc.light_count as usize,
        color_sort_key([dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]),
        color_sort_key([dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0]),
        color_sort_key([dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0]),
        dc.shininess.to_bits(),
    )
}
