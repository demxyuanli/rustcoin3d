use rc3d_actions::Ray;
use rc3d_core::math::Mat4;

pub(super) fn build_pick_ray(
    cursor_pos: (f64, f64),
    window_size: (u32, u32),
    camera_view: Mat4,
    camera_proj: Mat4,
) -> Ray {
    Ray::from_screen_point(
        cursor_pos.0 as f32,
        cursor_pos.1 as f32,
        window_size.0 as f32,
        window_size.1 as f32,
        camera_view,
        camera_proj,
    )
}
