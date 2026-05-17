/// Build an orthographic projection that maps pixel coordinates [0..w, 0..h]
/// to NDC [-1..1] with Y flipped (screen-space: origin top-left, Y down).
pub(super) fn screen_space_ortho(w: f32, h: f32) -> glam::Mat4 {
    glam::Mat4::from_cols(
        glam::Vec4::new(2.0 / w, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -2.0 / h, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
        glam::Vec4::new(-1.0, 1.0, 0.0, 1.0),
    )
}

/// Project a 3D world-space point to 2D screen coordinates.
pub(crate) fn project_point(
    pos: glam::Vec3,
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> Option<[f32; 2]> {
    let clip = proj * view * model * pos.extend(1.0);
    if clip.w.abs() < 0.0001 {
        return None;
    }
    let ndc = glam::vec3(clip.x, clip.y, clip.z) / clip.w;
    // Clip check: skip points behind camera or beyond far plane
    if depth_reversed_z {
        if ndc.z > 1.0 || ndc.z < 0.0 { return None; }
    } else {
        if ndc.z < 0.0 || ndc.z > 1.0 { return None; }
    }
    Some([
        (ndc.x * 0.5 + 0.5) * screen_w,
        (0.5 - ndc.y * 0.5) * screen_h,
    ])
}
