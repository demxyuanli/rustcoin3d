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

/// Project a local-space point to screen pixels using the scene view-projection (proj * view).
pub(crate) fn project_point_vp(
    pos: glam::Vec3,
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> Option<[f32; 2]> {
    let clip = scene_vp * model * pos.extend(1.0);
    if clip.w.abs() < 0.0001 {
        return None;
    }
    let ndc = glam::vec3(clip.x, clip.y, clip.z) / clip.w;
    if depth_reversed_z {
        if ndc.z > 1.0 || ndc.z < 0.0 {
            return None;
        }
    } else if ndc.z < 0.0 || ndc.z > 1.0 {
        return None;
    }
    Some([
        (ndc.x * 0.5 + 0.5) * screen_w,
        (0.5 - ndc.y * 0.5) * screen_h,
    ])
}

/// Pixels per world unit at `anchor` (model-local), using a short offset along +X.
pub(crate) fn pixels_per_world_unit_at(
    anchor: glam::Vec3,
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> f32 {
    let Some(ap) = project_point_vp(anchor, model, scene_vp, screen_w, screen_h, depth_reversed_z) else {
        return 1.0;
    };
    let test = anchor + glam::Vec3::new(0.1, 0.0, 0.0);
    let Some(tp) = project_point_vp(test, model, scene_vp, screen_w, screen_h, depth_reversed_z) else {
        return 1.0;
    };
    ((tp[0] - ap[0]).powi(2) + (tp[1] - ap[1]).powi(2)).sqrt().max(1.0) / 0.1
}
