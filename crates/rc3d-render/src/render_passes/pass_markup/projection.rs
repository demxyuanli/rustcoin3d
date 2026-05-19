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

/// Project to clip/NDC space (same camera as solid geometry — world-fixed on the model).
pub(crate) fn project_point_ndc(
    pos: glam::Vec3,
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    depth_reversed_z: bool,
) -> Option<[f32; 3]> {
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
    Some([ndc.x, ndc.y, ndc.z])
}

/// Convert NDC to screen pixels (inverse of `project_point_vp` xy mapping).
pub(crate) fn ndc_to_screen(ndc: [f32; 3], screen_w: f32, screen_h: f32) -> [f32; 2] {
    [
        (ndc[0] * 0.5 + 0.5) * screen_w,
        (0.5 - ndc[1] * 0.5) * screen_h,
    ]
}

/// Screen baseline angle (radians) from a model-local tangent projected to the viewport.
/// Text lies on the annotation plane in model space, not billboarded to the camera.
pub(crate) fn screen_baseline_from_model_tangent(
    at: glam::Vec3,
    tangent: glam::Vec3,
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> f32 {
    let len_sq = tangent.length_squared();
    if len_sq < 1e-12 {
        return 0.0;
    }
    let dir = tangent / len_sq.sqrt();
    let tip = at + dir * 0.1;
    let Some(a_ndc) = project_point_ndc(at, model, scene_vp, depth_reversed_z) else {
        return 0.0;
    };
    let Some(b_ndc) = project_point_ndc(tip, model, scene_vp, depth_reversed_z) else {
        return 0.0;
    };
    let a = ndc_to_screen(a_ndc, screen_w, screen_h);
    let b = ndc_to_screen(b_ndc, screen_w, screen_h);
    readable_baseline_angle(a, b)
}

/// Keep labels upright on screen (no upside-down mirrored text).
pub(crate) fn readable_baseline_angle(from: [f32; 2], to: [f32; 2]) -> f32 {
    let mut a = (to[1] - from[1]).atan2(to[0] - from[0]);
    let half_pi = std::f32::consts::FRAC_PI_2;
    if a > half_pi {
        a -= std::f32::consts::PI;
    } else if a < -half_pi {
        a += std::f32::consts::PI;
    }
    a
}

/// Leader/callout label anchor in model-local space: fixed offset along +X/+Y (not camera-derived).
pub(crate) fn leader_label_local_3d(
    anchor: [f32; 3],
    label_offset: [f32; 2],
    scale: f32,
) -> [f32; 3] {
    [
        anchor[0] + label_offset[0] * scale,
        anchor[1] - label_offset[1] * scale,
        anchor[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leader_local_offset_is_camera_independent() {
        let p = leader_label_local_3d([0.0, 0.0, -0.75], [40.0, -30.0], 0.015);
        assert_eq!(p, [0.6, 0.45, -0.75]);
    }

    #[test]
    fn model_tangent_baseline_uses_world_direction() {
        let model = glam::Mat4::IDENTITY;
        let scene_vp = glam::Mat4::look_at_rh(
            glam::Vec3::new(4.0, 3.0, 12.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        ) * glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
            0.1,
            100.0,
        );
        let a0 = screen_baseline_from_model_tangent(
            glam::Vec3::new(-3.5, 0.0, 0.0),
            glam::Vec3::X,
            model,
            scene_vp,
            800.0,
            600.0,
            false,
        );
        let scene_vp2 = glam::Mat4::look_at_rh(
            glam::Vec3::new(-8.0, 2.0, 14.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        ) * glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
            0.1,
            100.0,
        );
        let a1 = screen_baseline_from_model_tangent(
            glam::Vec3::new(-3.5, 0.0, 0.0),
            glam::Vec3::X,
            model,
            scene_vp2,
            800.0,
            600.0,
            false,
        );
        assert!(
            (a0 - a1).abs() < 0.15,
            "world +X tangent baseline should be stable across camera positions: {a0} vs {a1}"
        );
    }
}
