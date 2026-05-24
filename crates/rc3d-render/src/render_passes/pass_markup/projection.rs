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
#[allow(dead_code)]
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
        // Use approximate comparison due to floating-point precision
        assert!((p[0] - 0.6).abs() < 1e-6);
        assert!((p[1] - 0.45).abs() < 1e-6);
        assert!((p[2] - (-0.75)).abs() < 1e-6);
    }
}

use rc3d_scene::node_data::AnnotationElement;

/// Compute the annotation-plane normal in model-local space.
/// Returns `None` for types without a well-defined plane (Leader, Callout).
pub(crate) fn annotation_plane_normal(element: &AnnotationElement) -> Option<glam::Vec3> {
    match element {
        AnnotationElement::Dimension { start, end, offset_dir, .. } => {
            let s = glam::Vec3::from(start.coords());
            let e = glam::Vec3::from(end.coords());
            let off = glam::Vec3::from(*offset_dir);
            let n = (e - s).cross(off);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => {
            let c = glam::Vec3::from(center.coords());
            let a1 = glam::Vec3::from(arm1.coords());
            let a2 = glam::Vec3::from(arm2.coords());
            let n = (a1 - c).cross(a2 - c);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::RadialDimension { center, perimeter, .. } => {
            let d = glam::Vec3::from(perimeter.coords()) - glam::Vec3::from(center.coords());
            if d.length_squared() > 1e-8 { Some(d.normalize()) } else { None }
        }
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => {
            let c = glam::Vec3::from(center.coords());
            let a = glam::Vec3::from(p1.coords());
            let b = glam::Vec3::from(p2.coords());
            let n = (a - c).cross(b - c);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::Datum { .. } => Some(glam::Vec3::Z),
        AnnotationElement::Leader { .. } | AnnotationElement::Callout { .. } => None,
    }
}
