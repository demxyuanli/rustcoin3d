//! Math utilities for gizmo: screen projection, axis/plane constraint.

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

/// Project a world-space point to screen-space pixel coordinates.
/// Returns `None` if the point is behind the camera or outside NDC.
pub fn world_to_screen(pos: Vec3, view: Mat4, proj: Mat4, vp_w: f32, vp_h: f32, vp_x: f32, vp_y: f32) -> Option<(f32, f32)> {
    let clip = proj * view * Vec4::new(pos.x, pos.y, pos.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.xyz() / clip.w; // NDC x/y in [-1,1], z in [0,1]
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return None;
    }
    let sx = (ndc.x * 0.5 + 0.5) * vp_w + vp_x;
    let sy = (1.0 - ndc.y) * 0.5 * vp_h + vp_y; // flip Y for screen coords
    Some((sx, sy))
}

/// Closest point on a ray to a given point.
pub fn closest_point_on_ray(point: Vec3, origin: Vec3, dir: Vec3) -> Vec3 {
    let t = (point - origin).dot(dir).max(0.0);
    origin + dir * t
}

/// Closest point on a line segment.
pub fn closest_point_on_segment(point: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let t = (point - a).dot(ab) / ab.length_squared().max(1e-8);
    a + ab * t.clamp(0.0, 1.0)
}

/// Closest point on an infinite line (axis). Axis is auto-normalized.
pub fn closest_point_on_axis(point: Vec3, origin: Vec3, axis: Vec3) -> Vec3 {
    let ax = axis.normalize();
    let t = (point - origin).dot(ax);
    origin + ax * t
}

/// Closest point on a plane through origin with given normal. Normal is auto-normalized.
pub fn closest_point_on_plane(point: Vec3, origin: Vec3, normal: Vec3) -> Vec3 {
    let n = normal.normalize();
    let d = (point - origin).dot(n);
    point - n * d
}

/// Project a 3D vector onto screen-space direction along an axis.
/// Returns the projected screen-space delta magnitude.
pub fn project_on_axis_screen(
    world_origin: Vec3,
    world_axis: Vec3,
    mouse_dx: f32,
    mouse_dy: f32,
    view: Mat4,
    proj: Mat4,
    vp_w: f32,
    vp_h: f32,
    vp_x: f32,
    vp_y: f32,
) -> Option<f32> {
    let p0 = world_to_screen(world_origin, view, proj, vp_w, vp_h, vp_x, vp_y)?;
    let p1 = world_to_screen(world_origin + world_axis, view, proj, vp_w, vp_h, vp_x, vp_y)?;
    let screen_axis = (Vec3::new(p1.0 - p0.0, p1.1 - p0.1, 0.0)).normalize();
    let screen_delta = Vec3::new(mouse_dx, mouse_dy, 0.0);
    let projected = screen_delta.dot(screen_axis);
    Some(projected)
}

/// Compute the world-space delta along an axis from screen-space mouse delta.
pub fn axis_drag_delta(
    origin: Vec3,
    axis: Vec3,
    mouse_dx: f32,
    mouse_dy: f32,
    view: Mat4,
    proj: Mat4,
    vp_w: f32,
    vp_h: f32,
    vp_x: f32,
    vp_y: f32,
) -> Option<Vec3> {
    let screen_proj = project_on_axis_screen(origin, axis, mouse_dx, mouse_dy, view, proj, vp_w, vp_h, vp_x, vp_y)?;
    let p0 = world_to_screen(origin, view, proj, vp_w, vp_h, vp_x, vp_y)?;
    let p1 = world_to_screen(origin + axis, view, proj, vp_w, vp_h, vp_x, vp_y)?;
    let screen_len = (Vec3::new(p1.0 - p0.0, p1.1 - p0.1, 0.0)).length();
    if screen_len < 1e-6 {
        return None;
    }
    Some(axis * (screen_proj / screen_len))
}
