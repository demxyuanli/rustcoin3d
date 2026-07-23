//! Math utilities for gizmo: screen projection, axis/plane constraint.

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

/// Camera + viewport parameters for screen-space projection math.
#[derive(Clone, Copy, Debug)]
pub struct CameraScreen {
    pub view: Mat4,
    pub proj: Mat4,
    pub width: f32,
    pub height: f32,
    pub x: f32,
    pub y: f32,
}

impl CameraScreen {
    /// Create with zero viewport offset (the common case).
    pub fn new(view: Mat4, proj: Mat4, width: f32, height: f32) -> Self {
        Self { view, proj, width, height, x: 0.0, y: 0.0 }
    }
}

/// Project a world-space point to screen-space pixel coordinates.
/// Returns `None` if the point is behind the camera or outside NDC.
pub fn world_to_screen(pos: Vec3, camera: &CameraScreen) -> Option<(f32, f32)> {
    let clip = camera.proj * camera.view * Vec4::new(pos.x, pos.y, pos.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.xyz() / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return None;
    }
    let sx = (ndc.x * 0.5 + 0.5) * camera.width + camera.x;
    let sy = (1.0 - ndc.y) * 0.5 * camera.height + camera.y;
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
    camera: &CameraScreen,
) -> Option<f32> {
    let p0 = world_to_screen(world_origin, camera)?;
    let p1 = world_to_screen(world_origin + world_axis, camera)?;
    let screen_axis = Vec3::new(p1.0 - p0.0, p1.1 - p0.1, 0.0).normalize();
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
    camera: &CameraScreen,
) -> Option<Vec3> {
    let screen_proj = project_on_axis_screen(origin, axis, mouse_dx, mouse_dy, camera)?;
    let p0 = world_to_screen(origin, camera)?;
    let p1 = world_to_screen(origin + axis, camera)?;
    let screen_len = Vec3::new(p1.0 - p0.0, p1.1 - p0.1, 0.0).length();
    if screen_len < 1e-6 {
        return None;
    }
    Some(axis * (screen_proj / screen_len))
}
