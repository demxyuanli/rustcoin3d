//! Transform gizmo: translation, rotation, and scale manipulator.
//!
//! Follows the Coin3D SoTransformManip pattern: a self-contained overlay
//! that appears when a transform node is selected, with axis-constrained
//! dragging via screen-space projection math.

use glam::{Mat4, Vec3};
use rc3d_core::aabb::Aabb;
use rc3d_core::NodeId;
use rc3d_actions::{Action, Ray};
use rc3d_render::LineVertex;
use rc3d_scene::{NodeData, SceneGraph};

use crate::handles;

/// Gizmo interaction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

/// Which axis or plane is being manipulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    XY,
    YZ,
    ZX,
}

/// Which gizmo handle is being hovered/dragged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoHandle {
    TranslateArrow(GizmoAxis),
    RotateRing(GizmoAxis),
    ScaleHandle(GizmoAxis),
    TranslatePlane(GizmoAxis), // XY/YZ/ZX plane handles
}

/// Axis color mapping for standard RGB gizmo convention.
pub fn axis_color(axis: GizmoAxis) -> [f32; 4] {
    match axis {
        GizmoAxis::X => [1.0, 0.2, 0.2, 1.0],
        GizmoAxis::Y => [0.2, 1.0, 0.2, 1.0],
        GizmoAxis::Z => [0.2, 0.4, 1.0, 1.0],
        GizmoAxis::XY => [1.0, 1.0, 0.2, 0.4], // semi-transparent plane handles
        GizmoAxis::YZ => [0.2, 1.0, 0.2, 0.4],
        GizmoAxis::ZX => [0.2, 0.4, 1.0, 0.4],
    }
}

/// Highlighted (hovered) color override.
pub fn axis_highlight_color(axis: GizmoAxis) -> [f32; 4] {
    match axis {
        GizmoAxis::X => [1.0, 0.8, 0.2, 1.0],
        GizmoAxis::Y => [0.8, 1.0, 0.2, 1.0],
        GizmoAxis::Z => [0.4, 0.6, 1.0, 1.0],
        _ => axis_color(axis),
    }
}

/// Transform gizmo state.
pub struct Gizmo {
    pub mode: GizmoMode,
    pub target_node: Option<NodeId>,
    /// World-space position (bounding box center of the target).
    pub position: Vec3,
    /// Cached scale factor (keeps handle size roughly constant in screen space).
    pub screen_scale: f32,
    /// Currently hovered handle (for highlight).
    pub hovered: Option<GizmoHandle>,
    /// Currently active drag axis/plane.
    pub active_axis: Option<GizmoAxis>,
    /// Drag start point (world-space), for computing deltas.
    pub drag_start: Option<Vec3>,
    /// Original transform at drag start (for applying relative deltas).
    pub drag_start_transform: Option<Mat4>,
    /// Whether the gizmo is visible.
    pub visible: bool,
}

impl Gizmo {
    pub fn new() -> Self {
        Self {
            mode: GizmoMode::Translate,
            target_node: None,
            position: Vec3::ZERO,
            screen_scale: 1.0,
            hovered: None,
            active_axis: None,
            drag_start: None,
            drag_start_transform: None,
            visible: false,
        }
    }

    /// Position the gizmo at the bounding box center of the target node.
    pub fn update_target(&mut self, graph: &SceneGraph) {
        self.visible = false;
        let Some(node) = self.target_node else { return };
        let Some(entry) = graph.get(node) else { return };

        let bbox = bbox_of_node(graph, node);
        self.position = bbox.center();
        self.screen_scale = bbox.size().length().max(0.1);

        // Check for existing transform
        if matches!(&entry.data, NodeData::Transform(_)) {
            self.visible = true;
        }
    }

    /// Hit test: which handle (if any) does the ray intersect?
    /// Returns the handle and approximate distance along the ray.
    pub fn hit_test(&self, ray: &Ray) -> Option<(GizmoHandle, f32)> {
        if !self.visible {
            return None;
        }

        let handle_size = self.screen_scale * 0.15;
        let handles = self.active_handles();
        let mut closest: Option<(GizmoHandle, f32)> = None;

        for handle in handles {
            let dist = match handle {
                GizmoHandle::TranslateArrow(axis) => {
                    ray_intersect_cylinder(ray, self.position, axis_vector(axis), handle_size)
                }
                GizmoHandle::RotateRing(axis) => {
                    ray_intersect_torus(ray, self.position, axis_vector(axis), handle_size * 4.0)
                }
                GizmoHandle::ScaleHandle(axis) => {
                    ray_intersect_cube(ray, self.position + axis_vector(axis) * handle_size * 4.0, handle_size * 0.5)
                }
                GizmoHandle::TranslatePlane(axis) => {
                    ray_intersect_plane_rect(ray, self.position, axis_normal(axis), handle_size * 2.0)
                }
            };

            if let Some(d) = dist {
                if d > 0.0 && (closest.is_none() || d < closest.unwrap().1) {
                    closest = Some((handle, d));
                }
            }
        }
        closest
    }

    /// Start a drag operation on a handle.
    pub fn start_drag(&mut self, ray: &Ray, handle: GizmoHandle) {
        self.active_axis = Some(handle.axis());
        // Compute the world-space point on the constraint plane through gizmo position.
        let p = closest_point_on_axis_plane(ray, self.position, handle.axis());
        self.drag_start = Some(p);
        self.drag_start_transform = None;
    }

    /// Compute the delta transform during a drag.
    pub fn drag_delta(
        &self,
        ray: &Ray,
        _view: Mat4,
        _proj: Mat4,
        _vp_w: f32,
        _vp_h: f32,
        _vp_x: f32,
        _vp_y: f32,
    ) -> Option<Mat4> {
        let axis = self.active_axis?;
        let start = self.drag_start?;
        let new_point = closest_point_on_axis_plane(ray, self.position, axis);

        match axis {
            GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                let av = axis_vector(axis);
                let old_t = (start - self.position).dot(av);
                let new_t = (new_point - self.position).dot(av);
                let delta = new_t - old_t;
                Some(Mat4::from_translation(av * delta))
            }
            GizmoAxis::XY | GizmoAxis::YZ | GizmoAxis::ZX => {
                // Plane drag: delta is the full difference
                let delta = new_point - start;
                Some(Mat4::from_translation(delta))
            }
        }
    }

    /// End a drag operation.
    pub fn end_drag(&mut self) {
        self.active_axis = None;
        self.drag_start = None;
        self.drag_start_transform = None;
    }

    /// Get the list of handles visible in the current mode.
    fn active_handles(&self) -> Vec<GizmoHandle> {
        let axes = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];
        let planes = [GizmoAxis::XY, GizmoAxis::YZ, GizmoAxis::ZX];

        match self.mode {
            GizmoMode::Translate => {
                let mut h: Vec<_> = axes.iter().map(|&a| GizmoHandle::TranslateArrow(a)).collect();
                h.extend(planes.iter().map(|&a| GizmoHandle::TranslatePlane(a)));
                h
            }
            GizmoMode::Rotate => {
                axes.iter().map(|&a| GizmoHandle::RotateRing(a)).collect()
            }
            GizmoMode::Scale => {
                axes.iter().map(|&a| GizmoHandle::ScaleHandle(a)).collect()
            }
        }
    }

    /// Generate all draw-call line vertices for the gizmo.
    pub fn generate_lines(&self) -> Vec<(Vec<LineVertex>, [f32; 4])> {
        let handle_size = self.screen_scale * 0.15;
        let mut batches: Vec<(Vec<LineVertex>, [f32; 4])> = Vec::new();

        for handle in self.active_handles() {
            let highlighted = self.hovered == Some(handle) || self.active_axis == Some(handle.axis());
            let color = if highlighted {
                axis_highlight_color(handle.axis())
            } else {
                axis_color(handle.axis())
            };

            let lines = match handle {
                GizmoHandle::TranslateArrow(axis) => {
                    handles::translate_arrow(axis_vector(axis), self.position, handle_size * 4.0)
                }
                GizmoHandle::RotateRing(axis) => {
                    handles::rotate_ring(axis_vector(axis), self.position, handle_size * 4.0, 48)
                }
                GizmoHandle::ScaleHandle(axis) => {
                    handles::scale_handle(axis_vector(axis), self.position, handle_size * 4.0)
                }
                GizmoHandle::TranslatePlane(axis) => {
                    plane_handle_lines(self.position, axis, handle_size * 2.0)
                }
            };

            batches.push((lines, color));
        }

        batches
    }
}

impl Default for Gizmo {
    fn default() -> Self {
        Self::new()
    }
}

impl GizmoHandle {
    pub fn axis(&self) -> GizmoAxis {
        match *self {
            GizmoHandle::TranslateArrow(a) => a,
            GizmoHandle::RotateRing(a) => a,
            GizmoHandle::ScaleHandle(a) => a,
            GizmoHandle::TranslatePlane(a) => a,
        }
    }
}

// ── Helper functions ──

fn axis_vector(axis: GizmoAxis) -> Vec3 {
    match axis {
        GizmoAxis::X => Vec3::X,
        GizmoAxis::Y => Vec3::Y,
        GizmoAxis::Z => Vec3::Z,
        GizmoAxis::XY | GizmoAxis::YZ | GizmoAxis::ZX => Vec3::ZERO,
    }
}

fn axis_normal(plane: GizmoAxis) -> Vec3 {
    match plane {
        GizmoAxis::XY => Vec3::Z,
        GizmoAxis::YZ => Vec3::X,
        GizmoAxis::ZX => Vec3::Y,
        _ => Vec3::Y,
    }
}

fn plane_handle_lines(origin: Vec3, plane: GizmoAxis, size: f32) -> Vec<LineVertex> {
    let n = axis_normal(plane);
    let perp = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = n.cross(perp).normalize() * size;
    let v = n.cross(u).normalize() * size;
    let o = origin;

    let corners = [o + u + v, o + u - v, o - u - v, o - u + v];
    let mut lines = Vec::with_capacity(8);
    for i in 0..4 {
        let next = (i + 1) % 4;
        lines.push(LineVertex { position: corners[i].to_array() });
        lines.push(LineVertex { position: corners[next].to_array() });
    }
    lines
}

fn bbox_of_node(graph: &SceneGraph, node: NodeId) -> Aabb {
    let mut action = rc3d_actions::GetBoundingBoxAction::new();
    action.apply(graph, node);
    if action.bounding_box.min != action.bounding_box.max {
        action.bounding_box
    } else {
        Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) }
    }
}

/// Cylinder intersection (shaft of arrow). Returns distance along ray.
fn ray_intersect_cylinder(ray: &Ray, origin: Vec3, axis: Vec3, size: f32) -> Option<f32> {
    let shaft_len = size * 4.0 * 0.7;
    let shaft_start = origin;
    let _shaft_end = origin + axis * shaft_len;
    let radius = size * 0.12 * 0.4;

    // Ray vs infinite cylinder
    let ro = ray.origin - shaft_start;
    let rd = ray.direction;
    let a = rd.dot(rd) - rd.dot(axis) * rd.dot(axis);
    let b = 2.0 * (ro.dot(rd) - ro.dot(axis) * rd.dot(axis));
    let c = ro.dot(ro) - ro.dot(axis) * ro.dot(axis) - radius * radius;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }

    let t0 = (-b - disc.sqrt()) / (2.0 * a);
    let t1 = (-b + disc.sqrt()) / (2.0 * a);
    let t = if t0 > 0.0 { t0 } else { t1 };
    if t <= 0.0 {
        return None;
    }

    // Check against shaft segment
    let hit = ray.origin + ray.direction * t;
    let h = (hit - shaft_start).dot(axis);
    if h >= 0.0 && h <= shaft_len {
        Some(t)
    } else {
        None
    }
}

/// Torus intersection (rotation ring). Returns distance along ray.
fn ray_intersect_torus(ray: &Ray, center: Vec3, axis: Vec3, radius: f32) -> Option<f32> {
    // Simplified: treat the ring as a thick circle — test distance from ray
    // point to the circle.
    // Full torus intersection is complex; approximate with a ring of spheres.
    let tube_radius = radius * 0.08;
    let perp = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = axis.cross(perp).normalize();
    let v = axis.cross(u).normalize();

    let segments = 48; // spacing (~0.13r) < sphere diameter (0.16r) → no gaps
    let mut closest: Option<f32> = None;
    for i in 0..segments {
        let angle = i as f32 / segments as f32 * std::f32::consts::TAU;
        let ring_point = center + u * radius * angle.cos() + v * radius * angle.sin();
        if let Some(d) = ray_intersect_sphere(ray, ring_point, tube_radius) {
            if closest.is_none() || d < closest.unwrap() {
                closest = Some(d);
            }
        }
    }
    closest
}

/// Cube intersection. Returns distance along ray.
fn ray_intersect_cube(ray: &Ray, center: Vec3, half: f32) -> Option<f32> {
    let min = center - Vec3::splat(half);
    let max = center + Vec3::splat(half);
    let inv_dir = Vec3::new(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
    let t0 = (min - ray.origin) * inv_dir;
    let t1 = (max - ray.origin) * inv_dir;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let t_enter = tmin.x.max(tmin.y).max(tmin.z);
    let t_exit = tmax.x.min(tmax.y).min(tmax.z);
    if t_enter > t_exit || t_exit < 0.0 {
        None
    } else {
        Some(t_enter.max(0.0))
    }
}

/// Plane rectangle intersection.
fn ray_intersect_plane_rect(ray: &Ray, origin: Vec3, normal: Vec3, half: f32) -> Option<f32> {
    let denom = ray.direction.dot(normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (origin - ray.origin).dot(normal) / denom;
    if t <= 0.0 {
        return None;
    }
    let hit = ray.origin + ray.direction * t;
    let local = hit - origin;
    let perp = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = normal.cross(perp).normalize();
    let v = normal.cross(u).normalize();
    let du = local.dot(u).abs();
    let dv = local.dot(v).abs();
    if du <= half && dv <= half {
        Some(t)
    } else {
        None
    }
}

/// Ray-sphere intersection.
fn ray_intersect_sphere(ray: &Ray, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ray.origin - center;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / (2.0 * a);
    if t > 0.0 {
        Some(t)
    } else {
        let t2 = (-b + disc.sqrt()) / (2.0 * a);
        if t2 > 0.0 { Some(t2) } else { None }
    }
}

fn closest_point_on_axis_plane(ray: &Ray, origin: Vec3, axis: GizmoAxis) -> Vec3 {
    match axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
            let av = axis_vector(axis);
            let t = (origin - ray.origin).dot(av) / ray.direction.dot(av).max(1e-6);
            ray.origin + ray.direction * t
        }
        GizmoAxis::XY | GizmoAxis::YZ | GizmoAxis::ZX => {
            let n = axis_normal(axis);
            let t = (origin - ray.origin).dot(n) / ray.direction.dot(n).max(1e-6);
            ray.origin + ray.direction * t
        }
    }
}
