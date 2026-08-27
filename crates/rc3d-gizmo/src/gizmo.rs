//! Transform gizmo: translation, rotation, and scale manipulator.
//!
//! Follows the Coin3D SoTransformManip pattern: a self-contained overlay
//! that appears when a transform node is selected, with axis-constrained
//! dragging via screen-space projection math.

use glam::{Mat4, Vec3};
use rc3d_core::aabb::Aabb;
use rc3d_core::NodeId;
use rc3d_actions::Ray;
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
    /// World-from-local orientation (identity = world-aligned handles).
    pub orientation: Mat4,
    /// When set, these handles replace mode-based defaults (composed draggers).
    pub handle_filter: Option<Vec<GizmoHandle>>,
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
            orientation: Mat4::IDENTITY,
            handle_filter: None,
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
                    let av = self.axis_world(axis);
                    let cyl = ray_intersect_cylinder(ray, self.position, av, handle_size);
                    let cone = ray_intersect_cone(ray, self.position + av * handle_size * 4.0 * 0.7, av, handle_size * 4.0 * 0.3, handle_size * 0.12);
                    match (cyl, cone) {
                        (Some(c), Some(k)) => Some(c.min(k)),
                        (Some(c), None) => Some(c),
                        (None, Some(k)) => Some(k),
                        (None, None) => None,
                    }
                }
                GizmoHandle::RotateRing(axis) => {
                    ray_intersect_torus(ray, self.position, self.axis_world(axis), handle_size * 4.0)
                }
                GizmoHandle::ScaleHandle(axis) => {
                    ray_intersect_cube(ray, self.position + self.axis_world(axis) * handle_size * 4.0, handle_size * 0.5)
                }
                GizmoHandle::TranslatePlane(axis) => {
                    ray_intersect_plane_rect(ray, self.position, self.plane_normal_world(axis), handle_size * 2.0)
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
        let p = self.closest_point_on_axis_plane(ray, handle.axis());
        self.drag_start = Some(p);
        self.drag_start_transform = None;
    }

    /// Compute the delta transform during a drag.
    pub fn drag_delta(
        &self,
        ray: &Ray,
    ) -> Option<Mat4> {
        let axis = self.active_axis?;
        let start = self.drag_start?;
        let new_point = self.closest_point_on_axis_plane(ray, axis);

        match axis {
            GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                let av = self.axis_world(axis);
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

    /// Get the list of handles visible in the current mode (or composed filter).
    fn active_handles(&self) -> Vec<GizmoHandle> {
        if let Some(ref filter) = self.handle_filter {
            return filter.clone();
        }
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

    fn axis_world(&self, axis: GizmoAxis) -> Vec3 {
        let local = axis_vector(axis);
        if local.length_squared() < 1e-12 {
            return local;
        }
        let v = self.orientation.transform_vector3(local);
        if v.length_squared() < 1e-12 {
            local
        } else {
            v.normalize()
        }
    }

    fn plane_normal_world(&self, plane: GizmoAxis) -> Vec3 {
        let local = axis_normal(plane);
        let v = self.orientation.transform_vector3(local);
        if v.length_squared() < 1e-12 {
            local
        } else {
            v.normalize()
        }
    }

    fn closest_point_on_axis_plane(&self, ray: &Ray, axis: GizmoAxis) -> Vec3 {
        match axis {
            GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                let av = self.axis_world(axis);
                let to_eye = ray.origin - self.position;
                let mut n = av.cross(to_eye.cross(av));
                if n.length_squared() < 1e-10 {
                    let fallback = if av.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                    n = av.cross(fallback);
                }
                ray_plane_point(ray, self.position, n)
            }
            GizmoAxis::XY | GizmoAxis::YZ | GizmoAxis::ZX => {
                ray_plane_point(ray, self.position, self.plane_normal_world(axis))
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
                    handles::translate_arrow(self.axis_world(axis), self.position, handle_size * 4.0)
                }
                GizmoHandle::RotateRing(axis) => {
                    handles::rotate_ring(self.axis_world(axis), self.position, handle_size * 4.0, 48)
                }
                GizmoHandle::ScaleHandle(axis) => {
                    handles::scale_handle(self.axis_world(axis), self.position, handle_size * 4.0)
                }
                GizmoHandle::TranslatePlane(axis) => {
                    plane_handle_lines(self.position, self.plane_normal_world(axis), handle_size * 2.0)
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

fn plane_handle_lines(origin: Vec3, normal: Vec3, size: f32) -> Vec<LineVertex> {
    let n = if normal.length_squared() < 1e-12 {
        Vec3::Z
    } else {
        normal.normalize()
    };
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

fn aabb_usable(b: &Aabb) -> bool {
    b.min.x <= b.max.x && b.min.y <= b.max.y && b.min.z <= b.max.z
}

fn bbox_of_node(graph: &SceneGraph, node: NodeId) -> Aabb {
    let mut action = rc3d_actions::GetBoundingBoxAction::new();
    action.apply(graph, node);
    if aabb_usable(&action.bounding_box) {
        return action.bounding_box;
    }
    if let Some(parent) = graph.get(node).and_then(|e| e.parent) {
        let mut parent_bb = rc3d_actions::GetBoundingBoxAction::new();
        parent_bb.apply(graph, parent);
        if aabb_usable(&parent_bb.bounding_box) {
            return parent_bb.bounding_box;
        }
    }
    Aabb {
        min: Vec3::splat(-1.0),
        max: Vec3::splat(1.0),
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

    if a.abs() < 1e-12 {
        // Ray nearly parallel to cylinder axis: test perpendicular distance
        let ro_perp = ro - axis * ro.dot(axis);
        if ro_perp.length_squared() > radius * radius {
            return None;
        }
        // Project onto axis to find entry/exit t
        let t = ro.dot(axis).abs() / rd.dot(axis).abs().max(1e-6);
        return if t > 0.0 { Some(t) } else { None };
    }

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

/// Cone intersection (arrow tip). Cone apex at origin, extending along axis.
/// Tested as a cylinder with linearly decreasing radius toward apex.
fn ray_intersect_cone(ray: &Ray, base: Vec3, axis: Vec3, height: f32, base_radius: f32) -> Option<f32> {
    let apex = base + axis * height;
    // Transform: treat cone as a tapered cylinder from base (radius=r) to apex (radius=0)
    // Use the infinite cone equation: (p·axis)² = cos²α * (p·p) where cosα = h/√(h²+r²)
    let h2 = height * height;
    let r2 = base_radius * base_radius;
    let cos2 = h2 / (h2 + r2);

    let ro = ray.origin - apex;
    let rd = ray.direction;
    let a = rd.dot(axis);
    // Quadratic: a² * (rd·rd - cos²*rd·rd) + ...
    let rd_dot_rd = rd.dot(rd);
    let quad_a = a * a - cos2 * rd_dot_rd;
    let b = ro.dot(axis);
    let quad_b = 2.0 * (a * b - cos2 * ro.dot(rd));
    let ro_dot_ro = ro.dot(ro);
    let quad_c = b * b - cos2 * ro_dot_ro;

    let disc = quad_b * quad_b - 4.0 * quad_a * quad_c;
    if disc < 0.0 || quad_a.abs() < 1e-12 {
        return None;
    }

    let sqrt_disc = disc.sqrt();
    let t0 = (-quad_b - sqrt_disc) / (2.0 * quad_a);
    let t1 = (-quad_b + sqrt_disc) / (2.0 * quad_a);

    for &t in &[t0, t1] {
        if t > 0.001 {
            let hit = ray.origin + rd * t;
            // Verify hit is within cone segment [apex, base]
            let h = (hit - apex).dot(axis);
            if h >= 0.0 && h <= height {
                return Some(t);
            }
        }
    }
    None
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

fn ray_plane_point(ray: &Ray, origin: Vec3, normal: Vec3) -> Vec3 {
    let n = if normal.length_squared() < 1e-10 {
        Vec3::Y
    } else {
        normal.normalize()
    };
    let denom = ray.direction.dot(n);
    if denom.abs() < 1e-6 {
        return origin;
    }
    let t = (origin - ray.origin).dot(n) / denom;
    if !t.is_finite() {
        return origin;
    }
    ray.origin + ray.direction * t
}
