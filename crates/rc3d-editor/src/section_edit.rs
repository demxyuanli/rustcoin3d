//! Section-plane 3D widget: quad outline + normal arrow, drag along the normal.

use rc3d_actions::Ray;
use rc3d_core::math::Vec3;
use rc3d_core::NodeId;
use rc3d_gizmo::handles;
use rc3d_render::LineVertex;
use rc3d_scene::{NodeData, SceneGraph};

const WIDGET_SIZE: f32 = 2.5;
const ARROW_HIT_RADIUS: f32 = 0.18;

/// Unit normal, a point on the plane, and in-plane axes.
pub fn plane_frame(plane: [f32; 4]) -> Option<(Vec3, Vec3, Vec3, Vec3)> {
    let n = Vec3::new(plane[0], plane[1], plane[2]);
    let len = n.length();
    if len < 1e-6 {
        return None;
    }
    let n = n / len;
    let d = plane[3] / len;
    let origin = -n * d;
    let perp = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = n.cross(perp).normalize();
    let v = n.cross(u).normalize();
    Some((n, origin, u, v))
}

pub fn enabled_section_planes(graph: &SceneGraph) -> Vec<(NodeId, [f32; 4])> {
    let mut out = Vec::new();
    for id in graph.all_node_ids() {
        if let Some(e) = graph.get(id) {
            if let NodeData::SectionPlane(sp) = &e.data {
                if sp.enabled {
                    out.push((id, sp.plane));
                }
            }
        }
    }
    out
}

pub fn widget_batches(
    graph: &SceneGraph,
    hovered: Option<NodeId>,
) -> Vec<(Vec<LineVertex>, [f32; 4])> {
    let mut batches = Vec::new();
    for (id, plane) in enabled_section_planes(graph) {
        let Some((n, origin, u, v)) = plane_frame(plane) else {
            continue;
        };
        let size = WIDGET_SIZE;
        let color = if hovered == Some(id) {
            [1.0, 0.85, 0.2, 1.0]
        } else {
            [0.25, 0.85, 0.95, 1.0]
        };
        let mut lines = plane_quad_lines(origin, u, v, size);
        lines.extend(handles::translate_arrow(n, origin, size * 0.85));
        batches.push((lines, color));
    }
    batches
}

fn plane_quad_lines(origin: Vec3, u: Vec3, v: Vec3, size: f32) -> Vec<LineVertex> {
    let hu = u * size;
    let hv = v * size;
    let corners = [
        origin + hu + hv,
        origin + hu - hv,
        origin - hu - hv,
        origin - hu + hv,
    ];
    let mut lines = Vec::with_capacity(8);
    for i in 0..4 {
        let next = (i + 1) % 4;
        lines.push(LineVertex {
            position: corners[i].to_array(),
        });
        lines.push(LineVertex {
            position: corners[next].to_array(),
        });
    }
    lines
}

/// Hit-test the plane quad or the normal arrow. Returns world distance along the ray.
pub fn hit_test(ray: &Ray, plane: [f32; 4]) -> Option<f32> {
    let (n, origin, u, v) = plane_frame(plane)?;
    let size = WIDGET_SIZE;
    let quad = ray_hit_quad(ray, origin, n, u, v, size);
    let arrow = ray_hit_axis_capsule(ray, origin, n, size * 0.85, ARROW_HIT_RADIUS);
    match (quad, arrow) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn ray_hit_quad(ray: &Ray, origin: Vec3, n: Vec3, u: Vec3, v: Vec3, size: f32) -> Option<f32> {
    let denom = ray.direction.dot(n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (origin - ray.origin).dot(n) / denom;
    if t <= 0.0 {
        return None;
    }
    let p = ray.origin + ray.direction * t;
    let d = p - origin;
    if d.dot(u).abs() <= size && d.dot(v).abs() <= size {
        Some(t)
    } else {
        None
    }
}

fn ray_hit_axis_capsule(
    ray: &Ray,
    origin: Vec3,
    axis: Vec3,
    length: f32,
    radius: f32,
) -> Option<f32> {
    let ro = ray.origin - origin;
    let rd = ray.direction;
    let a = rd.dot(rd) - rd.dot(axis).powi(2);
    let b = 2.0 * (ro.dot(rd) - ro.dot(axis) * rd.dot(axis));
    let c = ro.dot(ro) - ro.dot(axis).powi(2) - radius * radius;
    if a.abs() < 1e-10 {
        return None;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / (2.0 * a);
    if t <= 0.0 {
        return None;
    }
    let p = ray.origin + rd * t;
    let s = (p - origin).dot(axis);
    if s >= 0.0 && s <= length {
        Some(t)
    } else {
        None
    }
}

/// Closest point on the plane normal line to the ray (for offset drag).
pub fn drag_point_on_normal(ray: &Ray, plane: [f32; 4]) -> Option<Vec3> {
    let (n, origin, _, _) = plane_frame(plane)?;
    Some(closest_point_on_axis(ray, origin, n))
}

fn closest_point_on_axis(ray: &Ray, origin: Vec3, axis: Vec3) -> Vec3 {
    let d1 = ray.direction;
    let d2 = axis;
    let r = ray.origin - origin;
    let a = d1.dot(d1);
    let b = d1.dot(d2);
    let c = d2.dot(d2);
    let d = d1.dot(r);
    let e = d2.dot(r);
    let denom = a * c - b * b;
    let t2 = if denom.abs() < 1e-10 {
        0.0
    } else {
        (a * e - b * d) / denom
    };
    origin + d2 * t2
}

pub fn plane_from_point(normal: Vec3, point: Vec3) -> [f32; 4] {
    let n = if normal.length_squared() < 1e-12 {
        Vec3::Y
    } else {
        normal.normalize()
    };
    [n.x, n.y, n.z, -n.dot(point)]
}

pub fn set_section_plane(graph: &mut SceneGraph, id: NodeId, plane: [f32; 4]) {
    if let Some(e) = graph.get_mut(id) {
        if let NodeData::SectionPlane(sp) = &mut e.data {
            sp.plane = plane;
        }
    }
}

pub fn set_section_enabled(graph: &mut SceneGraph, id: NodeId, enabled: bool) {
    if let Some(e) = graph.get_mut(id) {
        if let NodeData::SectionPlane(sp) = &mut e.data {
            sp.enabled = enabled;
        }
    }
}
