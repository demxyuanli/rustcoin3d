//! Directional shadow: orthographic light projection and scene bounds helpers.

use glam::{Mat4, Vec3};
use rc3d_core::Aabb;
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// WebGPU-style orthographic projection: clip Z in [0, 1], right-handed, -Z forward in view space.
pub fn orthographic_wgpu_rh(left: f32, right: f32, bottom: f32, top: f32, z_near: f32, z_far: f32) -> Mat4 {
    let rml = right - left;
    let tmb = top - bottom;
    let fmn = z_far - z_near;
    let inv_rml = 1.0 / rml;
    let inv_tmb = 1.0 / tmb;
    let inv_fmn = 1.0 / fmn;
    Mat4::from_cols_array_2d(&[
        [2.0 * inv_rml, 0.0, 0.0, 0.0],
        [0.0, 2.0 * inv_tmb, 0.0, 0.0],
        [0.0, 0.0, -inv_fmn, 0.0],
        [-(left + right) * inv_rml, -(top + bottom) * inv_tmb, -z_near * inv_fmn, 1.0],
    ])
}

/// Fit orthographic bounds to world AABB corners in light view space.
pub fn directional_light_view_proj(light_dir_world: Vec3, world_aabb: &Aabb, z_margin: f32) -> Mat4 {
    let dir = light_dir_world.normalize();
    let center = world_aabb.center();
    let extent = world_aabb.size().length();
    let eye = center - dir * (extent * 0.5 + 5.0);
    let view = Mat4::look_at_rh(eye, center, Vec3::Y);
    let corners = [
        Vec3::new(world_aabb.min.x, world_aabb.min.y, world_aabb.min.z),
        Vec3::new(world_aabb.max.x, world_aabb.min.y, world_aabb.min.z),
        Vec3::new(world_aabb.min.x, world_aabb.max.y, world_aabb.min.z),
        Vec3::new(world_aabb.max.x, world_aabb.max.y, world_aabb.min.z),
        Vec3::new(world_aabb.min.x, world_aabb.min.y, world_aabb.max.z),
        Vec3::new(world_aabb.max.x, world_aabb.min.y, world_aabb.max.z),
        Vec3::new(world_aabb.min.x, world_aabb.max.y, world_aabb.max.z),
        Vec3::new(world_aabb.max.x, world_aabb.max.y, world_aabb.max.z),
    ];
    let mut min_v = Vec3::splat(f32::MAX);
    let mut max_v = Vec3::splat(f32::MIN);
    for c in &corners {
        let v = view.transform_point3(*c);
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let proj = orthographic_wgpu_rh(
        min_v.x,
        max_v.x,
        min_v.y,
        max_v.y,
        min_v.z - z_margin,
        max_v.z + z_margin,
    );
    proj * view
}

pub fn aabb_from_scene(graph: &SceneGraph) -> Option<Aabb> {
    let mut bb = rc3d_actions::GetBoundingBoxAction::new();
    for &root in graph.roots() {
        bb.apply(graph, root);
    }
    if bb.bounding_box.min.x <= bb.bounding_box.max.x {
        Some(bb.bounding_box)
    } else {
        None
    }
}

pub fn primary_directional_light_dir(graph: &SceneGraph) -> Option<Vec3> {
    for &root in graph.roots() {
        if let Some(d) = walk_dir_light(graph, root) {
            return Some(d);
        }
    }
    None
}

fn walk_dir_light(graph: &SceneGraph, node: NodeId) -> Option<Vec3> {
    let entry = graph.get(node)?;
    match &entry.data {
        NodeData::DirectionalLight(l) => Some(l.direction.normalize()),
        NodeData::Separator(_) => {
            for &c in &entry.children {
                if let Some(d) = walk_dir_light(graph, c) {
                    return Some(d);
                }
            }
            None
        }
        NodeData::Group(_) | NodeData::Transform(_) => {
            for &c in &entry.children {
                if let Some(d) = walk_dir_light(graph, c) {
                    return Some(d);
                }
            }
            None
        }
        _ => None,
    }
}

pub fn union_draw_call_aabbs<'a, I>(draw_calls: I) -> Option<Aabb>
where
    I: IntoIterator<Item = &'a crate::render_action::DrawCall>,
{
    let mut acc: Option<Aabb> = None;
    for dc in draw_calls {
        if let Some(ref a) = dc.aabb {
            acc = Some(match acc {
                None => a.clone(),
                Some(u) => u.union(a),
            });
        }
    }
    acc
}
