//! Directional shadow: orthographic light projection, scene bounds, and CSM helpers.

use glam::{Mat4, Vec3, Vec4};
use rc3d_core::Aabb;
use rc3d_scene::SceneGraph;

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

/// Compute CSM split depths using the practical split scheme (logarithmic-uniform blend).
/// Returns `cascade_count + 1` depths: [near, split1, split2, ..., far].
/// `lambda` controls the blend: 0.0 = uniform, 1.0 = logarithmic.
pub fn compute_csm_splits(near: f32, far: f32, cascade_count: u32, lambda: f32) -> Vec<f32> {
    let count = cascade_count.max(1);
    let mut splits = Vec::with_capacity(count as usize + 1);
    splits.push(near);
    for i in 1..=count {
        let p = i as f32 / count as f32;
        let log_split = near * (far / near).powf(p);
        let uni_split = near + (far - near) * p;
        let split = lambda * log_split + (1.0 - lambda) * uni_split;
        splits.push(split);
    }
    splits
}

/// Extract the 8 corners of a camera frustum sub-volume defined by near/far planes.
pub fn frustum_slice_corners(
    inv_view_proj: Mat4,
    near: f32,
    far: f32,
) -> [Vec3; 8] {
    let ndc_corners = [
        Vec4::new(-1.0, -1.0, 0.0, 1.0),
        Vec4::new(1.0, -1.0, 0.0, 1.0),
        Vec4::new(-1.0, 1.0, 0.0, 1.0),
        Vec4::new(1.0, 1.0, 0.0, 1.0),
        Vec4::new(-1.0, -1.0, 1.0, 1.0),
        Vec4::new(1.0, -1.0, 1.0, 1.0),
        Vec4::new(-1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
    ];
    let mut world_corners = [Vec3::ZERO; 8];
    for (i, ndc) in ndc_corners.iter().enumerate() {
        let world = inv_view_proj * *ndc;
        world_corners[i] = (world / world.w).truncate();
    }
    // Interpolate between near and far planes
    let mut result = [Vec3::ZERO; 8];
    for i in 0..4 {
        let near_pt = world_corners[i];
        let far_pt = world_corners[i + 4];
        result[i] = near_pt + (far_pt - near_pt) * near;
        result[i + 4] = near_pt + (far_pt - near_pt) * far;
    }
    result
}

/// Compute the world-space AABB of a frustum slice.
fn frustum_slice_aabb(inv_view_proj: Mat4, near: f32, far: f32) -> Aabb {
    let ndc_corners = [
        Vec4::new(-1.0, -1.0, 0.0, 1.0),
        Vec4::new(1.0, -1.0, 0.0, 1.0),
        Vec4::new(-1.0, 1.0, 0.0, 1.0),
        Vec4::new(1.0, 1.0, 0.0, 1.0),
        Vec4::new(-1.0, -1.0, 1.0, 1.0),
        Vec4::new(1.0, -1.0, 1.0, 1.0),
        Vec4::new(-1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
    ];
    let mut min_v = Vec3::splat(f32::MAX);
    let mut max_v = Vec3::splat(f32::MIN);
    for ndc in &ndc_corners {
        let world = inv_view_proj * *ndc;
        let p = (world / world.w).truncate();
        min_v = min_v.min(p);
        max_v = max_v.max(p);
    }
    // Lerp near/far
    let near_lerp = near;  // simplified: NDC z=0 maps to near, z=1 maps to far
    let far_lerp = far;
    let rng = max_v - min_v;
    let zn = min_v.z + rng.z * near_lerp;
    let zf = min_v.z + rng.z * far_lerp;
    Aabb {
        min: Vec3::new(min_v.x, min_v.y, zn),
        max: Vec3::new(max_v.x, max_v.y, zf),
    }
}

/// Compute light view-proj for each CSM cascade.
pub fn csm_light_view_projs(
    light_dir: Vec3,
    inv_view_proj: Mat4,
    splits: &[f32],
    z_margin: f32,
) -> Vec<Mat4> {
    let mut vps = Vec::with_capacity(splits.len() - 1);
    for i in 0..(splits.len() - 1) {
        let near = splits[i];
        let far = splits[i + 1];
        let slice_aabb = frustum_slice_aabb(inv_view_proj, near, far);
        let vp = directional_light_view_proj(light_dir, &slice_aabb, z_margin);
        vps.push(vp);
    }
    vps
}

pub fn aabb_from_scene(graph: &SceneGraph) -> Option<Aabb> {
    let mut bb = rc3d_actions::GetBoundingBoxAction::new();
    rc3d_actions::apply_to_all_roots(&mut bb, graph);
    if bb.bounding_box.min.x <= bb.bounding_box.max.x {
        Some(bb.bounding_box)
    } else {
        None
    }
}

pub fn primary_directional_light_dir(graph: &SceneGraph) -> Option<Vec3> {
    rc3d_actions::LightSubsystem::primary_directional_dir(graph)
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
