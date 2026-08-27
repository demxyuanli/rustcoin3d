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
    let up = if dir.cross(Vec3::Y).length_squared() < 1e-4 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(eye, center, up);
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
    let proj = {
        // RH look-at: points in front have negative view Z. glam/wgpu ortho
        // `near`/`far` are positive distances along the view axis.
        let mut near_dist = -max_v.z;
        let mut far_dist = -min_v.z;
        if far_dist < near_dist {
            core::mem::swap(&mut near_dist, &mut far_dist);
        }
        let near_dist = (near_dist - z_margin).max(0.01);
        let far_dist = (far_dist + z_margin).max(near_dist + 0.01);
        Mat4::orthographic_rh(min_v.x, max_v.x, min_v.y, max_v.y, near_dist, far_dist)
    };
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

fn unproject_ndc(inv_view_proj: Mat4, x: f32, y: f32, z: f32) -> Vec3 {
    let world = inv_view_proj * Vec4::new(x, y, z, 1.0);
    world.truncate() / world.w.max(1e-8)
}

/// Camera-space distances to the projection near/far plane centers.
/// Used as the NDC 0..1 interpolant span so cascade slices match PBR sampling.
pub fn projection_near_far(
    inv_view_proj: Mat4,
    camera_pos: Vec3,
    reversed_z: bool,
) -> (f32, f32) {
    let (ndc_z_near, ndc_z_far) = if reversed_z { (1.0, 0.0) } else { (0.0, 1.0) };
    let near_c = unproject_ndc(inv_view_proj, 0.0, 0.0, ndc_z_near);
    let far_c = unproject_ndc(inv_view_proj, 0.0, 0.0, ndc_z_far);
    let proj_near = (near_c - camera_pos).length().max(1e-3);
    let proj_far = (far_c - camera_pos).length().max(proj_near + 1e-3);
    (proj_near, proj_far)
}

/// Practical CSM split far: follow the camera projection so slices stay inside
/// the view frustum, but never exceed the scene extent. CAD/import cameras set
/// `far = max(distance * 20, 100)` which is often 100..4000 and would stretch
/// cascades across empty space if used as the split far.
pub fn practical_csm_far(scene_far: f32, proj_far: f32) -> f32 {
    scene_far.min(proj_far).clamp(24.0, 1000.0)
}

/// Extract the 8 corners of a camera frustum sub-volume defined by near/far planes.
/// `near` / `far` are 0..1 along camera near-to-far frustum edges.
/// `ndc_z_near` / `ndc_z_far` are the clip-space Z of those planes (0/1 forward-Z, 1/0 reverse-Z).
pub fn frustum_slice_corners(
    inv_view_proj: Mat4,
    near: f32,
    far: f32,
    ndc_z_near: f32,
    ndc_z_far: f32,
) -> [Vec3; 8] {
    let xy = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
    let mut result = [Vec3::ZERO; 8];
    for i in 0..4 {
        let near_pt = unproject_ndc(inv_view_proj, xy[i][0], xy[i][1], ndc_z_near);
        let far_pt = unproject_ndc(inv_view_proj, xy[i][0], xy[i][1], ndc_z_far);
        result[i] = near_pt + (far_pt - near_pt) * near;
        result[i + 4] = near_pt + (far_pt - near_pt) * far;
    }
    result
}

/// Compute the world-space AABB of a frustum slice.
/// `near_t` / `far_t` are 0..1 along camera near-to-far frustum edges.
fn frustum_slice_aabb(
    inv_view_proj: Mat4,
    near_t: f32,
    far_t: f32,
    ndc_z_near: f32,
    ndc_z_far: f32,
) -> Aabb {
    let corners = frustum_slice_corners(
        inv_view_proj,
        near_t.clamp(0.0, 1.0),
        far_t.clamp(0.0, 1.0),
        ndc_z_near,
        ndc_z_far,
    );
    let mut aabb = Aabb::empty();
    for c in &corners {
        aabb = aabb.union(&Aabb::from_point(*c));
    }
    aabb
}

/// Compute light view-proj for each CSM cascade.
/// `splits` are view-space distances from [`compute_csm_splits`] (not 0..1).
///
/// Slice interpolation must use the **projection** near/far (NDC 0..1 span), not the
/// tightened split far. Using the split far as the interpolant denominator places
/// cascades far behind the casters, so PBR sampling misses the shadow map.
pub fn csm_light_view_projs(
    light_dir: Vec3,
    inv_view_proj: Mat4,
    splits: &[f32],
    z_margin: f32,
    camera_pos: Vec3,
    reversed_z: bool,
) -> Vec<Mat4> {
    let (ndc_z_near, ndc_z_far) = if reversed_z { (1.0, 0.0) } else { (0.0, 1.0) };
    let (proj_near, proj_far) = projection_near_far(inv_view_proj, camera_pos, reversed_z);
    let denom = proj_far - proj_near;
    let mut vps = Vec::with_capacity(splits.len().saturating_sub(1));
    for i in 0..splits.len().saturating_sub(1) {
        let t_n = ((splits[i] - proj_near) / denom).clamp(0.0, 1.0);
        let t_f = ((splits[i + 1] - proj_near) / denom).clamp(0.0, 1.0);
        let slice_aabb = frustum_slice_aabb(inv_view_proj, t_n, t_f, ndc_z_near, ndc_z_far);
        vps.push(directional_light_view_proj(light_dir, &slice_aabb, z_margin));
    }
    vps
}

pub fn aabb_from_scene(graph: &SceneGraph) -> Option<Aabb> {
    rc3d_scene::GetBoundingBoxAction::compute_scene_aabb(graph)
}

pub fn primary_directional_light_dir(graph: &SceneGraph) -> Option<Vec3> {
    rc3d_scene::LightSubsystem::primary_directional_dir(graph)
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
