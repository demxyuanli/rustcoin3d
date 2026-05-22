//! Meshlet culling module for GPU-driven rendering.
//!
//! Runs meshlet cull compute passes using hierarchical z-buffer (HZB) for frustum culling.

use crate::render_passes::PassContext;

/// Run meshlet cull compute passes in the given encoder.
///
/// Uses the same encoder as subsequent render passes so wgpu inserts
/// implicit barriers between compute (STORAGE write) and render (INDIRECT/INDEX read).
pub(crate) fn submit_meshlet_cull(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &PassContext<'_>,
    hzb_enabled: bool,
    hzb_dims: (u32, u32),
    mip_max: u32,
    hzb_need_max: bool,
    hzb_need_min: bool,
) {
    let Some(cluster_renderer) = renderer.gpu.cluster_renderer.as_ref() else {
        return;
    };
    let Some(hzb) = renderer.gpu.hzb.as_ref() else {
        return;
    };

    let max_bind: &wgpu::TextureView = if hzb_need_max {
        &hzb.max_pyramid.full_view
    } else {
        &hzb.min_pyramid.full_view
    };
    let min_bind: &wgpu::TextureView = if hzb_need_min {
        &hzb.min_pyramid.full_view
    } else {
        &hzb.max_pyramid.full_view
    };

    for &vis_idx in ctx.meshlet_indices {
        let dc = ctx.visible[vis_idx];
        let Some(md) = dc.meshlet_data.as_ref() else {
            continue;
        };
        let ptr = std::sync::Arc::as_ptr(md) as u64;
        if let Some(cluster_set) = renderer.gpu.assets.cluster_get(&ptr) {
            let model_inv = dc.model_matrix.inverse();
            let cam_model = model_inv.transform_point3(glam::Vec3::from(ctx.camera_pos));
            cluster_renderer.cull_and_compact(
                &renderer.device,
                &renderer.queue,
                encoder,
                cluster_set,
                dc.mvp.to_cols_array_2d(),
                [cam_model.x, cam_model.y, cam_model.z],
                1,
                0,
                false,
                max_bind,
                min_bind,
                hzb_dims,
                mip_max,
                hzb_enabled,
                dc.depth_reversed_z,
                dc.projection_orthographic,
            );
        }
    }
}
