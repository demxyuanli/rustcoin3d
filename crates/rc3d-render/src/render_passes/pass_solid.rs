use super::PassContext;

pub(super) fn pass_depth_prepass(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let prepass_pipeline = scene_pl.solid_depth_prepass.clone();
    let depth_clear = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
    let load_op = if renderer.gpu.bg_pass.is_some() {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(ctx.bg_color)
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Depth prepass (HZB)"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: load_op,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(depth_clear),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0u32),
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None, multiview_mask: None,
    });
    // Prepass uploads the shared instance/indirect buffers; the solid pass
    // re-records the identical layout without re-uploading (see draw_opaque).
    super::draw_opaque_triangle_batches(renderer, &mut pass, ctx, &prepass_pipeline, &scene_pl.flat_solid, true);
}

pub(super) fn pass_solid_and_outline(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    reuse_prepass_depth: bool,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let solid_pipeline = scene_pl.solid.clone();
    let depth_clear = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
    let depth_load = if reuse_prepass_depth {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(depth_clear)
    };
    let stencil_load = if reuse_prepass_depth {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(0u32)
    };
    let color_load = if reuse_prepass_depth || renderer.gpu.bg_pass.is_some() {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(ctx.bg_color)
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Solid Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: color_load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: depth_load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: stencil_load,
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None, multiview_mask: None,
    });

    // Meshlet draw path: uses basic (full-buffer direct draw) or GPU cull
    // (compact+indirect) depending on meshlet_gpu_cull_enabled.
    // When the depth prepass already ran this frame it uploaded identical
    // instance/indirect data; skip the redundant upload here.
    super::draw_opaque_triangle_batches(renderer, &mut pass, ctx, &solid_pipeline, &scene_pl.flat_solid, !reuse_prepass_depth);
}
