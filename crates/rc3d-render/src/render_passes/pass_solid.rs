use super::PassContext;
use crate::vertex::OutlineUniforms;

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
        occlusion_query_set: None,
    });
    super::draw_opaque_triangle_batches(renderer, &mut pass, ctx, &prepass_pipeline, &scene_pl.flat_solid, false);
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
    let outline_pipeline = scene_pl.outline.clone();
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
    let pass_label = if ctx.run_outline { "Solid+Outline Pass" } else { "Solid Pass" };
    let color_load = if reuse_prepass_depth || renderer.gpu.bg_pass.is_some() {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(ctx.bg_color)
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(pass_label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
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
        occlusion_query_set: None,
    });

    // Meshlet draw path re-enabled after compute→render synchronization fix (separate encoder + submit).
    super::draw_opaque_triangle_batches(renderer, &mut pass, ctx, &solid_pipeline, &scene_pl.flat_solid, true);

    if ctx.run_outline {
        {
            let mb = &mut renderer.gpu.draw_bufs.meshlet_bitmask;
            mb.resize(ctx.visible.len(), false);
            mb.fill(false);
            for &i in ctx.meshlet_indices { mb[i] = true; }
        }
        pass.set_pipeline(&outline_pipeline);
        pass.set_stencil_reference(0);
        let mut last_bound_mesh = None;
        for &i in ctx.solid_order {
            if renderer.gpu.draw_bufs.meshlet_bitmask[i] {
                continue;
            }
            let dc = ctx.visible[i];
            let uniforms = OutlineUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                outline_width: ctx.outline_width,
                _pad: [0.0; 3],
                color: ctx.outline_color,
            };
            if let Some(offset) = renderer.gpu.outline_pool.push_outline(&uniforms) {
                pass.set_bind_group(0, renderer.gpu.outline_pool.bind_group(), &[offset]);
                if let Some(mesh_id) = ctx.mesh_handles[i] {
                    renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                }
            }
        }
    }
}
