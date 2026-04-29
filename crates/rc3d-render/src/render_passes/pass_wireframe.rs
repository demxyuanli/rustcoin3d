use super::PassContext;
use crate::vertex::FlatUniforms;

pub(super) fn pass_wireframe(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let pl = scene_pl;
    let depth_clear = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Wireframe Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(ctx.bg_color), store: wgpu::StoreOp::Store },
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

    pass.set_pipeline(&pl.wireframe);
    let line_color = [0.2, 1.0, 0.4, 1.0];
    let mut last_bound_mesh = None;
    for &i in ctx.solid_order {
        let dc = ctx.visible[i];
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: line_color,
        };
        if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
            }
        }
    }
}
