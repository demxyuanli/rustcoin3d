use super::PassContext;
use crate::vertex::FlatUniforms;

pub(super) fn pass_edge_overlay(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    edge_worthy: bool,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let pl = scene_pl;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Edge Overlay Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_stencil_reference(0);
    pass.set_pipeline(&pl.edge_overlay);

    let mut last_bound_edge_mesh = None;
    for &i in ctx.edge_order {
        let dc = ctx.visible[i];
        if !edge_worthy && dc.overlay_color.is_none() {
            continue;
        }
        let edge_color = dc.overlay_color.unwrap_or(ctx.outline_color);
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: edge_color,
        };
        if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
            let drawn_from_cache = if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_edges_batched(&mut pass, mesh_id, &mut last_bound_edge_mesh)
            } else {
                false
            };
            if !drawn_from_cache {
                renderer.bind_and_draw_edges(&mut pass, dc, ctx.mesh_handles[i]);
            }
        }
    }
}
