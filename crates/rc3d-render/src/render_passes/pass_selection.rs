use super::PassContext;
use crate::vertex::FlatUniforms;

pub(super) fn pass_selection_fill(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let pl = scene_pl;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Selection Fill Pass"),
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
    pass.set_pipeline(&pl.selection_fill);
    let mut last_bound_mesh = None;
    for &i in ctx.selected_order {
        let dc = ctx.visible[i];
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: [1.0, 0.6, 0.0, 0.35],
        };
        if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
            }
        }
    }
}

pub(super) fn pass_selection_edge(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let pl = scene_pl;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Selection Edge Pass"),
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
    pass.set_pipeline(&pl.wireframe);
    let sel_color = [1.0, 0.5, 0.0, 1.0];
    let mut last_bound_mesh = None;
    for &i in ctx.selected_order {
        let dc = ctx.visible[i];
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: sel_color,
        };
        if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
            }
        }
    }
}
