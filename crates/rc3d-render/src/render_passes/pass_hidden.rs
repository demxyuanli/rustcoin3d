use super::PassContext;
use crate::gpu_resource::EdgeLineKind;
use crate::vertex::FlatUniforms;
use rc3d_core::Appearance;

fn hidden_edge_kind(app: Appearance) -> EdgeLineKind {
    if app.wants_full_edges() {
        EdgeLineKind::WireframeFull
    } else {
        EdgeLineKind::Feature
    }
}

/// Fast Hidden Line: dashed fragments that fail the visible-edge depth test.
pub(super) fn pass_hidden_edges(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    if ctx.performance_mode_active {
        return;
    }
    if !ctx.visible.iter().any(|dc| dc.appearance().wants_hidden_dashes()) {
        return;
    }

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Hidden Line Pass (dashed)"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
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
        occlusion_query_set: None, multiview_mask: None,
    });
    renderer.apply_scene_viewport(&mut pass);

    pass.set_stencil_reference(0);
    pass.set_pipeline(&scene_pl.edge_overlay_hidden);

    let mut last_bound_edge_mesh = None;
    for (i, dc) in ctx.visible.iter().enumerate() {
        if !dc.appearance().wants_hidden_dashes() {
            continue;
        }
        let kind = hidden_edge_kind(dc.appearance());
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: renderer.hidden_edge_color,
            model: dc.model_matrix.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) else {
            continue;
        };
        pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);

        let cpu_overlay = dc.appearance().wants_cpu_edge_overlay();
        let drawn = if cpu_overlay {
            false
        } else if let Some(mesh_id) = ctx.mesh_handles[i] {
            renderer.draw_edges_batched(&mut pass, mesh_id, &mut last_bound_edge_mesh, kind)
        } else {
            false
        };
        if !drawn {
            renderer.bind_and_draw_edges(
                &mut pass,
                dc,
                if cpu_overlay {
                    None
                } else {
                    ctx.mesh_handles[i]
                },
                kind,
            );
            last_bound_edge_mesh = None;
        }
    }
}
