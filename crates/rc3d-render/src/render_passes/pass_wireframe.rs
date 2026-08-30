use super::PassContext;
use crate::gpu_resource::EdgeLineKind;
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
    // When any sibling is filled (subtree SoDrawStyle), keep the solid pass.
    // Full-scene wireframe still clears because the solid pass did not run.
    let preserve_solid = ctx.visible.iter().any(|dc| dc.appearance().wants_filled());
    let color_load = if preserve_solid {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(ctx.bg_color)
    };
    let depth_load = if preserve_solid {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(depth_clear)
    };
    let stencil_load = if preserve_solid {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(0u32)
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Wireframe Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations { load: color_load, store: wgpu::StoreOp::Store },
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
    renderer.apply_scene_viewport(&mut pass);

    pass.set_stencil_reference(0);
    // Line list per topological edge (same as edge overlay). PolygonMode::Line draws each
    // triangle edge twice with near-identical depth and no depth write, causing z-fighting
    // and broken-looking segments on shared edges.
    pass.set_pipeline(&pl.edge_overlay);
    let mut last_bound_edge_mesh = None;
    for &i in ctx.solid_order {
        let dc = ctx.visible[i];
        if !dc.appearance().wants_full_edges() {
            continue;
        }
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: renderer.wireframe_edge_color,
            model: dc.model_matrix.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            let drawn = if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_edges_batched(
                    &mut pass,
                    mesh_id,
                    &mut last_bound_edge_mesh,
                    EdgeLineKind::WireframeFull,
                )
            } else {
                false
            };
            if !drawn {
                renderer.bind_and_draw_edges(
                    &mut pass,
                    dc,
                    ctx.mesh_handles[i],
                    EdgeLineKind::WireframeFull,
                );
            }
        }
    }
}
