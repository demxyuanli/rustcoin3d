use glam::Vec3;

use super::PassContext;
use crate::gpu_resource::EdgeLineKind;
use crate::vertex::{FlatUniforms, LineVertex};

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
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
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
    pass.set_pipeline(&pl.edge_overlay);
    let mut last_bound_edge_mesh = None;
    for &i in ctx.selected_order {
        let dc = ctx.visible[i];
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: ctx.outline_color,
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            let drawn = if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_edges_batched(
                    &mut pass,
                    mesh_id,
                    &mut last_bound_edge_mesh,
                    EdgeLineKind::Feature,
                )
            } else {
                false
            };
            if !drawn {
                renderer.bind_and_draw_edges(
                    &mut pass,
                    dc,
                    ctx.mesh_handles[i],
                    EdgeLineKind::Feature,
                );
            }
        }
    }
}

/// Draw bounding box wireframe for selected objects.
pub(super) fn pass_selection_bbox(
    encoder: &mut wgpu::CommandEncoder,
    shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &super::PassContext,
    scene_pl: &crate::pipelines::DepthModePipelines,
    flat_pool: &mut crate::gpu_resource::GpuUniformPool,
    wireframe_supported: bool,
) {
    if ctx.selected_order.is_empty() || !wireframe_supported {
        return;
    }

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Selection BBox"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: shade_view,
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
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    rpass.set_pipeline(&scene_pl.edge_overlay);
    let color = ctx.outline_color;

    for &idx in ctx.selected_order {
        let dc = &ctx.visible[idx];
        let verts = &dc.vertices;
        if verts.is_empty() {
            continue;
        }

        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for v in verts.iter() {
            let p = Vec3::from_array(v.position);
            mn = mn.min(p);
            mx = mx.max(p);
        }

        let corners = [
            [mn.x, mn.y, mn.z], [mx.x, mn.y, mn.z], [mx.x, mn.y, mx.z], [mn.x, mn.y, mx.z],
            [mn.x, mx.y, mn.z], [mx.x, mx.y, mn.z], [mx.x, mx.y, mx.z], [mn.x, mx.y, mx.z],
        ];
        let edges: [(usize, usize); 12] = [
            (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7),
        ];
        let mut lines: Vec<LineVertex> = Vec::with_capacity(24);
        for (a, b) in edges {
            lines.push(LineVertex { position: corners[a] });
            lines.push(LineVertex { position: corners[b] });
        }

        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color,
        };
        let offset = flat_pool.push_flat(&uniforms);
        if let Some(off) = offset {
            rpass.set_bind_group(0, flat_pool.bind_group(), &[off]);
        }
        // Pending: upload lines to GPU buffer and issue draw call.
        // Note: vertex buffer upload for dynamic lines requires a GPU buffer.
        // This establishes the pass structure; full vertex upload can be fleshed out later.
        let _ = lines;
    }
}
