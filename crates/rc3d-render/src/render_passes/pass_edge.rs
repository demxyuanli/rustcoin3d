use super::PassContext;
use crate::gpu_resource::EdgeLineKind;
use crate::vertex::{FlatUniforms, LineUniforms};

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

    // Reset line pool cursor for this pass
    renderer.gpu.line_pool.reset();

    let viewport_w = renderer.current_pass_viewport().width.max(1) as f32;
    let viewport_h = renderer.current_pass_viewport().height.max(1) as f32;

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Edge Overlay Pass (AA)"),
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
    renderer.apply_scene_viewport(&mut pass);

    pass.set_stencil_reference(0);
    pass.set_pipeline(&pl.edge_overlay_aa);

    let edge_kind = if renderer.wireframe_overlay {
        EdgeLineKind::WireframeFull
    } else {
        EdgeLineKind::Feature
    };
    let default_edge_color = match edge_kind {
        EdgeLineKind::WireframeFull => renderer.wireframe_edge_color,
        EdgeLineKind::Feature => renderer.feature_edge_color,
    };

    let mut last_bound_edge_mesh = None;
    for &i in ctx.edge_order {
        let dc = ctx.visible[i];
        if dc.overlay_color.is_none() && !dc.appearance().wants_edge_overlay() {
            continue;
        }
        let edge_color = dc.overlay_color.unwrap_or(default_edge_color);

        // Wireframe overlay uses the legacy path (expanded buffer only has feature edges)
        if renderer.wireframe_overlay {
            drop(pass);
            pass_fallback_edge(renderer, encoder, view, depth_view, ctx, edge_worthy, scene_pl, edge_kind);
            return;
        }

        // Classified overlays (silhouette / perimeter / hard / adjacent) live on
        // DrawCall.edge_positions, not the shared GPU crease buffer.
        if dc.appearance().wants_cpu_edge_overlay() {
            pass.set_pipeline(&pl.edge_overlay);
            let uniforms = FlatUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                color: edge_color,
                model: dc.model_matrix.to_cols_array_2d(),
                clip_planes: [[0.0; 4]; 6],
                clip_count: [0.0, 0.0, 0.0, 0.0],
                ..Default::default()
            };
            if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
                pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
                renderer.bind_and_draw_edges(&mut pass, dc, None, EdgeLineKind::Feature);
            }
            pass.set_pipeline(&pl.edge_overlay_aa);
            last_bound_edge_mesh = None;
            continue;
        }

        // Default line width 1.5px with 1.5px AA feather
        let half_width_px = 1.0;
        let aa_px = 1.0;

        // First try the AA pipeline with expanded edge geometry
        let line_uniforms = LineUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: edge_color,
            line_params: [half_width_px, aa_px, viewport_w, viewport_h],
        };
        if let Some(offset) = renderer.gpu.line_pool.push_line(&line_uniforms) {
            pass.set_bind_group(0, renderer.gpu.line_pool.bind_group(), &[offset]);
            let drawn = if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_expanded_edges_batched(
                    &mut pass,
                    mesh_id,
                    &mut last_bound_edge_mesh,
                )
            } else {
                false
            };
            if !drawn {
                // Fallback: use legacy LineList rendering
                drop(pass);
                pass_fallback_edge(renderer, encoder, view, depth_view, ctx, edge_worthy, scene_pl, edge_kind);
                return;
            }
        }
    }
}

/// Fallback edge pass using the original LineList pipeline.
fn pass_fallback_edge(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    _edge_worthy: bool,
    scene_pl: &crate::pipelines::DepthModePipelines,
    edge_kind: EdgeLineKind,
) {
    let pl = scene_pl;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Edge Overlay Pass (fallback)"),
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
    renderer.apply_scene_viewport(&mut pass);

    pass.set_stencil_reference(0);
    pass.set_pipeline(&pl.edge_overlay);

    let default_edge_color = match edge_kind {
        EdgeLineKind::WireframeFull => renderer.wireframe_edge_color,
        EdgeLineKind::Feature => renderer.feature_edge_color,
    };
    let mut last_bound_edge_mesh = None;
    for &i in ctx.edge_order {
        let dc = ctx.visible[i];
        if dc.overlay_color.is_none() && !dc.appearance().wants_edge_overlay() {
            continue;
        }
        let edge_color = dc.overlay_color.unwrap_or(default_edge_color);
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: edge_color,
            model: dc.model_matrix.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            let drawn_from_cache = if dc.appearance().wants_cpu_edge_overlay() {
                false
            } else if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_edges_batched(
                    &mut pass,
                    mesh_id,
                    &mut last_bound_edge_mesh,
                    edge_kind,
                )
            } else {
                false
            };
            if !drawn_from_cache {
                renderer.bind_and_draw_edges(
                    &mut pass,
                    dc,
                    if dc.appearance().wants_cpu_edge_overlay() {
                        None
                    } else {
                        ctx.mesh_handles[i]
                    },
                    edge_kind,
                );
            }
        }
    }
}
