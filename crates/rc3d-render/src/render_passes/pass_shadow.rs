use super::PassContext;
use crate::vertex::{ShadowDrawUniforms, CSM_CASCADE_COUNT};

pub(super) fn pass_shadow_depth(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &PassContext<'_>,
) {
    let Some(ref csm) = renderer.gpu.csm_shadow else {
        return;
    };

    let cascade_count = csm.cascade_count.min(CSM_CASCADE_COUNT as u32);

    if renderer.gpu.multiview_supported {
        // ── Layered rendering: single pass, instanced draws into D2Array ──
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("CSM Shadow (layered)"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &csm.array_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&renderer.gpu.pipelines.shadow_depth);
        let mut last_bound_mesh = None;

        for &draw_idx in ctx.solid_order {
            let dc = ctx.visible[draw_idx];
            let mesh_id = match ctx.mesh_handles[draw_idx] {
                Some(id) => id,
                None => continue,
            };

            let uniforms: [ShadowDrawUniforms; CSM_CASCADE_COUNT] = std::array::from_fn(|c| {
                let vp = ctx.csm_view_proj[c] * dc.model_matrix;
                ShadowDrawUniforms {
                    shadow_mvp: vp.to_cols_array_2d(),
                }
            });

            if let Some(offset) = renderer.gpu.shadow_pool.push_shadow_array(&uniforms) {
                pass.set_bind_group(0, renderer.gpu.shadow_pool.bind_group(), &[offset]);
                renderer.draw_mesh_instanced(&mut pass, mesh_id, cascade_count, &mut last_bound_mesh);
            }
        }
    } else {
        // ── Fallback: per-cascade render passes (MULTIVIEW unsupported) ──
        // Uses per-layer D2 views and non-instanced draws.
        // Pushes a full [ShadowDrawUniforms; 4] array to match the shader binding,
        // but the vertex shader only reads su[0] since instance_idx defaults to 0.
        let dummy = ShadowDrawUniforms {
            shadow_mvp: [[0.0; 4]; 4],
        };

        for cascade_idx in 0..cascade_count {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("CSM Shadow cascade {}", cascade_idx)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &csm.cascade_views[cascade_idx as usize],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&renderer.gpu.pipelines.shadow_depth);
            let mut last_bound_mesh = None;

            for &draw_idx in ctx.solid_order {
                let dc = ctx.visible[draw_idx];
                let mesh_id = match ctx.mesh_handles[draw_idx] {
                    Some(id) => id,
                    None => continue,
                };

                let vp = ctx.csm_view_proj[cascade_idx as usize] * dc.model_matrix;
                let active = ShadowDrawUniforms {
                    shadow_mvp: vp.to_cols_array_2d(),
                };
                let uniforms = [active, dummy, dummy, dummy];

                if let Some(offset) = renderer.gpu.shadow_pool.push_shadow_array(&uniforms) {
                    pass.set_bind_group(0, renderer.gpu.shadow_pool.bind_group(), &[offset]);
                    renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                }
            }
        }
    }
}
