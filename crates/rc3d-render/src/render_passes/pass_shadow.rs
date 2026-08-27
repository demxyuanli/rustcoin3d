use super::PassContext;
use crate::frustum::Frustum;
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

    // Build per-cascade frustums once for CPU-side pre-culling.
    // Each cascade only submits draws whose AABB intersects its frustum.
    let cascade_frustums: Vec<Frustum> = (0..cascade_count)
        // CSM ortho is forward-Z (near=0, far=1), not the camera reverse-Z flag.
        .map(|c| Frustum::from_view_projection(ctx.csm_view_proj[c as usize], false))
        .collect();

    let dummy = ShadowDrawUniforms {
        shadow_mvp: [[0.0; 4]; 4],
    };

    for cascade_idx in 0..cascade_count {
        let cascade_frustum = &cascade_frustums[cascade_idx as usize];

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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
            if !dc.appearance().wants_filled() {
                continue;
            }

            // ── Cascade frustum culling ──
            // Skip draws whose AABB does not intersect this cascade's frustum.
            if let Some(ref aabb) = dc.aabb {
                if !cascade_frustum.intersects_aabb(aabb) {
                    continue;
                }
            }

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
                renderer.draw_mesh_batched(
                    &mut pass,
                    mesh_id,
                    dc.index_draw_range(),
                    &mut last_bound_mesh,
                );
            }
        }

        // Transparent casters write depth so glass / ghost parts darken the receiver.
        for &draw_idx in ctx.transparent_order {
            let dc = ctx.visible[draw_idx];
            if dc.opacity < 0.08 || !dc.appearance().wants_filled() {
                continue;
            }
            if let Some(ref aabb) = dc.aabb {
                if !cascade_frustum.intersects_aabb(aabb) {
                    continue;
                }
            }
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
                renderer.draw_mesh_batched(
                    &mut pass,
                    mesh_id,
                    dc.index_draw_range(),
                    &mut last_bound_mesh,
                );
            }
        }
    }
}
