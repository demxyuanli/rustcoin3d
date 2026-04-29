use super::PassContext;
use crate::vertex::{ShadowDrawUniforms, CSM_CASCADE_COUNT};

pub(super) fn pass_shadow_depth(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &PassContext<'_>,
) {
    let Some(ref csm) = renderer.csm_shadow else {
        return;
    };

    let cascade_count = csm.cascade_count.min(CSM_CASCADE_COUNT as u32);

    for cascade_idx in 0..cascade_count {
        let light_vp = ctx.csm_view_proj[cascade_idx as usize];

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("CSM Shadow depth"),
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

        pass.set_pipeline(&renderer.pipelines.shadow_depth);
        let mut last_bound_mesh = None;
        for &i in ctx.solid_order {
            let dc = ctx.visible[i];
            let shadow_mvp = light_vp * dc.model_matrix;
            let uniforms = ShadowDrawUniforms {
                shadow_mvp: shadow_mvp.to_cols_array_2d(),
            };
            if let Some(offset) = renderer.shadow_pool.push_shadow(&uniforms) {
                if let Some(mesh_id) = ctx.mesh_handles[i] {
                    pass.set_bind_group(0, renderer.shadow_pool.bind_group(), &[offset]);
                    renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                }
            }
        }
    }
}
