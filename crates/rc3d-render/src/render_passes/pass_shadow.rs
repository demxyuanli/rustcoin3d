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

    // Single layered render pass: full array_view as depth attachment.
    // Each instanced draw writes to all array layers; the vertex shader
    // uses @builtin(instance_index) to select the correct cascade VP.
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

        // Build all 4 cascade VP uniforms into one array.
        // The vertex shader indexes into this array by gl_InstanceIndex.
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
}
