//! Screen-space edge detection pass.
//!
//! Replaces geometry edges during interaction using screen-space edge detection.

use wgpu::TextureView;

/// GPU resources required by the screen-space edge detection pass.
///
/// Extract these from `Renderer` once per frame and pass them in,
/// instead of letting the pass reach deep into `Renderer` internals.
pub(crate) struct SsEdgeResources<'a> {
    pub pipeline: &'a wgpu::RenderPipeline,
    pub bind_group_layout: &'a wgpu::BindGroupLayout,
    pub uniform_buf: &'a wgpu::Buffer,
    pub sampler: &'a wgpu::Sampler,
}

/// Runtime parameters for the screen-space edge detection pass.
pub(crate) struct SsEdgeParams {
    pub edge_color: [f32; 4],
    pub threshold: f32,
}

/// Encode screen-space edge detection pass.
///
/// Returns early (no-op) if `resources` is `None`, meaning the pass is
/// not active or GPU resources are not initialized.
pub fn encode_ss_edge_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: Option<&SsEdgeResources<'_>>,
    params: &SsEdgeParams,
    encoder: &mut wgpu::CommandEncoder,
    view: &TextureView,
    depth_read_view: &TextureView,
    ew: u32,
    eh: u32,
) {
    let Some(res) = resources else {
        return;
    };

    let tw = ew.max(1) as f32;
    let th = eh.max(1) as f32;

    let locals = [
        1.0 / tw, 1.0 / th, // texel_size: vec2<f32>  offset 0
        params.threshold,     // threshold: f32    offset 8
        0.0,                  // _pad             offset 12
        params.edge_color[0], params.edge_color[1], params.edge_color[2], // edge_color(vec3) offset 16
        0.0,                  // struct pad       offset 28
    ];

    queue.write_buffer(res.uniform_buf, 0, bytemuck::bytes_of(&locals));

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SS Edge BG"),
        layout: res.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_read_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(res.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: res.uniform_buf.as_entire_binding(),
            },
        ],
    });

    let mut edge_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("SS Edge Detection"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    edge_pass.set_pipeline(res.pipeline);
    edge_pass.set_bind_group(0, &bg, &[]);
    edge_pass.draw(0..3, 0..1);
}
