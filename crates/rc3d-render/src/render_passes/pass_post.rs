use crate::post_processor::{PostFxPipelines, PostFxTextures, SsaoParamsUniform};
use wgpu::util::DeviceExt;

pub(super) fn pass_bloom_prefilter(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
) {
    let bloom_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bloom prefilter BG"),
        layout: &pl.bloom_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&fx.hdr_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&pl.tonemap_sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&fx.bloom_view) },
        ],
    });
    let bloom_size = fx.bloom_tex.size();
    let wg_x = bloom_size.width.div_ceil(8);
    let wg_y = bloom_size.height.div_ceil(8);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bloom prefilter"), timestamp_writes: None,
        });
        pass.set_pipeline(&pl.bloom_prefilter);
        pass.set_bind_group(0, &bloom_bg, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

pub(super) fn pass_ssao(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
    depth_view: &wgpu::TextureView,
    noise_view: &wgpu::TextureView,
    proj: &[[f32; 4]; 4],
    inv_proj: &[[f32; 4]; 4],
) {
    let params = SsaoParamsUniform {
        // Keep AO conservative so streamed LOD/full-res transitions do not cause visible brightness jumps.
        proj: *proj, inv_proj: *inv_proj, radius: 0.8, bias: 0.02, power: 1.0, _pad: [0.0, 0.0], _tail_pad: [0.0, 0.0, 0.0],
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SSAO params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let ssao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SSAO BG"), layout: &pl.ssao_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(noise_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&pl.ssao_sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("SSAO"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &fx.ssao_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.ssao_pipeline);
    pass.set_bind_group(0, &ssao_bg, &[]);
    pass.draw(0..3, 0..1);
}

pub(super) fn pass_ssao_blur(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
    depth_view: &wgpu::TextureView,
) {
    let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SSAO Blur BG"), layout: &pl.ssao_blur_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&fx.ssao_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&pl.ssao_sampler) },
        ],
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("SSAO Blur"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &fx.ssao_blur_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.ssao_blur_pipeline);
    pass.set_bind_group(0, &blur_bg, &[]);
    pass.draw(0..3, 0..1);
}

pub(super) fn pass_tonemap_hdr_to_post_ldr(
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
    bg_color: wgpu::Color,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("ACES+FXAA+Bloom+SSAO"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &fx.post_ldr_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_color), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.tonemap_pipeline);
    pass.set_bind_group(0, &fx.tonemap_bg, &[]);
    pass.draw(0..3, 0..1);
}

/// Full-resolution texture copy (src → dst) via compute dispatch.
/// Used for ping-pong buffer resolution in post-processing chain.
pub(super) fn pass_copy_texture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &crate::post_processor::PostFxPipelines,
    src: &wgpu::TextureView,
    dst: &wgpu::TextureView,
    width: u32,
    height: u32,
) {
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Copy BG"),
        layout: &pl.bloom_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&pl.tonemap_sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(dst) },
        ],
    });
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Copy texture"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pl.copy_pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(wg_x, wg_y, 1);
}

pub(super) fn pass_fxaa_ldr_to_swapchain(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    ldr_scene_view: &wgpu::TextureView,
    swap_view: &wgpu::TextureView,
    bg_color: wgpu::Color,
) {
    let fxaa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FXAA LDR BG"),
        layout: &pl.fxaa_ldr_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(ldr_scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&pl.tonemap_sampler),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("FXAA LDR to swapchain"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: swap_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(bg_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.fxaa_ldr_pipeline);
    pass.set_bind_group(0, &fxaa_bg, &[]);
    pass.draw(0..3, 0..1);
}

pub(super) fn pass_blit_post_ldr_to_swapchain(
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
    swap_view: &wgpu::TextureView,
    bg_color: wgpu::Color,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Blit LDR to swapchain"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: swap_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_color), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.blit_pipeline);
    pass.set_bind_group(0, &fx.blit_bg, &[]);
    pass.draw(0..3, 0..1);
}
