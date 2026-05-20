use crate::pipelines::PipelineSet;

pub struct CsmShadowResources {
    pub texture: wgpu::Texture,
    /// Full array view (all layers), used for shader sampling.
    pub array_view: wgpu::TextureView,
    /// Per-cascade depth views for rendering (base_array_layer = cascade_idx, array_layer_count = 1).
    pub cascade_views: Vec<wgpu::TextureView>,
    /// Full-array bind group for PBR shader (texture_depth_2d_array + comparison sampler).
    pub bind_group: wgpu::BindGroup,
    pub resolution: u32,
    pub cascade_count: u32,
}

pub(super) fn create_csm_shadow_resources(
    device: &wgpu::Device,
    pipelines: &PipelineSet,
    compare_sampler: &wgpu::Sampler,
    global_frame_buffer: &wgpu::Buffer,
    omni_shadow: &crate::shadow_omni::OmniShadowMap,
    resolution: u32,
    cascade_count: u32,
) -> CsmShadowResources {
    let resolution = resolution.max(1);
    let cascade_count = cascade_count.max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("CSM Shadow Map Array"),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: cascade_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let array_view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("CSM Shadow Array View"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let mut cascade_views = Vec::with_capacity(cascade_count as usize);
    for layer in 0..cascade_count {
        cascade_views.push(tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("CSM Cascade Depth View"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
        }));
    }

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("CSM Shadow resources"),
        layout: &pipelines.shadow_resource_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&array_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(compare_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: global_frame_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&omni_shadow.depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&omni_shadow.sampler),
            },
        ],
    });

    CsmShadowResources {
        texture: tex,
        array_view,
        cascade_views,
        bind_group: bg,
        resolution,
        cascade_count,
    }
}

pub(super) fn create_shadow_compare_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Shadow compare"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::Less),
        ..Default::default()
    })
}
