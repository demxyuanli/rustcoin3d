use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SsaoParamsUniform {
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
    pub _pad: [f32; 2],
    pub _tail_pad: [f32; 3],
}

/// Post-processing effect parameters (uploaded to GPU uniform).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostEffectParams {
    pub vignette: f32,
    pub chromatic: f32,
    pub bloom_str: f32,
    pub grain: f32,
    pub exposure: f32,
    pub _pad: [f32; 3],
}

impl Default for PostEffectParams {
    fn default() -> Self {
        Self { vignette: 0.3, chromatic: 0.0, bloom_str: 0.8, grain: 0.0, exposure: 1.0, _pad: [0.0; 3] }
    }
}

/// All post-FX pipelines and bind-group layouts.
pub struct PostFxPipelines {
    pub tonemap_bgl: wgpu::BindGroupLayout,
    pub tonemap_sampler: wgpu::Sampler,
    pub tonemap_pipeline: wgpu::RenderPipeline,
    pub post_params_buf: wgpu::Buffer,
    pub fxaa_ldr_bgl: wgpu::BindGroupLayout,
    pub fxaa_ldr_pipeline: wgpu::RenderPipeline,
    pub smaa_edge_pipeline: wgpu::RenderPipeline,
    pub smaa_blend_pipeline: wgpu::RenderPipeline,
    pub blit_bgl: wgpu::BindGroupLayout,
    pub blit_pipeline: wgpu::RenderPipeline,
    pub copy_pipeline: wgpu::ComputePipeline,
    pub velocity_bgl: wgpu::BindGroupLayout,
    pub velocity_pipeline: wgpu::ComputePipeline,
    pub xray_bgl: wgpu::BindGroupLayout,
    pub xray_pipeline: wgpu::ComputePipeline,
    pub bloom_bgl: wgpu::BindGroupLayout,
    pub bloom_prefilter: wgpu::ComputePipeline,
    pub ssao_sampler: wgpu::Sampler,
    pub ssao_bgl: wgpu::BindGroupLayout,
    pub ssao_blur_bgl: wgpu::BindGroupLayout,
    pub ssao_pipeline: wgpu::RenderPipeline,
    pub ssao_blur_pipeline: wgpu::RenderPipeline,
    /// WBOIT composite: accum/revealage bind group layout (group 1).
    pub wboit_accum_bgl: wgpu::BindGroupLayout,
    pub wboit_composite_pipeline: wgpu::RenderPipeline,
}

pub fn create_post_fx_pipelines(device: &wgpu::Device, surface_format: wgpu::TextureFormat, hdr_format: wgpu::TextureFormat) -> PostFxPipelines {
    let tonemap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Post tonemap ACES+FXAA"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post_tonemap_fxaa.wgsl").into()),
    });
    let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Blit LDR"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit_tex.wgsl").into()),
    });
    let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Bloom prefilter"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom_prefilter.wgsl").into()),
    });
    let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Copy texture"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/copy_texture.wgsl").into()),
    });
    let velocity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Velocity"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/velocity.wgsl").into()),
    });
    let xray_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("X-Ray"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/xray.wgsl").into()),
    });
    let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SSAO"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ssao.wgsl").into()),
    });
    let ssao_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SSAO blur"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ssao_blur.wgsl").into()),
    });
    let fxaa_ldr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FXAA LDR"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fxaa_ldr.wgsl").into()),
    });
    let smaa_edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SMAA Edge"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/smaa_edge.wgsl").into()),
    });
    let smaa_blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SMAA Blend"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/smaa_blend.wgsl").into()),
    });

    let tonemap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Tonemap BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let post_params = PostEffectParams::default();
    let post_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("PostEffectParams"),
        contents: bytemuck::bytes_of(&post_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let tonemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Post process sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });

    let ssao_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("SSAO point sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });

    let tonemap_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Tonemap PLL"), bind_group_layouts: &[&tonemap_bgl], push_constant_ranges: &[],
    });
    let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Post ACES+FXAA+Bloom+SSAO"),
        layout: Some(&tonemap_pll),
        vertex: wgpu::VertexState { module: &tonemap_shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &tonemap_shader, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,
    });

    let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Blit BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let blit_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Blit PLL"), bind_group_layouts: &[&blit_bgl], push_constant_ranges: &[],
    });
    let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Blit post LDR to swapchain"),
        layout: Some(&blit_pll),
        vertex: wgpu::VertexState { module: &blit_shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &blit_shader, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,
    });

    let bloom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bloom BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 },
                count: None,
            },
        ],
    });
    let bloom_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Bloom PLL"), bind_group_layouts: &[&bloom_bgl], push_constant_ranges: &[],
    });
    let bloom_prefilter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Bloom prefilter"), layout: Some(&bloom_pll), module: &bloom_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });

    let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Copy texture"), layout: Some(&bloom_pll), module: &copy_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });

    let velocity_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Velocity BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 },
                count: None,
            },
        ],
    });
    let velocity_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Velocity PLL"), bind_group_layouts: &[&velocity_bgl], push_constant_ranges: &[],
    });
    let velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Velocity"), layout: Some(&velocity_pll), module: &velocity_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });

    let xray_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("X-Ray BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 },
                count: None,
            },
        ],
    });
    let xray_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("X-Ray PLL"), bind_group_layouts: &[&xray_bgl], push_constant_ranges: &[],
    });
    let xray_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("X-Ray"), layout: Some(&xray_pll), module: &xray_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });

    let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SSAO BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let ssao_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SSAO PLL"), bind_group_layouts: &[&ssao_bgl], push_constant_ranges: &[],
    });
    let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SSAO"),
        layout: Some(&ssao_pll),
        vertex: wgpu::VertexState { module: &ssao_shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &ssao_shader, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::R8Unorm, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,
    });

    let ssao_blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SSAO Blur BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let ssao_blur_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SSAO Blur PLL"), bind_group_layouts: &[&ssao_blur_bgl], push_constant_ranges: &[],
    });
    let ssao_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SSAO Blur"),
        layout: Some(&ssao_blur_pll),
        vertex: wgpu::VertexState { module: &ssao_blur_shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &ssao_blur_shader, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::R8Unorm, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil: None, multisample: wgpu::MultisampleState::default(),         multiview: None, cache: None,
    });

    let fxaa_ldr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FXAA LDR BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let fxaa_ldr_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FXAA LDR PLL"),
        bind_group_layouts: &[&fxaa_ldr_bgl],
        push_constant_ranges: &[],
    });
    let fxaa_ldr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("FXAA LDR"),
        layout: Some(&fxaa_ldr_pll),
        vertex: wgpu::VertexState {
            module: &fxaa_ldr_shader,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fxaa_ldr_shader,
            entry_point: Some("fs_fxaa"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // ── SMAA Edge Detection Pipeline ──
    let smaa_edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SMAA Edge"),
        layout: Some(&fxaa_ldr_pll), // same single-texture BGL
        vertex: wgpu::VertexState {
            module: &smaa_edge_shader,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &smaa_edge_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // ── SMAA Blend Pipeline ──
    let smaa_blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SMAA Blend"),
        layout: Some(&fxaa_ldr_pll),
        vertex: wgpu::VertexState {
            module: &smaa_blend_shader,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &smaa_blend_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // ── WBOIT Composite Pipeline ──
    let wboit_composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("WBOIT Composite"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wboit_composite.wgsl").into()),
    });

    let wboit_accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("WBOIT Accum BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let wboit_composite_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("WBOIT Composite PLL"),
        bind_group_layouts: &[&wboit_accum_bgl],
        push_constant_ranges: &[],
    });

    let wboit_composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("WBOIT Composite"),
        layout: Some(&wboit_composite_pll),
        vertex: wgpu::VertexState {
            module: &wboit_composite_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &wboit_composite_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: hdr_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    PostFxPipelines {
        tonemap_bgl,
        tonemap_sampler,
        tonemap_pipeline,
        post_params_buf,
        fxaa_ldr_bgl,
        fxaa_ldr_pipeline,
        smaa_edge_pipeline,
        smaa_blend_pipeline,
        blit_bgl,
        blit_pipeline,
        bloom_bgl,
        bloom_prefilter,
        copy_pipeline,
        velocity_bgl,
        velocity_pipeline,
        xray_bgl,
        xray_pipeline,
        ssao_sampler,
        ssao_bgl,
        ssao_blur_bgl,
        ssao_pipeline,
        ssao_blur_pipeline,
        wboit_accum_bgl,
        wboit_composite_pipeline,
    }
}

pub struct PostFxTextures {
    pub hdr_tex: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub tonemap_bg: wgpu::BindGroup,
    pub post_ldr_tex: wgpu::Texture,
    pub post_ldr_view: wgpu::TextureView,
    pub blit_bg: wgpu::BindGroup,
    pub bloom_tex: wgpu::Texture,
    pub bloom_view: wgpu::TextureView,
    pub ssao_tex: wgpu::Texture,
    pub ssao_view: wgpu::TextureView,
    pub ssao_blur_tex: wgpu::Texture,
    pub ssao_blur_view: wgpu::TextureView,
    /// SSR output texture (separate from HDR to avoid read-write conflict in compute dispatch).
    pub ssr_tex: wgpu::Texture,
    pub ssr_view: wgpu::TextureView,
    /// TAA output texture (separate from HDR to avoid read-write conflict).
    pub taa_tex: wgpu::Texture,
    pub taa_view: wgpu::TextureView,
    /// Post-FX scratch textures (avoid read-write conflict on HDR within single dispatch).
    pub scratch_tex: wgpu::Texture,
    pub scratch_view: wgpu::TextureView,
    /// Screen-space velocity (Rgba16Float) for motion blur.
    pub velocity_tex: wgpu::Texture,
    pub velocity_view: wgpu::TextureView,
    /// WBOIT accumulation buffer (Rgba16Float, additive blending).
    pub wboit_accum_tex: wgpu::Texture,
    pub wboit_accum_view: wgpu::TextureView,
    /// WBOIT revealage buffer (R8Unorm, multiplicative blending).
    pub wboit_revealage_tex: wgpu::Texture,
    pub wboit_revealage_view: wgpu::TextureView,
}

pub fn ensure_post_fx_textures(
    device: &wgpu::Device,
    p: &PostFxPipelines,
    width: u32,
    height: u32,
    _black_tex: &wgpu::Texture,
    _black_view: &wgpu::TextureView,
) -> PostFxTextures {
    let w = width.max(1);
    let h = height.max(1);

    let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("HDR scene"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let post_ldr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Post LDR"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
    });
    let post_ldr_view = post_ldr.create_view(&wgpu::TextureViewDescriptor::default());

    let hw = (w / 2).max(1);
    let hh = (h / 2).max(1);
    let bloom_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Bloom half-res"), size: wgpu::Extent3d { width: hw, height: hh, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING, view_formats: &[],
    });
    let bloom_view = bloom_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let ssao_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSAO"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
    });
    let ssao_view = ssao_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let ssao_blur_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSAO blurred"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
    });
    let ssao_blur_view = ssao_blur_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // SSR output — separate texture to avoid read-write conflict on HDR within single dispatch
    let ssr_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSR output"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING, view_formats: &[],
    });
    let ssr_view = ssr_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // TAA output — separate texture to avoid read-write conflict on HDR
    let taa_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("TAA output"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING, view_formats: &[],
    });
    let taa_view = taa_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Scratch texture for passes that read from and write to HDR within one dispatch.
    // COPY_SRC: TAA copies its resolved output (which may live here) into its history.
    let scratch_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("PostFX scratch"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
    });
    let scratch_view = scratch_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Velocity buffer for motion blur (Rgba16Float — STORAGE compatible)
    let velocity_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Velocity"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING, view_formats: &[],
    });
    let velocity_view = velocity_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let tonemap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tonemap BG"), layout: &p.tonemap_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&bloom_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&ssao_blur_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&p.tonemap_sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: p.post_params_buf.as_entire_binding() },
        ],
    });

    let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Blit BG"), layout: &p.blit_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&post_ldr_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&p.tonemap_sampler) },
        ],
    });

    // WBOIT accumulation buffer (Rgba16Float, additive blending target)
    let wboit_accum_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("WBOIT Accum"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let wboit_accum_view = wboit_accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // WBOIT revealage buffer (R8Unorm, multiplicative blending target)
    let wboit_revealage_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("WBOIT Revealage"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let wboit_revealage_view = wboit_revealage_tex.create_view(&wgpu::TextureViewDescriptor::default());

    PostFxTextures {
        hdr_tex, hdr_view, tonemap_bg,
        post_ldr_tex: post_ldr, post_ldr_view, blit_bg,
        bloom_tex, bloom_view,
        ssao_tex, ssao_view, ssao_blur_tex, ssao_blur_view,
        ssr_tex, ssr_view, taa_tex, taa_view, scratch_tex, scratch_view,
        velocity_tex, velocity_view,
        wboit_accum_tex, wboit_accum_view, wboit_revealage_tex, wboit_revealage_view,
    }
}

pub fn create_ssao_noise(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
    // Deterministic 4x4 random rotation vectors (Hilbert-spiral based)
    let noise_data: [f32; 64] = [
         0.538,  0.545, 0.0, 1.0,  -0.174,  0.841, 0.0, 1.0,
         0.707, -0.129, 0.0, 1.0,  -0.493, -0.493, 0.0, 1.0,
         0.956, -0.291, 0.0, 1.0,  -0.831,  0.556, 0.0, 1.0,
         0.331, -0.658, 0.0, 1.0,  -0.356, -0.729, 0.0, 1.0,
        -0.538, -0.545, 0.0, 1.0,   0.174, -0.841, 0.0, 1.0,
        -0.707,  0.129, 0.0, 1.0,   0.493,  0.493, 0.0, 1.0,
        -0.956,  0.291, 0.0, 1.0,   0.831, -0.556, 0.0, 1.0,
        -0.331,  0.658, 0.0, 1.0,   0.356,  0.729, 0.0, 1.0,
    ];
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSAO noise"), size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&noise_data),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(16 * 4), rows_per_image: Some(4) },
        wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
