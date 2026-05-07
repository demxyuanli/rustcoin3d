//! Advanced rendering effects: Decal, Volume, PointCloud.
//! Each effect is lazy-initialized when first needed.

/// Decal projection pipeline resources.
pub struct DecalPass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    pub params_buf: wgpu::Buffer,
}

impl DecalPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decal Project"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/decal_project.wgsl").into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Decal BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Decal PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Decal"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::LessEqual, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None, cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Decal Sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToBorder, address_mode_v: wgpu::AddressMode::ClampToBorder,
            border_color: Some(wgpu::SamplerBorderColor::TransparentBlack), ..Default::default()
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decal Params"), size: 256, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Self { bgl, pipeline, sampler, params_buf }
    }
}

/// Volume ray-march pipeline resources.
pub struct VolumePass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    pub params_buf: wgpu::Buffer,
}

impl VolumePass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volume Raymarch"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/volume_raymarch.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volume BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volume PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Volume Raymarch"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::LessEqual, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() }, multiview: None, cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Volume Sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToBorder, address_mode_v: wgpu::AddressMode::ClampToBorder,
            address_mode_w: wgpu::AddressMode::ClampToBorder, border_color: Some(wgpu::SamplerBorderColor::TransparentBlack), ..Default::default()
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volume Params"), size: 64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Self { bgl, pipeline, sampler, params_buf }
    }
}

/// Point cloud rendering pipeline resources.
pub struct PointCloudPass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl PointCloudPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/point_cloud.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PointCloud BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PointCloud PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PointCloud"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::PointList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::LessEqual, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None, cache: None,
        });
        Self { bgl, pipeline }
    }
}
