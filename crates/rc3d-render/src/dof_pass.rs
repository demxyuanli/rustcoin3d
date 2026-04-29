/// Depth of Field post-process effect.
///
/// Uses circle-of-confusion from linear depth to blur out-of-focus regions
/// with depth-aware sampling to prevent foreground-background bleeding.
pub struct DofPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sampler_linear: wgpu::Sampler,
    sampler_point: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DofParamsUniform {
    focus_distance: f32,
    aperture: f32,
    max_coc: f32,
    near_blur_scale: f32,
    far_blur_scale: f32,
    _pad0: f32,
}

impl DofPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DOF"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dof.wgsl").into()),
        });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DOF Linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let sampler_point = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DOF Point"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DOF BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DOF PLL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DOF Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DOF Params"),
            size: std::mem::size_of::<DofParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            sampler_linear,
            sampler_point,
            params_buffer,
        }
    }

    /// Apply depth of field.
    pub fn apply(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene_color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        width: u32,
        height: u32,
        focus_distance: f32,
        aperture: f32,
    ) {
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&DofParamsUniform {
                focus_distance,
                aperture: aperture.clamp(0.1, 16.0),
                max_coc: 16.0,
                near_blur_scale: 1.0,
                far_blur_scale: 1.0,
                _pad0: 0.0,
            }),
        );

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DOF BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_point),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(output),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Depth of Field"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
