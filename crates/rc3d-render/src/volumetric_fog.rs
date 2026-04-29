/// Volumetric fog using screen-space ray marching through frustum depth slices.
///
/// Computes exponential height fog with directional light in-scattering.
/// Output is a two-component texture: (scattered_light.rgb, transmittance)
/// that is composited with the scene color in a subsequent pass.
pub struct VolumetricFogPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sampler_point: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FogParamsUniform {
    inv_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    fog_color: [f32; 4],
    fog_density: f32,
    height_falloff: f32,
    global_density: f32,
    max_distance: f32,
    num_steps: u32,
    scattering: f32,
    _pad0: f32,
    _pad1: f32,
}

impl VolumetricFogPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volumetric Fog"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/volumetric_fog.wgsl").into(),
            ),
        });

        let sampler_point = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Fog Point"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VolumetricFog BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("VolumetricFog PLL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VolumetricFog Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fog Params"),
            size: std::mem::size_of::<FogParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            sampler_point,
            params_buffer,
        }
    }

    /// Compute volumetric fog. Output to `fog_output` (Rgba16Float):
    /// RGB = scattered light, A = transmittance.
    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        fog_output: &wgpu::TextureView,
        width: u32,
        height: u32,
        inv_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        light_dir: glam::Vec3,
        light_color: glam::Vec3,
        fog_color: glam::Vec3,
        fog_density: f32,
        height_falloff: f32,
        max_distance: f32,
        num_steps: u32,
    ) {
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&FogParamsUniform {
                inv_proj: inv_proj.to_cols_array_2d(),
                camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
                light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                light_color: [light_color.x, light_color.y, light_color.z, 1.0],
                fog_color: [fog_color.x, fog_color.y, fog_color.z, 1.0],
                fog_density,
                height_falloff,
                global_density: 0.01,
                max_distance,
                num_steps: num_steps.clamp(4, 128),
                scattering: 0.5,
                _pad0: 0.0,
                _pad1: 0.0,
            }),
        );

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VolumetricFog BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_point),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(fog_output),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Volumetric Fog"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
