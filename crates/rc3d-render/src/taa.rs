use glam::Vec2;

/// Halton sequence generator for jitter patterns.
pub struct HaltonSequence {
    base: u32,
    index: u32,
}

impl HaltonSequence {
    pub fn new(base: u32) -> Self {
        Self { base, index: 0 }
    }

    pub fn next_halton(&mut self) -> f32 {
        self.index += 1;
        let mut result = 0.0f32;
        let mut f = 1.0f32 / self.base as f32;
        let mut i = self.index;
        while i > 0 {
            result += f * (i % self.base) as f32;
            i /= self.base;
            f /= self.base as f32;
        }
        result
    }
}

/// TAA jitter pattern: returns (offset_x, offset_y) in NDC space [-1, 1].
pub struct TaaJitter {
    halton_x: HaltonSequence,
    halton_y: HaltonSequence,
    sample_count: u32,
}

impl TaaJitter {
    pub fn new() -> Self {
        Self {
            halton_x: HaltonSequence::new(2),
            halton_y: HaltonSequence::new(3),
            sample_count: 0,
        }
    }

    /// Advance to next sample in the jitter sequence.
    /// Returns the jitter offset in NDC units.
    pub fn next_jitter(&mut self, width: u32, height: u32) -> Vec2 {
        self.sample_count = self.sample_count.wrapping_add(1);
        let x = self.halton_x.next_halton();
        let y = self.halton_y.next_halton();
        Vec2::new(
            (x - 0.5) * 2.0 / width as f32,
            (y - 0.5) * 2.0 / height as f32,
        )
    }

    pub fn sample_index(&self) -> u32 {
        self.sample_count
    }
}

impl Default for TaaJitter {
    fn default() -> Self {
        Self::new()
    }
}

/// TAA post-processing pass.
pub struct TaaPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sampler_linear: wgpu::Sampler,
    sampler_point: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
    pub history_texture: Option<wgpu::Texture>,
    pub history_view: Option<wgpu::TextureView>,
    /// False until the history texture has been seeded with a real frame.
    history_valid: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaParamsUniform {
    blend_factor: f32,
    clip_factor: f32,
    _pad0: f32,
    _pad1: f32,
}

impl TaaPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA Resolve"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/taa_resolve.wgsl").into()),
        });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let sampler_point = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Point Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA BGL"),
            entries: &[
                // 0: current color
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
                // 1: history color
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 2: velocity
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // 3: depth
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // 4: linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 5: point sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // 6: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: output storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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
            label: Some("TAA PLL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TAA Resolve Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("taa_resolve"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TAA Params"),
            size: std::mem::size_of::<TaaParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            sampler_linear,
            sampler_point,
            params_buffer,
            history_texture: None,
            history_view: None,
            history_valid: false,
        }
    }

    /// Ensure history buffer matches the current output size.
    pub fn ensure_history(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let needs_create = match &self.history_texture {
            Some(t) => t.width() != width || t.height() != height,
            None => true,
        };
        if needs_create {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("TAA History"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.history_texture = Some(tex);
            self.history_view = Some(view);
            self.history_valid = false;
        }
    }

    /// Run TAA resolve. `current_color` is the HDR scene output.
    /// `velocity_tex` is the screen-space motion vector texture (Rg16Float).
    /// `depth_tex` is the linear depth texture.
    /// `current_color_tex`/`output_tex` are the textures behind the views;
    /// they are used to seed the history on the first frame and to copy the
    /// resolved result into the history for the next frame (both must have
    /// COPY_SRC usage).
    pub fn resolve(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        current_color_view: &wgpu::TextureView,
        current_color_tex: &wgpu::Texture,
        velocity_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        output_tex: &wgpu::Texture,
        blend_factor: f32,
        clip_factor: f32,
    ) {
        let Some(history_view) = &self.history_view else {
            return;
        };
        let Some(history_tex) = &self.history_texture else {
            return;
        };
        let copy_extent = wgpu::Extent3d {
            width: history_tex.width(),
            height: history_tex.height(),
            depth_or_array_layers: 1,
        };

        // Seed history with the current frame so the first resolve is an
        // identity blend instead of mixing 95% black into the output.
        if !self.history_valid {
            encoder.copy_texture_to_texture(
                current_color_tex.as_image_copy(),
                history_tex.as_image_copy(),
                copy_extent,
            );
        }

        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&TaaParamsUniform {
                blend_factor,
                clip_factor,
                _pad0: 0.0,
                _pad1: 0.0,
            }),
        );

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA Resolve BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(velocity_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_point),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TAA Resolve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        // Dispatch based on history texture dimensions
        let w = copy_extent.width;
        let h = copy_extent.height;
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        drop(pass);

        // Carry the resolved frame into the history for the next frame's
        // temporal accumulation.
        if let Some(history_tex) = &self.history_texture {
            encoder.copy_texture_to_texture(
                output_tex.as_image_copy(),
                history_tex.as_image_copy(),
                copy_extent,
            );
        }
        self.history_valid = true;
    }
}
