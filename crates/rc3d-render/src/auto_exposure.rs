/// Manages automatic exposure (eye adaptation) based on scene luminance.
///
/// A compute shader computes per-block log-luminance averages into a GPU buffer.
/// The CPU reads back the buffer, computes the overall average, and applies
/// temporal adaptation with separate up/down speeds (eyes adapt faster to bright than dark).
#[allow(dead_code)]
pub struct AutoExposure {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    luminance_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    buffer_element_count: u32,
    current_exposure: f32,
    /// Parameters
    pub min_log_lum: f32,
    pub max_log_lum: f32,
    pub adapt_speed_up: f32,
    pub adapt_speed_down: f32,
    pub key_value: f32,
    /// Limits on exposure
    pub min_exposure: f32,
    pub max_exposure: f32,
}

impl AutoExposure {
    pub fn new(device: &wgpu::Device, max_width: u32, max_height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Auto Exposure"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/auto_exposure.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AutoExposure BGL"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("AutoExposure PLL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AutoExposure Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Buffer size: one f32 per 8x8 block
        let blocks_x = (max_width + 7) / 8;
        let blocks_y = (max_height + 7) / 8;
        let buf_count = blocks_x * blocks_y;
        let buf_size = (buf_count * 4) as u64;

        let luminance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Luminance Buffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Luminance Staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            luminance_buffer,
            staging_buffer,
            buffer_element_count: buf_count,
            current_exposure: 1.0,
            min_log_lum: -8.0,
            max_log_lum: 4.0,
            adapt_speed_up: 1.5,
            adapt_speed_down: 1.0,
            key_value: 0.18,
            min_exposure: 0.01,
            max_exposure: 100.0,
        }
    }

    /// Run the luminance compute shader and update exposure.
    /// Call this each frame before tonemapping. Returns the current exposure value.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
        hdr_width: u32,
        hdr_height: u32,
        delta_time: f32,
    ) -> f32 {
        let blocks_x = (hdr_width + 7) / 8;
        let blocks_y = (hdr_height + 7) / 8;

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("AutoExposure BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.luminance_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Auto Exposure Luminance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(blocks_x, blocks_y, 1);
        }

        // Copy to staging buffer for CPU readback
        encoder.copy_buffer_to_buffer(
            &self.luminance_buffer,
            0,
            &self.staging_buffer,
            0,
            (blocks_x * blocks_y * 4) as u64,
        );

        // Read from staging (from previous frame — one frame of latency)
        let avg_log_lum = self.read_staging();
        let avg_lum = (avg_log_lum.exp2())
            .clamp(self.min_log_lum.exp2(), self.max_log_lum.exp2());

        let target_exposure = self.key_value / avg_lum.max(1e-6);
        let target = target_exposure.clamp(self.min_exposure, self.max_exposure);

        // Temporal adaptation: asymmetric speeds
        let speed = if target > self.current_exposure {
            self.adapt_speed_up
        } else {
            self.adapt_speed_down
        };

        let t = 1.0 - (-speed * delta_time.max(0.001)).exp();
        self.current_exposure += (target - self.current_exposure) * t;

        self.current_exposure
    }

    /// Read the staging buffer from the previous frame to get average log luminance.
    fn read_staging(&self) -> f32 {
        // GPU readback is not wired yet (would need map_async + device.poll + unmap on a
        // previous-frame copy). Until then, avoid leaving the staging buffer mapped, which
        // would break Queue::submit.
        -1.0 // log2(0.5) — neutral grey
    }

    /// Get the current exposure value (after temporal adaptation).
    pub fn exposure(&self) -> f32 {
        self.current_exposure
    }

    /// Reset exposure to a specific value (e.g. on scene change).
    pub fn reset(&mut self, exposure: f32) {
        self.current_exposure = exposure;
    }
}
