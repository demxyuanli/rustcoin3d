use std::path::Path;

/// Parse a .cube LUT file into raw RGBA16Float pixel data.
///
/// Format specification:
/// ```text
/// TITLE "name"
/// LUT_3D_SIZE 33
/// DOMAIN_MIN 0.0 0.0 0.0
/// DOMAIN_MAX 1.0 1.0 1.0
/// r g b
/// ...
/// ```
/// Data is stored in BGR order (blue varies fastest), then green, then red.
/// Each channel value is in the domain [DOMAIN_MIN, DOMAIN_MAX].
pub fn parse_cube_lut(contents: &str) -> Result<(u32, Vec<f32>), String> {
    let mut size: u32 = 0;
    let mut domain_min = [0.0f32; 3];
    let mut domain_max = [1.0f32; 3];
    let mut values: Vec<f32> = Vec::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let upper = trimmed.to_uppercase();
        if upper.starts_with("TITLE") {
            continue;
        } else if upper.starts_with("LUT_3D_SIZE") {
            size = upper
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if upper.starts_with("DOMAIN_MIN") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                domain_min[0] = parts[1].parse().unwrap_or(0.0);
                domain_min[1] = parts[2].parse().unwrap_or(0.0);
                domain_min[2] = parts[3].parse().unwrap_or(0.0);
            }
        } else if upper.starts_with("DOMAIN_MAX") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                domain_max[0] = parts[1].parse().unwrap_or(1.0);
                domain_max[1] = parts[2].parse().unwrap_or(1.0);
                domain_max[2] = parts[3].parse().unwrap_or(1.0);
            }
        } else {
            // Data line: three floats
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                let r: f32 = parts[0].parse().unwrap_or(0.0);
                let g: f32 = parts[1].parse().unwrap_or(0.0);
                let b: f32 = parts[2].parse().unwrap_or(0.0);
                // Remap from [domain_min, domain_max] to [0, 1]
                let rn = (r - domain_min[0]) / (domain_max[0] - domain_min[0]).max(0.001);
                let gn = (g - domain_min[1]) / (domain_max[1] - domain_min[1]).max(0.001);
                let bn = (b - domain_min[2]) / (domain_max[2] - domain_min[2]).max(0.001);
                values.extend_from_slice(&[rn, gn, bn, 1.0]);
            }
        }
    }

    if size == 0 {
        return Err("LUT_3D_SIZE not found".into());
    }
    let expected = (size * size * size * 4) as usize;
    if values.len() != expected {
        // Allow missing alpha by padding
        values.resize(expected, 1.0);
    }

    Ok((size, values))
}

/// Generate an identity 3D LUT (input = output) of given size.
pub fn generate_identity_lut(size: u32) -> Vec<f32> {
    let n = size as usize;
    let mut data = Vec::with_capacity(n * n * n * 4);
    for r in 0..n {
        let rf = r as f32 / (n - 1) as f32;
        for g in 0..n {
            let gf = g as f32 / (n - 1) as f32;
            for b in 0..n {
                let bf = b as f32 / (n - 1) as f32;
                data.extend_from_slice(&[rf, gf, bf, 1.0]);
            }
        }
    }
    data
}

/// GPU color grading pass using a 3D LUT texture.
pub struct ColorGradingPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
    /// Current LUT 3D texture (identity or loaded).
    lut_texture: wgpu::Texture,
    lut_view: wgpu::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LutParamsUniform {
    lut_size: f32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
}

impl ColorGradingPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Color Grading"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/color_grading.wgsl").into(),
            ),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("LUT Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ColorGrading BGL"),
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
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("ColorGrading PLL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ColorGrading Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LUT Params"),
            size: std::mem::size_of::<LutParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Default identity LUT (17x17x17)
        let lut_size = 17u32;
        let identity_data = generate_identity_lut(lut_size);
        let (lut_tex, lut_view) = upload_lut_3d(device, queue, lut_size, &identity_data, "Identity LUT");

        Self {
            pipeline,
            bgl,
            sampler,
            params_buffer,
            lut_texture: lut_tex,
            lut_view,
        }
    }

    /// Load a .cube LUT file.
    pub fn load_cube(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, path: &Path) -> Result<(), String> {
        let contents = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let (size, data) = parse_cube_lut(&contents)?;
        let (tex, view) = upload_lut_3d(device, queue, size, &data, "Custom LUT");
        self.lut_texture = tex;
        self.lut_view = view;
        log::info!("Loaded color grading LUT: {}x{}x{}", size, size, size);
        Ok(())
    }

    /// Apply color grading to the input HDR texture.
    pub fn apply(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        width: u32,
        height: u32,
        intensity: f32,
    ) {
        let lut_size = 17u32; // Match the default LUT size

        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&LutParamsUniform {
                lut_size: lut_size as f32,
                intensity: intensity.clamp(0.0, 1.0),
                _pad0: 0.0,
                _pad1: 0.0,
            }),
        );

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ColorGrading BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(output),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Color Grading"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
}

fn upload_lut_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
    data: &[f32],
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let byte_data: &[u8] = bytemuck::cast_slice(data);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        byte_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8 * size), // Rgba16Float = 8 bytes per pixel
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("LUT 3D View"),
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });

    (texture, view)
}
