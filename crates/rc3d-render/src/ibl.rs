use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IblPreset {
    Neutral,
    Studio,
    Warm,
}

impl IblPreset {
    pub const fn next(self) -> Self {
        match self {
            Self::Neutral => Self::Studio,
            Self::Studio => Self::Warm,
            Self::Warm => Self::Neutral,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::Studio => "studio",
            Self::Warm => "warm",
        }
    }
}

/// GPU resources for image-based lighting.
/// Uses an equirectangular HDR environment map + split-sum BRDF LUT.
pub struct IblResources {
    pub env_map: wgpu::Texture,
    pub env_map_view: wgpu::TextureView,
    pub brdf_lut: wgpu::Texture,
    pub brdf_lut_view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    /// Average scene luminance factors for fallback diffuse/specular
    pub ibl_diffuse: [f32; 4],
    pub ibl_specular: [f32; 4],
}

impl IblResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bgl: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        env_path: &Path,
        preset: IblPreset,
    ) -> Self {
        // Load and upload environment map
        let (env_tex, env_view, ibl_diffuse, ibl_specular) =
            load_equirectangular_hdr(device, queue, env_path, preset);

        // Create BRDF LUT
        let brdf_lut = create_brdf_lut(device, queue);

        let brdf_lut_view = brdf_lut.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL Resources BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&brdf_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        Self {
            env_map: env_tex,
            env_map_view: env_view,
            brdf_lut,
            brdf_lut_view,
            bind_group,
            ibl_diffuse,
            ibl_specular,
        }
    }
}

fn load_equirectangular_hdr(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &Path,
    preset: IblPreset,
) -> (wgpu::Texture, wgpu::TextureView, [f32; 4], [f32; 4]) {
    // Prefer external HDR file; fallback to built-in generic environment map.
    let try_external = !path.as_os_str().is_empty() && path.is_file();
    let (width, height, pixels, ibl_diffuse, ibl_specular) = match try_external.then(|| image::open(path)) {
        Some(Ok(img)) => {
            let rgba = img.to_rgba32f();
            let dims = rgba.dimensions();
            let (diffuse, specular) = compute_ibl_factors_from_rgb32f(&img.to_rgb32f());
            (dims.0, dims.1, rgba.into_raw(), diffuse, specular)
        }
        Some(Err(e)) => {
            log::warn!("Failed to load env map {:?}: {e}", path);
            let (w, h, px, diffuse, specular) = create_builtin_env_map(preset);
            log::warn!("Using built-in HDR preset: {}", preset.name());
            (w, h, px, diffuse, specular)
        }
        None => {
            log::info!("External env map not found, using built-in HDR preset: {}", preset.name());
            let (w, h, px, diffuse, specular) = create_builtin_env_map(preset);
            (w, h, px, diffuse, specular)
        }
    };

    let w = width.max(1);
    let h = height.max(1);
    let _pixel_bytes: &[u8] = bytemuck::cast_slice(&pixels);

    // Use Rgba16Float for the env map: supports filtering on all devices and
    // provides sufficient HDR range (±65504) for IBL specular prefiltering.
    let env_format = wgpu::TextureFormat::Rgba16Float;
    let mip_level_count = (w.max(h) as f32).log2().floor() as u32 + 1;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("IBL Env Map"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: env_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    // Convert f32 pixels to f16 and upload to mip 0
    {
        let pixels_f16: Vec<u16> = pixels
            .chunks_exact(4)
            .flat_map(|rgba| {
                [
                    half::f16::from_f32(rgba[0]).to_bits(),
                    half::f16::from_f32(rgba[1]).to_bits(),
                    half::f16::from_f32(rgba[2]).to_bits(),
                    half::f16::from_f32(rgba[3]).to_bits(),
                ]
            })
            .collect();
        let pixel_bytes_f16: &[u8] = bytemuck::cast_slice(&pixels_f16);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixel_bytes_f16,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8 * w), // 4 * f16 = 8 bytes per pixel
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    // Generate mipmaps via blit passes for roughness-varying specular IBL
    if mip_level_count > 1 {
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IBL Mip Blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit_tex.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Mip Blit BGL"),
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

        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("IBL Mip Blit PLL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("IBL Mip Blit Pipeline"),
            layout: Some(&pll),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: env_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL Mip Blit Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut mip_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("IBL Mipmap Gen"),
        });

        let mut src_w = w;
        let mut src_h = h;
        for mip in 1..mip_level_count {
            let dst_w = (src_w + 1) / 2;
            let dst_h = (src_h + 1) / 2;

            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("IBL Mip Src"),
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("IBL Mip Blit BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("IBL Mip Dst"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let mut pass = mip_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("IBL Mip Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);

            src_w = dst_w;
            src_h = dst_h;
        }

        queue.submit(std::iter::once(mip_encoder.finish()));
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("IBL Env Map View"),
        ..Default::default()
    });

    (texture, view, ibl_diffuse, ibl_specular)
}

fn create_builtin_env_map(preset: IblPreset) -> (u32, u32, Vec<f32>, [f32; 4], [f32; 4]) {
    let width: u32 = 256;
    let height: u32 = 128;
    let mut pixels = vec![0.0_f32; (width * height * 4) as usize];

    let (sky, horizon, ground, sun_boost, ibl_diffuse, ibl_specular) = match preset {
        IblPreset::Neutral => (
            [0.10_f32, 0.11_f32, 0.12_f32],
            [0.08_f32, 0.08_f32, 0.08_f32],
            [0.02_f32, 0.02_f32, 0.02_f32],
            1.2_f32,
            [0.08, 0.08, 0.08, 1.0],
            [0.18, 0.18, 0.18, 1.0],
        ),
        IblPreset::Studio => (
            [0.12_f32, 0.16_f32, 0.24_f32],
            [0.10_f32, 0.10_f32, 0.10_f32],
            [0.02_f32, 0.02_f32, 0.02_f32],
            2.4_f32,
            [0.09, 0.09, 0.09, 1.0],
            [0.22, 0.22, 0.22, 1.0],
        ),
        IblPreset::Warm => (
            [0.20_f32, 0.16_f32, 0.12_f32],
            [0.13_f32, 0.10_f32, 0.08_f32],
            [0.03_f32, 0.02_f32, 0.01_f32],
            2.0_f32,
            [0.11, 0.09, 0.07, 1.0],
            [0.24, 0.19, 0.14, 1.0],
        ),
    };

    for y in 0..height {
        let v = y as f32 / (height - 1) as f32;
        let t = (v * 2.0 - 1.0).clamp(-1.0, 1.0);
        let base = if t >= 0.0 {
            [
                horizon[0] * (1.0 - t) + sky[0] * t,
                horizon[1] * (1.0 - t) + sky[1] * t,
                horizon[2] * (1.0 - t) + sky[2] * t,
            ]
        } else {
            let k = -t;
            [
                horizon[0] * (1.0 - k) + ground[0] * k,
                horizon[1] * (1.0 - k) + ground[1] * k,
                horizon[2] * (1.0 - k) + ground[2] * k,
            ]
        };

        for x in 0..width {
            let u = x as f32 / (width - 1) as f32;
            // Add one soft sun lobe for specular highlights.
            let du = (u - 0.82).abs();
            let dv = (v - 0.26).abs();
            let sun = (-((du * du) / 0.0015 + (dv * dv) / 0.006)).exp() * sun_boost;
            let i = ((y * width + x) * 4) as usize;
            pixels[i] = base[0] + sun;
            pixels[i + 1] = base[1] + sun * 0.95;
            pixels[i + 2] = base[2] + sun * 0.85;
            pixels[i + 3] = 1.0;
        }
    }

    (width, height, pixels, ibl_diffuse, ibl_specular)
}

fn create_brdf_lut(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("BRDF LUT Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brdf_lut.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("BRDF LUT BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }],
    });

    let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("BRDF LUT PLL"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("BRDF LUT Pipeline"),
        layout: Some(&pll),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let lut_size: u32 = 256;
    let lut_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BRDF LUT"),
        size: wgpu::Extent3d {
            width: lut_size,
            height: lut_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let lut_view = lut_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BRDF LUT BG"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&lut_view),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("BRDF LUT Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let wg = (lut_size + 7) / 8;
        pass.dispatch_workgroups(wg, wg, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    lut_tex
}

fn compute_ibl_factors_from_rgb32f(img: &image::Rgb32FImage) -> ([f32; 4], [f32; 4]) {
    let mut sum = [0.0f64; 3];
    let mut n = 0u64;
    for p in img.pixels() {
        let c = p.0;
        sum[0] += c[0] as f64;
        sum[1] += c[1] as f64;
        sum[2] += c[2] as f64;
        n += 1;
    }
    if n > 0 {
        let avg = [
            (sum[0] / n as f64) as f32,
            (sum[1] / n as f64) as f32,
            (sum[2] / n as f64) as f32,
        ];
        let diffuse = [avg[0] * 0.22, avg[1] * 0.22, avg[2] * 0.22, 1.0];
        let specular = [avg[0] * 0.55, avg[1] * 0.55, avg[2] * 0.55, 1.0];
        return (diffuse, specular);
    }
    ([0.07, 0.07, 0.07, 1.0], [0.15, 0.15, 0.15, 1.0])
}

/// Create the IBL bind group layout for the renderer.
/// Binding 0: env map texture_2d, Binding 1: BRDF LUT texture_2d, Binding 2: sampler
pub fn create_ibl_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("IBL BGL"),
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
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}
