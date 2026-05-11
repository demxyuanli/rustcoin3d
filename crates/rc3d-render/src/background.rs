//! Background rendering pass: gradient, image, or solid color.
//! Renders before the main solid pass so geometry overlays correctly.

use wgpu::util::DeviceExt;

/// Background rendering mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BgMode {
    Gradient,
    Image,
    Solid,
}

/// Background configuration.
#[derive(Clone, Debug)]
pub struct BgSettings {
    pub mode: BgMode,
    pub top_color: [f32; 4],
    pub bot_color: [f32; 4],
    pub image_path: Option<String>,
}

impl Default for BgSettings {
    fn default() -> Self {
        Self {
            mode: BgMode::Solid,
            top_color: [0.02, 0.02, 0.02, 1.0],
            bot_color: [0.02, 0.02, 0.02, 1.0],
            image_path: None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BgUniforms {
    mode: u32,
    top_color: [f32; 4],
    bot_color: [f32; 4],
    _pad: [f32; 2],
}

pub struct BgPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    /// Background image texture (if set via set_image)
    pub image_tex: Option<wgpu::Texture>,
    pub image_view: Option<wgpu::TextureView>,
    pub sampler: wgpu::Sampler,
    fallback_view: wgpu::TextureView,
    fallback_tex: wgpu::Texture,
}

impl BgPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("background.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/background.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Background BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Background PPL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Background"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None,
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bg Uniforms"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bg Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // 1x1 white fallback texture for image mode when no image is loaded
        let fallback_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Bg fallback"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255u8, 255, 255, 255],
        );
        let fallback_view = fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            pipeline, bgl, uniform_buf,
            image_tex: None, image_view: None, sampler,
            fallback_view, fallback_tex,
        }
    }

    /// Upload background image from a file.
    pub fn set_image(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, path: &str) {
        let p = std::path::Path::new(path);
        if !p.is_file() {
            log::warn!("Background image not found: {}", path);
            return;
        }
        let img = match image::open(p) {
            Ok(i) => i.to_rgba8(),
            Err(e) => {
                log::warn!("Failed to load background image: {e}");
                return;
            }
        };
        let dims = img.dimensions();
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Background Image"),
                size: wgpu::Extent3d { width: dims.0, height: dims.1, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &img.into_raw(),
        );
        self.image_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.image_tex = Some(tex);
    }

    /// Encode background pass.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        settings: &BgSettings,
    ) {
        // Write uniforms
        let mode = match settings.mode {
            BgMode::Gradient => 0u32,
            BgMode::Image => 1u32,
            BgMode::Solid => 2u32,
        };
        let uniforms = BgUniforms {
            mode,
            top_color: settings.top_color,
            bot_color: settings.bot_color,
            _pad: [0.0; 2],
        };

        let img_view = self.image_view.as_ref().unwrap_or(&self.fallback_view);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Background BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(img_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Background Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: settings.top_color[0] as f64,
                        g: settings.top_color[1] as f64,
                        b: settings.top_color[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
