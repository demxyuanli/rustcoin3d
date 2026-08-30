//! Background rendering pass: gradients, image (stretch/tile/original), cubemap skybox, or solid color.
//! Renders before the main solid pass so geometry overlays correctly.

use wgpu::util::DeviceExt;

/// Background rendering mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BgMode {
    VerticalGradient,
    HorizontalGradient,
    CenterGradient,
    DiagonalGradient,
    Image,
    Solid,
    SkyGround,
}

/// How an image background is fitted to the screen.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ImageFit {
    #[default]
    Stretch,
    Tile,
    Original,
    /// Cubemap skybox (uses `cube_faces` paths; falls back to single image on all 6 faces).
    CubeMap,
}

/// Background configuration.
#[derive(Clone, Debug)]
pub struct BgSettings {
    pub mode: BgMode,
    pub image_fit: ImageFit,
    pub top_color: [f32; 4],
    pub bot_color: [f32; 4],
    pub image_path: Option<String>,
    /// Six cube-face image paths: +X, -X, +Y, -Y, +Z, -Z.
    pub cube_faces: [Option<String>; 6],
}

impl Default for BgSettings {
    fn default() -> Self {
        Self {
            mode: BgMode::Solid,
            image_fit: ImageFit::default(),
            top_color: [0.02, 0.02, 0.02, 1.0],
            bot_color: [0.02, 0.02, 0.02, 1.0],
            image_path: None,
            cube_faces: Default::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BgUniforms {
    mode: u32,
    image_fit: u32,
    _pad0: [u32; 2],
    top_color: [f32; 4],
    bot_color: [f32; 4],
    image_size: [f32; 2],
    screen_size: [f32; 2],
    /// Inverse projection (column-major) for cubemap view-direction reconstruction.
    inv_proj: [[f32; 4]; 4],
}

pub struct BgPass {
    pipeline_2d: wgpu::RenderPipeline,
    bgl_2d: wgpu::BindGroupLayout,
    pipeline_cubemap: wgpu::RenderPipeline,
    bgl_cubemap: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    pub image_view: Option<wgpu::TextureView>,
    image_dims: (u32, u32),
    sampler_clamp: wgpu::Sampler,
    sampler_repeat: wgpu::Sampler,
    /// D2-array with 6 layers, viewed as Cube.
    cube_view: Option<wgpu::TextureView>,
    cube_tex: Option<wgpu::Texture>,
    /// Fallback 1×1×6 cube (white).
    fallback_cube_view: wgpu::TextureView,
    fallback_2d_view: wgpu::TextureView,
}

impl BgPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader_2d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("background.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/background.wgsl").into()),
        });
        let shader_cube = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("background_cubemap.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/background_cubemap.wgsl").into()),
        });

        // ── 2D BGL ──
        let bgl_2d = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bg 2D BGL"),
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
        let layout_2d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bg 2D PPL"),
            bind_group_layouts: &[&bgl_2d],
            push_constant_ranges: &[],
        });
        let pipeline_2d = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Background 2D"),
            layout: Some(&layout_2d),
            vertex: wgpu::VertexState {
                module: &shader_2d, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_2d, entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None, cache: None,
        });

        // ── Cubemap BGL (uniform + texture_cube + sampler) ──
        let bgl_cubemap = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bg Cube BGL"),
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
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
        let layout_cubemap = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bg Cube PPL"),
            bind_group_layouts: &[&bgl_cubemap],
            push_constant_ranges: &[],
        });
        let pipeline_cubemap = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Background Cubemap"),
            layout: Some(&layout_cubemap),
            vertex: wgpu::VertexState {
                module: &shader_cube, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_cube, entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None, cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bg Uniforms"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bg Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let sampler_repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bg Tile Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // 1×1 white fallback (2D)
        let (_, fallback_2d_view) = make_white_tex_2d(device, queue, 1, 1);
        // 1×1×6 white fallback (Cube)
        let (fallback_cube_tex, fallback_cube_view) = make_white_cube(device, queue, 1, 1);

        Self {
            pipeline_2d, bgl_2d,
            pipeline_cubemap, bgl_cubemap,
            uniform_buf,
            image_view: None, image_dims: (1, 1),
            sampler_clamp, sampler_repeat,
            cube_view: Some(fallback_cube_view.clone()),
            cube_tex: Some(fallback_cube_tex),
            fallback_cube_view, fallback_2d_view,
        }
    }

    /// Upload a 2D background image.
    pub fn set_image(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, path: &str) {
        let img = match load_image_rgba8(path) {
            Some(d) => d, None => return,
        };
        self.image_dims = img.dimensions();
        let (_, view) = make_tex_2d(device, queue, self.image_dims.0, self.image_dims.1, &img.clone().into_raw());
        self.image_view = Some(view);
        // Also rebuild the cube from this single image.
        self.rebuild_cube_from_image(device, queue, &img);
    }

    fn rebuild_cube_from_image(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, img: &image::RgbaImage) {
        let dim = img.width().max(img.height());
        let squared = image::imageops::resize(img, dim, dim, image::imageops::FilterType::Lanczos3);
        let (tex, view) = make_cube_tex(device, queue, dim, dim, |_face| squared.clone());
        self.cube_tex = Some(tex);
        self.cube_view = Some(view);
    }

    /// Load all six cube faces from file paths. Missing faces fall back to the current 2D image.
    pub fn set_cube_faces(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        paths: &[Option<String>; 6],
    ) {
        let mut faces: [Option<image::RgbaImage>; 6] = Default::default();
        let mut max_dim: u32 = 4;
        for (i, path_opt) in paths.iter().enumerate() {
            if let Some(ref p) = path_opt {
                if let Some(img) = load_image_rgba8(p) {
                    max_dim = max_dim.max(img.width()).max(img.height());
                    faces[i] = Some(img);
                }
            }
        }
        // Fallback for missing faces: grey.
        let fallback = image::RgbaImage::from_pixel(max_dim, max_dim, image::Rgba([64u8, 64, 64, 255]));
        let (tex, view) = make_cube_tex(device, queue, max_dim, max_dim, |face_idx| {
            faces[face_idx].clone().unwrap_or_else(|| fallback.clone())
        });
        self.cube_tex = Some(tex);
        self.cube_view = Some(view);
    }

    /// Encode the background pass.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        settings: &BgSettings,
        screen_w: u32,
        screen_h: u32,
        camera_inv_proj: glam::Mat4,
        scene_region: crate::viewport::ViewportRect,
    ) {
        let mode = match settings.mode {
            BgMode::VerticalGradient => 0u32,
            BgMode::HorizontalGradient => 1u32,
            BgMode::CenterGradient => 2u32,
            BgMode::DiagonalGradient => 3u32,
            BgMode::Image => 4u32,
            BgMode::Solid => 5u32,
            BgMode::SkyGround => 6u32,
        };
        let image_fit = match settings.image_fit {
            ImageFit::Stretch => 0u32,
            ImageFit::Tile => 1u32,
            ImageFit::Original => 2u32,
            ImageFit::CubeMap => 3u32,
        };
        let uniforms = BgUniforms {
            mode, image_fit,
            _pad0: [0u32; 2],
            top_color: settings.top_color,
            bot_color: settings.bot_color,
            image_size: [self.image_dims.0 as f32, self.image_dims.1 as f32],
            screen_size: [screen_w as f32, screen_h as f32],
            inv_proj: camera_inv_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let is_cubemap = settings.mode == BgMode::Image && settings.image_fit == ImageFit::CubeMap;

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Background Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
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
        scene_region.apply_to_pass(&mut pass);

        if is_cubemap {
            let cube_v = self.cube_view.as_ref().unwrap_or(&self.fallback_cube_view);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Cubemap BG"),
                layout: &self.bgl_cubemap,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(cube_v) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler_clamp) },
                ],
            });
            pass.set_pipeline(&self.pipeline_cubemap);
            pass.set_bind_group(0, &bg, &[]);
        } else {
            let img_view = self.image_view.as_ref().unwrap_or(&self.fallback_2d_view);
            let sampler = if settings.image_fit == ImageFit::Tile { &self.sampler_repeat } else { &self.sampler_clamp };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Background BG"),
                layout: &self.bgl_2d,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(img_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });
            pass.set_pipeline(&self.pipeline_2d);
            pass.set_bind_group(0, &bg, &[]);
        }
        pass.draw(0..3, 0..1);
    }
}

// ── helpers ──

fn load_image_rgba8(path: &str) -> Option<image::RgbaImage> {
    let p = std::path::Path::new(path);
    if !p.is_file() { log::warn!("image not found: {path}"); return None; }
    match image::open(p) {
        Ok(i) => Some(i.to_rgba8()),
        Err(e) => { log::warn!("failed to load image {path}: {e}"); None }
    }
}

/// Load HDR/EXR environment map, returning float RGBA data suitable for
/// cubemap IBL. Supports `.hdr` (RGBE) and `.exr` via the `image` crate.
pub fn load_hdr_image(path: &str) -> Option<Vec<[f32; 4]>> {
    let p = std::path::Path::new(path);
    if !p.is_file() {
        log::warn!("HDR envmap not found: {path}");
        return None;
    }
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "hdr" => {
            match image::open(p) {
                Ok(img) => {
                    let rgba = img.to_rgba32f();
                    // Convert 32f RGBA flat buffer to [f32; 4] vec
                    let pixels: Vec<[f32; 4]> = rgba.chunks(4)
                        .map(|c| [c[0], c[1], c[2], c[3]])
                        .collect();
                    Some(pixels)
                }
                Err(e) => {
                    log::warn!("failed to load HDR {path}: {e}");
                    None
                }
            }
        }
        "exr" => {
            log::info!("EXR envmap loading via image crate: {path}");
            match image::open(p) {
                Ok(img) => {
                    let rgba = img.to_rgba32f();
                    let pixels: Vec<[f32; 4]> = rgba.chunks(4)
                        .map(|c| [c[0], c[1], c[2], c[3]])
                        .collect();
                    Some(pixels)
                }
                Err(e) => {
                    log::warn!("failed to load EXR {path}: {e}");
                    None
                }
            }
        }
        _ => {
            log::warn!("unsupported HDR format: {ext}");
            None
        }
    }
}

fn make_tex_2d(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, data: &[u8]) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("Bg 2D"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor, data,
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_white_tex_2d(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let data = vec![255u8; (w * h * 4) as usize];
    make_tex_2d(device, queue, w, h, &data)
}

fn make_cube_tex(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w: u32,
    h: u32,
    face_provider: impl Fn(usize) -> image::RgbaImage,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Bg Cube"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 6 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let pad_to_256 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let row_size = (w as usize * 4).max(pad_to_256 as usize);
    let layer_size = row_size * h as usize;
    let mut staging = vec![0u8; layer_size * 6];
    for face in 0..6 {
        let img = face_provider(face);
        let resized = image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3);
        let src_row = w as usize * 4;
        let dst_row = row_size;
        for y in 0..(h as usize) {
            let src_off = y * src_row;
            let dst_off = face * layer_size + y * dst_row;
            staging[dst_off..dst_off + src_row].copy_from_slice(&resized.as_raw()[src_off..src_off + src_row]);
        }
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &staging,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_size as u32),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 6 },
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    });
    (tex, view)
}

fn make_white_cube(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let white = image::RgbaImage::from_pixel(w, h, image::Rgba([255u8; 4]));
    make_cube_tex(device, queue, w, h, |_| white.clone())
}
