//! Minimal egui renderer using raw wgpu 24 API.

use std::borrow::Cow;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScreenUniform {
    screen_size: [f32; 2],
    _pad: [f32; 2],
}

pub struct EguiPainter {
    pipeline: wgpu::RenderPipeline,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    white_bind_group: wgpu::BindGroup,
    screen_uniform_buf: wgpu::Buffer,
    screen_bind_group: wgpu::BindGroup,
    texture_map: std::collections::HashMap<egui::TextureId, wgpu::BindGroup>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    vertex_capacity: u64,
    index_capacity: u64,
    sampler: wgpu::Sampler,
    screen_size: [f32; 2],
}

impl EguiPainter {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("egui shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EGUI_SHADER)),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui uniform"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui texture"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("egui pipeline"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("egui pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<EguiVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm8x4,
                            offset: 16,
                            shader_location: 2,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            multisample: wgpu::MultisampleState::default(),
            depth_stencil: None,
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("egui sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let white_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("egui white"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let white_view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let white_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui white"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&white_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let screen_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("egui screen uniform"),
            size: std::mem::size_of::<ScreenUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let screen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui screen bind"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_uniform_buf.as_entire_binding(),
            }],
        });

        let vc = 65536u64;
        let ic = 65536u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("egui vbuf"),
            size: vc * std::mem::size_of::<EguiVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("egui ibuf"),
            size: ic * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_bind_group_layout,
            texture_bind_group_layout,
            white_bind_group,
            screen_uniform_buf,
            screen_bind_group,
            texture_map: std::collections::HashMap::new(),
            vertex_buffer,
            index_buffer,
            vertex_capacity: vc,
            index_capacity: ic,
            sampler,
            screen_size: [1280.0, 800.0],
        }
    }

    pub fn update_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
    ) {
        let (pixels_rgba, size) = match &delta.image {
            egui::epaint::ImageData::Color(img) => {
                let rgba: Vec<u8> = img
                    .pixels
                    .iter()
                    .flat_map(|c| [c.r(), c.g(), c.b(), c.a()])
                    .collect();
                (rgba, img.size)
            }
            egui::epaint::ImageData::Font(img) => {
                let srgba: Vec<u8> = img
                    .srgba_pixels(None)
                    .flat_map(|c| [c.r(), c.g(), c.b(), c.a()])
                    .collect();
                (srgba, img.size)
            }
        };

        let w = size[0] as u32;
        let h = size[1] as u32;
        let tex_size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("egui user tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui user bind"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            tex_size,
        );

        self.texture_map.insert(id, bind_group);
    }

    pub fn free_texture(&mut self, id: &egui::TextureId) {
        self.texture_map.remove(id);
    }

    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rpass: &mut wgpu::RenderPass<'_>,
        paint_jobs: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
        screen_width: u32,
        screen_height: u32,
    ) {
        let ss = [screen_width as f32, screen_height as f32];
        if ss != self.screen_size {
            self.screen_size = ss;
            queue.write_buffer(
                &self.screen_uniform_buf,
                0,
                bytemuck::cast_slice(&[ScreenUniform {
                    screen_size: ss,
                    _pad: [0.0; 2],
                }]),
            );
        }

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.screen_bind_group, &[]);

        for job in paint_jobs {
            let rect = egui::Rect::from_min_max(
                egui::pos2(
                    job.clip_rect.min.x * pixels_per_point,
                    job.clip_rect.min.y * pixels_per_point,
                ),
                egui::pos2(
                    job.clip_rect.max.x * pixels_per_point,
                    job.clip_rect.max.y * pixels_per_point,
                ),
            );
            rpass.set_scissor_rect(
                rect.min.x as u32,
                rect.min.y as u32,
                rect.width() as u32,
                rect.height() as u32,
            );

            if let egui::epaint::Primitive::Mesh(mesh) = &job.primitive {
                let vtx_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
                let idx_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);
                let vtx_size = vtx_bytes.len() as u64;
                let idx_size = idx_bytes.len() as u64;

                if vtx_size > self.vertex_capacity {
                    self.vertex_capacity = vtx_size.max(self.vertex_capacity * 2);
                    self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("egui vbuf"),
                        size: self.vertex_capacity,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                if idx_size > self.index_capacity {
                    self.index_capacity = idx_size.max(self.index_capacity * 2);
                    self.index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("egui ibuf"),
                        size: self.index_capacity,
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }

                queue.write_buffer(&self.vertex_buffer, 0, vtx_bytes);
                queue.write_buffer(&self.index_buffer, 0, idx_bytes);

                let tex_bg = self
                    .texture_map
                    .get(&mesh.texture_id)
                    .unwrap_or(&self.white_bind_group);
                rpass.set_bind_group(1, tex_bg, &[]);
                rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EguiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [u8; 4],
}

const EGUI_SHADER: &str = r#"
struct Uniforms {
    screen_size: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Transform from screen-space (0..W, 0..H) to clip-space (-1..1, -1..1, flipped Y)
    out.position = vec4<f32>(
        2.0 * in.pos.x / u.screen_size.x - 1.0,
        1.0 - 2.0 * in.pos.y / u.screen_size.y,
        0.0,
        1.0,
    );
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(t_diffuse, s_diffuse, in.uv);
    return tex_color * in.color;
}
"#;
