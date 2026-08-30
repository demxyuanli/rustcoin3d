//! Minimal egui renderer using raw wgpu 24 API.

use std::borrow::Cow;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScreenUniform {
    screen_size: [f32; 2],
    _pad: [f32; 2],
}

/// Pre-uploaded draw data for one clipped primitive.
pub struct DrawBatch {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    bind_group: wgpu::BindGroup,
    scissor: [u32; 4], // x, y, w, h
}

pub struct EguiPainter {
    pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    white_bind_group: wgpu::BindGroup,
    screen_uniform_buf: wgpu::Buffer,
    screen_bind_group: wgpu::BindGroup,
    texture_map: std::collections::HashMap<egui::TextureId, wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    screen_size: [f32; 2],
}

impl EguiPainter {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("egui shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EGUI_SHADER)),
        });

        let uniform_bgl =
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

        let texture_bgl =
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
            bind_group_layouts: &[Some(&uniform_bgl), Some(&texture_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("egui pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
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
                })],
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
            multiview_mask: None,
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
            layout: &texture_bgl,
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
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_uniform_buf.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            texture_bind_group_layout: texture_bgl,
            white_bind_group,
            screen_uniform_buf,
            screen_bind_group,
            texture_map: std::collections::HashMap::new(),
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
                let rgba: Vec<u8> = img.pixels.iter()
                    .flat_map(|c| [c.r(), c.g(), c.b(), c.a()]).collect();
                (rgba, img.size)
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
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
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

    /// Sample an existing GPU color target (Viewport RT) from egui.
    pub fn bind_user_texture_view(
        &mut self,
        device: &wgpu::Device,
        texture_id: egui::TextureId,
        texture_view: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui user native"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.texture_map.insert(texture_id, bind_group);
    }

    /// Upload all mesh data to GPU buffers (call BEFORE creating encoder/render pass).
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        paint_jobs: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
        screen_width: u32,
        screen_height: u32,
    ) -> Vec<DrawBatch> {
        let ss = [screen_width as f32, screen_height as f32];
        if ss != self.screen_size {
            self.screen_size = ss;
            queue.write_buffer(
                &self.screen_uniform_buf,
                0,
                bytemuck::cast_slice(&[ScreenUniform { screen_size: ss, _pad: [0.0; 2] }]),
            );
        }

        let mut batches = Vec::with_capacity(paint_jobs.len());

        for job in paint_jobs {
            let scissor = {
                let min_x = (job.clip_rect.min.x * pixels_per_point) as u32;
                let min_y = (job.clip_rect.min.y * pixels_per_point) as u32;
                let max_x = (job.clip_rect.max.x * pixels_per_point) as u32;
                let max_y = (job.clip_rect.max.y * pixels_per_point) as u32;
                [min_x, min_y, max_x.saturating_sub(min_x), max_y.saturating_sub(min_y)]
            };

            if let egui::epaint::Primitive::Mesh(mesh) = &job.primitive {
                let vtx_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
                let idx_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);

                let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("egui vbuf"),
                    size: (vtx_bytes.len() as u64).max(64),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("egui ibuf"),
                    size: (idx_bytes.len() as u64).max(64),
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                queue.write_buffer(&vbuf, 0, vtx_bytes);
                queue.write_buffer(&ibuf, 0, idx_bytes);

                let bg = self.texture_map.get(&mesh.texture_id)
                    .cloned()
                    .unwrap_or_else(|| self.white_bind_group.clone());

                batches.push(DrawBatch {
                    vbuf,
                    ibuf,
                    index_count: mesh.indices.len() as u32,
                    bind_group: bg,
                    scissor,
                });
            }
        }

        batches
    }

    /// Draw pre-uploaded batches (call INSIDE render pass).
    pub fn draw_batches<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, batches: &'a [DrawBatch]) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.screen_bind_group, &[]);

        for batch in batches {
            rpass.set_scissor_rect(batch.scissor[0], batch.scissor[1], batch.scissor[2], batch.scissor[3]);
            rpass.set_bind_group(1, &batch.bind_group, &[]);
            rpass.set_vertex_buffer(0, batch.vbuf.slice(..));
            rpass.set_index_buffer(batch.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..batch.index_count, 0, 0..1);
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
