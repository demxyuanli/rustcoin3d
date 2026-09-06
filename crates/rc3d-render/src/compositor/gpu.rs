//! GPU ping-pong compositor: HDR ops, LDR film, blit to the main viewport.

use bytemuck::{Pod, Zeroable};

use super::graph::{CompExecStep, CompPing};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompParamsGpu {
    op: u32,
    mix_mode: u32,
    _p0: u32,
    _p1: u32,
    fac: f32,
    radius: f32,
    brightness: f32,
    contrast: f32,
    dir: [f32; 2],
    stop_count: u32,
    _p2: u32,
    stops: [[f32; 4]; 4],
    const_color: [f32; 4],
    scalars0: [f32; 4],
    scalars1: [f32; 4],
    scalars2: [f32; 4],
    crop: [f32; 4],
    morph: f32,
    _p3: [f32; 3],
}

pub struct CompositorGpu {
    hdr_pipeline: wgpu::ComputePipeline,
    hdr_bgl: wgpu::BindGroupLayout,
    preview_pipeline: wgpu::ComputePipeline,
    preview_bgl: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params_buf: wgpu::Buffer,
    black_tex: wgpu::Texture,
    black_view: wgpu::TextureView,
    ping_a: Option<(wgpu::Texture, wgpu::TextureView)>,
    ping_b: Option<(wgpu::Texture, wgpu::TextureView)>,
    preview: Option<(wgpu::Texture, wgpu::TextureView)>,
    size: [u32; 2],
}

impl CompositorGpu {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compositor"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compositor.wgsl").into()),
        });
        let hdr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compositor hdr BGL"),
            entries: &[
                tex_entry(0, true),
                tex_entry(1, true),
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
        let hdr_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compositor hdr PLL"),
            bind_group_layouts: &[Some(&hdr_bgl)],
            immediate_size: 0,
        });
        let hdr_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compositor hdr"),
            layout: Some(&hdr_pll),
            module: &shader,
            entry_point: Some("hdr_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let preview_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compositor preview BGL"),
            entries: &[
                tex_entry(0, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let preview_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compositor preview PLL"),
            bind_group_layouts: &[Some(&preview_bgl)],
            immediate_size: 0,
        });
        let preview_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compositor preview"),
            layout: Some(&preview_pll),
            module: &shader,
            entry_point: Some("preview_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compositor blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit_tex.wgsl").into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compositor blit BGL"),
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
        let blit_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compositor blit PLL"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compositor blit"),
            layout: Some(&blit_pll),
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
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("compositor sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compositor params"),
            size: std::mem::size_of::<CompParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let black_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compositor black"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let black_view = black_tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            hdr_pipeline,
            hdr_bgl,
            preview_pipeline,
            preview_bgl,
            blit_pipeline,
            blit_bgl,
            sampler,
            params_buf,
            black_tex,
            black_view,
            ping_a: None,
            ping_b: None,
            preview: None,
            size: [0, 0],
        }
    }

    pub fn ensure_size(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.clamp(64, 4096);
        let h = height.clamp(64, 4096);
        if self.size == [w, h] && self.preview.is_some() {
            return;
        }
        self.size = [w, h];
        self.ping_a = Some(make_hdr(device, "comp ping A", w, h));
        self.ping_b = Some(make_hdr(device, "comp ping B", w, h));
        self.preview = Some(make_ldr(device, "comp preview", w, h));
    }

    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: Option<&wgpu::TextureView>,
        steps: &[CompExecStep],
        dest: &wgpu::TextureView,
        scissor: [u32; 4],
    ) {
        let Some((_, ping_a)) = self.ping_a.as_ref() else {
            return;
        };
        let Some((_, ping_b)) = self.ping_b.as_ref() else {
            return;
        };
        let Some((_, preview)) = self.preview.as_ref() else {
            return;
        };
        let w = self.size[0];
        let h = self.size[1];
        let wg = (w.div_ceil(8), h.div_ceil(8));
        let black = &self.black_view;
        let scene_view = scene.unwrap_or(black);

        for step in steps {
            if step.write_preview {
                let src = ping_view(step.src_a, scene_view, black, ping_a, ping_b);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("compositor preview BG"),
                    layout: &self.preview_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(preview),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compositor preview"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.preview_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg.0, wg.1, 1);
                continue;
            }

            let params = CompParamsGpu {
                op: step.op_code,
                mix_mode: step.mix_mode,
                _p0: 0,
                _p1: 0,
                fac: step.fac,
                radius: step.radius,
                brightness: step.brightness,
                contrast: step.contrast,
                dir: step.dir,
                stop_count: step.stop_count,
                _p2: 0,
                stops: step.stops,
                const_color: step.color,
                scalars0: step.scalars0,
                scalars1: step.scalars1,
                scalars2: step.scalars2,
                crop: step.crop,
                morph: step.morph as f32,
                _p3: [0.0; 3],
            };
            queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
            let src_a = ping_view(step.src_a, scene_view, black, ping_a, ping_b);
            let src_b = ping_view(step.src_b, scene_view, black, ping_a, ping_b);
            let dst = if step.dst_is_b { ping_b } else { ping_a };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compositor hdr BG"),
                layout: &self.hdr_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_a),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src_b),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compositor hdr"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hdr_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
        }

        if steps.is_empty() {
            return;
        }

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor film blit BG"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(preview),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compositor film blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dest,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_scissor_rect(scissor[0], scissor[1], scissor[2].max(1), scissor[3].max(1));
            pass.draw(0..3, 0..1);
        }

        let _ = &self.black_tex;
    }
}

fn ping_view<'a>(
    ping: CompPing,
    scene: &'a wgpu::TextureView,
    black: &'a wgpu::TextureView,
    a: &'a wgpu::TextureView,
    b: &'a wgpu::TextureView,
) -> &'a wgpu::TextureView {
    match ping {
        CompPing::Scene => scene,
        CompPing::Black => black,
        CompPing::A => a,
        CompPing::B => b,
    }
}

fn tex_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type: wgpu::TextureSampleType::Float { filterable },
        },
        count: None,
    }
}

fn make_hdr(
    device: &wgpu::Device,
    label: &'static str,
    w: u32,
    h: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_ldr(
    device: &wgpu::Device,
    label: &'static str,
    w: u32,
    h: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
