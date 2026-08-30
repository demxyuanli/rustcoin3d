use wgpu::util::DeviceExt;

use super::collect::{DecalDrawCommand, PointCloudDrawCommand, VolumeDrawCommand};
use crate::gpu_particles::GpuParticleSim;

/// Decal projection pipeline resources.
pub struct DecalPass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    pub params_buf: wgpu::Buffer,
}

impl DecalPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decal Project"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/decal_project.wgsl").into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Decal BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Decal PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Decal"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::Always, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None, cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Decal Sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge, address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decal Params"), size: 256, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Self { bgl, pipeline, sampler, params_buf }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        commands: &[DecalDrawCommand],
        _viewport_w: u32,
        _viewport_h: u32,
        scene_region: crate::viewport::ViewportRect,
    ) {
        if commands.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Decal Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        scene_region.apply_to_pass(&mut pass);
        pass.set_pipeline(&self.pipeline);

        // Load decal texture from file (shared across all commands).
        let mut decal_tex: Option<wgpu::Texture> = None;
        let mut decal_view: Option<wgpu::TextureView> = None;

        for cmd in commands {
            if cmd.texture_path.is_empty() {
                continue;
            }
            // Lazy-load texture on first command
            if decal_view.is_none() {
                let path = std::path::Path::new(&cmd.texture_path);
                let (w, h, pixels) = if path.is_file() {
                    match image::open(path) {
                        Ok(img) => {
                            let rgba = img.to_rgba8();
                            let dims = rgba.dimensions();
                            (dims.0, dims.1, rgba.into_raw())
                        }
                        Err(e) => {
                            log::warn!("Failed to load decal {:?}: {e}, using fallback", path);
                            fallback_decal_pixels()
                        }
                    }
                } else {
                    log::info!("Decal texture not found: {:?}, using fallback", path);
                    fallback_decal_pixels()
                };
                let tex = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("Decal Texture"),
                        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1, sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &pixels,
                );
                decal_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
                decal_tex = Some(tex);
            }

            let tex_view = decal_view.as_ref().unwrap();
            // Pack uniform data: model(16) + position(4) + direction(4) + size(2) + color(4) + opacity(1) + pad(1)
            let mut uniform_data = Vec::<f32>::with_capacity(32);
            let m = cmd.model_matrix.to_cols_array_2d();
            for row in &m { uniform_data.extend_from_slice(row); }
            uniform_data.extend_from_slice(&[cmd.position.x, cmd.position.y, cmd.position.z, 1.0]);
            uniform_data.extend_from_slice(&[cmd.direction.x, cmd.direction.y, cmd.direction.z, 0.0]);
            uniform_data.extend_from_slice(&[cmd.size[0], cmd.size[1]]);
            uniform_data.extend_from_slice(&[cmd.color[0], cmd.color[1], cmd.color[2], cmd.color[3]]);
            uniform_data.extend_from_slice(&[cmd.opacity, 0.0]);
            queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&uniform_data));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Decal BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(tex_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding { buffer: &self.params_buf, offset: 0, size: None }
                    ) },
                ],
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..6, 0..1);
        }
        drop(decal_tex);
    }
}

/// Generate a fallback 64×64 decal texture: orange circle on dark background.
fn fallback_decal_pixels() -> (u32, u32, Vec<u8>) {
    let w = 64u32;
    let h = 64u32;
    let c = 32;
    let r = 24;
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let dx = x as i32 - c;
            let dy = y as i32 - c;
            let i = (y * w as usize + x) * 4;
            if dx * dx + dy * dy < r * r {
                px[i] = 255; px[i+1] = 128; px[i+2] = 0; px[i+3] = 255;
            } else {
                px[i] = 30; px[i+1] = 30; px[i+2] = 30; px[i+3] = 255;
            }
        }
    }
    (w, h, px)
}

/// Volume ray-march pipeline resources.
pub struct VolumePass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    pub params_buf: wgpu::Buffer,
}

impl VolumePass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volume Raymarch"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/volume_raymarch.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volume BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volume PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Volume Raymarch"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::LessEqual, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() }, multiview: None, cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Volume Sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToBorder, address_mode_v: wgpu::AddressMode::ClampToBorder,
            address_mode_w: wgpu::AddressMode::ClampToBorder, border_color: Some(wgpu::SamplerBorderColor::TransparentBlack), ..Default::default()
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volume Params"), size: 64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Self { bgl, pipeline, sampler, params_buf }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        commands: &[VolumeDrawCommand],
        _viewport_w: u32,
        _viewport_h: u32,
        scene_region: crate::viewport::ViewportRect,
    ) {
        if commands.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Volume Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        scene_region.apply_to_pass(&mut pass);
        pass.set_pipeline(&self.pipeline);

        let placeholder_vol = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let placeholder_view = placeholder_vol.create_view(&wgpu::TextureViewDescriptor::default());

        for cmd in commands {
            if cmd.texture_path.is_empty() {
                continue;
            }
            let mut uniform_data = Vec::<f32>::with_capacity(80);
            let m = cmd.model_matrix.to_cols_array_2d();
            for row in &m { uniform_data.extend_from_slice(row); }
            uniform_data.extend_from_slice(&[cmd.dimensions[0] as f32, cmd.dimensions[1] as f32, cmd.dimensions[2] as f32, 0.0]);
            uniform_data.push(cmd.density_scale);
            for cm in &cmd.color_map { uniform_data.extend_from_slice(cm); }
            uniform_data.extend_from_slice(&[0.0; 3]);
            queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&uniform_data));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Volume BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&placeholder_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding { buffer: &self.params_buf, offset: 0, size: None }
                    ) },
                ],
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

/// Point cloud rendering pipeline resources.
pub struct PointCloudPass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub sim: GpuParticleSim,
}

impl PointCloudPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/point_cloud.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PointCloud BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PointCloud PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PointCloud"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::LessEqual, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None, cache: None,
        });
        Self { bgl, pipeline, sim: GpuParticleSim::new(device) }
    }

    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        commands: &[PointCloudDrawCommand],
        view_proj: glam::Mat4,
        _inv_projection: glam::Mat4,
        viewport_w: u32,
        viewport_h: u32,
        scene_region: crate::viewport::ViewportRect,
    ) {
        if commands.is_empty() {
            return;
        }

        let (dt, time) = self.sim.tick_dt();
        for cmd in commands {
            if let Some(ref emitter) = cmd.emitter {
                self.sim.update(device, queue, encoder, cmd.sim_key, emitter, dt, time);
            }
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("PointCloud Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        scene_region.apply_to_pass(&mut pass);
        pass.set_pipeline(&self.pipeline);

        for cmd in commands {
            let gpu_sim = cmd.emitter.is_some();
            if !gpu_sim && cmd.points.is_empty() {
                continue;
            }
            let mvp = view_proj * cmd.model_matrix;
            let uniforms = PointCloudUniforms {
                mvp: mvp.to_cols_array_2d(),
                view: view_proj.to_cols_array_2d(),
                point_size: cmd.point_size.max(1.0),
                viewport_x: viewport_w.max(1) as f32,
                viewport_y: viewport_h.max(1) as f32,
                _pad: 0.0,
            };
            let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PC uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            if gpu_sim {
                let Some((particle_buf, count)) = self.sim.buffer(cmd.sim_key) else {
                    continue;
                };
                if count == 0 {
                    continue;
                }
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PointCloud BG"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: particle_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(
                            wgpu::BufferBinding { buffer: &ub, offset: 0, size: None }
                        ) },
                    ],
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..(count * 6), 0..1);
            } else {
                let count = cmd.points.len().min(cmd.max_visible_points as usize) as u32;
                if count == 0 {
                    continue;
                }
                let point_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("PC points"),
                    contents: bytemuck::cast_slice(&cmd.points[..count as usize]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PointCloud BG"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: point_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(
                            wgpu::BufferBinding { buffer: &ub, offset: 0, size: None }
                        ) },
                    ],
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..(count * 6), 0..1);
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PointCloudUniforms {
    mvp: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    point_size: f32,
    viewport_x: f32,
    viewport_y: f32,
    _pad: f32,
}
