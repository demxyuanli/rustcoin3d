//! Dynamic CubeCamera capture: six 90-degree faces packed into a cubemap, then
//! converted to an equirectangular env map for the existing PBR IBL path.

use std::num::NonZeroU64;

use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::node_data::{CubeCameraNode, NodeData};
use rc3d_scene::SceneGraph;
use wgpu::util::DeviceExt;

use crate::render_action::DrawCall;
use crate::vertex::Vertex;
use crate::Renderer;

const FACE_DIRS: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y),
    (Vec3::NEG_X, Vec3::Y),
    (Vec3::Y, Vec3::NEG_Z),
    (Vec3::NEG_Y, Vec3::Z),
    (Vec3::Z, Vec3::Y),
    (Vec3::NEG_Z, Vec3::Y),
];

const UNIFORM_STRIDE: u64 = 256;
const MAX_CAPTURE_DRAWS: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CaptureUniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    color: [f32; 4],
    _pad: [f32; 28],
}

pub(crate) struct CubeCameraGpu {
    resolution: u32,
    #[allow(dead_code)]
    cube: wgpu::Texture,
    #[allow(dead_code)]
    cube_view: wgpu::TextureView,
    #[allow(dead_code)]
    depth: wgpu::Texture,
    face_color: Vec<wgpu::TextureView>,
    face_depth: Vec<wgpu::TextureView>,
    capture_pipeline: wgpu::RenderPipeline,
    capture_bgl: wgpu::BindGroupLayout,
    capture_bg: wgpu::BindGroup,
    capture_uniform: wgpu::Buffer,
    equirect: wgpu::Texture,
    equirect_view: wgpu::TextureView,
    to_eq_pipeline: wgpu::RenderPipeline,
    to_eq_bg: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    captured: bool,
    frames_since: u32,
}

impl CubeCameraGpu {
    pub(crate) fn mark_dirty(&mut self) {
        self.captured = false;
        self.frames_since = 0;
    }

    pub fn new(device: &wgpu::Device, resolution: u32) -> Self {
        let resolution = resolution.clamp(32, 512);
        let cube = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CubeCamera cube"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cube_view = cube.create_view(&wgpu::TextureViewDescriptor {
            label: Some("CubeCamera cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let mut face_color = Vec::with_capacity(6);
        for face in 0..6 {
            face_color.push(cube.create_view(&wgpu::TextureViewDescriptor {
                label: Some("CubeCamera face color"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face,
                array_layer_count: Some(1),
                ..Default::default()
            }));
        }
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CubeCamera depth"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mut face_depth = Vec::with_capacity(6);
        for face in 0..6 {
            face_depth.push(depth.create_view(&wgpu::TextureViewDescriptor {
                label: Some("CubeCamera face depth"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face,
                array_layer_count: Some(1),
                ..Default::default()
            }));
        }

        let capture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CubeCamera capture BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(NonZeroU64::new(UNIFORM_STRIDE).unwrap()),
                },
                count: None,
            }],
        });
        let capture_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeCamera capture UBO"),
            size: UNIFORM_STRIDE * 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let capture_bg = bind_capture_bg(device, &capture_bgl, &capture_uniform);
        let capture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CubeCamera capture"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cube_capture.wgsl").into()),
        });
        let capture_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CubeCamera capture PLL"),
            bind_group_layouts: &[Some(&capture_bgl)],
            immediate_size: 0,
        });
        let capture_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CubeCamera capture pipeline"),
            layout: Some(&capture_pll),
            vertex: wgpu::VertexState {
                module: &capture_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(Vertex::desc())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &capture_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let eq_w = resolution * 2;
        let eq_h = resolution;
        let mip_count = (eq_w.max(eq_h) as f32).log2().floor() as u32 + 1;
        let equirect = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CubeCamera equirect"),
            size: wgpu::Extent3d {
                width: eq_w,
                height: eq_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let equirect_view = equirect.create_view(&wgpu::TextureViewDescriptor {
            label: Some("CubeCamera equirect view"),
            ..Default::default()
        });

        let cube_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("CubeCamera cube sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let to_eq_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CubeCamera to equirect"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cube_to_equirect.wgsl").into()),
        });
        let to_eq_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CubeCamera to-eq BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
        let to_eq_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CubeCamera to-eq PLL"),
            bind_group_layouts: &[Some(&to_eq_bgl)],
            immediate_size: 0,
        });
        let to_eq_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CubeCamera to-eq pipeline"),
            layout: Some(&to_eq_pll),
            vertex: wgpu::VertexState {
                module: &to_eq_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &to_eq_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let to_eq_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CubeCamera to-eq BG"),
            layout: &to_eq_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&cube_samp),
                },
            ],
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CubeCamera mip blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit_tex.wgsl").into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CubeCamera mip blit BGL"),
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
            label: Some("CubeCamera mip blit PLL"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CubeCamera mip blit"),
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
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("CubeCamera mip sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            resolution,
            cube,
            cube_view,
            depth,
            face_color,
            face_depth,
            capture_pipeline,
            capture_bgl,
            capture_bg,
            capture_uniform,
            equirect,
            equirect_view,
            to_eq_pipeline,
            to_eq_bg,
            blit_pipeline,
            blit_bgl,
            blit_sampler,
            captured: false,
            frames_since: 0,
        }
    }

    fn ensure_uniform_capacity(&mut self, device: &wgpu::Device, slots: usize) {
        let needed = (slots as u64).max(1) * UNIFORM_STRIDE;
        if self.capture_uniform.size() >= needed {
            return;
        }
        let size = needed.next_power_of_two().max(UNIFORM_STRIDE * 64);
        self.capture_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeCamera capture UBO"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.capture_bg = bind_capture_bg(device, &self.capture_bgl, &self.capture_uniform);
    }
}

fn bind_capture_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("CubeCamera capture BG"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: Some(NonZeroU64::new(UNIFORM_STRIDE).unwrap()),
            }),
        }],
    })
}

fn face_view_proj(eye: Vec3, face: usize, z_near: f32, z_far: f32) -> Mat4 {
    let (dir, up) = FACE_DIRS[face];
    let view = Mat4::look_at_rh(eye, eye + dir, up);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, z_near, z_far);
    proj * view
}

struct ProbeDesc {
    world_pos: Vec3,
    node: CubeCameraNode,
}

fn collect_probes(graph: &SceneGraph) -> Vec<ProbeDesc> {
    let mut out = Vec::new();
    for &root in graph.roots() {
        walk_probes(graph, root, Mat4::IDENTITY, &mut out);
    }
    out
}

fn walk_probes(graph: &SceneGraph, node: rc3d_core::NodeId, model: Mat4, out: &mut Vec<ProbeDesc>) {
    let Some(entry) = graph.get(node) else {
        return;
    };
    let model = match entry.data.local_matrix() {
        Some(lm) => model * lm,
        None => model,
    };
    if let NodeData::CubeCamera(cc) = &entry.data {
        if cc.enabled {
            out.push(ProbeDesc {
                world_pos: model.transform_point3(cc.position),
                node: cc.clone(),
            });
        }
    }
    for &child in &entry.children {
        walk_probes(graph, child, model, out);
    }
}

pub fn update_cube_cameras(renderer: &mut Renderer, scene: &SceneGraph, draws: &[DrawCall]) {
    let probes = collect_probes(scene);
    let Some(probe) = probes.into_iter().next() else {
        return;
    };
    if !probe.node.enabled {
        return;
    }
    let res = probe.node.resolution.clamp(32, 512);
    let needs_rebuild = renderer
        .gpu
        .cube_camera
        .as_ref()
        .map(|g| g.resolution != res)
        .unwrap_or(true);
    if needs_rebuild {
        renderer.gpu.cube_camera = Some(CubeCameraGpu::new(&renderer.device, res));
    }
    let gpu = renderer.gpu.cube_camera.as_mut().expect("cube camera gpu");
    gpu.frames_since = gpu.frames_since.saturating_add(1);
    let period = probe.node.update_period;
    let should = if period == 0 {
        !gpu.captured
    } else {
        gpu.frames_since >= period
    };
    if !should {
        return;
    }
    gpu.frames_since = 0;

    let eye = probe.world_pos;
    let z_near = probe.node.near.max(0.01);
    let z_far = probe.node.far.max(z_near + 0.1);

    let mut mesh_bufs: Vec<(wgpu::Buffer, Option<wgpu::Buffer>, u32, Mat4, [f32; 4])> =
        Vec::new();
    for dc in draws {
        if mesh_bufs.len() >= MAX_CAPTURE_DRAWS {
            break;
        }
        if dc.vertices.is_empty() {
            continue;
        }
        if let Some(aabb) = dc.aabb.as_ref() {
            if aabb.min.x <= eye.x
                && eye.x <= aabb.max.x
                && aabb.min.y <= eye.y
                && eye.y <= aabb.max.y
                && aabb.min.z <= eye.z
                && eye.z <= aabb.max.z
            {
                continue;
            }
        }
        let vb = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CubeCamera vb"),
                contents: bytemuck::cast_slice(dc.vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let (ib, count) = if let Some(idx) = dc.indices.as_ref() {
            if idx.is_empty() {
                continue;
            }
            let buf = renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CubeCamera ib"),
                    contents: bytemuck::cast_slice(idx.as_slice()),
                    usage: wgpu::BufferUsages::INDEX,
                });
            (Some(buf), idx.len() as u32)
        } else {
            (None, dc.vertices.len() as u32)
        };
        let color = [
            dc.base_color.x,
            dc.base_color.y,
            dc.base_color.z,
            dc.opacity.max(0.15),
        ];
        mesh_bufs.push((vb, ib, count, dc.model_matrix, color));
    }

    let n = mesh_bufs.len();
    gpu.ensure_uniform_capacity(&renderer.device, n * 6);
    if n > 0 {
        let mut bytes = vec![0u8; n * 6 * UNIFORM_STRIDE as usize];
        for face in 0..6 {
            let vp = face_view_proj(eye, face, z_near, z_far);
            for (i, (_, _, _, model, color)) in mesh_bufs.iter().enumerate() {
                let uniforms = CaptureUniforms {
                    mvp: (vp * *model).to_cols_array_2d(),
                    model: model.to_cols_array_2d(),
                    color: *color,
                    _pad: [0.0; 28],
                };
                let offset = (face * n + i) * UNIFORM_STRIDE as usize;
                bytes[offset..offset + std::mem::size_of::<CaptureUniforms>()]
                    .copy_from_slice(bytemuck::bytes_of(&uniforms));
            }
        }
        renderer
            .queue
            .write_buffer(&gpu.capture_uniform, 0, &bytes);
    }

    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CubeCamera capture"),
        });
    for face in 0..6 {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("CubeCamera face"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gpu.face_color[face],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.045,
                            b: 0.055,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.face_depth[face],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(&gpu.capture_pipeline);
            for (i, (vb, ib, count, _, _)) in mesh_bufs.iter().enumerate() {
                let offset = ((face * n + i) as u32) * UNIFORM_STRIDE as u32;
                pass.set_bind_group(0, &gpu.capture_bg, &[offset]);
                pass.set_vertex_buffer(0, vb.slice(..));
                if let Some(ib) = ib {
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                } else {
                    pass.draw(0..*count, 0..1);
                }
            }
        }
    }

    let eq_mip0 = gpu.equirect.create_view(&wgpu::TextureViewDescriptor {
        label: Some("CubeCamera eq mip0"),
        base_mip_level: 0,
        mip_level_count: Some(1),
        ..Default::default()
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("CubeCamera cube to equirect"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &eq_mip0,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None, multiview_mask: None,
        });
        pass.set_pipeline(&gpu.to_eq_pipeline);
        pass.set_bind_group(0, &gpu.to_eq_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    let mip_count = gpu.equirect.mip_level_count();
    for mip in 1..mip_count {
        let src_view = gpu.equirect.create_view(&wgpu::TextureViewDescriptor {
            label: Some("CubeCamera mip src"),
            base_mip_level: mip - 1,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = gpu.equirect.create_view(&wgpu::TextureViewDescriptor {
            label: Some("CubeCamera mip dst"),
            base_mip_level: mip,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let bg = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("CubeCamera mip blit BG"),
                layout: &gpu.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&gpu.blit_sampler),
                    },
                ],
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("CubeCamera mip blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(&gpu.blit_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    renderer.queue.submit(std::iter::once(encoder.finish()));
    gpu.captured = true;
    bind_probe_as_ibl(renderer);
}

fn bind_probe_as_ibl(renderer: &mut Renderer) {
    let env_view = match renderer.gpu.cube_camera.as_ref() {
        Some(g) => g.equirect_view.clone(),
        None => return,
    };
    let brdf_view = match renderer.gpu.ibl.as_ref() {
        Some(i) => i.brdf_lut_view.clone(),
        None => return,
    };
    let sampler = renderer.gpu.ibl_sampler.clone();
    renderer.gpu.ibl_instance_bind_group = renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("IBL + Instance BG (CubeCamera)"),
        layout: &renderer.gpu.pipelines.ibl_instance_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&env_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: renderer.gpu.instance_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: renderer.gpu.morph_dummy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: renderer.gpu.morph_dummy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: renderer.gpu.morph_params_dummy.as_entire_binding(),
            },
        ],
    });
}

impl Renderer {
    pub fn update_cube_cameras(&mut self, scene: &SceneGraph, draws: &[DrawCall]) {
        update_cube_cameras(self, scene, draws);
    }
}
