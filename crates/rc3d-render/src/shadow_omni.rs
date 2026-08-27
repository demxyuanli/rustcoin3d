use glam::{Mat4, Vec3};

pub const OMNISHADOW_RESOLUTION: u32 = 512;
pub const MAX_OMNI_SHADOWS: u32 = 4;

/// Cube map face directions for omnidirectional shadow rendering.
const CUBE_FACE_DIRS: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y),    // +X (right)
    (Vec3::NEG_X, Vec3::Y), // -X (left)
    (Vec3::Y, Vec3::NEG_Z), // +Y (top)
    (Vec3::NEG_Y, Vec3::Z), // -Y (bottom)
    (Vec3::Z, Vec3::Y),    // +Z (front)
    (Vec3::NEG_Z, Vec3::Y), // -Z (back)
];

/// View-projection matrix for one cube face of a point light shadow map.
fn cube_face_view_proj(light_pos: Vec3, face_index: usize, z_near: f32, z_far: f32) -> Mat4 {
    let (target_dir, up_dir) = CUBE_FACE_DIRS[face_index];
    let target = light_pos + target_dir;
    let view = Mat4::look_at_rh(light_pos, target, up_dir);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, z_near, z_far);
    proj * view
}

/// Returns the 6 view-projection matrices for omnidirectional shadow rendering.
pub fn omni_view_proj_matrices(light_pos: Vec3, z_near: f32, z_far: f32) -> [Mat4; 6] {
    [
        cube_face_view_proj(light_pos, 0, z_near, z_far),
        cube_face_view_proj(light_pos, 1, z_near, z_far),
        cube_face_view_proj(light_pos, 2, z_near, z_far),
        cube_face_view_proj(light_pos, 3, z_near, z_far),
        cube_face_view_proj(light_pos, 4, z_near, z_far),
        cube_face_view_proj(light_pos, 5, z_near, z_far),
    ]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OmniShadowUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_pos: [f32; 3],
    pub far_plane: f32,
    pub model: [[f32; 4]; 4],
}

/// GPU resources for rendering omnidirectional shadow maps for one point light.
pub struct OmniShadowMap {
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// Per-face 2D views: layer = cube * 6 + face.
    pub face_views: Vec<wgpu::TextureView>,
    pub bind_group: wgpu::BindGroup,
    pub sampler: wgpu::Sampler,
    pub resolution: u32,
    pub cube_count: u32,
}

impl OmniShadowMap {
    pub fn new(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        resolution: u32,
    ) -> Self {
        Self::with_cubes(device, bgl, resolution, MAX_OMNI_SHADOWS)
    }

    pub fn with_cubes(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        resolution: u32,
        cube_count: u32,
    ) -> Self {
        let cube_count = cube_count.clamp(1, MAX_OMNI_SHADOWS);
        let size = wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 6 * cube_count,
        };

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Omni Shadow Depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Omni Shadow Cube Array"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });

        let face_count = (6 * cube_count) as usize;
        let face_views: Vec<wgpu::TextureView> = (0..face_count)
            .map(|layer| {
                depth_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Omni Shadow Face"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Omni Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Omni Shadow BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            depth_texture,
            depth_view,
            face_views,
            bind_group,
            sampler,
            resolution,
            cube_count,
        }
    }
}

/// Render pipeline for omnidirectional shadow pass.
pub struct OmniShadowRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub shadow_bgl: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    omni_resource_bgl: wgpu::BindGroupLayout,
    /// Pre-created bind group for uniform buffer (avoids per-frame allocation).
    uniform_bind_group: wgpu::BindGroup,
}

impl OmniShadowRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Omni Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/shadow_omni.wgsl").into(),
            ),
        });

        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Omni Shadow BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let omni_resource_bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Omni Shadow Resource BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::CubeArray,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            },
        );

        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Omni Shadow PLL"),
            bind_group_layouts: &[&shadow_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Omni Shadow Pipe"),
            layout: Some(&pll),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::vertex::Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Omni Shadow Uniforms"),
            size: std::mem::size_of::<OmniShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-create bind group for uniform buffer to avoid per-frame allocation
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Omni Shadow Uniform BG"),
            layout: &shadow_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            shadow_bgl,
            uniform_buffer,
            omni_resource_bgl,
            uniform_bind_group,
        }
    }

    /// Get the bind group layout for sampling omnidirectional shadow maps in the PBR shader.
    pub fn resource_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.omni_resource_bgl
    }
}

fn collect_omni_lights(ctx: &crate::render_passes::PassContext<'_>) -> Vec<Vec3> {
    let light_set = ctx.light_sets.get(0);
    let light_positions = light_set.3;
    let light_types = light_set.2;
    let light_count = light_set.5 as usize;
    let mut out = Vec::new();
    for i in 0..light_count.min(16) {
        if light_types[i][0] as i32 != 1 {
            continue;
        }
        out.push(Vec3::from_array([
            light_positions[i][0],
            light_positions[i][1],
            light_positions[i][2],
        ]));
        if out.len() >= MAX_OMNI_SHADOWS as usize {
            break;
        }
    }
    out
}

fn draw_omni_casters(
    renderer: &mut crate::renderer::Renderer,
    pass: &mut wgpu::RenderPass<'_>,
    ctx: &crate::render_passes::PassContext<'_>,
    light_vp: Mat4,
    light_pos: Vec3,
    z_far: f32,
) {
    let Some(omni) = renderer.gpu.omni_shadow.as_ref() else {
        return;
    };
    let uniform_buffer = omni.uniform_buffer.clone();
    let uniform_bg = omni.uniform_bind_group.clone();
    let mut last_bound = None;
    for (transparent, order) in [
        (false, ctx.solid_order),
        (true, ctx.transparent_order),
    ] {
        for &vis_idx in order {
            let dc = ctx.visible[vis_idx];
            if !dc.appearance().wants_filled() {
                continue;
            }
            if transparent && dc.opacity < 0.08 {
                continue;
            }
            let Some(mesh_id) = ctx.mesh_handles[vis_idx] else {
                continue;
            };
            let u = OmniShadowUniforms {
                view_proj: light_vp.to_cols_array_2d(),
                light_pos: light_pos.to_array(),
                far_plane: z_far,
                model: dc.model_matrix.to_cols_array_2d(),
            };
            renderer
                .queue
                .write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&u));
            pass.set_bind_group(0, &uniform_bg, &[]);
            renderer.draw_mesh_batched(pass, mesh_id, dc.index_draw_range(), &mut last_bound);
        }
    }
}

/// Render cube-array omni shadows for up to [`MAX_OMNI_SHADOWS`] point lights.
pub(crate) fn render_omni_shadow_pass(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &crate::render_passes::PassContext<'_>,
) {
    if renderer.gpu.omni_shadow.is_none() || renderer.gpu.omni_shadow_map.is_none() {
        return;
    }
    let lights = collect_omni_lights(ctx);
    if lights.is_empty() {
        return;
    }
    let z_near = 0.1;
    let z_far = 100.0;
    let depth_clear = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
    let cube_count = renderer
        .gpu
        .omni_shadow_map
        .as_ref()
        .map(|m| m.cube_count)
        .unwrap_or(1);
    let pipeline = renderer.gpu.omni_shadow.as_ref().unwrap().pipeline.clone();

    for (slot, pl_pos) in lights.iter().enumerate() {
        if slot as u32 >= cube_count {
            break;
        }
        let vps = omni_view_proj_matrices(*pl_pos, z_near, z_far);
        for face in 0..6u32 {
            let layer = slot as u32 * 6 + face;
            let face_view = {
                let map = renderer.gpu.omni_shadow_map.as_ref().unwrap();
                match map.face_views.get(layer as usize) {
                    Some(v) => v.clone(),
                    None => continue,
                }
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Omni Shadow Face"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &face_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(depth_clear),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            draw_omni_casters(renderer, &mut pass, ctx, vps[face as usize], *pl_pos, z_far);
        }
    }
}

/// Stamp 1-based cube slots into `light_types[i].y` for the first N point lights.
pub fn assign_omni_shadow_slots(light_types: &mut [[f32; 4]; crate::vertex::MAX_LIGHTS], light_count: u32) {
    let mut slot = 0u32;
    for i in 0..light_count.min(crate::vertex::MAX_LIGHTS as u32) as usize {
        if light_types[i][0] as i32 == 1 && slot < MAX_OMNI_SHADOWS {
            light_types[i][1] = (slot + 1) as f32;
            slot += 1;
        } else if light_types[i][0] as i32 == 1 {
            light_types[i][1] = 0.0;
        }
    }
}
