use glam::{Mat4, Vec3};

pub const OMNISHADOW_RESOLUTION: u32 = 512;

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
}

/// GPU resources for rendering omnidirectional shadow maps for one point light.
pub struct OmniShadowMap {
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// Per-face 2D views for rendering (base_array_layer = face_index, array_layer_count = 1).
    pub face_views: [wgpu::TextureView; 6],
    pub bind_group: wgpu::BindGroup,
    pub sampler: wgpu::Sampler,
    pub resolution: u32,
}

impl OmniShadowMap {
    pub fn new(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        resolution: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 6,
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
            label: Some("Omni Shadow View"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // Per-face 2D views for rendering individual cube faces
        let face_views: [wgpu::TextureView; 6] = std::array::from_fn(|face| {
            depth_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Omni Shadow Face"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });

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
                            view_dimension: wgpu::TextureViewDimension::Cube,
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

/// Render the omni-directional shadow map for the first point light in the scene.
/// Renders all visible geometry to 6 cube faces from the light's perspective.
pub(crate) fn render_omni_shadow_pass(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &crate::render_passes::PassContext<'_>,
) {
    let Some(ref omni) = renderer.gpu.omni_shadow else { return };
    let Some(ref omni_map) = renderer.gpu.omni_shadow_map else { return };

    // Find first point light from light sets
    let light_set = ctx.light_sets.get(0);
    let light_positions = light_set.3;
    let light_types = light_set.2;
    let light_count = light_set.5 as usize;
    let mut pl_pos = glam::Vec3::ZERO;
    let mut found = false;
    for i in 0..light_count.min(16) {
        let lt = light_types[i][0] as i32;
        if lt == 1 {
            pl_pos = glam::Vec3::from_array([
                light_positions[i][0],
                light_positions[i][1],
                light_positions[i][2],
            ]);
            found = true;
            break;
        }
    }
    if !found { return; }

    let z_near = 0.1;
    let z_far = 100.0;
    let vps = omni_view_proj_matrices(pl_pos, z_near, z_far);

    for face in 0..6u32 {
        let u = OmniShadowUniforms {
            view_proj: vps[face as usize].to_cols_array_2d(),
            light_pos: pl_pos.to_array(),
            far_plane: z_far,
        };
        // Reuse uniform buffer and bind group instead of allocating each frame
        renderer.queue.write_buffer(
            &omni.uniform_buffer,
            0,
            bytemuck::bytes_of(&u),
        );
        let depth_clear = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
        let mut pass = encoder.begin_render_pass(
            &wgpu::RenderPassDescriptor {
                label: Some("Omni Shadow Face"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &omni_map.face_views[face as usize],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(depth_clear),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            },
        );
        pass.set_pipeline(&omni.pipeline);
        pass.set_bind_group(0, &omni.uniform_bind_group, &[]);

        let mut last_bound = None;
        for &vis_idx in ctx.solid_order {
            if let Some(mesh_id) = ctx.mesh_handles[vis_idx] {
                renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound);
            }
        }
    }
}
