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
struct OmniShadowUniforms {
    view_proj: [[f32; 4]; 4],
    light_pos: [f32; 3],
    far_plane: f32,
}

/// GPU resources for rendering omnidirectional shadow maps for one point light.
pub struct OmniShadowMap {
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
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
            bind_group,
            sampler,
            resolution,
        }
    }
}

/// Render pipeline for omnidirectional shadow pass.
#[allow(dead_code)]
pub struct OmniShadowRenderer {
    pipeline: wgpu::RenderPipeline,
    shadow_bgl: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    omni_resource_bgl: wgpu::BindGroupLayout,
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

        Self {
            pipeline,
            shadow_bgl,
            uniform_buffer,
            omni_resource_bgl,
        }
    }

    /// Get the bind group layout for sampling omnidirectional shadow maps in the PBR shader.
    pub fn resource_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.omni_resource_bgl
    }
}
