use bytemuck::{Pod, Zeroable};

pub const MAX_MORPH_WEIGHTS: usize = 8;

/// Per-instance data stored in a read-only SSBO for instanced rendering.
/// One entry per instance; the vertex shader reads model/mvp/material from here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub mvp: [[f32; 4]; 4],
    pub diffuse_color: [f32; 4],
    pub base_color: [f32; 4],
    pub metallic_roughness: [f32; 4],
    /// xyz = emissive color, w = alpha_cutoff
    pub emissive_alpha: [f32; 4],
    /// Morph target weights (up to MAX_MORPH_WEIGHTS targets).
    pub morph_weights: [f32; MAX_MORPH_WEIGHTS],
    /// x = morph target count, yzw = unused.
    pub morph_count: [f32; 4],
}

pub const MAX_INSTANCES: usize = 65536;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub texcoord: [f32; 2],
    /// Tangent vector (xyz) + handedness (w ∈ {-1, 1})
    pub tangent: [f32; 4],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
}

impl LineVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 1] =
            wgpu::vertex_attr_array![0 => Float32x3];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

pub const MAX_LIGHTS: usize = 16;
pub const CSM_CASCADE_COUNT: usize = 4;
pub const GPU_OBJECT_TRANSFORM_SIZE: u64 = 128;

/// Per-object transform for GPU compute culling.
/// 128 bytes, two cache lines.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuObjectTransform {
    pub model_matrix: [[f32; 4]; 4],   // 64B
    pub aabb_min: [f32; 3],            // 12B
    pub flags: u32,                    // 4B
    pub aabb_max: [f32; 3],            // 12B
    pub mesh_id: u32,                  // 4B
    pub material_id: u32,              // 4B
    pub _pad: [u32; 7],               // 28B — align to 128B
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SceneUniforms {
    pub mvp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub diffuse_color: [f32; 4],
    pub ambient_color: [f32; 4],
    pub specular_color: [f32; 4],
    pub shininess: [f32; 4],
    pub clip_planes: [[f32; 4]; 6],
    pub clip_count: [f32; 4],
    pub pbr_base_color: [f32; 4],
    pub pbr_metallic_roughness: [f32; 4],
    pub pbr_emissive_alpha: [f32; 4],
    pub pbr_alpha_flags: [f32; 4],
    pub light_set_index: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GlobalFrameUniforms {
    pub light_dirs: [[f32; 4]; MAX_LIGHTS],
    pub light_colors: [[f32; 4]; MAX_LIGHTS],
    pub light_types: [[f32; 4]; MAX_LIGHTS],
    pub light_positions: [[f32; 4]; MAX_LIGHTS],
    pub spot_params: [[f32; 4]; MAX_LIGHTS],
    pub light_count: [f32; 4],
    pub ibl_diffuse: [f32; 4],
    pub ibl_specular: [f32; 4],
    pub csm_view_proj: [[f32; 4]; 16],
    pub csm_split_depths: [f32; 4],
    pub shadow_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowDrawUniforms {
    pub shadow_mvp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FlatUniforms {
    pub mvp: [[f32; 4]; 4],
    pub color: [f32; 4],
}

/// Uniforms for procedural section-cap fill (plane-distance discard in fragment shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SectionCapUniforms {
    pub mvp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub plane: [f32; 4],
    /// x = minimum world-space half-thickness; yzw unused.
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OutlineUniforms {
    pub mvp: [[f32; 4]; 4],
    pub outline_width: f32,
    pub _pad: [f32; 3],
    pub color: [f32; 4],
}
