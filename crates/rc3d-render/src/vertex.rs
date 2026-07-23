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

/// Per-vertex colored line for markup overlays — carries its own color.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MarkupVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl MarkupVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MarkupVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

/// Textured quad vertex for world-space annotation labels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WorldLabelVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

impl WorldLabelVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WorldLabelVertex>() as wgpu::BufferAddress,
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
    /// x=clearcoat_factor, y=clearcoat_roughness, zw=pad
    pub pbr_clearcoat: [f32; 4],
    /// x=sheen_color.r, y=sheen_color.g, z=sheen_color.b, w=sheen_roughness
    pub pbr_sheen: [f32; 4],
    /// KHR_materials_specular: xyz=specular_color_factor, w=specular_factor
    pub pbr_specular: [f32; 4],
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
    /// Model matrix (for world-space operations like section cap clip planes).
    pub model: [[f32; 4]; 4],
    /// Clip planes for section cap (up to 6 planes).
    pub clip_planes: [[f32; 4]; 6],
    /// x = clip plane count, yzw unused.
    pub clip_count: [f32; 4],
}

/// Per-vertex data for expanded (AA-capable) line segments.
/// Each line segment produces 6 vertices (2 triangles forming a camera-facing quad).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineVertexExpanded {
    /// One endpoint of the line segment (world space).
    pub position: [f32; 3],
    /// The other endpoint of the line segment (world space).
    pub partner: [f32; 3],
    /// Which side of the line: -1.0 (left of direction) or +1.0 (right).
    pub side: f32,
}

impl LineVertexExpanded {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertexExpanded>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

/// Uniforms for anti-aliased line rendering (expanded quads).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineUniforms {
    pub mvp: [[f32; 4]; 4],
    pub color: [f32; 4],
    /// (half_width_px, aa_px, viewport_w, viewport_h)
    pub line_params: [f32; 4],
}

/// Uniforms for procedural section-cap fill (full-screen triangle + ray-plane intersection).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SectionCapUniforms {
    pub clip_to_world: [[f32; 4]; 4],
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
