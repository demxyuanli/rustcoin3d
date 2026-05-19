//! FlatDrawCache: persistent frame-to-frame cache with hot-cold data split.
//!
//! Replaces the heap-heavy `DrawCall` struct with:
//! - `GpuDrawData` (64B, cache-aligned) — uploaded to GPU uniform ring buffer each frame.
//! - `CachedDrawMetadata` (~72B) — only updated when a node's dirty flag triggers re-traversal.

use std::ops::Range;

use crate::render_passes::pass_effects::EffectCommands;

// ── Hot data (~100 bytes, no padding) ──
// Uploaded to GPU uniform ring buffer each frame.
// model_matrix occupies the first 64B (one cache line).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuDrawData {
    pub model_matrix: [[f32; 4]; 4], // 64B — first cache line
    pub material_id: u32,
    pub light_set_id: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub draw_flags: u32,
    pub instance_count: u32,
    pub _pad: u32,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Default)]
    pub struct DrawFlags: u32 {
        const HAS_EDGES      = 1 << 0;
        const TRANSPARENT    = 1 << 1;
        const OVERLAY        = 1 << 2;
        const INSTANCED      = 1 << 3;
        const SELECTED       = 1 << 4;
        const DEPTH_REVERSED = 1 << 5;
        const ORTHOGRAPHIC   = 1 << 6;
        const WIREFRAME      = 1 << 7;
    }
}

impl GpuDrawData {
    /// Zeroed default — used as a tombstone in the cache (invalidated entry).
    pub fn zeroed() -> Self {
        Self::default()
    }
}

// ── Cold data (~72 bytes) ──
// Only updated when a node's dirty flag triggers re-traversal.
#[derive(Clone)]
pub struct CachedDrawMetadata {
    pub mesh_hash: u64,
    pub bvh_node_id: Option<u32>,
    pub material_params: MaterialUniform,
    pub albedo_tex_id: u16,
    pub normal_tex_id: u16,
    pub mr_tex_id: u16,
    pub emissive_tex_id: u16,
    pub occlusion_tex_id: u16,
    pub atlas_layer: u16,
    pub alpha_mode: u32,
    pub alpha_cutoff: f32,
    pub double_sided: u32,
    pub _pad: [u32; 2],
}

impl Default for CachedDrawMetadata {
    fn default() -> Self {
        Self {
            mesh_hash: 0,
            bvh_node_id: None,
            material_params: MaterialUniform::default(),
            albedo_tex_id: u16::MAX,
            normal_tex_id: u16::MAX,
            mr_tex_id: u16::MAX,
            emissive_tex_id: u16::MAX,
            occlusion_tex_id: u16::MAX,
            atlas_layer: 0,
            alpha_mode: 0,
            alpha_cutoff: 0.5,
            double_sided: 0,
            _pad: [0; 2],
        }
    }
}

// ── Compact PBR material uniform ──
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    pub emissive_color: [f32; 4],
    pub metallic_roughness_anisotropic: [f32; 4], // (metallic, roughness, anisotropic, _unused)
}

impl Default for MaterialUniform {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            emissive_color: [0.0; 4],
            metallic_roughness_anisotropic: [0.0, 0.5, 0.0, 0.0],
        }
    }
}

// ── Persistent frame-to-frame cache ──
pub struct FlatDrawCache {
    pub gpu_data: Vec<GpuDrawData>,
    pub metadata: Vec<CachedDrawMetadata>,
    pub effect_commands: EffectCommands,

    // Draw ordering groups (indices into gpu_data/metadata)
    pub opaque_order: Vec<usize>,
    pub transparent_order: Vec<usize>,
    pub edge_order: Vec<usize>,
    pub selected_order: Vec<usize>,

    // Ranges for quick iteration without resorting
    pub opaque_range: Range<u32>,
    pub transparent_range: Range<u32>,

    /// Node-to-cache mapping: graph NodeId -> index into gpu_data[].
    pub node_to_draw: std::collections::HashMap<rc3d_core::NodeId, usize>,

    pub groups_dirty: bool,
    pub total_triangles: u64,
}

impl Default for FlatDrawCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatDrawCache {
    pub fn new() -> Self {
        Self {
            gpu_data: Vec::with_capacity(1024),
            metadata: Vec::with_capacity(1024),
            effect_commands: EffectCommands::default(),
            opaque_order: Vec::with_capacity(1024),
            transparent_order: Vec::with_capacity(128),
            edge_order: Vec::with_capacity(1024),
            selected_order: Vec::with_capacity(64),
            opaque_range: 0..0,
            transparent_range: 0..0,
            node_to_draw: Default::default(),
            groups_dirty: true,
            total_triangles: 0,
        }
    }

    pub fn clear(&mut self) {
        self.gpu_data.clear();
        self.metadata.clear();
        self.effect_commands = EffectCommands::default();
        self.opaque_order.clear();
        self.transparent_order.clear();
        self.edge_order.clear();
        self.selected_order.clear();
        self.node_to_draw.clear();
        self.groups_dirty = true;
        self.total_triangles = 0;
    }

    /// Rebuild draw order groups when dirty. Called after traversal.
    /// Sorts opaque draws by mesh_hash for batching (minimize pipeline/buffer switches).
    pub fn ensure_groups_sorted(&mut self) {
        if !self.groups_dirty {
            return;
        }
        self.opaque_order.clear();
        self.transparent_order.clear();
        self.edge_order.clear();
        self.selected_order.clear();

        for (i, gpu) in self.gpu_data.iter().enumerate() {
            let flags = DrawFlags::from_bits_truncate(gpu.draw_flags);
            if flags.contains(DrawFlags::TRANSPARENT) {
                self.transparent_order.push(i);
            } else {
                self.opaque_order.push(i);
            }
            if flags.contains(DrawFlags::HAS_EDGES) && !flags.contains(DrawFlags::TRANSPARENT) {
                self.edge_order.push(i);
            }
            if flags.contains(DrawFlags::SELECTED) {
                self.selected_order.push(i);
            }
        }
        // Sort opaque draws by mesh_hash for GPU batching
        self.opaque_order.sort_by_key(|&i| self.metadata[i].mesh_hash);

        self.opaque_range = 0..(self.opaque_order.len() as u32);
        self.transparent_range = 0..(self.transparent_order.len() as u32);
        self.groups_dirty = false;
    }
}
