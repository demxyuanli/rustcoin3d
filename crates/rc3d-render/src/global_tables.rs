//! Global render tables: interned texture paths and shared GPU light buffer.
//!
//! These replace per-DrawCall heap allocations with index-based lookups:
//! - `TexturePathTable`: stores `Arc<str>` once, DrawCalls hold a `u16` index
//! - `GlobalLightUniform`: single GPU-light array shared by all draw calls

use std::sync::Arc;

use rc3d_core::NodeId;

use crate::vertex::MAX_LIGHTS;

// ---------------------------------------------------------------------------
// TexturePathTable
// ---------------------------------------------------------------------------

/// Interned string table for texture paths.
///
/// DrawCalls store a `u16` index instead of `Arc<str>`, eliminating redundant
/// heap allocations across thousands of draw calls that reference the same texture.
#[derive(Default)]
pub struct TexturePathTable {
    paths: Vec<Arc<str>>,
}

impl TexturePathTable {
    pub fn new() -> Self {
        Self { paths: Vec::new() }
    }

    /// Insert a path and return its index. If the path already exists,
    /// the existing index is returned (no duplicate entries).
    pub fn intern(&mut self, path: &str) -> u16 {
        if let Some(idx) = self.paths.iter().position(|p| p.as_ref() == path) {
            return idx as u16;
        }
        let idx = self.paths.len() as u16;
        self.paths.push(Arc::from(path));
        idx
    }

    /// Look up a path by its index.
    pub fn get(&self, idx: u16) -> Option<&Arc<str>> {
        self.paths.get(idx as usize)
    }

    /// Returns `true` if the table contains no entries.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Returns the number of interned paths.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Build the table from the scene graph by scanning all material nodes.
    /// Called once at initialization or on scene load.
    pub fn build_from_scene(&mut self, graph: &rc3d_scene::SceneGraph) {
        self.paths.clear();
        for &root in graph.roots() {
            self.intern_recursive(graph, root);
        }
    }

    fn intern_recursive(&mut self, graph: &rc3d_scene::SceneGraph, node_id: NodeId) {
        let Some(entry) = graph.get(node_id) else { return };
        if let rc3d_scene::NodeData::Material(mat) = &entry.data {
            if let Some(path) = &mat.albedo_texture {
                if !path.is_empty() {
                    self.intern(path);
                }
            }
            if let Some(path) = &mat.normal_texture {
                if !path.is_empty() {
                    self.intern(path);
                }
            }
            if let Some(path) = &mat.metallic_roughness_texture {
                if !path.is_empty() {
                    self.intern(path);
                }
            }
            if let Some(path) = &mat.emissive_texture {
                if !path.is_empty() {
                    self.intern(path);
                }
            }
            if let Some(path) = &mat.occlusion_texture {
                if !path.is_empty() {
                    self.intern(path);
                }
            }
        }
        if let Some(children) = graph.children(node_id) {
            for &child in children {
                self.intern_recursive(graph, child);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpuLight + GlobalLightUniform
// ---------------------------------------------------------------------------

/// A single GPU light, packed for direct upload to a uniform/storage buffer.
///
/// Each field uses `[f32; 4]` for alignment (std140-compatible layout).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub direction: [f32; 4],
    pub color: [f32; 4],
    pub position: [f32; 4],
    /// (inner_angle, outer_angle, falloff, light_type)
    /// light_type: 0 = directional, 1 = point, 2 = spot
    pub spot_params: [f32; 4],
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            direction: [0.0, 0.0, -1.0, 0.0],
            color: [0.0; 4],
            position: [0.0; 4],
            spot_params: [0.0; 4],
        }
    }
}

/// Shared GPU light uniform block.
///
/// Replaces the per-DrawCall `[MAX_LIGHTS]` arrays (320 bytes waste per draw
/// call for 16 lights x 5 arrays of [f32; 4]), consolidating lights into a
/// single buffer shared by all draw calls in the frame.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalLightUniform {
    pub lights: [GpuLight; MAX_LIGHTS],
    pub count: u32,
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 4],
}

impl Default for GlobalLightUniform {
    fn default() -> Self {
        Self {
            lights: [GpuLight::default(); MAX_LIGHTS],
            count: 0,
            ambient_intensity: 0.2,
            ambient_color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}
