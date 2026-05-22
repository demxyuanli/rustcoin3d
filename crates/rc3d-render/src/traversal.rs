//! Traversal module: cache management and draw call conversion utilities.
//!
//! This module contains code for:
//! - `TraversalChunk`: intermediate representation for traversal results
//! - Cache management: merge, invalidate, populate
//! - DrawCall -> FlatDrawCache conversion utilities

use rc3d_core::NodeId;
use rc3d_scene::AlphaMode;
use std::collections::HashMap;

use crate::flat_draw_cache::{
    CachedDrawMetadata, DrawFlags, FlatDrawCache, GpuDrawData, MaterialUniform,
};
use crate::global_tables::TexturePathTable;
use crate::render_passes::pass_effects::EffectCommands;
// Import DrawCall for use in conversion functions (private to avoid circular re-export)
use crate::render_action::DrawCall;

// ── TraversalChunk ──────────────────────────────────────────────────────────

/// This is merged into FlatDrawCache by the caller.
#[derive(Default)]
pub struct TraversalChunk {
    pub gpu_data: Vec<GpuDrawData>,
    pub metadata: Vec<CachedDrawMetadata>,
    pub effects: EffectCommands,
    pub node_to_draw: HashMap<NodeId, usize>,
    pub total_triangles: u64,
}

// ── Cache management ────────────────────────────────────────────────────────

/// Merge a traversal chunk into the FlatDrawCache.
pub fn merge_chunk_into_cache(cache: &mut FlatDrawCache, chunk: TraversalChunk) {
    let base = cache.gpu_data.len();
    cache.gpu_data.extend(chunk.gpu_data);
    cache.metadata.extend(chunk.metadata);
    cache.total_triangles += chunk.total_triangles;
    for (node_id, local_idx) in chunk.node_to_draw {
        cache.node_to_draw.insert(node_id, base + local_idx);
    }
    // Effect commands are appended
    cache.effect_commands.decals.extend(chunk.effects.decals);
    cache.effect_commands.volumes.extend(chunk.effects.volumes);
    cache
        .effect_commands
        .point_clouds
        .extend(chunk.effects.point_clouds);
}

/// Invalidate a cached subtree — remove its entries from the cache.
/// Zeroes out the gpu_data entry so it will be skipped during rendering.
pub fn invalidate_cache_for_subtree(cache: &mut FlatDrawCache, node: NodeId) {
    if let Some(&idx) = cache.node_to_draw.get(&node) {
        cache.gpu_data[idx] = GpuDrawData::zeroed();
    }
}

// ── Node counting helpers ───────────────────────────────────────────────────

pub(crate) fn count_all_nodes(graph: &rc3d_scene::SceneGraph) -> usize {
    graph
        .roots()
        .iter()
        .map(|&r| count_subtree(graph, r))
        .sum()
}

fn count_subtree(graph: &rc3d_scene::SceneGraph, node: NodeId) -> usize {
    let Some(entry) = graph.get(node) else {
        return 0;
    };
    1 + entry
        .children
        .iter()
        .map(|&c| count_subtree(graph, c))
        .sum::<usize>()
}

// ── DrawCall -> FlatDrawCache conversion ──────────────────────────────────

/// Compute texture IDs for a DrawCall by interning paths into the texture table.
/// Returns `[albedo, normal, mr, emissive, occlusion]` IDs.
pub(crate) fn intern_draw_call_tex_ids(
    dc: &DrawCall,
    texture_table: &mut TexturePathTable,
) -> [u16; 5] {
    [
        dc.albedo_path.as_deref().map_or(u16::MAX, |p| texture_table.intern(p)),
        dc.normal_path.as_deref().map_or(u16::MAX, |p| texture_table.intern(p)),
        dc.metallic_roughness_path.as_deref().map_or(u16::MAX, |p| texture_table.intern(p)),
        dc.emissive_path.as_deref().map_or(u16::MAX, |p| texture_table.intern(p)),
        dc.occlusion_path.as_deref().map_or(u16::MAX, |p| texture_table.intern(p)),
    ]
}

/// Intern texture paths from draw calls into the texture table.
/// Convenience wrapper for batch texture path registration.
pub fn convert_draw_calls_to_cache_textures(
    draw_calls: &[DrawCall],
    texture_table: &mut TexturePathTable,
) {
    for dc in draw_calls {
        let _ = intern_draw_call_tex_ids(dc, texture_table);
    }
}

/// Build `(GpuDrawData, CachedDrawMetadata)` from a `DrawCall` for cache insertion.
///
/// `tex_ids` contains pre-computed texture IDs: `[albedo, normal, mr, emissive, occlusion]`.
/// Use `intern_draw_call_tex_ids` to compute them, or pass `[u16::MAX; 5]` if
/// texture IDs are not yet available (they will need backfilling later).
pub(crate) fn draw_call_to_cache_entries(
    dc: &DrawCall,
    tex_ids: [u16; 5],
) -> (GpuDrawData, CachedDrawMetadata) {
    let mut flags = DrawFlags::empty();
    if !dc.edge_positions.is_empty() {
        flags |= DrawFlags::HAS_EDGES;
    }
    if dc.is_overlay {
        flags |= DrawFlags::OVERLAY;
    }
    if dc.selected {
        flags |= DrawFlags::SELECTED;
    }
    if dc.alpha_mode != AlphaMode::Opaque {
        flags |= DrawFlags::TRANSPARENT;
    }
    if dc.depth_reversed_z {
        flags |= DrawFlags::DEPTH_REVERSED;
    }
    if dc.projection_orthographic {
        flags |= DrawFlags::ORTHOGRAPHIC;
    }

    let gpu = GpuDrawData {
        model_matrix: dc.model_matrix.to_cols_array_2d(),
        material_id: 0,
        light_set_id: dc.light_set_id,
        vertex_offset: 0,
        vertex_count: dc.vertices.len() as u32,
        index_offset: 0,
        index_count: dc.indices.as_ref().map_or(0, |i| i.len() as u32),
        draw_flags: flags.bits(),
        instance_count: 1,
        _pad: 0,
    };

    let meta = CachedDrawMetadata {
        mesh_hash: dc.mesh_hash.unwrap_or(0),
        bvh_node_id: None,
        material_params: MaterialUniform {
            base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, dc.opacity],
            emissive_color: [
                dc.emissive_color.x,
                dc.emissive_color.y,
                dc.emissive_color.z,
                1.0,
            ],
            metallic_roughness_anisotropic: [
                dc.metallic,
                dc.roughness,
                dc.anisotropic,
                0.0,
            ],
        },
        albedo_tex_id: tex_ids[0],
        normal_tex_id: tex_ids[1],
        mr_tex_id: tex_ids[2],
        emissive_tex_id: tex_ids[3],
        occlusion_tex_id: tex_ids[4],
        atlas_layer: 0,
        alpha_mode: match dc.alpha_mode {
            AlphaMode::Opaque => 0,
            AlphaMode::Mask => 1,
            AlphaMode::Blend => 2,
        },
        alpha_cutoff: dc.alpha_cutoff,
        double_sided: dc.double_sided as u32,
        _pad: [0; 2],
    };

    (gpu, meta)
}

/// Populate FlatDrawCache from existing DrawCall data.
/// Called each frame as a side effect: the render loop consumes DrawCalls
/// directly, while the cache is built for future incremental traversal.
pub fn populate_cache_from_draw_calls(
    cache: &mut FlatDrawCache,
    draw_calls: &[DrawCall],
    texture_table: &mut TexturePathTable,
) {
    cache.clear();
    for dc in draw_calls {
        let tex_ids = intern_draw_call_tex_ids(dc, texture_table);
        let (gpu, meta) = draw_call_to_cache_entries(dc, tex_ids);
        cache.gpu_data.push(gpu);
        cache.metadata.push(meta);
    }
}
