//! Parallel dirty-subtree traversal using rayon + staging belt upload.
//!
//! Each dirty subtree is traversed independently on a thread pool (no shared
//! state between subtrees). Results are merged into the FlatDrawCache on the
//! calling thread. The staging belt upload copies all GpuDrawData to a mapped
//! buffer in one shot.

use rayon::prelude::*;
use rc3d_scene::SceneGraph;

use crate::flat_draw_cache::{FlatDrawCache, GpuDrawData};
use crate::global_tables::TexturePathTable;
use crate::render_action::TraversalChunk;

/// Parallel traversal of dirty subtrees using rayon.
///
/// Each dirty subtree is traversed on a thread via the existing RenderCollector,
/// then results are merged into the FlatDrawCache on the calling thread.
pub fn parallel_traverse_into_cache(
    graph: &SceneGraph,
    cache: &mut FlatDrawCache,
    texture_table: &TexturePathTable,
    hidden_nodes: &std::collections::HashSet<rc3d_core::NodeId>,
) {
    let dirty_roots = crate::dirty_flags::collect_dirty_roots(graph);

    if dirty_roots.is_empty() {
        cache.ensure_groups_sorted();
        return;
    }

    // Count total nodes for full vs incremental decision
    let total_nodes = count_all_nodes(graph);
    if dirty_roots.len() as f64 > total_nodes as f64 * 0.5 {
        // Full rebuild — parallel across root nodes
        cache.clear();
        let root_chunks: Vec<TraversalChunk> = graph
            .roots()
            .par_iter()
            .map(|&root| {
                let mut collector = super::render_action::RenderCollector::new();
                collector.set_hidden_nodes(hidden_nodes);
                collector.traverse(graph, root);
                collector_to_chunk(&collector, texture_table)
            })
            .collect();
        for chunk in root_chunks {
            super::render_action::merge_chunk_into_cache(cache, chunk);
        }
    } else {
        // Incremental — invalidate first (must be sequential), then parallel traverse
        for &dirty_root in &dirty_roots {
            super::render_action::invalidate_cache_for_subtree(cache, dirty_root);
        }
        let chunks: Vec<TraversalChunk> = dirty_roots
            .par_iter()
            .map(|&dirty_root| {
                let mut collector = super::render_action::RenderCollector::new();
                collector.set_hidden_nodes(hidden_nodes);
                collector.traverse(graph, dirty_root);
                collector_to_chunk(&collector, texture_table)
            })
            .collect();
        for chunk in chunks {
            super::render_action::merge_chunk_into_cache(cache, chunk);
        }
    }

    cache.groups_dirty = true;
    cache.ensure_groups_sorted();
}

/// Convert RenderCollector output to a TraversalChunk for parallel collection.
/// Note: texture IDs are set to `u16::MAX` because the parallel path cannot
/// mutate the texture table; interning must happen on the calling thread.
fn collector_to_chunk(
    collector: &super::render_action::RenderCollector,
    _texture_table: &TexturePathTable,
) -> TraversalChunk {
    let mut chunk = TraversalChunk::default();

    for dc in &collector.draw_calls {
        // Texture IDs deferred to post-merge interning on the calling thread
        let (gpu, meta) = crate::traversal::draw_call_to_cache_entries(dc, [u16::MAX; 5]);
        chunk.gpu_data.push(gpu);
        chunk.metadata.push(meta);
    }

    chunk.effects.decals.extend(
        collector
            .effect_commands
            .decals
            .clone(),
    );
    chunk.effects.volumes.extend(
        collector
            .effect_commands
            .volumes
            .clone(),
    );
    chunk
        .effects
        .point_clouds
        .extend(collector.effect_commands.point_clouds.clone());
    chunk.total_triangles = 0; // Computed from vertices later if needed

    chunk
}

fn count_all_nodes(graph: &SceneGraph) -> usize {
    crate::traversal::count_all_nodes(graph)
}

/// Parallel upload of GpuDrawData from FlatDrawCache into a staging buffer.
///
/// Uses a single staging buffer with mapped-at-creation write for the full
/// GpuDrawData slice, then copies to the destination buffer via the encoder.
pub fn parallel_upload_gpu_data(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    cache: &FlatDrawCache,
    dst_buffer: &wgpu::Buffer,
) {
    if cache.gpu_data.is_empty() {
        return;
    }
    let total_size = (cache.gpu_data.len() * std::mem::size_of::<GpuDrawData>()) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GpuData staging"),
        size: total_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: true,
    });

    {
        let mut mapped = staging.slice(..).get_mapped_range_mut().expect("staging map");
        let src: &[u8] = bytemuck::cast_slice(&cache.gpu_data);
        // Copy sequentially for correctness — parallel memcpy to a single mapped range
        // is not safe in Rust's aliasing model.
        // For large datasets (>10K draws), split into multiple staging buffers.
        mapped.copy_from_slice(src);
    }
    staging.unmap();

    encoder.copy_buffer_to_buffer(&staging, 0, dst_buffer, 0, total_size);
    // staging buffer is dropped here — GPU retains the copy in dst_buffer
}
