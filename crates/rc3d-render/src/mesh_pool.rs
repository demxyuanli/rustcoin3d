//! Streaming GPU mesh pool with LRU eviction.
//!
//! Caps GPU memory while supporting unlimited unique meshes. When a mesh
//! becomes visible but isn't in GPU memory, it's uploaded to an available
//! slot (evicting LRU if full). Already-present meshes are promoted to MRU
//! without re-upload.
//!
//! Uploads are throttled to `MAX_UPLOADS_PER_FRAME` to avoid pipeline stalls.

use std::num::NonZeroUsize;

use lru::LruCache;

use crate::vertex::Vertex;

pub const DEFAULT_POOL_SIZE: usize = 4096;
pub const MAX_UPLOADS_PER_FRAME: usize = 16;

/// One slot in the mesh pool — owns GPU buffers for a single mesh.
pub struct GpuMeshSlot {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    pub edge_buffer: Option<wgpu::Buffer>,
    pub wireframe_edge_buffer: Option<wgpu::Buffer>,
    pub vertex_count: u32,
    pub index_count: u32,
    pub edge_count: u32,
    pub wireframe_edge_count: u32,
    pub last_used_frame: u64,
}

/// Fixed-capacity pool of GPU mesh slots with LRU eviction.
pub struct GpuMeshPool {
    slots: Vec<Option<GpuMeshSlot>>,
    hash_to_slot: LruCache<u64, usize>,
    free_slots: Vec<usize>,
    uploads_this_frame: usize,
}

impl GpuMeshPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            hash_to_slot: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
            free_slots: (0..capacity).rev().collect(),
            uploads_this_frame: 0,
        }
    }

    /// Check if a mesh is in the pool. Promotes to MRU on hit.
    pub fn get(&mut self, mesh_hash: u64, frame: u64) -> Option<&GpuMeshSlot> {
        let &slot_idx = self.hash_to_slot.get(&mesh_hash)?;
        let slot = self.slots[slot_idx].as_mut()?;
        slot.last_used_frame = frame;
        Some(slot)
    }

    /// Upload a mesh, returning the slot index. Evicts LRU if full.
    /// Returns None if the upload throttle is exceeded (try next frame).
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        mesh_hash: u64,
        vertices: &[Vertex],
        indices: Option<&[u32]>,
        edge_positions: Option<&[[f32; 3]]>,
        wireframe_edges: Option<&[[f32; 3]]>,
        frame: u64,
    ) -> Option<usize> {
        if self.uploads_this_frame >= MAX_UPLOADS_PER_FRAME {
            return None;
        }

        let slot_idx = if let Some(idx) = self.free_slots.pop() {
            idx
        } else {
            // Pool full — evict LRU
            let (&evicted_hash, &evicted_idx) = self.hash_to_slot.peek_lru()?;
            self.hash_to_slot.pop(&evicted_hash);
            self.slots[evicted_idx] = None; // drop GPU buffers
            evicted_idx
        };

        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshPool Vertex"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = indices.map(|idx| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MeshPool Index"),
                contents: bytemuck::cast_slice(idx),
                usage: wgpu::BufferUsages::INDEX,
            })
        });
        let edge_buffer = edge_positions.map(|e| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MeshPool Edge"),
                contents: bytemuck::cast_slice(e),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let wireframe_edge_buffer = wireframe_edges.map(|e| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MeshPool WireEdge"),
                contents: bytemuck::cast_slice(e),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });

        let index_count = indices.map_or(0, |i| i.len() as u32);
        let edge_count = edge_positions.map_or(0, |e| e.len() as u32);
        let wireframe_edge_count = wireframe_edges.map_or(0, |e| e.len() as u32);

        self.slots[slot_idx] = Some(GpuMeshSlot {
            vertex_buffer,
            index_buffer,
            edge_buffer,
            wireframe_edge_buffer,
            vertex_count: vertices.len() as u32,
            index_count,
            edge_count,
            wireframe_edge_count,
            last_used_frame: frame,
        });
        self.hash_to_slot.push(mesh_hash, slot_idx);
        self.uploads_this_frame += 1;
        Some(slot_idx)
    }

    /// Reset the per-frame upload counter. Call at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.uploads_this_frame = 0;
    }

    /// Number of meshes currently in the pool.
    pub fn len(&self) -> usize {
        self.hash_to_slot.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.hash_to_slot.is_empty()
    }

    /// Number of uploads performed this frame.
    pub fn uploads_this_frame(&self) -> usize {
        self.uploads_this_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_is_empty() {
        let pool = GpuMeshPool::new(64);
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.uploads_this_frame(), 0);
    }

    #[test]
    fn begin_frame_resets_counter() {
        let mut pool = GpuMeshPool::new(64);
        // Can't test upload without GPU device
        pool.begin_frame();
        assert_eq!(pool.uploads_this_frame(), 0);
    }
}
