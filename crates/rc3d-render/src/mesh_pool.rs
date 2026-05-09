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
/// Default max GPU memory for pool slots: 512 MB.
pub const DEFAULT_MAX_POOL_BYTES: u64 = 512 * 1024 * 1024;

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

impl GpuMeshSlot {
    /// Estimate GPU memory used by this slot's buffers.
    fn byte_size(&self) -> u64 {
        let v = self.vertex_count as u64 * std::mem::size_of::<Vertex>() as u64;
        let i = self.index_count as u64 * 4;
        let e = self.edge_count as u64 * 12;
        let w = self.wireframe_edge_count as u64 * 12;
        v + i + e + w
    }
}

/// Fixed-capacity pool of GPU mesh slots with LRU eviction.
pub struct GpuMeshPool {
    slots: Vec<Option<GpuMeshSlot>>,
    hash_to_slot: LruCache<u64, usize>,
    free_slots: Vec<usize>,
    uploads_this_frame: usize,
    /// Total bytes of all GPU buffers currently in the pool.
    total_bytes: u64,
    /// Evict LRU entries when total_bytes exceeds this limit.
    max_bytes: u64,
}

impl GpuMeshPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            hash_to_slot: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
            free_slots: (0..capacity).rev().collect(),
            uploads_this_frame: 0,
            total_bytes: 0,
            max_bytes: DEFAULT_MAX_POOL_BYTES,
        }
    }

    /// Set the maximum GPU memory budget for pool slots.
    pub fn set_max_bytes(&mut self, max_bytes: u64) {
        self.max_bytes = max_bytes;
    }

    /// Total GPU bytes currently used by pool slots.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
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

        // Estimate bytes for the new mesh buffers
        let new_bytes = vertices.len() as u64 * std::mem::size_of::<Vertex>() as u64
            + indices.map_or(0, |i| i.len() as u64 * 4)
            + edge_positions.map_or(0, |e| e.len() as u64 * 12)
            + wireframe_edges.map_or(0, |e| e.len() as u64 * 12);

        // Evict LRU entries until we have room under the budget
        while self.total_bytes + new_bytes > self.max_bytes && !self.hash_to_slot.is_empty() {
            if let Some((&evicted_hash, &evicted_idx)) = self.hash_to_slot.peek_lru() {
                if let Some(slot) = self.slots[evicted_idx].take() {
                    let slot_bytes = slot.byte_size();
                    self.total_bytes = self.total_bytes.saturating_sub(slot_bytes);
                }
                self.hash_to_slot.pop(&evicted_hash);
                self.free_slots.push(evicted_idx);
            }
        }

        let slot_idx = if let Some(idx) = self.free_slots.pop() {
            idx
        } else {
            // Pool full — evict one more LRU
            let (&evicted_hash, &evicted_idx) = self.hash_to_slot.peek_lru()?;
            if let Some(slot) = self.slots[evicted_idx].take() {
                self.total_bytes = self.total_bytes.saturating_sub(slot.byte_size());
            }
            self.hash_to_slot.pop(&evicted_hash);
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
        self.total_bytes += new_bytes;
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
