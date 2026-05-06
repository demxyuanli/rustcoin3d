use std::num::NonZeroUsize;

use lru::LruCache;

use crate::cluster::ClusterSet;
use crate::gpu_resource::{GpuResourceManager, MeshId};
use crate::gpu_skinning::GpuSkinningResources;
use std::collections::HashMap;

pub const MESH_CACHE_MAX: usize = 256;
pub const CLUSTER_CACHE_MAX: usize = 128;
pub const MESH_CACHE_IDLE_FRAMES: u64 = 120;

/// GPU-side mesh and cluster resources with LRU eviction.
pub struct GpuAssetManager {
    pub mesh_cache: LruCache<u64, (MeshId, u64)>,
    pub cluster_cache: LruCache<u64, ClusterSet>,
}

impl GpuAssetManager {
    pub fn new() -> Self {
        Self {
            mesh_cache: LruCache::new(NonZeroUsize::new(MESH_CACHE_MAX).unwrap()),
            cluster_cache: LruCache::new(NonZeroUsize::new(CLUSTER_CACHE_MAX).unwrap()),
        }
    }

    pub fn invalidate_all(&mut self) {
        self.mesh_cache.clear();
        self.cluster_cache.clear();
    }

    /// Insert or touch a mesh entry. If the cache is full, the least recently used entry
    /// is evicted. Caller is responsible for freeing the evicted GPU resources.
    pub fn mesh_insert(&mut self, key: u64, mesh_id: MeshId, frame: u64) -> Option<(MeshId, u64)> {
        self.mesh_cache.push(key, (mesh_id, frame)).map(|(_, v)| v)
    }

    /// Access a mesh entry, promoting it to most-recently used.
    pub fn mesh_get(&mut self, key: &u64) -> Option<&(MeshId, u64)> {
        self.mesh_cache.get(key)
    }

    /// Touch a mesh entry: update its `last_used` frame and promote to MRU.
    /// Returns the `MeshId` if the entry exists.
    pub fn mesh_touch(&mut self, key: &u64, frame: u64) -> Option<MeshId> {
        if let Some((_mesh_id, _last_used)) = self.mesh_cache.pop(key) {
            let id = _mesh_id;
            self.mesh_cache.push(*key, (id, frame));
            Some(id)
        } else {
            None
        }
    }

    /// Remove a mesh entry by key.
    pub fn mesh_remove(&mut self, key: &u64) -> Option<(MeshId, u64)> {
        self.mesh_cache.pop(key)
    }

    /// Returns the current number of cached meshes.
    pub fn mesh_cache_len(&self) -> usize {
        self.mesh_cache.len()
    }

    /// Insert a cluster set, evicting LRU if at capacity.
    pub fn cluster_insert(&mut self, key: u64, cs: ClusterSet) -> Option<(u64, ClusterSet)> {
        self.cluster_cache.push(key, cs)
    }

    /// Check if a cluster key exists, promoting to MRU.
    pub fn cluster_contains(&mut self, key: &u64) -> bool {
        self.cluster_cache.contains(key)
    }

    /// Access a cluster set, promoting to MRU.
    pub fn cluster_get(&mut self, key: &u64) -> Option<&ClusterSet> {
        self.cluster_cache.get(key)
    }

    /// Remove stale mesh entries that haven't been used in `MESH_CACHE_IDLE_FRAMES` frames.
    pub fn prune_stale_meshes(
        &mut self,
        frame_counter: u64,
        gpu_meshes: &mut GpuResourceManager,
        mut skinned_mesh_resources: Option<&mut HashMap<MeshId, GpuSkinningResources>>,
    ) {
        // Collect stale keys (LRU doesn't support filter-remove, so iterate)
        let stale_keys: Vec<u64> = self
            .mesh_cache
            .iter()
            .filter(|(_, (_, last_used))| {
                frame_counter.saturating_sub(*last_used) > MESH_CACHE_IDLE_FRAMES
            })
            .map(|(k, _)| *k)
            .collect();
        for key in stale_keys {
            if let Some((mesh_id, _)) = self.mesh_cache.pop(&key) {
                gpu_meshes.remove(mesh_id);
                if let Some(map) = skinned_mesh_resources.as_mut() {
                    map.remove(&mesh_id);
                }
            }
        }
    }
}

impl Default for GpuAssetManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_manager_empty() {
        let m = GpuAssetManager::new();
        assert_eq!(m.mesh_cache_len(), 0);
    }

    #[test]
    fn test_mesh_insert_and_touch() {
        let mut m = GpuAssetManager::new();
        let id = MeshId::default();
        m.mesh_insert(1, id, 10);
        assert_eq!(m.mesh_cache_len(), 1);

        // Touch updates frame and promotes
        let touched = m.mesh_touch(&1, 20);
        assert!(touched.is_some());
        assert_eq!(m.mesh_cache_len(), 1);
    }

    #[test]
    fn test_mesh_insert_evicts_lru_when_full() {
        let mut m = GpuAssetManager::new();
        // Fill to capacity (256)
        for i in 0..MESH_CACHE_MAX as u64 {
            m.mesh_insert(i, MeshId::default(), i);
        }
        assert_eq!(m.mesh_cache_len(), MESH_CACHE_MAX);

        // Insert one more — should evict LRU (key 0)
        m.mesh_insert(MESH_CACHE_MAX as u64 + 1, MeshId::default(), 0);
        assert_eq!(m.mesh_cache_len(), MESH_CACHE_MAX);
        assert!(m.mesh_get(&0).is_none()); // evicted
        assert!(m.mesh_get(&1).is_some()); // still there
    }

    #[test]
    fn test_mesh_touch_promotes() {
        let mut m = GpuAssetManager::new();
        for i in 0..(MESH_CACHE_MAX / 2) as u64 {
            m.mesh_insert(i, MeshId::default(), i);
        }
        // Touch key 0 to promote it to MRU
        m.mesh_touch(&0, 100);
        // Fill up to capacity — key 0 should survive evictions
        for i in (MESH_CACHE_MAX / 2) as u64..MESH_CACHE_MAX as u64 + 10 {
            m.mesh_insert(i, MeshId::default(), i);
        }
        assert!(m.mesh_get(&0).is_some(), "touched mesh should survive eviction");
    }

    #[test]
    fn test_mesh_remove() {
        let mut m = GpuAssetManager::new();
        m.mesh_insert(42, MeshId::default(), 0);
        let removed = m.mesh_remove(&42);
        assert!(removed.is_some());
        assert_eq!(m.mesh_cache_len(), 0);
    }

    #[test]
    fn test_invalidate_all() {
        let mut m = GpuAssetManager::new();
        m.mesh_insert(1, MeshId::default(), 0);
        m.invalidate_all();
        assert_eq!(m.mesh_cache_len(), 0);
    }

    #[test]
    fn test_prune_stale_meshes() {
        let mut m = GpuAssetManager::new();
        let mut gpu = GpuResourceManager::new();
        m.mesh_insert(100, MeshId::default(), 10);
        // At frame 200 (> 120 idle), the mesh should be pruned
        m.prune_stale_meshes(200, &mut gpu, None);
        assert_eq!(m.mesh_cache_len(), 0);
    }
}
