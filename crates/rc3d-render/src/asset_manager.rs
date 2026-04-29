use std::collections::HashMap;

use crate::cluster::ClusterSet;
use crate::gpu_resource::{GpuResourceManager, MeshId};

pub const MESH_CACHE_MAX: usize = 256;
pub const MESH_CACHE_IDLE_FRAMES: u64 = 120;

/// GPU-side mesh and cluster resources keyed by stable hashes / pointer keys.
#[derive(Default)]
pub struct GpuAssetManager {
    pub mesh_cache: HashMap<u64, (MeshId, u64)>,
    pub cluster_cache: HashMap<u64, ClusterSet>,
}

impl GpuAssetManager {
    pub fn new() -> Self {
        Self {
            mesh_cache: HashMap::new(),
            cluster_cache: HashMap::new(),
        }
    }

    pub fn invalidate_all(&mut self) {
        self.mesh_cache.clear();
        self.cluster_cache.clear();
    }

    pub fn prune_stale_meshes(&mut self, frame_counter: u64, gpu_meshes: &mut GpuResourceManager) {
        let stale_keys: Vec<u64> = self
            .mesh_cache
            .iter()
            .filter(|(_, (_, last_used))| frame_counter.saturating_sub(*last_used) > MESH_CACHE_IDLE_FRAMES)
            .map(|(k, _)| *k)
            .collect();
        for key in stale_keys {
            if let Some((mesh_id, _)) = self.mesh_cache.remove(&key) {
                gpu_meshes.remove(mesh_id);
            }
        }
    }
}
