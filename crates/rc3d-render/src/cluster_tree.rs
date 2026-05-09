//! Hierarchical LOD cluster tree for GPU-driven meshlet culling.
//!
//! For large meshes (500K+ triangles), meshlet clusters are organized into
//! multiple LOD levels. GPU culls top-down: coarsest level first, then refines
//! visible clusters to finer levels. Uses indirect dispatch so only visible
//! clusters generate work.
//!
//! See `shaders/cluster_tree_cull.wgsl` for the compute shader.

use crate::cluster::ClusterSet;

/// One level in the cluster LOD tree.
pub struct ClusterLodLevel {
    /// Meshlet clusters at this LOD (coarsest = level 0).
    pub set: ClusterSet,
    pub cluster_count: u32,
    /// For each cluster: index of first child in the next level.
    pub first_child: Vec<u32>,
    /// For each cluster: number of children in the next level.
    pub child_count: Vec<u32>,
}

/// Hierarchical LOD tree for a single large mesh.
///
/// Level 0 is the coarsest LOD, level N is the finest.
/// GPU culling dispatches one workgroup per cluster at each level,
/// testing frustum + HZB occlusion and writing visible children
/// to the next level's indirect buffer.
pub struct ClusterTree {
    pub levels: Vec<ClusterLodLevel>,
    pub max_lod: u32,
    pub total_triangles: u64,
}

impl ClusterTree {
    /// Build a cluster tree from meshlet data at each LOD level.
    ///
    /// `lod_meshlets[0]` is the coarsest LOD, `lod_meshlets[N]` is the finest.
    pub fn build(
        device: &wgpu::Device,
        lod_meshlets: &[rc3d_mesh::MeshletData],
        cull_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let mut levels = Vec::with_capacity(lod_meshlets.len());
        let max_lod = lod_meshlets.len().saturating_sub(1) as u32;
        let mut total_triangles = 0u64;

        for (i, md) in lod_meshlets.iter().enumerate() {
            let cs = ClusterSet::from_meshlet_data(
                device, md, cull_bgl, cull_bgl, cull_bgl,
            );
            total_triangles += md.total_triangles as u64;

            let (first_child, child_count) = if i + 1 < lod_meshlets.len() {
                let next_count = lod_meshlets[i + 1].total_meshlets;
                // Initial: all children linked to cluster 0
                (vec![0u32; cs.meshlet_count as usize],
                 if cs.meshlet_count > 0 { vec![next_count as u32] } else { vec![] })
            } else {
                (vec![0u32; cs.meshlet_count as usize],
                 vec![0u32; cs.meshlet_count as usize])
            };

            levels.push(ClusterLodLevel {
                set: cs,
                cluster_count: md.total_meshlets as u32,
                first_child,
                child_count,
            });
        }

        Self { levels, max_lod, total_triangles }
    }

    pub fn lod_count(&self) -> usize {
        self.levels.len()
    }
}
