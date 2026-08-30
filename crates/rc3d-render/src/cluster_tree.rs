//! Hierarchical LOD cluster tree for GPU-driven meshlet culling.
//!
//! For large meshes (500K+ triangles), meshlet clusters are organized into
//! multiple LOD levels. GPU culls top-down: coarsest level first, then refines
//! visible clusters to finer levels. Uses indirect dispatch so only visible
//! clusters generate work.
//!
//! See `shaders/cluster_tree_cull.wgsl` for the compute shader.

use crate::cluster::ClusterSet;
use wgpu::util::DeviceExt as _;

/// One level in the cluster LOD tree.
pub struct ClusterLodLevel {
    /// Meshlet clusters at this LOD (coarsest = level 0).
    pub set: ClusterSet,
    pub cluster_count: u32,
    /// For each cluster: index of first child in the next level.
    pub first_child: Vec<u32>,
    /// For each cluster: number of children in the next level.
    pub child_count: Vec<u32>,
    /// GPU buffer holding first_child data.
    pub first_child_buf: wgpu::Buffer,
    /// GPU buffer holding child_count data.
    pub child_count_buf: wgpu::Buffer,
    /// Output buffer: visible child cluster indices for the next level.
    pub next_visible_buf: wgpu::Buffer,
    /// Atomic counter for next_visible_buf writes.
    pub next_count_buf: wgpu::Buffer,
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
    /// Compute pipeline for hierarchical culling (shared across all levels).
    pub cull_pipeline: wgpu::ComputePipeline,
    /// Bind group layout matching cluster_tree_cull.wgsl.
    pub cull_bgl: wgpu::BindGroupLayout,
}

impl ClusterTree {
    /// Create the bind group layout for the hierarchical culling compute shader.
    pub fn create_cull_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ClusterTree Cull BGL"),
            entries: &[
                // binding 0: bounds (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: first_child (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: child_count (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: frustum (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: HZB texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 5: next_visible (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6: next_visible_count (storage, read_write, atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the compute pipeline for hierarchical culling.
    pub fn create_cull_pipeline(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster_tree_cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/cluster_tree_cull.wgsl"
            ))),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ClusterTree Cull Layout"),
            bind_group_layouts: &[Some(bgl)],
            immediate_size: 16,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ClusterTree Cull Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Build a cluster tree from meshlet data at each LOD level.
    ///
    /// `lod_meshlets[0]` is the coarsest LOD, `lod_meshlets[N]` is the finest.
    pub fn build(
        device: &wgpu::Device,
        lod_meshlets: &[rc3d_mesh::MeshletData],
        cull_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let cull_pipeline = Self::create_cull_pipeline(device, cull_bgl);
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
                (vec![0u32; cs.meshlet_count as usize],
                 if cs.meshlet_count > 0 { vec![next_count] } else { vec![] })
            } else {
                (vec![0u32; cs.meshlet_count as usize],
                 vec![0u32; cs.meshlet_count as usize])
            };

            let first_child_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("ClusterTree first_child L{}", i)),
                contents: bytemuck::cast_slice(&first_child),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let child_count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("ClusterTree child_count L{}", i)),
                contents: bytemuck::cast_slice(&child_count),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // Per-level output buffers
            let next_visible_size = if i + 1 < lod_meshlets.len() {
                lod_meshlets[i + 1].total_meshlets
            } else {
                cs.meshlet_count
            } as u64 * 4;
            let next_visible_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ClusterTree next_visible L{}", i)),
                size: next_visible_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let next_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ClusterTree next_count L{}", i)),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            levels.push(ClusterLodLevel {
                set: cs,
                cluster_count: md.total_meshlets,
                first_child,
                child_count,
                first_child_buf,
                child_count_buf,
                next_visible_buf,
                next_count_buf,
            });
        }

        Self { levels, max_lod, total_triangles, cull_pipeline, cull_bgl: cull_bgl.clone() }
    }

    pub fn lod_count(&self) -> usize {
        self.levels.len()
    }

    /// Create a bind group for a specific LOD level.
    pub fn create_level_bind_group(
        &self,
        device: &wgpu::Device,
        level: usize,
        frustum_uniform: &wgpu::Buffer,
        hzb_view: &wgpu::TextureView,
        next_visible_buf: &wgpu::Buffer,
        next_count_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let lvl = &self.levels[level];
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("ClusterTree BG L{}", level)),
            layout: &self.cull_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lvl.set.bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lvl.first_child_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lvl.child_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: frustum_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: next_visible_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: next_count_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch culling for one LOD level.
    pub fn cull_level(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        cluster_count: u32,
        hzb_mip: u32,
        use_hzb: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ClusterTree Cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.cull_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let pc: [u32; 4] = [cluster_count, hzb_mip, use_hzb, 0];
        pass.set_immediates(0, bytemuck::bytes_of(&pc));
        let wg_count = cluster_count.div_ceil(64);
        pass.dispatch_workgroups(wg_count, 1, 1);
    }
}
