use wgpu::util::DeviceExt;

use rc3d_mesh::MeshletData;

pub struct ClusterSet {
    pub meshlet_buffer: wgpu::Buffer,
    pub bounds_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub visible_buffer: wgpu::Buffer,
    pub compact_index_buffer: wgpu::Buffer,
    pub indirect_buffer: wgpu::Buffer,
    pub meshlet_count: u32,
    pub total_triangles: u32,
    pub total_indices: u32,
    /// Pre-built static bind group for the cull pass (meshlet + bounds + visible buffers).
    pub cull_static_bg: wgpu::BindGroup,
    /// Pre-built bind group for the compact pass.
    pub compact_bg: wgpu::BindGroup,
    /// Pre-built bind group for the finalize pass.
    pub finalize_bg: wgpu::BindGroup,
}

impl ClusterSet {
    pub fn from_meshlet_data(
        device: &wgpu::Device,
        data: &MeshletData,
        cull_static_bgl: &wgpu::BindGroupLayout,
        compact_bgl: &wgpu::BindGroupLayout,
        finalize_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let meshlet_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Meshlet Buf"),
            contents: bytemuck::cast_slice(&data.meshlets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Meshlet Bounds Buf"),
            contents: bytemuck::cast_slice(&data.meshlet_bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cluster Vert Buf"),
            contents: bytemuck::cast_slice(&data.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cluster Idx Buf"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
        });

        let meshlets_n = data.total_meshlets.max(1);
        let indices_n = data.indices.len().max(1) as u64;

        let visible_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible List"),
            size: 4 + (meshlets_n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compact_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compact Idx"),
            size: indices_n * 4,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Args"),
            size: 24,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-build static bind groups
        let cull_static_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cull Static BG"),
            layout: cull_static_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meshlet_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: visible_buffer.as_entire_binding(),
                },
            ],
        });

        let compact_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compact BG"),
            layout: compact_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visible_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meshlet_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: compact_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect_buffer.as_entire_binding(),
                },
            ],
        });

        let finalize_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Finalize BG"),
            layout: finalize_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: indirect_buffer.as_entire_binding(),
            }],
        });

        ClusterSet {
            meshlet_buffer,
            bounds_buffer,
            vertex_buffer,
            index_buffer,
            visible_buffer,
            compact_index_buffer,
            indirect_buffer,
            meshlet_count: data.total_meshlets,
            total_triangles: data.total_triangles,
            total_indices: data.indices.len() as u32,
            cull_static_bg,
            compact_bg,
            finalize_bg,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub view_pos: [f32; 4],
    pub meshlet_count: u32,
    pub lod_stride: u32,
    pub meshlet_phase: u32,
    pub meshlet_stride_spatial: u32,
    pub hzb_dims: [u32; 4],
    pub hzb_mip_max: u32,
    pub hzb_enabled: u32,
    pub depth_reversed_z: u32,
    pub orthographic_projection: u32,
}

pub struct ClusterRenderer {
    cull_pipeline: wgpu::ComputePipeline,
    compact_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    cull_dynamic_bgl: wgpu::BindGroupLayout,
    cull_static_bgl: wgpu::BindGroupLayout,
    compact_bgl: wgpu::BindGroupLayout,
    finalize_bgl: wgpu::BindGroupLayout,
    cull_uniform_buffer: wgpu::Buffer,
}

impl ClusterRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cluster Cull"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cluster_cull.wgsl").into()),
        });

        let compact_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cluster Compact"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cluster_compact.wgsl").into()),
        });

        let finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compact Finalize"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compact_finalize.wgsl").into()),
        });

        // Dynamic bind group (group 0): uniforms + HZB textures (per-frame)
        let cull_dynamic_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cull Dynamic BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
            ],
        });

        // Static bind group (group 1): meshlet storage buffers (per-ClusterSet, pre-built once)
        let cull_static_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cull Static BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let cull_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cull PLL"),
            bind_group_layouts: &[&cull_dynamic_bgl, &cull_static_bgl],
            push_constant_ranges: &[],
        });

        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cull Pipe"),
            layout: Some(&cull_pll),
            module: &cull_shader,
            entry_point: Some("cull_meshlets"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compact BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compact_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compact PLL"),
            bind_group_layouts: &[&compact_bgl],
            push_constant_ranges: &[],
        });

        let compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compact Pipe"),
            layout: Some(&compact_pll),
            module: &compact_shader,
            entry_point: Some("compact_meshlets"),
            compilation_options: Default::default(),
            cache: None,
        });

        let finalize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Finalize BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let finalize_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Finalize PLL"),
            bind_group_layouts: &[&finalize_bgl],
            push_constant_ranges: &[],
        });

        let finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Finalize Pipe"),
            layout: Some(&finalize_pll),
            module: &finalize_shader,
            entry_point: Some("finalize_args"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cull_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Uniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            cull_pipeline,
            compact_pipeline,
            finalize_pipeline,
            cull_dynamic_bgl,
            cull_static_bgl,
            compact_bgl,
            finalize_bgl,
            cull_uniform_buffer,
        }
    }

    /// Returns the bind group layouts needed to construct ClusterSets.
    pub fn bind_group_layouts(&self) -> (&wgpu::BindGroupLayout, &wgpu::BindGroupLayout, &wgpu::BindGroupLayout) {
        (&self.cull_static_bgl, &self.compact_bgl, &self.finalize_bgl)
    }

    pub fn cull_and_compact(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cluster_set: &ClusterSet,
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        lod_stride: u32,
        meshlet_phase: u32,
        meshlet_stride_spatial: bool,
        hzb_max_view: &wgpu::TextureView,
        hzb_min_view: &wgpu::TextureView,
        hzb_dims: (u32, u32),
        hzb_mip_max: u32,
        hzb_enabled: bool,
        depth_reversed_z: bool,
        orthographic_projection: bool,
    ) {
        let meshlet_count = cluster_set.meshlet_count;
        if meshlet_count == 0 {
            return;
        }

        queue.write_buffer(
            &self.cull_uniform_buffer,
            0,
            bytemuck::bytes_of(&CullUniforms {
                view_proj,
                view_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
                meshlet_count,
                lod_stride: lod_stride.max(1),
                meshlet_phase,
                meshlet_stride_spatial: meshlet_stride_spatial as u32,
                hzb_dims: [hzb_dims.0.max(1), hzb_dims.1.max(1), 0, 0],
                hzb_mip_max,
                hzb_enabled: hzb_enabled as u32,
                depth_reversed_z: depth_reversed_z as u32,
                orthographic_projection: orthographic_projection as u32,
            }),
        );

        // Reset visible list atomic count and indirect args using encoder.clear_buffer
        // (not queue.write_buffer) so the clears are ordered on the encoder timeline
        // with subsequent compute dispatches, avoiding GPU-level race conditions.
        encoder.clear_buffer(&cluster_set.visible_buffer, 0, Some(4));
        encoder.clear_buffer(&cluster_set.indirect_buffer, 0, Some(24));

        // Create per-frame dynamic bind group (uniform + HZB views)
        let cull_dynamic_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cull Dynamic BG"),
            layout: &self.cull_dynamic_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cull_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(hzb_max_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(hzb_min_view),
                },
            ],
        });

        // Cull pass: group(0)=dynamic, group(1)=static (pre-built per ClusterSet)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cull Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_pipeline);
            pass.set_bind_group(0, &cull_dynamic_bg, &[]);
            pass.set_bind_group(1, &cluster_set.cull_static_bg, &[]);
            let wg = meshlet_count.div_ceil(64);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        // Compact pass: uses pre-built compact_bg
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compact Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compact_pipeline);
            pass.set_bind_group(0, &cluster_set.compact_bg, &[]);
            let wg = meshlet_count.div_ceil(64);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        // Finalize pass: uses pre-built finalize_bg
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Finalize Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finalize_pipeline);
            pass.set_bind_group(0, &cluster_set.finalize_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    pub fn draw_clustered(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        cluster_set: &ClusterSet,
        first_instance: u32,
    ) {
        pass.set_vertex_buffer(0, cluster_set.vertex_buffer.slice(..));
        pass.set_index_buffer(cluster_set.compact_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        // Note: indirect buffer has first_instance=0 from clear; instance data
        // at index 0 is written by caller before drawing meshlet draws.
        pass.draw_indexed_indirect(&cluster_set.indirect_buffer, 4);
    }

    /// Draw ALL meshlet indices without culling, using uncompacted index buffer.
    pub fn draw_clustered_full_diag(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        cluster_set: &ClusterSet,
        first_instance: u32,
    ) {
        pass.set_vertex_buffer(0, cluster_set.vertex_buffer.slice(..));
        pass.set_index_buffer(cluster_set.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..cluster_set.total_indices, 0, first_instance..first_instance + 1);
    }
}
