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
}

impl ClusterSet {
    pub fn from_meshlet_data(device: &wgpu::Device, data: &MeshletData) -> Self {
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
            size: 20,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
    cull_bgl: wgpu::BindGroupLayout,
    compact_bgl: wgpu::BindGroupLayout,
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

        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cull BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let cull_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cull PLL"),
            bind_group_layouts: &[&cull_bgl],
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
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
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

        let cull_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Uniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            cull_pipeline,
            compact_pipeline,
            cull_bgl,
            compact_bgl,
            cull_uniform_buffer,
        }
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

        queue.write_buffer(&self.cull_uniform_buffer, 0, bytemuck::bytes_of(&CullUniforms {
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
        }));

        queue.write_buffer(&cluster_set.visible_buffer, 0, bytemuck::bytes_of(&0u32));

        let clear_indirect: [u32; 5] = [0, 1, 0, 0, 0];
        queue.write_buffer(&cluster_set.indirect_buffer, 0, bytemuck::cast_slice(&clear_indirect));

        {
            let cull_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Cull BG"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.cull_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cluster_set.meshlet_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cluster_set.bounds_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cluster_set.visible_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(hzb_max_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(hzb_min_view),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cull Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_pipeline);
            pass.set_bind_group(0, &cull_bg, &[]);
            let wg = meshlet_count.div_ceil(64);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        {
            let compact_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compact BG"),
                layout: &self.compact_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cluster_set.visible_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cluster_set.meshlet_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cluster_set.index_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cluster_set.compact_index_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: cluster_set.indirect_buffer.as_entire_binding() },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compact Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compact_pipeline);
            pass.set_bind_group(0, &compact_bg, &[]);
            let wg = meshlet_count.div_ceil(64);
            pass.dispatch_workgroups(wg, 1, 1);
        }
    }

    pub fn draw_clustered(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        cluster_set: &ClusterSet,
    ) {
        pass.set_vertex_buffer(0, cluster_set.vertex_buffer.slice(..));
        pass.set_index_buffer(cluster_set.compact_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(&cluster_set.indirect_buffer, 0);
    }
}
