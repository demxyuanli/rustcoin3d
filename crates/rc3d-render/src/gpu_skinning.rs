/// GPU-accelerated skeletal mesh skinning via compute shader.
///
/// Uploads bind-pose vertex data + skin weights + bone matrices,
/// dispatches a compute pass, and outputs animated vertices into
/// a GPU buffer ready for rendering.

use glam::Mat4;
use rc3d_scene::animation::VertexSkinData;
use wgpu::util::DeviceExt;

/// Packed per-vertex data for GPU skinning: 4 bone indices (u32) + 4 weights (f32 via bitcast).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSkinVertex {
    bone_indices: [u32; 4],
    bone_weights: [u32; 4], // bitcast from f32
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkinningUniforms {
    joint_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU skinning resources for one skinned mesh.
pub struct GpuSkinningResources {
    /// Source vertex buffer: interleaved pos(3)+norm(3)+uv(2)+tan(4) = 12 f32 per vertex.
    pub src_vertex_buffer: wgpu::Buffer,
    /// Destination vertex buffer (written by compute shader).
    pub dst_vertex_buffer: wgpu::Buffer,
    /// Skin data buffer (bone indices + weights per vertex).
    pub skin_data_buffer: wgpu::Buffer,
    /// Bone matrices buffer (uniform, updated each frame).
    pub bone_mat_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    /// Pre-built bind group for the skinning compute dispatch.
    pub skin_bg: wgpu::BindGroup,
}

/// Compute pipeline that runs GPU skinning for all skinned meshes.
pub struct GpuSkinningPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
}

impl GpuSkinningPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Skinning"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/gpu_skinning.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skinning BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skinning PLL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Skinning Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skinning Uniforms"),
            size: std::mem::size_of::<SkinningUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            uniform_buffer,
        }
    }

    /// Create GPU skinning resources for a skinned mesh.
    pub fn create_skinned_mesh(
        &self,
        device: &wgpu::Device,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        texcoords: &[[f32; 2]],
        tangents: &[[f32; 4]],
        skin_data: &[VertexSkinData],
        max_bones: u32,
    ) -> GpuSkinningResources {
        let vertex_count = positions.len();
        assert_eq!(vertex_count, skin_data.len());

        // Interleaved vertices: pos(3), norm(3), uv(2), tangent(4) = 12 f32
        let mut interleaved = Vec::with_capacity(vertex_count * 12);
        for i in 0..vertex_count {
            interleaved.extend_from_slice(&positions[i]);
            interleaved.extend_from_slice(&normals[i]);
            interleaved.extend_from_slice(&texcoords.get(i).copied().unwrap_or([0.0, 0.0]));
            interleaved.extend_from_slice(&tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]));
        }

        let src_bytes: &[u8] = bytemuck::cast_slice(&interleaved);
        let src_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skin Src Verts"),
            contents: src_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let dst_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Dst Verts"),
            size: src_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pack skin data: 4 indices (u32) + 4 weights (f32→u32 bitcast)
        let packed_skin: Vec<u32> = skin_data
            .iter()
            .flat_map(|s| {
                let w0: u32 = bytemuck::cast(s.bone_weights[0]);
                let w1: u32 = bytemuck::cast(s.bone_weights[1]);
                let w2: u32 = bytemuck::cast(s.bone_weights[2]);
                let w3: u32 = bytemuck::cast(s.bone_weights[3]);
                vec![
                    s.bone_indices[0],
                    s.bone_indices[1],
                    s.bone_indices[2],
                    s.bone_indices[3],
                    w0, w1, w2, w3,
                ]
            })
            .collect();

        let skin_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skin Data"),
            contents: bytemuck::cast_slice(&packed_skin),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bone_mat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bone Mats"),
            size: (max_bones as u64) * 64, // 4x4 f32 = 64 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skinning BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: skin_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bone_mat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dst_vertex_buffer.as_entire_binding(),
                },
            ],
        });

        GpuSkinningResources {
            src_vertex_buffer,
            dst_vertex_buffer,
            skin_data_buffer,
            bone_mat_buffer,
            vertex_count: vertex_count as u32,
            skin_bg: bg,
        }
    }

    /// Run skinning for a mesh: upload bones, dispatch compute.
    pub fn skin_mesh(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuSkinningResources,
        skinning_matrices: &[Mat4],
    ) {
        let joint_count = skinning_matrices.len() as u32;
        if joint_count == 0 || resources.vertex_count == 0 {
            return;
        }

        // Upload bone matrices
        let mats_flat: Vec<[[f32; 4]; 4]> = skinning_matrices
            .iter()
            .map(|m| m.to_cols_array_2d())
            .collect();
        let bone_bytes: &[u8] = bytemuck::cast_slice(&mats_flat);
        queue.write_buffer(&resources.bone_mat_buffer, 0, bone_bytes);

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&SkinningUniforms {
                joint_count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU Skinning"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &resources.skin_bg, &[]);
        pass.dispatch_workgroups((resources.vertex_count + 63) / 64, 1, 1);
    }
}
