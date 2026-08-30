//! GPU compute-culling pass: frustum-cull per-object transforms,
//! writing visible instance indices into an indirect draw buffer.

use crate::vertex::GpuObjectTransform;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrustumUniforms {
    planes: [[f32; 4]; 6],
    // object_count rides in the uniform: the device is created without the
    // PUSH_CONSTANTS feature, so push constants would fail validation.
    object_count: u32,
    _pad: [u32; 3],
}

/// Byte size of the frustum uniform buffer (planes + object_count + padding).
pub const FRUSTUM_UNIFORM_SIZE: u64 = std::mem::size_of::<FrustumUniforms>() as u64;

/// GPU compute pipeline for object-level frustum culling.
pub struct GpuCullPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuCullPass {
    /// Create the compute pipeline and bind group layout.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("object_cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/object_cull.wgsl"
            ))),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Object Cull BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
            ],
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Object Cull Layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Object Cull Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self { pipeline, bgl }
    }

    /// Create a bind group for the current frame's buffers.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        transform_buffer: &wgpu::Buffer,
        indirect_args_buffer: &wgpu::Buffer,
        frustum_uniform: &wgpu::Buffer,
        instance_indices_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Object Cull BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frustum_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_indices_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Upload frustum plane data and the object count to the uniform buffer.
    pub fn write_frustum(
        &self,
        queue: &wgpu::Queue,
        frustum_uniform: &wgpu::Buffer,
        planes: &[[f32; 4]; 6],
        object_count: u32,
    ) {
        let data = FrustumUniforms {
            planes: *planes,
            object_count,
            _pad: [0; 3],
        };
        queue.write_buffer(frustum_uniform, 0, bytemuck::bytes_of(&data));
    }

    /// Upload dirty object transforms to the transform buffer.
    pub fn write_transforms(
        &self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        transforms: &[GpuObjectTransform],
    ) {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(transforms));
    }

    /// Reset indirect args counters to zero (called before dispatch).
    /// Only the first entry is used today (all objects upload mesh_id = 0),
    /// so avoid re-uploading the full 64 KB table every frame.
    pub fn reset_indirect_args(
        &self,
        queue: &wgpu::Queue,
        indirect_buffer: &wgpu::Buffer,
        _max_entries: u32,
    ) {
        let entry_size = std::mem::size_of::<wgpu::util::DrawIndirectArgs>();
        let zeroed = vec![0u8; entry_size];
        queue.write_buffer(indirect_buffer, 0, &zeroed);
    }

    /// Dispatch the compute cull pass. `object_count` must match the count
    /// previously written via `write_frustum`.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        object_count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Object Cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let workgroup_count = object_count.div_ceil(256);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}
