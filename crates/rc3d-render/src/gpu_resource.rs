use slotmap::new_key_type;
use wgpu::util::DeviceExt;

use crate::vertex::{FlatUniforms, LineVertex, OutlineUniforms, SceneUniforms, Vertex};

new_key_type! {
    pub struct MeshId;
}

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub index_buffer: Option<wgpu::Buffer>,
    pub index_count: u32,
    pub edge_vertex_buffer: Option<wgpu::Buffer>,
    pub edge_vertex_count: u32,
    pub generation: u32,
}

pub struct GpuResourceManager {
    meshes: slotmap::SlotMap<MeshId, GpuMesh>,
}

impl GpuResourceManager {
    pub fn new() -> Self {
        Self {
            meshes: slotmap::SlotMap::with_key(),
        }
    }

    pub fn upload_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: Option<&[u32]>,
        edge_positions: &[[f32; 3]],
    ) -> MeshId {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let (index_buffer, index_count) = if let Some(idx) = indices {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Indices"),
                contents: bytemuck::cast_slice(idx),
                usage: wgpu::BufferUsages::INDEX,
            });
            (Some(buf), idx.len() as u32)
        } else {
            (None, 0)
        };

        let (edge_vertex_buffer, edge_vertex_count) = if !edge_positions.is_empty() {
            let line_verts: Vec<LineVertex> = edge_positions
                .iter()
                .map(|&p| LineVertex { position: p })
                .collect();
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Edge Vertices"),
                contents: bytemuck::cast_slice(&line_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            (Some(buf), line_verts.len() as u32)
        } else {
            (None, 0)
        };

        self.meshes.insert(GpuMesh {
            vertex_buffer,
            vertex_count: vertices.len() as u32,
            index_buffer,
            index_count,
            edge_vertex_buffer,
            edge_vertex_count,
            generation: 0,
        })
    }

    pub fn get(&self, id: MeshId) -> Option<&GpuMesh> {
        self.meshes.get(id)
    }

    pub fn remove(&mut self, id: MeshId) {
        self.meshes.remove(id);
    }
}

pub struct GpuUniformPool {
    buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    stride: u64,
    capacity: usize,
    cursor: usize,
}

impl GpuUniformPool {
    pub fn new_phong(device: &wgpu::Device, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<SceneUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = ((raw_stride + alignment - 1) / alignment) * alignment;
        let size = stride * capacity as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Phong Uniform Pool"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Phong BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self { buffer, bind_group_layout, stride, capacity, cursor: 0 }
    }

    pub fn new_flat(device: &wgpu::Device, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<FlatUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = ((raw_stride + alignment - 1) / alignment) * alignment;
        let size = stride * capacity as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flat Uniform Pool"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flat BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self { buffer, bind_group_layout, stride, capacity, cursor: 0 }
    }

    pub fn new_outline(device: &wgpu::Device, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<OutlineUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = ((raw_stride + alignment - 1) / alignment) * alignment;
        let size = stride * capacity as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Outline Uniform Pool"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Outline BGL Pool"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self { buffer, bind_group_layout, stride, capacity, cursor: 0 }
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn push_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, uniforms: &SceneUniforms) -> Option<(wgpu::BindGroup, u32)> {
        if self.cursor >= self.capacity {
            return None;
        }
        let offset = (self.cursor * self.stride as usize) as u64;
        queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(&[*uniforms]));
        let offset_u32 = offset as u32;
        self.cursor += 1;
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.buffer,
                    offset: 0,
                    size: Some(std::num::NonZero::new(self.stride).unwrap()),
                }),
            }],
            label: None,
        });
        Some((bg, offset_u32))
    }

    pub fn push_flat(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, uniforms: &FlatUniforms) -> Option<(wgpu::BindGroup, u32)> {
        if self.cursor >= self.capacity {
            return None;
        }
        let offset = (self.cursor * self.stride as usize) as u64;
        queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(&[*uniforms]));
        let offset_u32 = offset as u32;
        self.cursor += 1;
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.buffer,
                    offset: 0,
                    size: Some(std::num::NonZero::new(self.stride).unwrap()),
                }),
            }],
            label: None,
        });
        Some((bg, offset_u32))
    }

    pub fn push_outline(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, uniforms: &OutlineUniforms) -> Option<(wgpu::BindGroup, u32)> {
        if self.cursor >= self.capacity {
            return None;
        }
        let offset = (self.cursor * self.stride as usize) as u64;
        queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(&[*uniforms]));
        let offset_u32 = offset as u32;
        self.cursor += 1;
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.buffer,
                    offset: 0,
                    size: Some(std::num::NonZero::new(self.stride).unwrap()),
                }),
            }],
            label: None,
        });
        Some((bg, offset_u32))
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
