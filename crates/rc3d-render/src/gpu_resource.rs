use slotmap::new_key_type;
use wgpu::util::DeviceExt;

use crate::vertex::{FlatUniforms, LineVertex, OutlineUniforms, SceneUniforms, ShadowDrawUniforms, Vertex};

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

impl Default for GpuResourceManager {
    fn default() -> Self {
        Self::new()
    }
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
    bind_group: wgpu::BindGroup,
    stride: u64,
    capacity: usize,
    cursor: usize,
    staging: Vec<u8>,
    written_end: usize,
}

impl GpuUniformPool {
    pub fn new_phong(device: &wgpu::Device, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<SceneUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = raw_stride.div_ceil(alignment) * alignment;
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: Some(std::num::NonZero::new(stride).expect("uniform stride must be non-zero")),
                }),
            }],
            label: Some("Phong Uniform Pool BG"),
        });
        Self {
            buffer,
            bind_group_layout,
            bind_group,
            stride,
            capacity,
            cursor: 0,
            staging: vec![0; size as usize],
            written_end: 0,
        }
    }

    pub fn new_flat(device: &wgpu::Device, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<FlatUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = raw_stride.div_ceil(alignment) * alignment;
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: Some(std::num::NonZero::new(stride).expect("uniform stride must be non-zero")),
                }),
            }],
            label: Some("Flat Uniform Pool BG"),
        });
        Self {
            buffer,
            bind_group_layout,
            bind_group,
            stride,
            capacity,
            cursor: 0,
            staging: vec![0; size as usize],
            written_end: 0,
        }
    }

    /// Pool for per-draw shadow MVP; `layout` must match [PipelineSet::shadow_draw_bgl].
    pub fn new_shadow_pool(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<ShadowDrawUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = raw_stride.div_ceil(alignment) * alignment;
        let size = stride * capacity as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Uniform Pool"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = layout.clone();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: Some(std::num::NonZero::new(stride).expect("uniform stride must be non-zero")),
                }),
            }],
            label: Some("Shadow Uniform Pool BG"),
        });
        Self {
            buffer,
            bind_group_layout,
            bind_group,
            stride,
            capacity,
            cursor: 0,
            staging: vec![0; size as usize],
            written_end: 0,
        }
    }

    pub fn new_outline(device: &wgpu::Device, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<OutlineUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = raw_stride.div_ceil(alignment) * alignment;
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: Some(std::num::NonZero::new(stride).expect("uniform stride must be non-zero")),
                }),
            }],
            label: Some("Outline Uniform Pool BG"),
        });
        Self {
            buffer,
            bind_group_layout,
            bind_group,
            stride,
            capacity,
            cursor: 0,
            staging: vec![0; size as usize],
            written_end: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
        self.written_end = 0;
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> Option<u32> {
        if self.cursor >= self.capacity {
            return None;
        }
        let stride = self.stride as usize;
        let offset = self.cursor * stride;
        let end = offset + bytes.len();
        self.staging[offset..end].copy_from_slice(bytes);
        self.written_end = self.written_end.max(offset + stride);
        let offset_u32 = offset as u32;
        self.cursor += 1;
        Some(offset_u32)
    }

    pub fn push_scene(&mut self, uniforms: &SceneUniforms) -> Option<u32> {
        self.push_bytes(bytemuck::bytes_of(uniforms))
    }

    pub fn push_flat(&mut self, uniforms: &FlatUniforms) -> Option<u32> {
        self.push_bytes(bytemuck::bytes_of(uniforms))
    }

    pub fn push_outline(&mut self, uniforms: &OutlineUniforms) -> Option<u32> {
        self.push_bytes(bytemuck::bytes_of(uniforms))
    }

    pub fn push_shadow(&mut self, uniforms: &ShadowDrawUniforms) -> Option<u32> {
        self.push_bytes(bytemuck::bytes_of(uniforms))
    }

    pub fn flush(&self, queue: &wgpu::Queue) {
        if self.written_end > 0 {
            queue.write_buffer(&self.buffer, 0, &self.staging[..self.written_end]);
        }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
