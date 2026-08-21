use slotmap::new_key_type;
use wgpu::util::DeviceExt;

use crate::vertex::{
    FlatUniforms, LineUniforms, LineVertex, LineVertexExpanded, SceneUniforms,
    SectionCapUniforms, ShadowDrawUniforms, Vertex,
};

new_key_type! {
    pub struct MeshId;
}

/// Which line-list buffer to use on [`GpuMesh`] / fall back data on [`crate::render_action::DrawCall`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeLineKind {
    /// Crease + boundary edges (shaded overlay).
    Feature,
    /// Full triangle mesh topology (wireframe mode).
    WireframeFull,
}

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub index_buffer: Option<wgpu::Buffer>,
    pub index_count: u32,
    pub edge_vertex_buffer: Option<wgpu::Buffer>,
    pub edge_vertex_count: u32,
    pub wireframe_edge_vertex_buffer: Option<wgpu::Buffer>,
    pub wireframe_edge_vertex_count: u32,
    /// Expanded edge vertices for anti-aliased line rendering (6 vertices per segment).
    pub edge_expanded_buffer: Option<wgpu::Buffer>,
    pub edge_expanded_count: u32,
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

    fn edge_buffer_from_positions(
        device: &wgpu::Device,
        label: &'static str,
        edge_positions: &[[f32; 3]],
    ) -> (Option<wgpu::Buffer>, u32) {
        if edge_positions.is_empty() {
            return (None, 0);
        }
        let line_verts: Vec<LineVertex> = edge_positions
            .iter()
            .map(|&p| LineVertex { position: p })
            .collect();
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&line_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        (Some(buf), line_verts.len() as u32)
    }

    /// Build expanded edge buffer for anti-aliased line rendering.
    /// Each segment (A, B) produces 6 `LineVertexExpanded` vertices (2 triangles),
    /// encoding the partner endpoint and which side (-1/+1) of the quad each vertex belongs to.
    fn expanded_edge_buffer_from_positions(
        device: &wgpu::Device,
        edge_positions: &[[f32; 3]],
    ) -> (Option<wgpu::Buffer>, u32) {
        if edge_positions.len() < 2 {
            return (None, 0);
        }
        let segments = edge_positions.len() / 2;
        let mut verts: Vec<LineVertexExpanded> = Vec::with_capacity(segments * 6);
        for seg in edge_positions.chunks_exact(2) {
            let a = seg[0];
            let b = seg[1];
            // Triangle 1: (A,side=-1), (B,side=-1), (A,side=+1)
            verts.push(LineVertexExpanded { position: a, partner: b, side: -1.0 });
            verts.push(LineVertexExpanded { position: b, partner: a, side: -1.0 });
            verts.push(LineVertexExpanded { position: a, partner: b, side:  1.0 });
            // Triangle 2: (B,side=-1), (B,side=+1), (A,side=+1)
            verts.push(LineVertexExpanded { position: b, partner: a, side: -1.0 });
            verts.push(LineVertexExpanded { position: b, partner: a, side:  1.0 });
            verts.push(LineVertexExpanded { position: a, partner: b, side:  1.0 });
        }
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Edge Expanded Vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        (Some(buf), verts.len() as u32)
    }

    pub fn upload_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: Option<&[u32]>,
        feature_edge_positions: &[[f32; 3]],
        wireframe_edge_positions: &[[f32; 3]],
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

        let (edge_vertex_buffer, edge_vertex_count) =
            Self::edge_buffer_from_positions(device, "Edge Vertices (feature)", feature_edge_positions);
        let (wireframe_edge_vertex_buffer, wireframe_edge_vertex_count) = Self::edge_buffer_from_positions(
            device,
            "Edge Vertices (wireframe full)",
            wireframe_edge_positions,
        );
        let (edge_expanded_buffer, edge_expanded_count) =
            Self::expanded_edge_buffer_from_positions(device, feature_edge_positions);

        self.meshes.insert(GpuMesh {
            vertex_buffer,
            vertex_count: vertices.len() as u32,
            index_buffer,
            index_count,
            edge_vertex_buffer,
            edge_vertex_count,
            wireframe_edge_vertex_buffer,
            wireframe_edge_vertex_count,
            edge_expanded_buffer,
            edge_expanded_count,
            generation: 0,
        })
    }

    /// Mesh whose vertex buffer is the skinning compute output (same layout as [`Vertex`]).
    pub fn insert_skinned_mesh(
        &mut self,
        device: &wgpu::Device,
        vertex_buffer: wgpu::Buffer,
        vertex_count: u32,
        indices: Option<&[u32]>,
        feature_edge_positions: &[[f32; 3]],
        wireframe_edge_positions: &[[f32; 3]],
    ) -> MeshId {
        let (index_buffer, index_count) = if let Some(idx) = indices {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Skinned mesh indices"),
                contents: bytemuck::cast_slice(idx),
                usage: wgpu::BufferUsages::INDEX,
            });
            (Some(buf), idx.len() as u32)
        } else {
            (None, 0)
        };

        let (edge_vertex_buffer, edge_vertex_count) =
            Self::edge_buffer_from_positions(device, "Skinned mesh edges (feature)", feature_edge_positions);
        let (wireframe_edge_vertex_buffer, wireframe_edge_vertex_count) = Self::edge_buffer_from_positions(
            device,
            "Skinned mesh edges (wireframe full)",
            wireframe_edge_positions,
        );

        self.meshes.insert(GpuMesh {
            vertex_buffer,
            vertex_count,
            index_buffer,
            index_count,
            edge_vertex_buffer,
            edge_vertex_count,
            wireframe_edge_vertex_buffer,
            wireframe_edge_vertex_count,
            edge_expanded_buffer: None,
            edge_expanded_count: 0,
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
    pub(crate) buffer: wgpu::Buffer,
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

    /// Same bind group layout as [`PipelineSet::flat_bgl`] / flat passes; buffer holds [`SectionCapUniforms`] strides.
    pub fn new_section_cap(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<SectionCapUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = raw_stride.div_ceil(alignment) * alignment;
        let size = stride * capacity as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Section Cap Uniform Pool"),
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
            label: Some("Section Cap Uniform Pool BG"),
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

    pub fn stride(&self) -> u64 {
        self.stride
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Pool for anti-aliased line uniforms; reuses the same bind group layout as flat.
    pub fn new_line(device: &wgpu::Device, flat_layout: &wgpu::BindGroupLayout, capacity: usize) -> Self {
        let raw_stride = std::mem::size_of::<LineUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = raw_stride.div_ceil(alignment) * alignment;
        let size = stride * capacity as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Line Uniform Pool"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = flat_layout.clone();
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
            label: Some("Line Uniform Pool BG"),
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
    /// The shader expects `array<ShadowDrawUniforms, 4>` so the binding must span 4 strides.
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
                    size: Some(
                        std::num::NonZero::new(stride * 4)
                            .expect("shadow uniform stride * 4 must be non-zero"),
                    ),
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

    pub fn reset(&mut self) {
        self.cursor = 0;
        self.written_end = 0;
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> Option<u32> {
        if self.cursor >= self.capacity {
            log::warn!(
                "GpuUniformPool overflow: cursor={}, capacity={}, consider increasing pool size",
                self.cursor, self.capacity
            );
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

    pub fn push_line(&mut self, uniforms: &LineUniforms) -> Option<u32> {
        self.push_bytes(bytemuck::bytes_of(uniforms))
    }

    pub fn push_section_cap(&mut self, uniforms: &SectionCapUniforms) -> Option<u32> {
        self.push_bytes(bytemuck::bytes_of(uniforms))
    }

    pub fn push_shadow(&mut self, uniforms: &ShadowDrawUniforms) -> Option<u32> {
        self.push_bytes(bytemuck::bytes_of(uniforms))
    }

    /// Push an array of shadow uniforms for layered CSM rendering.
    /// Writes 4 `ShadowDrawUniforms` at consecutive aligned offsets and returns the base offset.
    pub fn push_shadow_array(&mut self, uniforms: &[ShadowDrawUniforms; 4]) -> Option<u32> {
        if self.cursor + 4 > self.capacity {
            return None;
        }
        let stride = self.stride as usize;
        let base_offset = self.cursor * stride;
        for (i, u) in uniforms.iter().enumerate() {
            let offset = base_offset + i * stride;
            self.staging[offset..offset + std::mem::size_of::<ShadowDrawUniforms>()]
                .copy_from_slice(bytemuck::bytes_of(u));
        }
        let end = base_offset + 4 * stride;
        self.written_end = self.written_end.max(end);
        let offset_u32 = base_offset as u32;
        self.cursor += 4;
        Some(offset_u32)
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
