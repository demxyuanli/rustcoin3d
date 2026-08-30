//! Runtime WGSL injection for `MaterialNode.custom_wgsl` (Three.js ShaderMaterial analogue).

use std::collections::HashMap;
use std::hash::Hasher;

use wgpu::util::DeviceExt;

use crate::render_action::DrawCall;
use crate::vertex::Vertex;

const PRELUDE: &str = include_str!("shaders/custom_material_prelude.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CustomDrawUniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    custom: [f32; 4],
    base_color: [f32; 4],
    extra: [f32; 4],
}

#[derive(Clone)]
struct CachedPipeline {
    pipeline: wgpu::RenderPipeline,
}

/// Compiles and caches user WGSL pipelines, then draws matching `DrawCall`s.
pub struct CustomShaderPass {
    bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    pipelines: HashMap<u64, CachedPipeline>,
}

impl CustomShaderPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CustomShader BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CustomShader PLL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        Self {
            bgl,
            layout,
            format,
            pipelines: HashMap::new(),
        }
    }

    pub fn compose_source(user: &str) -> String {
        let trimmed = user.trim();
        if trimmed.contains("@fragment") {
            trimmed.to_string()
        } else {
            format!(
                "{PRELUDE}\n{trimmed}\n@fragment\nfn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{\n    return material_fs(in, u);\n}}\n"
            )
        }
    }

    fn source_hash(src: &str) -> u64 {
        let mut h = twox_hash::XxHash64::with_seed(0);
        h.write(src.as_bytes());
        h.finish()
    }

    fn pipeline_for(&mut self, device: &wgpu::Device, user: &str) -> Option<&wgpu::RenderPipeline> {
        let src = Self::compose_source(user);
        let key = Self::source_hash(&src);
        if !self.pipelines.contains_key(&key) {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("CustomShader"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("CustomShader"),
                layout: Some(&self.layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(Vertex::desc())],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32FloatStencil8,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });
            self.pipelines.insert(key, CachedPipeline { pipeline });
        }
        self.pipelines.get(&key).map(|c| &c.pipeline)
    }

    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        draws: &[&DrawCall],
        time: f32,
        scene_region: crate::viewport::ViewportRect,
    ) {
        let mut items: Vec<&DrawCall> = draws
            .iter()
            .copied()
            .filter(|dc| {
                dc.custom_wgsl
                    .as_ref()
                    .map(|s| !s.is_empty())
                    .unwrap_or(false)
                    && !dc.vertices.is_empty()
            })
            .collect();
        if items.is_empty() {
            return;
        }
        items.sort_by_key(|dc| {
            dc.custom_wgsl
                .as_ref()
                .map(|s| Self::source_hash(s))
                .unwrap_or(0)
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("CustomShader Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None, multiview_mask: None,
        });
        scene_region.apply_to_pass(&mut pass);

        for dc in items {
            let Some(src) = dc.custom_wgsl.as_deref() else {
                continue;
            };
            let Some(pipeline) = self.pipeline_for(device, src).cloned() else {
                continue;
            };
            pass.set_pipeline(&pipeline);

            let uniforms = CustomDrawUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                model: dc.model_matrix.to_cols_array_2d(),
                camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                custom: dc.custom_uniforms,
                base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, dc.opacity],
                extra: [dc.opacity, time, 0.0, 0.0],
            };
            let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CustomShader uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("CustomShader BG"),
                layout: &self.bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ub.as_entire_binding(),
                }],
            });
            pass.set_bind_group(0, &bg, &[]);

            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CustomShader VB"),
                contents: bytemuck::cast_slice(dc.vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });
            pass.set_vertex_buffer(0, vb.slice(..));
            if let Some(ref indices) = dc.indices {
                if !indices.is_empty() {
                    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("CustomShader IB"),
                        contents: bytemuck::cast_slice(indices.as_slice()),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                    continue;
                }
            }
            pass.draw(0..dc.vertices.len() as u32, 0..1);
        }
    }
}

/// True when this draw should skip the PBR solid/transparent paths.
#[inline]
pub fn is_custom_shader_draw(dc: &DrawCall) -> bool {
    dc.custom_wgsl
        .as_ref()
        .map(|s| !s.is_empty())
        .unwrap_or(false)
}
