//! Screen-space selection outline (three.js `OutlinePass`-style):
//! non-selected depth prepass, selected mask with depth compare, Sobel edge, composite.

use crate::render_passes::PassContext;
use crate::vertex::{FlatUniforms, Vertex};
use bytemuck::{Pod, Zeroable};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const PREPASS_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const EDGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskParams {
    reversed_z: f32,
    depth_bias: f32,
    far_ndc_z: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EdgeParams {
    texel_size: [f32; 2],
    _pad0: [f32; 2],
    visible_color: [f32; 3],
    _pad1: f32,
    hidden_color: [f32; 3],
    edge_strength: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeStrength {
    params: [f32; 4],
    texel_size: [f32; 2],
    _pad: [f32; 2],
}

pub struct SelectionOutlinePipelines {
    pub mask_aux_bgl: wgpu::BindGroupLayout,
    pub edge_bgl: wgpu::BindGroupLayout,
    pub composite_bgl: wgpu::BindGroupLayout,
    composite_pll: wgpu::PipelineLayout,
    pub depth_prepass_fwd: wgpu::RenderPipeline,
    pub depth_prepass_rev: wgpu::RenderPipeline,
    pub mask_fwd: wgpu::RenderPipeline,
    pub mask_rev: wgpu::RenderPipeline,
    pub edge_pipeline: wgpu::RenderPipeline,
    pub linear_sampler: wgpu::Sampler,
    composite_shader: wgpu::ShaderModule,
}

impl SelectionOutlinePipelines {
    pub fn new(device: &wgpu::Device, flat_bgl: &wgpu::BindGroupLayout) -> Self {
        let depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection_outline_depth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection_outline_depth.wgsl").into()),
        });
        let mask_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection_outline_mask"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection_outline_mask.wgsl").into()),
        });
        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection_outline_edge"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection_outline_edge.wgsl").into()),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection_outline_composite"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection_outline_composite.wgsl").into()),
        });

        let mask_aux_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selection_outline_mask_aux"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<MaskParams>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        let edge_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selection_outline_edge_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<EdgeParams>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selection_composite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<CompositeStrength>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        let depth_prepass_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection_depth_prepass_pll"),
            bind_group_layouts: &[flat_bgl],
            push_constant_ranges: &[],
        });

        let mask_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection_mask_pll"),
            bind_group_layouts: &[flat_bgl, &mask_aux_bgl],
            push_constant_ranges: &[],
        });

        let edge_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection_edge_pll"),
            bind_group_layouts: &[&edge_bgl],
            push_constant_ranges: &[],
        });

        let composite_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection_composite_pll"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });

        let prepass_targets = &[Some(wgpu::ColorTargetState {
            format: PREPASS_COLOR_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let ms = wgpu::MultisampleState::default();

        let depth_prepass_fwd = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection_depth_prepass_fwd"),
            layout: Some(&depth_prepass_pll),
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: Some("fs_main"),
                targets: prepass_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: ms,
            multiview: None,
            cache: None,
        });

        let depth_prepass_rev = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection_depth_prepass_rev"),
            layout: Some(&depth_prepass_pll),
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: Some("fs_main"),
                targets: prepass_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: ms,
            multiview: None,
            cache: None,
        });

        let mask_targets = &[Some(wgpu::ColorTargetState {
            format: MASK_FORMAT,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let mask_fwd = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection_mask_fwd"),
            layout: Some(&mask_pll),
            vertex: wgpu::VertexState {
                module: &mask_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mask_shader,
                entry_point: Some("fs_main"),
                targets: mask_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: ms,
            multiview: None,
            cache: None,
        });

        let mask_rev = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection_mask_rev"),
            layout: Some(&mask_pll),
            vertex: wgpu::VertexState {
                module: &mask_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mask_shader,
                entry_point: Some("fs_main"),
                targets: mask_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: ms,
            multiview: None,
            cache: None,
        });

        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection_edge"),
            layout: Some(&edge_pll),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_edge"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: EDGE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: ms,
            multiview: None,
            cache: None,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("selection_outline_linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            mask_aux_bgl,
            edge_bgl,
            composite_bgl,
            composite_pll,
            depth_prepass_fwd,
            depth_prepass_rev,
            mask_fwd,
            mask_rev,
            edge_pipeline,
            linear_sampler,
            composite_shader,
        }
    }

    pub fn create_composite_pipeline(
        &self,
        device: &wgpu::Device,
        out_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection_composite"),
            layout: Some(&self.composite_pll),
            vertex: wgpu::VertexState {
                module: &self.composite_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.composite_shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: out_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }
}

#[allow(dead_code)]
pub struct SelectionOutlineTargets {
    pub width: u32,
    pub height: u32,
    pub shade_format: wgpu::TextureFormat,
    pub reversed_z: bool,
    pub prepass_color_view: wgpu::TextureView,
    prepass_color: wgpu::Texture,
    prepass_depth: wgpu::Texture,
    pub prepass_depth_view: wgpu::TextureView,
    pub mask_view: wgpu::TextureView,
    mask_tex: wgpu::Texture,
    mask_depth: wgpu::Texture,
    pub mask_depth_view: wgpu::TextureView,
    pub edge_view: wgpu::TextureView,
    edge_tex: wgpu::Texture,
    pub scene_scratch_view: wgpu::TextureView,
    scene_scratch: wgpu::Texture,
    pub mask_params_buf: wgpu::Buffer,
    pub edge_params_buf: wgpu::Buffer,
    pub composite_strength_buf: wgpu::Buffer,
    pub mask_aux_bg: wgpu::BindGroup,
    pub edge_bg: wgpu::BindGroup,
    pub composite_bg: wgpu::BindGroup,
    pub composite_pl: wgpu::RenderPipeline,
}

impl SelectionOutlineTargets {
    pub fn ensure(
        device: &wgpu::Device,
        pl: &SelectionOutlinePipelines,
        width: u32,
        height: u32,
        shade_format: wgpu::TextureFormat,
        reversed_z: bool,
        existing: Option<Self>,
    ) -> Self {
        if let Some(e) = existing {
            if e.width == width
                && e.height == height
                && e.shade_format == shade_format
                && e.reversed_z == reversed_z
            {
                return e;
            }
        }

        let w = width.max(1);
        let h = height.max(1);
        let sz = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };

        let prepass_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sel_outline_prepass_color"),
            size: sz,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PREPASS_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let prepass_color_view = prepass_color.create_view(&wgpu::TextureViewDescriptor::default());

        let prepass_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sel_outline_prepass_depth"),
            size: sz,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let prepass_depth_view = prepass_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let mask_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sel_outline_mask"),
            size: sz,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MASK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mask_view = mask_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mask_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sel_outline_mask_depth"),
            size: sz,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mask_depth_view = mask_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let edge_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sel_outline_edge"),
            size: sz,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: EDGE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let edge_view = edge_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let scene_scratch = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sel_outline_scene_scratch"),
            size: sz,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: shade_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let scene_scratch_view = scene_scratch.create_view(&wgpu::TextureViewDescriptor::default());

        let mask_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sel_outline_mask_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let edge_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sel_outline_edge_params"),
            size: std::mem::size_of::<EdgeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_strength_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sel_outline_composite_str"),
            size: std::mem::size_of::<CompositeStrength>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mask_aux_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sel_outline_mask_aux_bg"),
            layout: &pl.mask_aux_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&prepass_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mask_params_buf.as_entire_binding(),
                },
            ],
        });

        let edge_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sel_outline_edge_bg"),
            layout: &pl.edge_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&pl.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: edge_params_buf.as_entire_binding(),
                },
            ],
        });

        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sel_outline_composite_bg"),
            layout: &pl.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_scratch_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&edge_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&pl.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: composite_strength_buf.as_entire_binding(),
                },
            ],
        });

        let composite_pl = pl.create_composite_pipeline(device, shade_format);

        Self {
            width: w,
            height: h,
            shade_format,
            reversed_z,
            prepass_color_view,
            prepass_color,
            prepass_depth,
            prepass_depth_view,
            mask_view,
            mask_tex,
            mask_depth,
            mask_depth_view,
            edge_view,
            edge_tex,
            scene_scratch_view,
            scene_scratch,
            mask_params_buf,
            edge_params_buf,
            composite_strength_buf,
            mask_aux_bg,
            edge_bg,
            composite_bg,
            composite_pl,
        }
    }
}

fn ndc_far_clear(depth_reversed_z: bool) -> f32 {
    if depth_reversed_z {
        0.0
    } else {
        1.0
    }
}

fn wgpu_depth_clear(depth_reversed_z: bool) -> f32 {
    if depth_reversed_z {
        0.0
    } else {
        1.0
    }
}

/// Encodes the selection outline pass. `scene_source_tex` must point at the shaded scene texture
/// (HDR, LDR offscreen, swapchain, or viewport RT) and remain valid for the encode duration.
/// `target_width_px` / `target_height_px` must match that texture and `shade_view` size (typically
/// swapchain dims or an off-screen viewport extent — not blindly `renderer.config` when embedding).
pub(crate) fn encode_selection_outline_pass(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &PassContext<'_>,
    shade_view: &wgpu::TextureView,
    shade_format: wgpu::TextureFormat,
    scene_source_tex: *const wgpu::Texture,
    target_width_px: u32,
    target_height_px: u32,
) {
    let Some(ref pl) = renderer.gpu.selection_outline_pipelines else {
        return;
    };
    let w = target_width_px.max(1);
    let h = target_height_px.max(1);

    let tg = SelectionOutlineTargets::ensure(
        &renderer.device,
        pl,
        w,
        h,
        shade_format,
        ctx.depth_reversed_z,
        renderer.gpu.selection_outline_targets.take(),
    );

    // SAFETY: Caller provides a pointer to a texture that outlives this encode; copy uses it read-only.
    let scene_ptr = scene_source_tex;

    let far_ndc = ndc_far_clear(ctx.depth_reversed_z);
    let depth_clear = wgpu_depth_clear(ctx.depth_reversed_z);
    let depth_prepass_pl = if ctx.depth_reversed_z {
        &pl.depth_prepass_rev
    } else {
        &pl.depth_prepass_fwd
    };
    let mask_pl = if ctx.depth_reversed_z {
        &pl.mask_rev
    } else {
        &pl.mask_fwd
    };

    // Uniforms for mask compare (written each frame — buffer recreated with targets; still valid)
    let mask_params = MaskParams {
        reversed_z: if ctx.depth_reversed_z { 1.0 } else { 0.0 },
        depth_bias: 1e-5,
        far_ndc_z: far_ndc,
        _pad: 0.0,
    };
    renderer.queue.write_buffer(&tg.mask_params_buf, 0, bytemuck::bytes_of(&mask_params));

    let oc = ctx.outline_color;
    let visible_rgb = [oc[0], oc[1], oc[2]];
    let hidden_rgb = [oc[0] * 0.35, oc[1] * 0.35, oc[2] * 0.35];
    let edge_params = EdgeParams {
        texel_size: [1.0 / tg.width as f32, 1.0 / tg.height as f32],
        _pad0: [0.0; 2],
        visible_color: visible_rgb,
        _pad1: 0.0,
        hidden_color: hidden_rgb,
        edge_strength: 3.0,
    };
    renderer
        .queue
        .write_buffer(&tg.edge_params_buf, 0, bytemuck::bytes_of(&edge_params));

    let composite_u = CompositeStrength {
        params: [1.0, 0.0, 0.0, 0.0],
        texel_size: [1.0 / tg.width as f32, 1.0 / tg.height as f32],
        _pad: [0.0; 2],
    };
    renderer
        .queue
        .write_buffer(&tg.composite_strength_buf, 0, bytemuck::bytes_of(&composite_u));

    // Copy scene -> scratch
    unsafe {
        let scene_tex = &*scene_ptr;
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: scene_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &tg.scene_scratch,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: tg.width,
                height: tg.height,
                depth_or_array_layers: 1,
            },
        );
    }

    // 1) Non-selected depth prepass
    {
        let clear_c = wgpu::Color { r: far_ndc as f64, g: 0.0, b: 0.0, a: 1.0 };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("selection outline depth prepass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &tg.prepass_color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_c),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &tg.prepass_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(depth_clear),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(depth_prepass_pl);
        let mut last_mesh = None;
        for &i in ctx.solid_order {
            if ctx.visible[i].selected {
                continue;
            }
            let dc = ctx.visible[i];
            if dc.vertices.is_empty() && dc.meshlet_data.is_none() {
                continue;
            }
            let dum = FlatUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                color: [0.0; 4],
            };
            let Some(off) = renderer.gpu.flat_pool.push_flat(&dum) else { break };
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[off]);
            if let Some(mid) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mid, &mut last_mesh);
            }
        }
    }

    // 2) Selected mask
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("selection outline mask"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &tg.mask_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &tg.mask_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(depth_clear),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(mask_pl);
        let mut last_mesh = None;
        for &i in ctx.selected_order {
            let dc = ctx.visible[i];
            let dum = FlatUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                color: [0.0; 4],
            };
            let Some(off) = renderer.gpu.flat_pool.push_flat(&dum) else { break };
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[off]);
            pass.set_bind_group(1, &tg.mask_aux_bg, &[]);
            if let Some(mid) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mid, &mut last_mesh);
            }
        }
    }

    // 3) Edge fullscreen
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("selection outline edge"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &tg.edge_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pl.edge_pipeline);
        pass.set_bind_group(0, &tg.edge_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    // 4) Composite -> shade_view
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("selection outline composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&tg.composite_pl);
        pass.set_bind_group(0, &tg.composite_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    renderer.gpu.selection_outline_targets = Some(tg);
}
