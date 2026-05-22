use crate::shader_permutation::{ShaderFeatures, ShaderVariantCache};
use crate::vertex::{LineVertex, MarkupVertex, Vertex, WorldLabelVertex};
use bitflags::bitflags;

#[path = "pipelines_build_depth.rs"]
mod pipelines_build_depth;
use pipelines_build_depth::build_depth_mode_pipelines;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PbrFeatures: u32 {
        const HAS_ALBEDO_TEX    = 1 << 0;
        const HAS_NORMAL_TEX    = 1 << 1;
        const HAS_MR_TEX        = 1 << 2;
        const HAS_EMISSIVE_TEX  = 1 << 3;
        const HAS_OCCLUSION_TEX = 1 << 4;
        const HAS_IBL           = 1 << 5;
        const HAS_SHADOWS       = 1 << 6;
        const IS_TRANSPARENT    = 1 << 7;
    }
}

/// Render pipelines for one depth convention (forward-Z: Less / clear 1, reverse-Z: Greater / clear 0).
#[derive(Clone)]
pub struct DepthModePipelines {
    pub solid: wgpu::RenderPipeline,
    pub solid_alpha: wgpu::RenderPipeline,
    /// Phong solid with color writes disabled (depth/stencil only), for same-frame HZB prepass.
    pub solid_depth_prepass: wgpu::RenderPipeline,
    /// Flat color solid: MVP + uniform color, no lighting/PBR.
    pub flat_solid: wgpu::RenderPipeline,
    pub wireframe: wgpu::RenderPipeline,
    pub edge_overlay: wgpu::RenderPipeline,
    pub edge_overlay_aa: wgpu::RenderPipeline,
    pub selection_fill: wgpu::RenderPipeline,
    pub selection_edge: wgpu::RenderPipeline,
    /// Procedural fill for mesh / section-plane intersection (triangle mesh + plane-distance discard).
    pub section_cap_fill: wgpu::RenderPipeline,
    /// Stencil prepass for section caps: marks cross-section interior (stencil=2)
    /// by rendering complete (unclipped) front faces with depth NotEqual.
    pub section_cap_stencil: wgpu::RenderPipeline,
    pub outline: wgpu::RenderPipeline,
    /// WBOIT accumulate: PBR with MRT output (accum + revealage), additive + multiplicative blending.
    pub wboit_accum: wgpu::RenderPipeline,
}

pub struct PipelineSet {
    pub phong_bgl: wgpu::BindGroupLayout,
    pub pbr_material_bgl: wgpu::BindGroupLayout,
    pub shadow_draw_bgl: wgpu::BindGroupLayout,
    pub shadow_resource_bgl: wgpu::BindGroupLayout,
    pub flat_bgl: wgpu::BindGroupLayout,
    pub outline_bgl: wgpu::BindGroupLayout,
    pub ibl_instance_bgl: wgpu::BindGroupLayout,
    /// Directional shadow depth pass (forward-Z depth only, not tied to reverse-Z camera).
    pub shadow_depth: wgpu::RenderPipeline,
    pub forward: DepthModePipelines,
    pub reverse: DepthModePipelines,
    /// Same as forward/reverse but color targets use `Rgba16Float` for HDR scene + post.
    pub forward_hdr: DepthModePipelines,
    pub reverse_hdr: DepthModePipelines,
    /// Line list on swapchain without depth stencil (viewport split borders overlay).
    pub viewport_border_lines: wgpu::RenderPipeline,
    /// Line list without depth (legacy 2D MarkupNode screen-space overlay).
    pub markup_lines_screen: wgpu::RenderPipeline,
    /// Projected 3D annotation lines with depth test (forward-Z: Less).
    pub markup_lines_forward: wgpu::RenderPipeline,
    /// Projected 3D annotation lines with depth test (reverse-Z: Greater).
    pub markup_lines_reverse: wgpu::RenderPipeline,
    /// Line list with depth-test (forward-Z: Less).
    pub grid_lines_forward: wgpu::RenderPipeline,
    /// Line list with depth-test (reverse-Z: Greater).
    pub grid_lines_reverse: wgpu::RenderPipeline,
    /// World-space annotation label quads (uniform + label texture).
    pub world_label_bgl: wgpu::BindGroupLayout,
    pub world_label_sampler: wgpu::Sampler,
    pub world_label_forward: wgpu::RenderPipeline,
    pub world_label_reverse: wgpu::RenderPipeline,
}

impl PipelineSet {
    #[inline]
    pub fn for_depth(&self, depth_reversed_z: bool) -> &DepthModePipelines {
        if depth_reversed_z {
            &self.reverse
        } else {
            &self.forward
        }
    }

    #[inline]
    pub fn for_shaded_target(&self, depth_reversed_z: bool, hdr_scene: bool) -> &DepthModePipelines {
        match (hdr_scene, depth_reversed_z) {
            (true, true) => &self.reverse_hdr,
            (true, false) => &self.forward_hdr,
            (false, true) => &self.reverse,
            (false, false) => &self.forward,
        }
    }

    pub fn create(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shader_cache: &mut ShaderVariantCache,
    ) -> Self {
        let depth_format = wgpu::TextureFormat::Depth32FloatStencil8;

        // Default PBR permutation: all features enabled
        let pbr_features = ShaderFeatures::HAS_NORMAL_MAP
            .with(ShaderFeatures::HAS_SHADOW)
            .with(ShaderFeatures::HAS_ALBEDO_TEX)
            .with(ShaderFeatures::HAS_IBL)
            .with(ShaderFeatures::HAS_MR_TEX)
            .with(ShaderFeatures::HAS_EMISSIVE_TEX)
            .with(ShaderFeatures::HAS_OCCLUSION_TEX);
        let pbr_shader = shader_cache.get_module(
            device,
            pbr_features.bits,
            "pbr",
            include_str!("shaders/pbr.wgsl"),
        );
        let shadow_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow depth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow_depth.wgsl").into()),
        });
        let flat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flat Color Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flat_color.wgsl").into()),
        });
        let world_label_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("World Label Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/world_label.wgsl").into()),
        });
        let section_cap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Section Cap Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/section_cap.wgsl").into()),
        });
        let section_cap_mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Section Cap Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/section_cap_mesh.wgsl").into()),
        });
        let outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Outline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/outline.wgsl").into()),
        });
        let line_aa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line AA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line_aa.wgsl").into()),
        });

        let phong_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let flat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let outline_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Outline BGL"),
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

        let pbr_material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PBR material BGL"),
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
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });

        let shadow_draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow draw BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_resource_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow + Global Frame BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        // Combined IBL + instance data BGL (group 3): bindings 0-2 = IBL, binding 3 = instance SSBO
        let ibl_instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL + Instance BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false, view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false, view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Morph target position deltas (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Morph target normal deltas (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Morph target params (uniform: vertex count stride)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shadow_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow depth PLL"),
            bind_group_layouts: &[&shadow_draw_bgl],
            push_constant_ranges: &[],
        });

        let shadow_depth_format = wgpu::TextureFormat::Depth32Float;
        let shadow_depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow depth"),
            layout: Some(&shadow_pll),
            vertex: wgpu::VertexState {
                module: &shadow_depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: shadow_depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let lit_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lit PBR PLL"),
            bind_group_layouts: &[&phong_bgl, &pbr_material_bgl, &shadow_resource_bgl, &ibl_instance_bgl],
            push_constant_ranges: &[],
        });
        let flat_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flat PLL"),
            bind_group_layouts: &[&flat_bgl],
            push_constant_ranges: &[],
        });
        let outline_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Outline PLL"),
            bind_group_layouts: &[&outline_bgl],
            push_constant_ranges: &[],
        });

        let forward = build_depth_mode_pipelines(
            device,
            format,
            depth_format,
            false,
            &lit_pll,
            &flat_pll,
            &outline_pll,
            pbr_shader,
            &flat_shader,
            &section_cap_shader,
            &section_cap_mesh_shader,
            &outline_shader,
            &line_aa_shader,
        );
        let reverse = build_depth_mode_pipelines(
            device,
            format,
            depth_format,
            true,
            &lit_pll,
            &flat_pll,
            &outline_pll,
            pbr_shader,
            &flat_shader,
            &section_cap_shader,
            &section_cap_mesh_shader,
            &outline_shader,
            &line_aa_shader,
        );

        let hdr_color = wgpu::TextureFormat::Rgba16Float;
        let forward_hdr = build_depth_mode_pipelines(
            device,
            hdr_color,
            depth_format,
            false,
            &lit_pll,
            &flat_pll,
            &outline_pll,
            pbr_shader,
            &flat_shader,
            &section_cap_shader,
            &section_cap_mesh_shader,
            &outline_shader,
            &line_aa_shader,
        );
        let reverse_hdr = build_depth_mode_pipelines(
            device,
            hdr_color,
            depth_format,
            true,
            &lit_pll,
            &flat_pll,
            &outline_pll,
            pbr_shader,
            &flat_shader,
            &section_cap_shader,
            &section_cap_mesh_shader,
            &outline_shader,
            &line_aa_shader,
        );

        let viewport_border_ms = wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };
        let viewport_border_lines = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Viewport border lines (no depth)"),
            layout: Some(&flat_pll),
            vertex: wgpu::VertexState {
                module: &flat_shader,
                entry_point: Some("vs_line"),
                buffers: &[LineVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &flat_shader,
                entry_point: Some("fs_line_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: viewport_border_ms,
            multiview: None,
            cache: None,
        });

        let markup_lines_screen = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Markup screen-space lines (no depth)"),
            layout: Some(&flat_pll),
            vertex: wgpu::VertexState {
                module: &flat_shader,
                entry_point: Some("vs_markup"),
                buffers: &[MarkupVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &flat_shader,
                entry_point: Some("fs_markup"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: viewport_border_ms,
            multiview: None,
            cache: None,
        });

        let make_markup_depth_pipeline = |label: &str, cmp: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&flat_pll),
                vertex: wgpu::VertexState {
                    module: &flat_shader,
                    entry_point: Some("vs_markup"),
                    buffers: &[MarkupVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &flat_shader,
                    entry_point: Some("fs_markup"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false,
                    depth_compare: cmp,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: viewport_border_ms,
                multiview: None,
                cache: None,
            })
        };
        let markup_lines_forward =
            make_markup_depth_pipeline("Markup 3D lines (forward-Z)", wgpu::CompareFunction::Less);
        let markup_lines_reverse =
            make_markup_depth_pipeline("Markup 3D lines (reverse-Z)", wgpu::CompareFunction::Greater);

        let make_grid_pipeline = |label: &str, cmp: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&flat_pll),
                vertex: wgpu::VertexState {
                    module: &flat_shader,
                    entry_point: Some("vs_line"),
                    buffers: &[LineVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &flat_shader,
                    entry_point: Some("fs_line_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false,
                    depth_compare: cmp,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: viewport_border_ms,
                multiview: None,
                cache: None,
            })
        };
        let grid_lines_forward = make_grid_pipeline("Grid lines (forward-Z)", wgpu::CompareFunction::Less);
        let grid_lines_reverse = make_grid_pipeline("Grid lines (reverse-Z)", wgpu::CompareFunction::Greater);

        let world_label_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("World Label BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
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
            ],
        });
        let world_label_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("World Label Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let world_label_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("World Label PLL"),
            bind_group_layouts: &[&world_label_bgl],
            push_constant_ranges: &[],
        });
        let make_world_label_pipeline = |label: &str, cmp: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&world_label_pll),
                vertex: wgpu::VertexState {
                    module: &world_label_shader,
                    entry_point: Some("vs_world_label"),
                    buffers: &[WorldLabelVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &world_label_shader,
                    entry_point: Some("fs_world_label"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false,
                    depth_compare: cmp,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: viewport_border_ms,
                multiview: None,
                cache: None,
            })
        };
        let world_label_forward =
            make_world_label_pipeline("World label (forward-Z)", wgpu::CompareFunction::Less);
        let world_label_reverse =
            make_world_label_pipeline("World label (reverse-Z)", wgpu::CompareFunction::Greater);

        Self {
            phong_bgl,
            pbr_material_bgl,
            shadow_draw_bgl,
            shadow_resource_bgl,
            flat_bgl,
            outline_bgl,
            ibl_instance_bgl,
            shadow_depth,
            forward,
            reverse,
            forward_hdr,
            reverse_hdr,
            viewport_border_lines,
            markup_lines_screen,
            markup_lines_forward,
            markup_lines_reverse,
            grid_lines_forward,
            grid_lines_reverse,
            world_label_bgl,
            world_label_sampler,
            world_label_forward,
            world_label_reverse,
        }
    }
}

/// Cache for per-material shader variants.
/// Uses LRU eviction (max 16 variants) to bound memory.
pub struct PbrVariantCache {
    cache: lru::LruCache<PbrFeatures, wgpu::ShaderModule>,
}

impl Default for PbrVariantCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PbrVariantCache {
    pub fn new() -> Self {
        Self {
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(16).unwrap()),
        }
    }

    /// Get or create a shader module for the given feature set.
    /// Shader variants are created by prepending `#define` directives.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        features: PbrFeatures,
    ) -> &wgpu::ShaderModule {
        self.cache.get_or_insert(features, || {
            Self::build_variant(device, features)
        })
    }

    fn build_variant(device: &wgpu::Device, features: PbrFeatures) -> wgpu::ShaderModule {
        let defines = format!(
            "const HAS_ALBEDO_TEX: u32 = {}u;\n\
             const HAS_NORMAL_TEX: u32 = {}u;\n\
             const HAS_MR_TEX: u32 = {}u;\n\
             const HAS_EMISSIVE_TEX: u32 = {}u;\n\
             const HAS_OCCLUSION_TEX: u32 = {}u;\n\
             const HAS_IBL: u32 = {}u;\n\
             const HAS_SHADOWS: u32 = {}u;\n\
             const IS_TRANSPARENT: u32 = {}u;\n",
            features.contains(PbrFeatures::HAS_ALBEDO_TEX) as u32,
            features.contains(PbrFeatures::HAS_NORMAL_TEX) as u32,
            features.contains(PbrFeatures::HAS_MR_TEX) as u32,
            features.contains(PbrFeatures::HAS_EMISSIVE_TEX) as u32,
            features.contains(PbrFeatures::HAS_OCCLUSION_TEX) as u32,
            features.contains(PbrFeatures::HAS_IBL) as u32,
            features.contains(PbrFeatures::HAS_SHADOWS) as u32,
            features.contains(PbrFeatures::IS_TRANSPARENT) as u32,
        );

        // Load base PBR shader and prepend defines
        let base = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/pbr.wgsl")
        ).unwrap_or_else(|_| String::from("// pbr shader not found"));

        let full = format!("// Auto-generated variant: {:?}\n{}\n{}", features, defines, base);

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("PBR variant {:?}", features)),
            source: wgpu::ShaderSource::Wgsl(full.into()),
        })
    }
}

/// Compute PbrFeatures from a DrawCall's material properties.
/// Used to select the appropriate shader variant at draw time.
pub fn features_for_draw(
    has_albedo_tex: bool,
    has_normal_tex: bool,
    has_mr_tex: bool,
    has_emissive_tex: bool,
    has_occlusion_tex: bool,
    has_ibl: bool,
    has_shadows: bool,
    is_transparent: bool,
) -> PbrFeatures {
    let mut f = PbrFeatures::empty();
    if has_albedo_tex { f |= PbrFeatures::HAS_ALBEDO_TEX; }
    if has_normal_tex { f |= PbrFeatures::HAS_NORMAL_TEX; }
    if has_mr_tex { f |= PbrFeatures::HAS_MR_TEX; }
    if has_emissive_tex { f |= PbrFeatures::HAS_EMISSIVE_TEX; }
    if has_occlusion_tex { f |= PbrFeatures::HAS_OCCLUSION_TEX; }
    if has_ibl { f |= PbrFeatures::HAS_IBL; }
    if has_shadows { f |= PbrFeatures::HAS_SHADOWS; }
    if is_transparent { f |= PbrFeatures::IS_TRANSPARENT; }
    f
}
