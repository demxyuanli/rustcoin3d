use crate::vertex::{LineVertex, Vertex};

/// Render pipelines for one depth convention (forward-Z: Less / clear 1, reverse-Z: Greater / clear 0).
#[derive(Clone)]
pub struct DepthModePipelines {
    pub solid: wgpu::RenderPipeline,
    /// Phong solid with color writes disabled (depth/stencil only), for same-frame HZB prepass.
    pub solid_depth_prepass: wgpu::RenderPipeline,
    pub wireframe: wgpu::RenderPipeline,
    pub edge_overlay: wgpu::RenderPipeline,
    pub selection_fill: wgpu::RenderPipeline,
    pub selection_edge: wgpu::RenderPipeline,
    pub outline: wgpu::RenderPipeline,
}

pub struct PipelineSet {
    pub phong_bgl: wgpu::BindGroupLayout,
    pub pbr_material_bgl: wgpu::BindGroupLayout,
    pub shadow_draw_bgl: wgpu::BindGroupLayout,
    pub shadow_resource_bgl: wgpu::BindGroupLayout,
    pub flat_bgl: wgpu::BindGroupLayout,
    pub outline_bgl: wgpu::BindGroupLayout,
    /// Directional shadow depth pass (forward-Z depth only, not tied to reverse-Z camera).
    pub shadow_depth: wgpu::RenderPipeline,
    pub forward: DepthModePipelines,
    pub reverse: DepthModePipelines,
    /// Same as forward/reverse but color targets use `Rgba16Float` for HDR scene + post.
    pub forward_hdr: DepthModePipelines,
    pub reverse_hdr: DepthModePipelines,
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

    pub fn create(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let depth_format = wgpu::TextureFormat::Depth32FloatStencil8;

        let pbr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBR Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pbr.wgsl").into()),
        });
        let shadow_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow depth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow_depth.wgsl").into()),
        });
        let flat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flat Color Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flat_color.wgsl").into()),
        });
        let outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Outline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/outline.wgsl").into()),
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
            label: Some("Shadow map BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
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
                cull_mode: Some(wgpu::Face::Back),
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
            bind_group_layouts: &[&phong_bgl, &pbr_material_bgl, &shadow_resource_bgl],
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
            &pbr_shader,
            &flat_shader,
            &outline_shader,
        );
        let reverse = build_depth_mode_pipelines(
            device,
            format,
            depth_format,
            true,
            &lit_pll,
            &flat_pll,
            &outline_pll,
            &pbr_shader,
            &flat_shader,
            &outline_shader,
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
            &pbr_shader,
            &flat_shader,
            &outline_shader,
        );
        let reverse_hdr = build_depth_mode_pipelines(
            device,
            hdr_color,
            depth_format,
            true,
            &lit_pll,
            &flat_pll,
            &outline_pll,
            &pbr_shader,
            &flat_shader,
            &outline_shader,
        );

        Self {
            phong_bgl,
            pbr_material_bgl,
            shadow_draw_bgl,
            shadow_resource_bgl,
            flat_bgl,
            outline_bgl,
            shadow_depth,
            forward,
            reverse,
            forward_hdr,
            reverse_hdr,
        }
    }
}

fn build_depth_mode_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    depth_reversed_z: bool,
    lit_pll: &wgpu::PipelineLayout,
    flat_pll: &wgpu::PipelineLayout,
    outline_pll: &wgpu::PipelineLayout,
    lit_shader: &wgpu::ShaderModule,
    flat_shader: &wgpu::ShaderModule,
    outline_shader: &wgpu::ShaderModule,
) -> DepthModePipelines {
    let stencil_op_keep = |cmp: wgpu::CompareFunction| wgpu::StencilFaceState {
        compare: cmp,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::Keep,
    };

    let stencil_unchanged = wgpu::StencilState {
        front: stencil_op_keep(wgpu::CompareFunction::Always),
        back: stencil_op_keep(wgpu::CompareFunction::Always),
        read_mask: 0xff,
        write_mask: 0x00,
    };

    let (depth_cmp, depth_cmp_overlay) = if depth_reversed_z {
        (
            wgpu::CompareFunction::Greater,
            wgpu::CompareFunction::GreaterEqual,
        )
    } else {
        (
            wgpu::CompareFunction::Less,
            wgpu::CompareFunction::LessEqual,
        )
    };

    let depth_stencil = wgpu::DepthStencilState {
        format: depth_format,
        depth_write_enabled: true,
        depth_compare: depth_cmp,
        stencil: stencil_unchanged.clone(),
        bias: wgpu::DepthBiasState::default(),
    };

    let depth_stencil_solid = wgpu::DepthStencilState {
        format: depth_format,
        depth_write_enabled: true,
        depth_compare: depth_cmp,
        stencil: wgpu::StencilState {
            front: wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Always,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Replace,
            },
            back: wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Always,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Replace,
            },
            read_mask: 0xff,
            write_mask: 0xff,
        },
        bias: wgpu::DepthBiasState::default(),
    };

    let depth_stencil_outline = wgpu::DepthStencilState {
        format: depth_format,
        depth_write_enabled: false,
        depth_compare: depth_cmp,
        stencil: wgpu::StencilState {
            front: stencil_op_keep(wgpu::CompareFunction::Always),
            back: wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            },
            read_mask: 0xff,
            write_mask: 0x00,
        },
        bias: wgpu::DepthBiasState::default(),
    };

    let ms = wgpu::MultisampleState {
        count: 1,
        mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let solid = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("PBR solid"),
        layout: Some(lit_pll),
        vertex: wgpu::VertexState {
            module: lit_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: lit_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(depth_stencil_solid.clone()),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    let solid_depth_prepass = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Solid depth prepass"),
        layout: Some(lit_pll),
        vertex: wgpu::VertexState {
            module: lit_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: lit_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::empty(),
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(depth_stencil_solid),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    let wireframe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(flat_pll),
        vertex: wgpu::VertexState {
            module: flat_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: flat_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Line,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            depth_write_enabled: false,
            ..depth_stencil.clone()
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    let edge_overlay = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(flat_pll),
        vertex: wgpu::VertexState {
            module: flat_shader,
            entry_point: Some("vs_line"),
            buffers: &[LineVertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: flat_shader,
            entry_point: Some("fs_main"),
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
            depth_write_enabled: false,
            depth_compare: depth_cmp_overlay,
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.0,
                clamp: 0.0,
            },
            ..depth_stencil.clone()
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    let selection_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(flat_pll),
        vertex: wgpu::VertexState {
            module: flat_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: flat_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            depth_write_enabled: false,
            depth_compare: depth_cmp_overlay,
            ..depth_stencil.clone()
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    let selection_edge = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(flat_pll),
        vertex: wgpu::VertexState {
            module: flat_shader,
            entry_point: Some("vs_line"),
            buffers: &[LineVertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: flat_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
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
            depth_write_enabled: false,
            depth_compare: depth_cmp_overlay,
            ..depth_stencil.clone()
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    let outline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(outline_pll),
        vertex: wgpu::VertexState {
            module: outline_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: outline_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Front),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(depth_stencil_outline),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    DepthModePipelines {
        solid,
        solid_depth_prepass,
        wireframe,
        edge_overlay,
        selection_fill,
        selection_edge,
        outline,
    }
}
