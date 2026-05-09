use crate::vertex::{LineVertex, Vertex};

pub(super) fn build_depth_mode_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    depth_reversed_z: bool,
    lit_pll: &wgpu::PipelineLayout,
    flat_pll: &wgpu::PipelineLayout,
    outline_pll: &wgpu::PipelineLayout,
    lit_shader: &wgpu::ShaderModule,
    flat_shader: &wgpu::ShaderModule,
    section_cap_shader: &wgpu::ShaderModule,
    outline_shader: &wgpu::ShaderModule,
) -> super::DepthModePipelines {
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

    let flat_solid = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Flat solid"),
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
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(depth_stencil.clone()),
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

    let section_cap_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Section cap fill"),
        layout: Some(flat_pll),
        vertex: wgpu::VertexState {
            module: section_cap_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: section_cap_shader,
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
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: depth_cmp_overlay,
            stencil: stencil_unchanged.clone(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.0,
                clamp: 0.0,
            },
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    let solid_alpha = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("PBR solid alpha"),
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: depth_cmp,
            stencil: stencil_unchanged.clone(),
            bias: wgpu::DepthBiasState::default(),
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

    super::DepthModePipelines {
        solid,
        solid_alpha,
        solid_depth_prepass,
        flat_solid,
        wireframe,
        edge_overlay,
        selection_fill,
        selection_edge,
        section_cap_fill,
        outline,
    }
}
