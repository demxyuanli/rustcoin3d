use crate::vertex::{LineVertex, LineVertexExpanded, Vertex};

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
    section_cap_mesh_shader: &wgpu::ShaderModule,
    outline_shader: &wgpu::ShaderModule,
    line_aa_shader: &wgpu::ShaderModule,
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

    let (depth_cmp_prepass, depth_cmp_main, depth_cmp_overlay) = if depth_reversed_z {
        (
            wgpu::CompareFunction::Greater,
            wgpu::CompareFunction::GreaterEqual,
            wgpu::CompareFunction::GreaterEqual,
        )
    } else {
        (
            wgpu::CompareFunction::Less,
            wgpu::CompareFunction::LessEqual,
            wgpu::CompareFunction::LessEqual,
        )
    };

    let depth_stencil = wgpu::DepthStencilState {
        format: depth_format,
        depth_write_enabled: true,
        depth_compare: depth_cmp_prepass,
        stencil: stencil_unchanged.clone(),
        bias: wgpu::DepthBiasState::default(),
    };

    let depth_stencil_solid = wgpu::DepthStencilState {
        format: depth_format,
        depth_write_enabled: true,
        depth_compare: depth_cmp_main,
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
                pass_op: wgpu::StencilOperation::Keep,
            },
            read_mask: 0xff,
            write_mask: 0xff,
        },
        bias: wgpu::DepthBiasState::default(),
    };

    let depth_stencil_outline = wgpu::DepthStencilState {
        format: depth_format,
        depth_write_enabled: false,
        depth_compare: depth_cmp_main,
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
            cull_mode: None,
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
            cull_mode: None,
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
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            depth_compare: depth_cmp_main,
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
            depth_write_enabled: false,
            depth_compare: depth_cmp_overlay,
            bias: wgpu::DepthBiasState {
                constant: 24,
                slope_scale: 3.0,
                clamp: 0.0,
            },
            ..depth_stencil.clone()
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    let edge_overlay_aa = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Edge Overlay AA"),
        layout: Some(flat_pll),
        vertex: wgpu::VertexState {
            module: line_aa_shader,
            entry_point: Some("vs_line_expanded"),
            buffers: &[LineVertexExpanded::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: line_aa_shader,
            entry_point: Some("fs_line_aa"),
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
            depth_write_enabled: false,
            depth_compare: depth_cmp_overlay,
            bias: wgpu::DepthBiasState {
                constant: 24,
                slope_scale: 3.0,
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
            entry_point: Some("fs_line_main"),
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

    // Section cap fill: renders back faces of the original mesh with the same clip plane
    // as the solid pass. Since cull=Front, only back faces render.
    // The clip plane in the fragment shader discards fragments above the plane,
    // leaving only the cross-section polygon (which is the silhouette of the back
    // face at the cut plane). Depth write disabled.
    // Depth compare: Only write where depth is >= existing (in front or equal).
    // For reverse-Z: GreaterEqual. For forward-Z: LessEqual.
    let section_cap_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Section cap fill (back faces)"),
        layout: Some(flat_pll),  // flat_pll matches flat_pool's bind group layout
        vertex: wgpu::VertexState {
            module: section_cap_mesh_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: section_cap_mesh_shader,
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
            cull_mode: Some(wgpu::Face::Front),  // back faces only
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: if depth_reversed_z {
                wgpu::CompareFunction::GreaterEqual
            } else {
                wgpu::CompareFunction::LessEqual
            },
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Always,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
                back: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Always,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
                read_mask: 0xff,
                write_mask: 0x00,
            },
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    // Stencil prepass (unused in this simplified approach, kept for API compatibility)
    let section_cap_stencil = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Section cap stencil prepass"),
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
                write_mask: wgpu::ColorWrites::empty(),
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
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
                    pass_op: wgpu::StencilOperation::Keep,
                },
                read_mask: 0xff,
                write_mask: 0xff,
            },
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    // Section cap fill: full-screen triangle restricted to stencil == 2
    // (the cross-section area identified by the stencil prepass above).
    // Uses depth Always because the cap plane may be behind back faces
    // drawn by the solid pass at the cross-section.
    // NOTE: This pipeline is for the stencil-based approach (section_cap.wgsl).
    // The mesh-based section_cap_fill above uses flat_shader instead.
    let _section_cap_fill_fullscreen = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Section cap fill"),
        layout: Some(flat_pll),
        vertex: wgpu::VertexState {
            module: section_cap_shader,
            entry_point: Some("vs_main"),
            buffers: &[],  // full-screen triangle — no vertex buffers
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
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Equal,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
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
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: depth_cmp_main,
            stencil: stencil_unchanged.clone(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });

    // WBOIT accumulate pipeline: PBR with MRT output (accum + revealage)
    // RT0 (accum):     additive blending — color * alpha * weight accumulates
    // RT1 (revealage): multiplicative blending — dst *= (1 - alpha)
    let wboit_accum = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("WBOIT Accumulate"),
        layout: Some(lit_pll),
        vertex: wgpu::VertexState {
            module: lit_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: lit_shader,
            entry_point: Some("fs_main_wboit"),
            targets: &[
                // RT0: accum (Rgba16Float, additive blend)
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // RT1: revealage (R8Unorm, multiplicative blend: dst *= src)
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::Src, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::Src, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // double-sided for transparent
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: depth_cmp_main,
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
        edge_overlay_aa,
        selection_fill,
        selection_edge,
        section_cap_stencil,
        section_cap_fill,
        outline,
        wboit_accum,
    }
}
