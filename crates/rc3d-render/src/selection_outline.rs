//! Screen-space selection outline aligned with three.js `OutlinePass`:
//! non-selected depth, selected mask, half-res Sobel, separable blur, additive overlay.

use crate::render_passes::PassContext;
use crate::vertex::{FlatUniforms, Vertex};
use bytemuck::{Pod, Zeroable};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const PREPASS_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const EDGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const EDGE_STRENGTH: f32 = 3.0;

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
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    texel_size: [f32; 2],
    direction: [f32; 2],
    kernel_radius: f32,
    _pad0: f32,
    _pad1: [f32; 2],
    // Match EdgeParams (48B) so edge/blur can share fs_bgl min_binding_size.
    _pad2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayParams {
    edge_strength: f32,
    // WGSL `vec3` _pad is 16-aligned, so OverlayParams is 32B.
    _pad: [f32; 7],
}

pub struct SelectionOutlinePipelines {
    pub mask_aux_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    fs_bgl: wgpu::BindGroupLayout,
    overlay_bgl: wgpu::BindGroupLayout,
    overlay_pll: wgpu::PipelineLayout,
    pub depth_prepass_fwd: wgpu::RenderPipeline,
    pub depth_prepass_rev: wgpu::RenderPipeline,
    pub mask_fwd: wgpu::RenderPipeline,
    pub mask_rev: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
    overlay_shader: wgpu::ShaderModule,
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
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection_outline_blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit_tex.wgsl").into()),
        });
        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection_outline_edge"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection_outline_edge.wgsl").into()),
        });
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection_outline_blur"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection_outline_blur.wgsl").into()),
        });
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection_outline_overlay"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/selection_outline_composite.wgsl").into(),
            ),
        });

        let mask_aux_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selection_outline_mask_aux"),
            entries: &[
                tex_entry(0, false),
                uniform_entry(1, std::mem::size_of::<MaskParams>() as u64),
            ],
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selection_outline_blit_bgl"),
            entries: &[tex_entry(0, true), sampler_entry(1)],
        });
        debug_assert_eq!(
            std::mem::size_of::<EdgeParams>(),
            std::mem::size_of::<BlurParams>()
        );
        let fs_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selection_outline_fs_bgl"),
            entries: &[
                tex_entry(0, true),
                sampler_entry(1),
                uniform_entry(2, std::mem::size_of::<EdgeParams>() as u64),
            ],
        });
        debug_assert_eq!(std::mem::size_of::<OverlayParams>(), 32);
        let overlay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selection_outline_overlay_bgl"),
            entries: &[
                tex_entry(0, true),
                tex_entry(1, true),
                sampler_entry(2),
                uniform_entry(3, std::mem::size_of::<OverlayParams>() as u64),
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
        let blit_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection_blit_pll"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });
        let fs_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection_fs_pll"),
            bind_group_layouts: &[&fs_bgl],
            push_constant_ranges: &[],
        });
        let overlay_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection_overlay_pll"),
            bind_group_layouts: &[&overlay_bgl],
            push_constant_ranges: &[],
        });

        let ms = wgpu::MultisampleState::default();
        let depth_prepass_fwd = mesh_depth_pipeline(
            device,
            "selection_depth_prepass_fwd",
            &depth_prepass_pll,
            &depth_shader,
            PREPASS_COLOR_FORMAT,
            wgpu::CompareFunction::Less,
            ms,
        );
        let depth_prepass_rev = mesh_depth_pipeline(
            device,
            "selection_depth_prepass_rev",
            &depth_prepass_pll,
            &depth_shader,
            PREPASS_COLOR_FORMAT,
            wgpu::CompareFunction::Greater,
            ms,
        );
        let mask_fwd = mesh_depth_pipeline(
            device,
            "selection_mask_fwd",
            &mask_pll,
            &mask_shader,
            MASK_FORMAT,
            wgpu::CompareFunction::Less,
            ms,
        );
        let mask_rev = mesh_depth_pipeline(
            device,
            "selection_mask_rev",
            &mask_pll,
            &mask_shader,
            MASK_FORMAT,
            wgpu::CompareFunction::Greater,
            ms,
        );

        let downsample_pipeline = fullscreen_pipeline(
            device,
            "selection_outline_downsample",
            &blit_pll,
            &blit_shader,
            "vs_main",
            "fs_main",
            EDGE_FORMAT,
            wgpu::BlendState::REPLACE,
            ms,
        );
        let edge_pipeline = fullscreen_pipeline(
            device,
            "selection_outline_edge",
            &fs_pll,
            &edge_shader,
            "vs_fullscreen",
            "fs_edge",
            EDGE_FORMAT,
            wgpu::BlendState::REPLACE,
            ms,
        );
        let blur_pipeline = fullscreen_pipeline(
            device,
            "selection_outline_blur",
            &fs_pll,
            &blur_shader,
            "vs_fullscreen",
            "fs_blur",
            EDGE_FORMAT,
            wgpu::BlendState::REPLACE,
            ms,
        );

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("selection_outline_linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("selection_outline_nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            mask_aux_bgl,
            blit_bgl,
            fs_bgl,
            overlay_bgl,
            overlay_pll,
            depth_prepass_fwd,
            depth_prepass_rev,
            mask_fwd,
            mask_rev,
            downsample_pipeline,
            edge_pipeline,
            blur_pipeline,
            linear_sampler,
            nearest_sampler,
            overlay_shader,
        }
    }

    fn create_overlay_pipeline(
        &self,
        device: &wgpu::Device,
        out_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        fullscreen_pipeline(
            device,
            "selection_outline_overlay",
            &self.overlay_pll,
            &self.overlay_shader,
            "vs_fullscreen",
            "fs_overlay",
            out_format,
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            wgpu::MultisampleState::default(),
        )
    }
}

#[allow(dead_code)]
pub struct SelectionOutlineTargets {
    pub width: u32,
    pub height: u32,
    half_width: u32,
    half_height: u32,
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
    downsample_view: wgpu::TextureView,
    downsample_tex: wgpu::Texture,
    edge_view: wgpu::TextureView,
    edge_tex: wgpu::Texture,
    blur_view: wgpu::TextureView,
    blur_tex: wgpu::Texture,
    blurred_view: wgpu::TextureView,
    blurred_tex: wgpu::Texture,
    pub mask_params_buf: wgpu::Buffer,
    edge_params_buf: wgpu::Buffer,
    blur_h_params_buf: wgpu::Buffer,
    blur_v_params_buf: wgpu::Buffer,
    overlay_params_buf: wgpu::Buffer,
    pub mask_aux_bg: wgpu::BindGroup,
    downsample_bg: wgpu::BindGroup,
    edge_bg: wgpu::BindGroup,
    blur_h_bg: wgpu::BindGroup,
    blur_v_bg: wgpu::BindGroup,
    overlay_bg: wgpu::BindGroup,
    overlay_pl: wgpu::RenderPipeline,
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
        let hw = (w / 2).max(1);
        let hh = (h / 2).max(1);
        let full = extent(w, h);
        let half = extent(hw, hh);

        let (prepass_color, prepass_color_view) =
            color_target(device, "sel_outline_prepass_color", full, PREPASS_COLOR_FORMAT);
        let (prepass_depth, prepass_depth_view) =
            depth_target(device, "sel_outline_prepass_depth", full);
        let (mask_tex, mask_view) = color_target(device, "sel_outline_mask", full, MASK_FORMAT);
        let (mask_depth, mask_depth_view) = depth_target(device, "sel_outline_mask_depth", full);
        let (downsample_tex, downsample_view) =
            color_target(device, "sel_outline_mask_half", half, EDGE_FORMAT);
        let (edge_tex, edge_view) = color_target(device, "sel_outline_edge", half, EDGE_FORMAT);
        let (blur_tex, blur_view) = color_target(device, "sel_outline_blur", half, EDGE_FORMAT);
        let (blurred_tex, blurred_view) =
            color_target(device, "sel_outline_blurred", half, EDGE_FORMAT);

        let mask_params_buf = uniform_buf(device, "sel_outline_mask_params", std::mem::size_of::<MaskParams>());
        let edge_params_buf = uniform_buf(device, "sel_outline_edge_params", std::mem::size_of::<EdgeParams>());
        let blur_h_params_buf = uniform_buf(device, "sel_outline_blur_h", std::mem::size_of::<BlurParams>());
        let blur_v_params_buf = uniform_buf(device, "sel_outline_blur_v", std::mem::size_of::<BlurParams>());
        let overlay_params_buf =
            uniform_buf(device, "sel_outline_overlay", std::mem::size_of::<OverlayParams>());

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
        let downsample_bg = tex_sampler_bg(
            device,
            "sel_outline_downsample_bg",
            &pl.blit_bgl,
            &mask_view,
            &pl.nearest_sampler,
        );
        let edge_bg = tex_sampler_uniform_bg(
            device,
            "sel_outline_edge_bg",
            &pl.fs_bgl,
            &downsample_view,
            &pl.nearest_sampler,
            &edge_params_buf,
        );
        let blur_h_bg = tex_sampler_uniform_bg(
            device,
            "sel_outline_blur_h_bg",
            &pl.fs_bgl,
            &edge_view,
            &pl.linear_sampler,
            &blur_h_params_buf,
        );
        let blur_v_bg = tex_sampler_uniform_bg(
            device,
            "sel_outline_blur_v_bg",
            &pl.fs_bgl,
            &blur_view,
            &pl.linear_sampler,
            &blur_v_params_buf,
        );
        let overlay_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sel_outline_overlay_bg"),
            layout: &pl.overlay_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blurred_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&pl.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: overlay_params_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            width: w,
            height: h,
            half_width: hw,
            half_height: hh,
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
            downsample_view,
            downsample_tex,
            edge_view,
            edge_tex,
            blur_view,
            blur_tex,
            blurred_view,
            blurred_tex,
            mask_params_buf,
            edge_params_buf,
            blur_h_params_buf,
            blur_v_params_buf,
            overlay_params_buf,
            mask_aux_bg,
            downsample_bg,
            edge_bg,
            blur_h_bg,
            blur_v_bg,
            overlay_bg,
            overlay_pl: pl.create_overlay_pipeline(device, shade_format),
        }
    }
}

pub(crate) fn encode_selection_outline_pass(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &PassContext<'_>,
    shade_view: &wgpu::TextureView,
    shade_format: wgpu::TextureFormat,
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

    let far_ndc = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
    let depth_clear = far_ndc;
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

    let mask_params = MaskParams {
        reversed_z: if ctx.depth_reversed_z { 1.0 } else { 0.0 },
        depth_bias: 1e-3,
        far_ndc_z: far_ndc,
        _pad: 0.0,
    };
    renderer
        .queue
        .write_buffer(&tg.mask_params_buf, 0, bytemuck::bytes_of(&mask_params));

    let oc = ctx.outline_color;
    let visible_rgb = [oc[0], oc[1], oc[2]];
    let hidden_rgb = [oc[0] * 0.15, oc[1] * 0.15, oc[2] * 0.15];
    let half_texel = [1.0 / tg.half_width as f32, 1.0 / tg.half_height as f32];
    let thickness = pixel_thickness(ctx.outline_width);
    renderer.queue.write_buffer(
        &tg.edge_params_buf,
        0,
        bytemuck::bytes_of(&EdgeParams {
            texel_size: half_texel,
            _pad0: [0.0; 2],
            visible_color: visible_rgb,
            _pad1: 0.0,
            hidden_color: hidden_rgb,
            _pad2: 0.0,
        }),
    );
    renderer.queue.write_buffer(
        &tg.blur_h_params_buf,
        0,
        bytemuck::bytes_of(&BlurParams {
            texel_size: half_texel,
            direction: [1.0, 0.0],
            kernel_radius: thickness,
            _pad0: 0.0,
            _pad1: [0.0; 2],
            _pad2: [0.0; 4],
        }),
    );
    renderer.queue.write_buffer(
        &tg.blur_v_params_buf,
        0,
        bytemuck::bytes_of(&BlurParams {
            texel_size: half_texel,
            direction: [0.0, 1.0],
            kernel_radius: thickness,
            _pad0: 0.0,
            _pad1: [0.0; 2],
            _pad2: [0.0; 4],
        }),
    );
    renderer.queue.write_buffer(
        &tg.overlay_params_buf,
        0,
        bytemuck::bytes_of(&OverlayParams {
            edge_strength: EDGE_STRENGTH,
            _pad: [0.0; 7],
        }),
    );

    {
        let clear_c = wgpu::Color {
            r: far_ndc as f64,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("selection outline depth prepass"),
            color_attachments: &[Some(color_att(&tg.prepass_color_view, wgpu::LoadOp::Clear(clear_c)))],
            depth_stencil_attachment: Some(depth_att(&tg.prepass_depth_view, depth_clear)),
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
            let dum = dummy_flat(dc);
            let Some(off) = renderer.gpu.flat_pool.push_flat(&dum) else {
                break;
            };
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[off]);
            if let Some(mid) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mid, &mut last_mesh);
            }
        }
    }

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("selection outline mask"),
            color_attachments: &[Some(color_att(&tg.mask_view, wgpu::LoadOp::Clear(wgpu::Color::WHITE)))],
            depth_stencil_attachment: Some(depth_att(&tg.mask_depth_view, depth_clear)),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(mask_pl);
        let mut last_mesh = None;
        for &i in ctx.selected_order {
            let dc = ctx.visible[i];
            let dum = dummy_flat(dc);
            let Some(off) = renderer.gpu.flat_pool.push_flat(&dum) else {
                break;
            };
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[off]);
            pass.set_bind_group(1, &tg.mask_aux_bg, &[]);
            if let Some(mid) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mid, &mut last_mesh);
            }
        }
    }

    fullscreen(
        encoder,
        "selection outline downsample",
        &tg.downsample_view,
        &pl.downsample_pipeline,
        &tg.downsample_bg,
        wgpu::LoadOp::Clear(wgpu::Color::WHITE),
    );
    fullscreen(
        encoder,
        "selection outline edge",
        &tg.edge_view,
        &pl.edge_pipeline,
        &tg.edge_bg,
        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
    );
    fullscreen(
        encoder,
        "selection outline blur h",
        &tg.blur_view,
        &pl.blur_pipeline,
        &tg.blur_h_bg,
        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
    );
    fullscreen(
        encoder,
        "selection outline blur v",
        &tg.blurred_view,
        &pl.blur_pipeline,
        &tg.blur_v_bg,
        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
    );
    fullscreen(
        encoder,
        "selection outline overlay",
        shade_view,
        &tg.overlay_pl,
        &tg.overlay_bg,
        wgpu::LoadOp::Load,
    );

    renderer.gpu.selection_outline_targets = Some(tg);
}

fn pixel_thickness(outline_width: f32) -> f32 {
    let t = if outline_width < 0.5 {
        1.0
    } else {
        outline_width
    };
    t.clamp(1.0, 4.0)
}

fn dummy_flat(dc: &crate::render_action::DrawCall) -> FlatUniforms {
    FlatUniforms {
        mvp: dc.mvp.to_cols_array_2d(),
        color: [0.0; 4],
        model: dc.model_matrix.to_cols_array_2d(),
        clip_planes: [[0.0; 4]; 6],
        clip_count: [0.0, 0.0, 0.0, 0.0],
    }
}

fn tex_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type: wgpu::TextureSampleType::Float { filterable },
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size),
        },
        count: None,
    }
}

fn mesh_depth_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    depth_compare: wgpu::CompareFunction,
    ms: wgpu::MultisampleState,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                // R32Float prepass is not blendable; mask/prepass both overwrite.
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
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
            depth_compare,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    })
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vs: &'static str,
    fs: &'static str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
    ms: wgpu::MultisampleState,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: ms,
        multiview: None,
        cache: None,
    })
}

fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

fn color_target(
    device: &wgpu::Device,
    label: &str,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn depth_target(
    device: &wgpu::Device,
    label: &str,
    size: wgpu::Extent3d,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn uniform_buf(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn tex_sampler_bg(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn tex_sampler_uniform_bg(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf.as_entire_binding(),
            },
        ],
    })
}

fn color_att<'a>(
    view: &'a wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPassColorAttachment<'a> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
    }
}

fn depth_att<'a>(
    view: &'a wgpu::TextureView,
    clear: f32,
) -> wgpu::RenderPassDepthStencilAttachment<'a> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(clear),
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    }
}

fn fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(color_att(view, load))],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[]);
    pass.draw(0..3, 0..1);
}
