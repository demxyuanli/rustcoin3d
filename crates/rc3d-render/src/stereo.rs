//! Dual-eye presentation for `StereoCameraNode` (side-by-side, top-bottom, anaglyph).

use rc3d_scene::node_data::StereoMode;
use rc3d_scene::SceneGraph;

use crate::render_action::{apply_world_camera_ex, DrawCall};
use crate::viewport::{QuadViewEye, ViewportRect};
use crate::{FrameStats, Renderer};

pub fn stereo_view_rects(mode: StereoMode, surface_w: u32, surface_h: u32) -> Vec<ViewportRect> {
    let w = surface_w.max(1);
    let h = surface_h.max(1);
    match mode {
        StereoMode::SideBySide => {
            let hw = (w / 2).max(1);
            let rw = w.saturating_sub(hw).max(1);
            vec![
                ViewportRect {
                    x: 0,
                    y: 0,
                    width: hw,
                    height: h,
                },
                ViewportRect {
                    x: hw,
                    y: 0,
                    width: rw,
                    height: h,
                },
            ]
        }
        StereoMode::TopBottom => {
            let hh = (h / 2).max(1);
            let bh = h.saturating_sub(hh).max(1);
            vec![
                ViewportRect {
                    x: 0,
                    y: 0,
                    width: w,
                    height: hh,
                },
                ViewportRect {
                    x: 0,
                    y: hh,
                    width: w,
                    height: bh,
                },
            ]
        }
        StereoMode::Anaglyph => vec![
            ViewportRect {
                x: 0,
                y: 0,
                width: w,
                height: h,
            },
            ViewportRect {
                x: 0,
                y: 0,
                width: w,
                height: h,
            },
        ],
    }
}

fn ensure_anaglyph_pipeline(renderer: &mut Renderer) {
    if renderer.gpu.anaglyph_pipeline.is_some() {
        return;
    }
    let bgl = renderer
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Anaglyph BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
    let shader = renderer
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Anaglyph"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/anaglyph.wgsl"
            ))),
        });
    let layout = renderer
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Anaglyph Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
    let pipeline = renderer
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Anaglyph Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_anaglyph"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_anaglyph"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: renderer.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            depth_stencil: None,
            cache: None,
        });
    renderer.gpu.anaglyph_bgl = Some(bgl);
    renderer.gpu.anaglyph_pipeline = Some(pipeline);
}

fn blit_anaglyph(renderer: &Renderer, target: &wgpu::TextureView) {
    let Some(pipeline) = renderer.gpu.anaglyph_pipeline.as_ref() else {
        return;
    };
    let Some(bgl) = renderer.gpu.anaglyph_bgl.as_ref() else {
        return;
    };
    let Some(sampler) = renderer.gpu.upscale_sampler.as_ref() else {
        return;
    };
    if renderer.gpu.quad_tiles.len() < 2 {
        return;
    }
    let bg = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Anaglyph blit"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&renderer.gpu.quad_tiles[0].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&renderer.gpu.quad_tiles[1].view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Anaglyph composite"),
        });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blit anaglyph"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
    renderer.queue.submit(std::iter::once(encoder.finish()));
}

impl Renderer {
    /// Render left/right eyes into split (or anaglyph) viewports. Does not mutate `ViewportLayout`.
    pub fn render_stereo_views(
        &mut self,
        draw_calls: &mut [DrawCall],
        scene: &SceneGraph,
        eyes: &[QuadViewEye],
        mode: StereoMode,
    ) -> FrameStats {
        if eyes.len() < 2 || self.gpu.upscale_pipeline.is_none() {
            apply_world_camera_ex(
                draw_calls,
                eyes.first().map(|e| e.view).unwrap_or(glam::Mat4::IDENTITY),
                eyes.first()
                    .map(|e| e.projection)
                    .unwrap_or(glam::Mat4::IDENTITY),
                eyes.first()
                    .map(|e| e.camera_pos)
                    .unwrap_or(glam::Vec3::ZERO),
                eyes.first().map(|e| e.orthographic).unwrap_or(false),
            );
            return self.render_draw_calls(draw_calls, scene);
        }
        let (sw, sh) = (self.config.width, self.config.height);
        let mut mode = mode;
        if mode == StereoMode::Anaglyph {
            ensure_anaglyph_pipeline(self);
            if self.gpu.anaglyph_pipeline.is_none() {
                mode = StereoMode::SideBySide;
            }
        }
        let rects = stereo_view_rects(mode, sw, sh);
        let stats = self.render_quad_tiles(draw_calls, scene, eyes, &rects);
        let Ok(frame) = self.surface.get_current_texture() else {
            return stats;
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        if mode == StereoMode::Anaglyph {
            blit_anaglyph(self, &view);
        } else {
            self.blit_quad_tiles_to_view(&view, &rects);
        }
        frame.present();
        stats
    }
}
