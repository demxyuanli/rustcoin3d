//! Viewport border overlay rendering.
//!
//! Draws viewport split borders and the active-viewport highlight border
//! using the existing edge overlay pipeline.

use crate::vertex::{FlatUniforms, LineVertex};
use crate::viewport::LayoutMode;
use glam::Mat4;
use wgpu::util::DeviceExt;

/// Generate line vertices for a viewport border rectangle.
fn border_lines(x: u32, y: u32, w: u32, h: u32) -> Vec<LineVertex> {
    let x0 = x as f32;
    let y0 = y as f32;
    let x1 = (x + w - 1) as f32;
    let y1 = (y + h - 1) as f32;
    let z = 0.0;
    vec![
        // Top edge
        LineVertex { position: [x0, y0, z] },
        LineVertex { position: [x1, y0, z] },
        // Right edge
        LineVertex { position: [x1, y0, z] },
        LineVertex { position: [x1, y1, z] },
        // Bottom edge
        LineVertex { position: [x1, y1, z] },
        LineVertex { position: [x0, y1, z] },
        // Left edge
        LineVertex { position: [x0, y1, z] },
        LineVertex { position: [x0, y0, z] },
    ]
}

/// Line vertices for all viewport borders + active-viewport highlight.
pub struct ViewportBorderGeometry {
    pub split_lines: Vec<LineVertex>,
    pub active_lines: Vec<LineVertex>,
}

impl ViewportBorderGeometry {
    /// Build border geometry from the current viewport layout.
    pub fn build(layout: &crate::viewport::ViewportLayout, surface_w: u32, surface_h: u32) -> Self {
        let mut split_lines = Vec::new();
        let mut active_lines = Vec::new();

        // Draw borders between viewports (internal split lines)
        let vp_count = layout.viewports.len();
        if vp_count > 1 {
            for vp in &layout.viewports {
                // Right border (except rightmost viewports)
                let right = vp.rect.x + vp.rect.width;
                if right < surface_w && !layout.viewports.iter().any(|v| v.rect.x == right) {
                    split_lines.extend(border_lines(right.saturating_sub(1), 0, 4, surface_h));
                }
                // Bottom border (except bottommost viewports)
                let bottom = vp.rect.y + vp.rect.height;
                if bottom < surface_h && !layout.viewports.iter().any(|v| v.rect.y == bottom) {
                    split_lines.extend(border_lines(0, bottom.saturating_sub(1), surface_w, 4));
                }
            }
        }

        // Active viewport highlight
        if let Some(active) = layout.active() {
            active_lines.extend(border_lines(
                active.rect.x, active.rect.y,
                active.rect.width, active.rect.height,
            ));
        }

        ViewportBorderGeometry { split_lines, active_lines }
    }
}

/// Encode viewport split borders and active-viewport highlight.
pub(crate) fn encode_viewport_borders(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    ew: u32,
    eh: u32,
) {
    let geom = ViewportBorderGeometry::build(
        &renderer.frame.viewport_layout,
        ew,
        eh,
    );
    let has_splits = !geom.split_lines.is_empty();
    let has_active = !geom.active_lines.is_empty();
    if !has_splits && !has_active {
        return;
    }
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Viewport Borders"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None, multiview_mask: None,
    });
    pass.set_pipeline(&renderer.gpu.pipelines.viewport_border_lines);
    let wpx = ew as f32;
    let hpx = eh as f32;
    let screen_mvp = Mat4::orthographic_rh_gl(0.0, wpx, hpx, 0.0, -1.0, 1.0).to_cols_array_2d();
    // Split borders
    if has_splits {
        let uniforms = FlatUniforms {
            mvp: screen_mvp,
            color: [0.4, 0.4, 0.4, 1.0],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            for chunk in geom.split_lines.chunks(2) {
                if chunk.len() < 2 { break; }
                let verts = [chunk[0], chunk[1]];
                let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("split border vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(0..2, 0..1);
            }
        }
    }
    // Active viewport highlight (hidden for single full-window viewport — no editor benefit).
    if has_active && renderer.frame.viewport_layout.layout_mode != LayoutMode::Single {
        let uniforms = FlatUniforms {
            mvp: screen_mvp,
            color: [1.0, 0.85, 0.1, 1.0],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            for chunk in geom.active_lines.chunks(2) {
                if chunk.len() < 2 { break; }
                let verts = [chunk[0], chunk[1]];
                let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("active border vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(0..2, 0..1);
            }
        }
    }
}
