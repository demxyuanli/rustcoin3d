//! Screen-space markup overlay rendering.
//!
//! Collects all MarkupNodes from the scene graph and renders their elements
//! as screen-space line geometry with depth_compare: Always (overlay).

use wgpu::util::DeviceExt;

use crate::vertex::LineVertex;
use rc3d_core::NodeId;
use rc3d_scene::node_data::{MarkupElement, NodeData};
use rc3d_scene::SceneGraph;

/// Flatten all visible markup elements from the scene graph into line vertices.
pub fn collect_markup_lines(
    graph: &SceneGraph,
    root: NodeId,
    _surface_w: u32,
    _surface_h: u32,
) -> Vec<LineVertex> {
    let mut vertices = Vec::new();
    collect_recursive(graph, root, &mut vertices);
    vertices
}

fn collect_recursive(graph: &SceneGraph, node: NodeId, out: &mut Vec<LineVertex>) {
    let Some(entry) = graph.get(node) else { return };

    if let NodeData::Markup(m) = &entry.data {
        if m.visible {
            for el in &m.elements {
                push_element_vertices(el, out);
            }
        }
    }

    for &child in &entry.children {
        collect_recursive(graph, child, out);
    }
}

fn push_element_vertices(el: &MarkupElement, out: &mut Vec<LineVertex>) {
    match el {
        MarkupElement::Line { start, end, .. } => {
            out.push(LineVertex { position: [start[0], start[1], 0.0] });
            out.push(LineVertex { position: [end[0], end[1], 0.0] });
        }
        MarkupElement::Rect {
            origin, size, ..
        } => {
            let x0 = origin[0];
            let y0 = origin[1];
            let x1 = x0 + size[0];
            let y1 = y0 + size[1];
            let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
            for i in 0..4 {
                out.push(LineVertex {
                    position: [corners[i][0], corners[i][1], 0.0],
                });
                out.push(LineVertex {
                    position: [corners[(i + 1) % 4][0], corners[(i + 1) % 4][1], 0.0],
                });
            }
        }
        MarkupElement::Circle { center, radius, .. } => {
            let n_segments = 64usize;
            let mut prev = [
                center[0] + radius,
                center[1],
            ];
            for i in 1..=n_segments {
                let angle = (i as f32 / n_segments as f32) * std::f32::consts::TAU;
                let curr = [
                    center[0] + radius * angle.cos(),
                    center[1] + radius * angle.sin(),
                ];
                out.push(LineVertex { position: [prev[0], prev[1], 0.0] });
                out.push(LineVertex { position: [curr[0], curr[1], 0.0] });
                prev = curr;
            }
        }
        MarkupElement::Freehand { points, .. } => {
            for w in points.windows(2) {
                out.push(LineVertex { position: [w[0][0], w[0][1], 0.0] });
                out.push(LineVertex { position: [w[1][0], w[1][1], 0.0] });
            }
        }
        MarkupElement::Dimension {
            start,
            end,
            offset_dir,
            extension_len,
            ..
        } => {
            let dl_start = [
                start[0] + offset_dir[0] * extension_len,
                start[1] + offset_dir[1] * extension_len,
            ];
            let dl_end = [
                end[0] + offset_dir[0] * extension_len,
                end[1] + offset_dir[1] * extension_len,
            ];
            out.push(LineVertex { position: [dl_start[0], dl_start[1], 0.0] });
            out.push(LineVertex { position: [dl_end[0], dl_end[1], 0.0] });
            out.push(LineVertex { position: [start[0], start[1], 0.0] });
            out.push(LineVertex { position: [dl_start[0], dl_start[1], 0.0] });
            out.push(LineVertex { position: [end[0], end[1], 0.0] });
            out.push(LineVertex { position: [dl_end[0], dl_end[1], 0.0] });
        }
        MarkupElement::Text { .. } => {
            // Text elements are deferred to HUD/glyphon layer.
        }
    }
}

/// Render markup lines as an overlay.
pub fn pass_markup(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) {
    if renderer.markup_vertices.is_empty() {
        return;
    }

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Markup Overlay"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    // Reuse viewport_border_lines pipeline (same config: no depth, alpha blend, line list)
    pass.set_pipeline(&renderer.pipelines.viewport_border_lines);

    let ident = glam::Mat4::IDENTITY.to_cols_array_2d();
    let uniforms = crate::vertex::FlatUniforms {
        mvp: ident,
        color: [1.0, 0.0, 0.0, 0.8],
    };

    if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
        let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("markup vb"),
            contents: bytemuck::cast_slice(&renderer.markup_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.draw(0..renderer.markup_vertices.len() as u32, 0..1);
    }
}
