//! Markup overlay rendering.
//!
//! - Legacy `MarkupNode` elements: 2D screen-pixel lines.
//! - `AnnotationSet` elements: model-local 3D geometry projected with the scene camera (NDC).

pub(crate) mod projection;
mod primitives;

use crate::vertex::MarkupVertex;
use rc3d_core::NodeId;
use rc3d_scene::node_data::{MarkupElement, NodeData};
use rc3d_scene::SceneGraph;
use super::pass_effects::EffectCommands;
use wgpu::util::DeviceExt;

use projection::screen_space_ortho;
use primitives::project_annotation_elements;

/// Flatten all visible markup elements from the scene graph into colored line vertices.
pub fn collect_markup_lines(
    graph: &SceneGraph,
    root: NodeId,
    _surface_w: u32,
    _surface_h: u32,
) -> Vec<MarkupVertex> {
    let mut vertices = Vec::new();
    collect_recursive(graph, root, &mut vertices);
    vertices
}

/// Collect all MarkupElement::Text strings from the scene graph for HUD overlay display.
pub fn collect_markup_text_lines(graph: &SceneGraph) -> Vec<String> {
    let mut lines = Vec::new();
    for &root in graph.roots() {
        collect_text_recursive(graph, root, &mut lines);
    }
    lines
}

fn collect_text_recursive(graph: &SceneGraph, node: NodeId, out: &mut Vec<String>) {
    let Some(entry) = graph.get(node) else { return };
    if let NodeData::Markup(m) = &entry.data {
        if m.visible {
            for el in &m.elements {
                if let MarkupElement::Text { string, position, .. } = el {
                    out.push(format!("{:>4.0},{:<4.0} {}", position[0], position[1], string));
                }
            }
        }
    }
    for &child in &entry.children {
        collect_text_recursive(graph, child, out);
    }
}

fn collect_recursive(graph: &SceneGraph, node: NodeId, out: &mut Vec<MarkupVertex>) {
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

fn push_element_vertices(el: &MarkupElement, out: &mut Vec<MarkupVertex>) {
    match *el {
        MarkupElement::Line { start, end, color, .. } => {
            out.push(MarkupVertex { position: [start[0], start[1], 0.0], color });
            out.push(MarkupVertex { position: [end[0], end[1], 0.0], color });
        }
        MarkupElement::Rect {
            origin, size, color, ..
        } => {
            let x0 = origin[0];
            let y0 = origin[1];
            let x1 = x0 + size[0];
            let y1 = y0 + size[1];
            let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
            for i in 0..4 {
                out.push(MarkupVertex { position: [corners[i][0], corners[i][1], 0.0], color });
                out.push(MarkupVertex { position: [corners[(i + 1) % 4][0], corners[(i + 1) % 4][1], 0.0], color });
            }
        }
        MarkupElement::Circle { center, radius, color, .. } => {
            let n_segments = 64usize;
            let mut prev = [center[0] + radius, center[1]];
            for i in 1..=n_segments {
                let angle = (i as f32 / n_segments as f32) * std::f32::consts::TAU;
                let curr = [center[0] + radius * angle.cos(), center[1] + radius * angle.sin()];
                out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev = curr;
            }
        }
        MarkupElement::Freehand { ref points, color, .. } => {
            for w in points.windows(2) {
                out.push(MarkupVertex { position: [w[0][0], w[0][1], 0.0], color });
                out.push(MarkupVertex { position: [w[1][0], w[1][1], 0.0], color });
            }
        }
        MarkupElement::Dimension {
            start, end, offset_dir, extension_len, arrow_size, color, ..
        } => {
            let dl_start = [start[0] + offset_dir[0] * extension_len, start[1] + offset_dir[1] * extension_len];
            let dl_end = [end[0] + offset_dir[0] * extension_len, end[1] + offset_dir[1] * extension_len];
            // Dimension line
            out.push(MarkupVertex { position: [dl_start[0], dl_start[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_end[0], dl_end[1], 0.0], color });
            // Extension lines
            out.push(MarkupVertex { position: [start[0], start[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_start[0], dl_start[1], 0.0], color });
            out.push(MarkupVertex { position: [end[0], end[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_end[0], dl_end[1], 0.0], color });
            // Arrowheads
            let dir = [(end[0] - start[0]), (end[1] - start[1])];
            let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(0.001);
            let ndir = [dir[0] / len, dir[1] / len];
            let perp = [-ndir[1], ndir[0]];
            let asz = arrow_size;
            // Start arrowhead
            let p1 = [dl_start[0] + perp[0] * asz * 0.5, dl_start[1] + perp[1] * asz * 0.5];
            let p2 = [dl_start[0] - perp[0] * asz * 0.5, dl_start[1] - perp[1] * asz * 0.5];
            out.push(MarkupVertex { position: [dl_start[0], dl_start[1], 0.0], color });
            out.push(MarkupVertex { position: [p1[0], p1[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_start[0], dl_start[1], 0.0], color });
            out.push(MarkupVertex { position: [p2[0], p2[1], 0.0], color });
            // End arrowhead
            let e1 = [dl_end[0] + perp[0] * asz * 0.5, dl_end[1] + perp[1] * asz * 0.5];
            let e2 = [dl_end[0] - perp[0] * asz * 0.5, dl_end[1] - perp[1] * asz * 0.5];
            out.push(MarkupVertex { position: [dl_end[0], dl_end[1], 0.0], color });
            out.push(MarkupVertex { position: [e1[0], e1[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_end[0], dl_end[1], 0.0], color });
            out.push(MarkupVertex { position: [e2[0], e2[1], 0.0], color });
        }
        MarkupElement::AngleDimension { center, arm1, arm2, radius, color, .. } => {
            let a1 = (arm1[1] - center[1]).atan2(arm1[0] - center[0]);
            let a2 = (arm2[1] - center[1]).atan2(arm2[0] - center[0]);
            let n = 32usize;
            let span = a2 - a1;
            let mut prev = [center[0] + radius * a1.cos(), center[1] + radius * a1.sin()];
            for i in 1..=n {
                let t = i as f32 / n as f32;
                let angle = a1 + span * t;
                let curr = [center[0] + radius * angle.cos(), center[1] + radius * angle.sin()];
                out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev = curr;
            }
            // Arms to center
            out.push(MarkupVertex { position: [center[0], center[1], 0.0], color });
            out.push(MarkupVertex { position: [arm1[0], arm1[1], 0.0], color });
            out.push(MarkupVertex { position: [center[0], center[1], 0.0], color });
            out.push(MarkupVertex { position: [arm2[0], arm2[1], 0.0], color });
        }
        MarkupElement::RadialDimension { center, perimeter, color, .. } => {
            out.push(MarkupVertex { position: [center[0], center[1], 0.0], color });
            out.push(MarkupVertex { position: [perimeter[0], perimeter[1], 0.0], color });
        }
        MarkupElement::DiameterDimension { p1, p2, center, color, .. } => {
            out.push(MarkupVertex { position: [p1[0], p1[1], 0.0], color });
            out.push(MarkupVertex { position: [p2[0], p2[1], 0.0], color });
            let s = 3.0;
            out.push(MarkupVertex { position: [center[0] - s, center[1] - s, 0.0], color });
            out.push(MarkupVertex { position: [center[0] + s, center[1] + s, 0.0], color });
            out.push(MarkupVertex { position: [center[0] + s, center[1] - s, 0.0], color });
            out.push(MarkupVertex { position: [center[0] - s, center[1] + s, 0.0], color });
        }
        MarkupElement::Leader { anchor, label_pos, color, .. } => {
            out.push(MarkupVertex { position: [anchor[0], anchor[1], 0.0], color });
            out.push(MarkupVertex { position: [label_pos[0], label_pos[1], 0.0], color });
            let r = 2.0;
            let n = 8;
            let mut prev = [anchor[0] + r, anchor[1]];
            for i in 1..=n {
                let a = (i as f32 / n as f32) * std::f32::consts::TAU;
                let curr = [anchor[0] + r * a.cos(), anchor[1] + r * a.sin()];
                out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev = curr;
            }
        }
        MarkupElement::Callout { anchor, label_pos, radius, color, .. } => {
            out.push(MarkupVertex { position: [anchor[0], anchor[1], 0.0], color });
            out.push(MarkupVertex { position: [label_pos[0], label_pos[1], 0.0], color });
            let n = 8;
            let sr = 2.0;
            let mut prev = [anchor[0] + sr, anchor[1]];
            for i in 1..=n {
                let a = (i as f32 / n as f32) * std::f32::consts::TAU;
                let curr = [anchor[0] + sr * a.cos(), anchor[1] + sr * a.sin()];
                out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev = curr;
            }
            let n_seg = 32usize;
            let mut prev_b = [label_pos[0] + radius, label_pos[1]];
            for i in 1..=n_seg {
                let a = (i as f32 / n_seg as f32) * std::f32::consts::TAU;
                let curr = [label_pos[0] + radius * a.cos(), label_pos[1] + radius * a.sin()];
                out.push(MarkupVertex { position: [prev_b[0], prev_b[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev_b = curr;
            }
        }
        MarkupElement::Text { position, color, .. } => {
            // Render a small underline marker at the text position so the
            // label is visible even without the glyphon text layer.
            let w = 8.0;
            out.push(MarkupVertex { position: [position[0], position[1], 0.0], color });
            out.push(MarkupVertex { position: [position[0] + w, position[1], 0.0], color });
        }
    }
}

/// Compute projected render data for annotations (CPU side, can be cached).
/// Returns (projected_markup_vertices, world_labels) for the GPU pass.
pub fn compute_projected_markup(
    renderer: &mut crate::renderer::Renderer,
    effect_commands: &EffectCommands,
    scene_vp: glam::Mat4,
    surface_w: f32,
    surface_h: f32,
    depth_reversed_z: bool,
) -> (Vec<MarkupVertex>, Vec<crate::world_label::WorldLabelCommand>) {
    let mut world_labels = std::mem::take(&mut renderer.frame.annotation_world_labels);
    let projected = project_annotation_elements(
        &effect_commands.annotation_elements,
        scene_vp,
        surface_w,
        surface_h,
        depth_reversed_z,
        renderer.frame.scene_camera_pos,
        &mut world_labels,
    );
    (projected, world_labels)
}

/// Render markup lines; 3D annotations depth-test against the scene depth buffer.
pub fn pass_markup(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    surface_w: u32,
    surface_h: u32,
    scene_vp: glam::Mat4,
    depth_reversed_z: bool,
    projected: &[MarkupVertex],
    world_labels: &[crate::world_label::WorldLabelCommand],
) {
    // ── Combine 2D screen-space markup + projected 3D annotations ──
    let legacy = &renderer.frame.markup_vertices;
    if projected.is_empty() && legacy.is_empty() && world_labels.is_empty() {
        return;
    }

    // 3D annotations: NDC lines + world-space label quads, depth-tested against solid pass.
    if !projected.is_empty() || !world_labels.is_empty() {
        let markup_pl = if depth_reversed_z {
            &renderer.gpu.pipelines.markup_lines_reverse
        } else {
            &renderer.gpu.pipelines.markup_lines_forward
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Markup 3D"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
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
            occlusion_query_set: None,
        });
        pass.set_pipeline(markup_pl);
        let identity = glam::Mat4::IDENTITY.to_cols_array_2d();
        let uniforms = crate::vertex::FlatUniforms {
            mvp: identity,
            color: [0.0; 4],
            model: identity,
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
        };
        if !projected.is_empty() {
            if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
                let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("annotation markup vb"),
                    contents: bytemuck::cast_slice(projected),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(0..projected.len() as u32, 0..1);
            }
        }
        if !world_labels.is_empty() {
            let device = &renderer.device;
            let queue = &renderer.queue;
            let flat_pool = &mut renderer.gpu.flat_pool;
            let pipelines = &renderer.gpu.pipelines;
            let font = &mut renderer.gpu.world_label_font;
            let label_attrs = font.label_attrs();
            crate::world_label::draw_world_labels(
                device,
                queue,
                flat_pool,
                pipelines,
                &mut pass,
                world_labels,
                scene_vp,
                surface_w as f32,
                surface_h as f32,
                depth_reversed_z,
                renderer.frame.scene_camera_pos,
                &mut font.font_system,
                &mut font.swash_cache,
                label_attrs,
            );
        }
    }

    // Legacy 2D MarkupNode overlay (screen pixels, no depth).
    if !legacy.is_empty() {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Markup Screen"),
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
        pass.set_pipeline(&renderer.gpu.pipelines.markup_lines_screen);
        let mvp = screen_space_ortho(surface_w as f32, surface_h as f32).to_cols_array_2d();
        let identity = glam::Mat4::IDENTITY.to_cols_array_2d();
        let uniforms = crate::vertex::FlatUniforms {
            mvp,
            color: [0.0; 4],
            model: identity,
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("legacy markup vb"),
                contents: bytemuck::cast_slice(legacy),
                usage: wgpu::BufferUsages::VERTEX,
            });
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.draw(0..legacy.len() as u32, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_scene::node_data::{MarkupElement, MarkupNode, NodeData};
    use rc3d_scene::SceneGraph;

    fn make_markup_graph(elements: Vec<MarkupElement>, visible: bool) -> (SceneGraph, NodeId) {
        let mut g = SceneGraph::new();
        let markup = MarkupNode {
            elements,
            visible,
            ..Default::default()
        };
        let id = g.add_root(NodeData::Markup(markup));
        (g, id)
    }

    // ── Empty / hidden ──

    #[test]
    fn test_empty_scene_yields_no_vertices() {
        let g = SceneGraph::new();
        let v = collect_markup_lines(&g, NodeId::default(), 800, 600);
        assert!(v.is_empty());
    }

    #[test]
    fn test_hidden_markup_yields_no_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Line {
                start: [0.0, 0.0],
                end: [10.0, 10.0],
                color: [1.0, 0.0, 0.0, 1.0],
                width: 1.0,
            }],
            false,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.is_empty());
    }

    // ── Line ──

    #[test]
    fn test_line_element_produces_two_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Line {
                start: [10.0, 20.0],
                end: [30.0, 40.0],
                color: [1.0, 0.0, 0.0, 0.8],
                width: 2.0,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].position, [10.0, 20.0, 0.0]);
        assert_eq!(v[1].position, [30.0, 40.0, 0.0]);
    }

    // ── Rect ──

    #[test]
    fn test_rect_element_produces_eight_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Rect {
                origin: [0.0, 0.0],
                size: [100.0, 50.0],
                color: [1.0, 0.0, 0.0, 0.6],
                filled: false,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // 4 edges × 2 vertices = 8
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_rect_covers_correct_corners() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Rect {
                origin: [50.0, 50.0],
                size: [100.0, 100.0],
                color: [1.0, 0.0, 0.0, 0.6],
                filled: false,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // Should have corners at (50,50), (150,50), (150,150), (50,150)
        let positions: Vec<[f32; 2]> = v.iter().map(|lv| [lv.position[0], lv.position[1]]).collect();
        assert!(positions.contains(&[50.0, 50.0]));
        assert!(positions.contains(&[150.0, 50.0]));
        assert!(positions.contains(&[150.0, 150.0]));
        assert!(positions.contains(&[50.0, 150.0]));
    }

    // ── Circle ──

    #[test]
    fn test_circle_produces_64_segment_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Circle {
                center: [400.0, 300.0],
                radius: 50.0,
                color: [1.0, 0.0, 0.0, 0.6],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // 64 segments × 2 = 128 vertices
        assert_eq!(v.len(), 128);
    }

    #[test]
    fn test_circle_vertices_near_center_radius() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Circle {
                center: [0.0, 0.0],
                radius: 10.0,
                color: [1.0, 0.0, 0.0, 0.6],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        for lv in &v {
            let dist = (lv.position[0].powi(2) + lv.position[1].powi(2)).sqrt();
            assert!((dist - 10.0).abs() < 0.2, "distance {} not ≈ 10", dist);
        }
    }

    // ── Freehand ──

    #[test]
    fn test_freehand_two_points_produces_two_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Freehand {
                points: vec![[0.0, 0.0], [10.0, 10.0]],
                color: [0.0, 1.0, 0.0, 0.8],
                width: 2.0,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_freehand_polyline_connects_points() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Freehand {
                points: vec![[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]],
                color: [0.0, 1.0, 0.0, 0.8],
                width: 2.0,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // 2 segments * 2 = 4 vertices
        assert_eq!(v.len(), 4);
        // First segment
        assert_eq!(v[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(v[1].position, [5.0, 5.0, 0.0]);
        // Second segment
        assert_eq!(v[2].position, [5.0, 5.0, 0.0]);
        assert_eq!(v[3].position, [10.0, 0.0, 0.0]);
    }

    #[test]
    fn test_freehand_single_point_no_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Freehand {
                points: vec![[0.0, 0.0]],
                color: [0.0, 1.0, 0.0, 0.8],
                width: 2.0,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // windows(2) on a single element gives empty iterator
        assert!(v.is_empty());
    }

    // ── Dimension ──

    #[test]
    fn test_dimension_produces_six_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Dimension {
                start: [10.0, 10.0],
                end: [110.0, 10.0],
                offset_dir: [0.0, -1.0],
                extension_len: 20.0,
                arrow_size: 5.0,
                label: String::new(),
                color: [1.0, 1.0, 0.0, 1.0],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // 3 extension/dim lines + 4 arrow lines = 14 vertices
        assert!(v.len() >= 14, "expected >=14 vertices, got {}", v.len());
    }

    // ── Text (deferred) ──

    #[test]
    fn test_text_element_produces_underline_marker() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Text {
                position: [100.0, 200.0],
                string: "hello".into(),
                size: 14.0,
                color: [1.0, 1.0, 1.0, 1.0],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // Text now renders an underline marker: 2 vertices
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].position, [100.0, 200.0, 0.0]);
        assert_eq!(v[1].position, [108.0, 200.0, 0.0]);
    }

    // ── Multiple elements ──

    // ── Angle Dimension ──

    #[test]
    fn test_angle_dimension_produces_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::AngleDimension {
                center: [100.0, 100.0],
                arm1: [150.0, 100.0],
                arm2: [100.0, 50.0],
                radius: 40.0,
                color: [1.0, 1.0, 0.0, 1.0],
                label: "45°".into(),
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.len() > 0, "angle dimension should produce vertices");
    }

    #[test]
    fn test_radial_dimension_produces_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::RadialDimension {
                center: [100.0, 100.0],
                perimeter: [150.0, 100.0],
                color: [1.0, 1.0, 0.0, 1.0],
                label: "R=50".into(),
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_diameter_dimension_produces_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::DiameterDimension {
                p1: [50.0, 100.0],
                p2: [150.0, 100.0],
                center: [100.0, 100.0],
                color: [1.0, 1.0, 0.0, 1.0],
                label: "D=100".into(),
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.len() >= 6, "diameter dimension should have main line + cross mark");
    }

    #[test]
    fn test_leader_produces_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Leader {
                anchor: [50.0, 50.0],
                label_pos: [150.0, 100.0],
                text: "note".into(),
                color: [1.0, 1.0, 0.0, 1.0],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.len() >= 2, "leader line + dot should produce vertices");
    }

    #[test]
    fn test_multiple_elements_combined() {
        let (g, root) = make_markup_graph(
            vec![
                MarkupElement::Line {
                    start: [0.0, 0.0],
                    end: [10.0, 10.0],
                    color: [1.0, 0.0, 0.0, 0.8],
                    width: 1.0,
                },
                MarkupElement::Line {
                    start: [10.0, 10.0],
                    end: [20.0, 20.0],
                    color: [1.0, 0.0, 0.0, 0.8],
                    width: 1.0,
                },
            ],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert_eq!(v.len(), 4);
    }
}
