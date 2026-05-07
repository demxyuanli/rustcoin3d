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
            start, end, offset_dir, extension_len, arrow_size, ..
        } => {
            let dl_start = [start[0] + offset_dir[0] * extension_len, start[1] + offset_dir[1] * extension_len];
            let dl_end = [end[0] + offset_dir[0] * extension_len, end[1] + offset_dir[1] * extension_len];
            // Dimension line
            out.push(LineVertex { position: [dl_start[0], dl_start[1], 0.0] });
            out.push(LineVertex { position: [dl_end[0], dl_end[1], 0.0] });
            // Extension lines
            out.push(LineVertex { position: [start[0], start[1], 0.0] });
            out.push(LineVertex { position: [dl_start[0], dl_start[1], 0.0] });
            out.push(LineVertex { position: [end[0], end[1], 0.0] });
            out.push(LineVertex { position: [dl_end[0], dl_end[1], 0.0] });
            // Arrowheads
            let dir = [(end[0] - start[0]), (end[1] - start[1])];
            let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(0.001);
            let ndir = [dir[0] / len, dir[1] / len];
            let perp = [-ndir[1], ndir[0]];
            let asz = *arrow_size;
            // Start arrowhead
            let _p0 = [dl_start[0] + ndir[0] * asz, dl_start[1] + ndir[1] * asz];
            let p1 = [dl_start[0] + perp[0] * asz * 0.5, dl_start[1] + perp[1] * asz * 0.5];
            let p2 = [dl_start[0] - perp[0] * asz * 0.5, dl_start[1] - perp[1] * asz * 0.5];
            out.push(LineVertex { position: [dl_start[0], dl_start[1], 0.0] });
            out.push(LineVertex { position: [p1[0], p1[1], 0.0] });
            out.push(LineVertex { position: [dl_start[0], dl_start[1], 0.0] });
            out.push(LineVertex { position: [p2[0], p2[1], 0.0] });
            // End arrowhead
            let _e0 = [dl_end[0] - ndir[0] * asz, dl_end[1] - ndir[1] * asz];
            let e1 = [dl_end[0] + perp[0] * asz * 0.5, dl_end[1] + perp[1] * asz * 0.5];
            let e2 = [dl_end[0] - perp[0] * asz * 0.5, dl_end[1] - perp[1] * asz * 0.5];
            out.push(LineVertex { position: [dl_end[0], dl_end[1], 0.0] });
            out.push(LineVertex { position: [e1[0], e1[1], 0.0] });
            out.push(LineVertex { position: [dl_end[0], dl_end[1], 0.0] });
            out.push(LineVertex { position: [e2[0], e2[1], 0.0] });
        }
        MarkupElement::AngleDimension { center, arm1, arm2, radius, .. } => {
            // Arc from arm1 to arm2
            let a1 = (arm1[1] - center[1]).atan2(arm1[0] - center[0]);
            let a2 = (arm2[1] - center[1]).atan2(arm2[0] - center[0]);
            let n = 32usize;
            let span = a2 - a1;
            let mut prev = [center[0] + radius * a1.cos(), center[1] + radius * a1.sin()];
            for i in 1..=n {
                let t = i as f32 / n as f32;
                let angle = a1 + span * t;
                let curr = [center[0] + radius * angle.cos(), center[1] + radius * angle.sin()];
                out.push(LineVertex { position: [prev[0], prev[1], 0.0] });
                out.push(LineVertex { position: [curr[0], curr[1], 0.0] });
                prev = curr;
            }
            // Arms to center
            out.push(LineVertex { position: [center[0], center[1], 0.0] });
            out.push(LineVertex { position: [arm1[0], arm1[1], 0.0] });
            out.push(LineVertex { position: [center[0], center[1], 0.0] });
            out.push(LineVertex { position: [arm2[0], arm2[1], 0.0] });
        }
        MarkupElement::RadialDimension { center, perimeter, .. } => {
            out.push(LineVertex { position: [center[0], center[1], 0.0] });
            out.push(LineVertex { position: [perimeter[0], perimeter[1], 0.0] });
        }
        MarkupElement::DiameterDimension { p1, p2, center, .. } => {
            out.push(LineVertex { position: [p1[0], p1[1], 0.0] });
            out.push(LineVertex { position: [p2[0], p2[1], 0.0] });
            // Cross mark at center
            let s = 3.0;
            out.push(LineVertex { position: [center[0] - s, center[1] - s, 0.0] });
            out.push(LineVertex { position: [center[0] + s, center[1] + s, 0.0] });
            out.push(LineVertex { position: [center[0] + s, center[1] - s, 0.0] });
            out.push(LineVertex { position: [center[0] - s, center[1] + s, 0.0] });
        }
        MarkupElement::Leader { anchor, label_pos, .. } => {
            out.push(LineVertex { position: [anchor[0], anchor[1], 0.0] });
            out.push(LineVertex { position: [label_pos[0], label_pos[1], 0.0] });
            // Small dot at anchor
            let r = 2.0;
            let n = 8;
            let mut prev = [anchor[0] + r, anchor[1]];
            for i in 1..=n {
                let a = (i as f32 / n as f32) * std::f32::consts::TAU;
                let curr = [anchor[0] + r * a.cos(), anchor[1] + r * a.sin()];
                out.push(LineVertex { position: [prev[0], prev[1], 0.0] });
                out.push(LineVertex { position: [curr[0], curr[1], 0.0] });
                prev = curr;
            }
        }
        MarkupElement::Callout { anchor, label_pos, radius, .. } => {
            // Leader line
            out.push(LineVertex { position: [anchor[0], anchor[1], 0.0] });
            out.push(LineVertex { position: [label_pos[0], label_pos[1], 0.0] });
            // Filled circle (dot) at anchor
            let n = 8;
            let sr = 2.0;
            let mut prev = [anchor[0] + sr, anchor[1]];
            for i in 1..=n {
                let a = (i as f32 / n as f32) * std::f32::consts::TAU;
                let curr = [anchor[0] + sr * a.cos(), anchor[1] + sr * a.sin()];
                out.push(LineVertex { position: [prev[0], prev[1], 0.0] });
                out.push(LineVertex { position: [curr[0], curr[1], 0.0] });
                prev = curr;
            }
            // Circle bubble at label_pos
            let n_seg = 32usize;
            let mut prev_b = [label_pos[0] + radius, label_pos[1]];
            for i in 1..=n_seg {
                let a = (i as f32 / n_seg as f32) * std::f32::consts::TAU;
                let curr = [label_pos[0] + radius * a.cos(), label_pos[1] + radius * a.sin()];
                out.push(LineVertex { position: [prev_b[0], prev_b[1], 0.0] });
                out.push(LineVertex { position: [curr[0], curr[1], 0.0] });
                prev_b = curr;
            }
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
    if renderer.frame.markup_vertices.is_empty() {
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

    pass.set_pipeline(&renderer.gpu.pipelines.markup_lines);

    let ident = glam::Mat4::IDENTITY.to_cols_array_2d();
    let uniforms = crate::vertex::FlatUniforms {
        mvp: ident,
        color: [1.0, 0.0, 0.0, 0.8],
    };

    if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
        let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("markup vb"),
            contents: bytemuck::cast_slice(&renderer.frame.markup_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.draw(0..renderer.frame.markup_vertices.len() as u32, 0..1);
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
    fn test_text_element_no_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Text {
                position: [0.0, 0.0],
                string: "hello".into(),
                size: 14.0,
                color: [1.0, 1.0, 1.0, 1.0],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.is_empty());
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
