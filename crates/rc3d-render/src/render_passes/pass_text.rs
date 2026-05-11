//! Text rendering pass for Text2/Text3 scene-graph nodes.
//! Collects text draw commands for rendering via the existing HUD glyphon system.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// A text draw command ready for rendering.
pub(crate) struct TextDrawCommand {
    pub(crate) string: String,
    pub(crate) screen_pos: [f32; 2],
    pub(crate) size: f32,
    pub(crate) color: [f32; 4],
    pub(crate) is_3d: bool,
}

/// Overlay lines (no positioning — rendered as HUD block) + positioned text entries.
pub(crate) struct TextCollection {
    pub overlay_lines: Vec<String>,
    pub positioned: Vec<TextDrawCommand>,
}

/// Collect all Text2/Text3 draw commands from the scene graph.
pub fn collect_text_nodes(
    graph: &SceneGraph,
    view_proj: Option<Mat4>,
    viewport: Option<(u32, u32)>,
    camera_pos: Option<Vec3>,
) -> TextCollection {
    let mut overlay_lines = Vec::new();
    let mut positioned = Vec::new();
    for &root in graph.roots() {
        collect_recursive(
            graph, root, Mat4::IDENTITY,
            view_proj, viewport, camera_pos,
            false,
            &mut overlay_lines, &mut positioned,
        );
    }
    TextCollection { overlay_lines, positioned }
}

fn collect_recursive(
    graph: &SceneGraph,
    node: NodeId,
    model: Mat4,
    view_proj: Option<Mat4>,
    viewport: Option<(u32, u32)>,
    camera_pos: Option<Vec3>,
    inside_billboard: bool,
    overlay_lines: &mut Vec<String>,
    positioned: &mut Vec<TextDrawCommand>,
) {
    let Some(entry) = graph.get(node) else { return };
    match &entry.data {
        NodeData::Text2(t) => {
            if inside_billboard {
                if let (Some(vp), Some((vw, vh))) = (view_proj, viewport) {
                    let world = model.transform_point3(Vec3::new(t.position[0], t.position[1], 0.0));
                    let clip = vp.project_point3(world);
                    // Skip if behind camera
                    if clip.z > 1.0 || clip.z < -1.0 {
                        return;
                    }
                    let sx = (clip.x * 0.5 + 0.5) * vw as f32;
                    let sy = (1.0 - (clip.y * 0.5 + 0.5)) * vh as f32;
                    positioned.push(TextDrawCommand {
                        string: t.string.clone(),
                        screen_pos: [sx, sy],
                        size: t.size,
                        color: t.color,
                        is_3d: false,
                    });
                }
            } else {
                overlay_lines.push(t.string.clone());
            }
        }
        NodeData::Text3(t) => {
            let world_pos = model.transform_point3(t.position);
            if let (Some(vp), Some((vw, vh))) = (view_proj, viewport) {
                let clip = vp.project_point3(world_pos);
                if clip.z > 1.0 || clip.z < -1.0 {
                    return;
                }
                let sx = (clip.x * 0.5 + 0.5) * vw as f32;
                let sy = (1.0 - (clip.y * 0.5 + 0.5)) * vh as f32;
                positioned.push(TextDrawCommand {
                    string: t.string.clone(),
                    screen_pos: [sx, sy],
                    size: t.size,
                    color: t.color,
                    is_3d: true,
                });
            }
        }
        NodeData::Billboard(b) => {
            let billboard_pos = model.w_axis.truncate();
            let facing = if let Some(cam) = camera_pos {
                compute_billboard_facing(b.axis_aligned, billboard_pos, cam)
            } else {
                Mat4::IDENTITY
            };
            let new_model = model * facing;
            for &child in &entry.children {
                collect_recursive(
                    graph, child, new_model,
                    view_proj, viewport, camera_pos,
                    true,
                    overlay_lines, positioned,
                );
            }
        }
        NodeData::Transform(t) => {
            let m = model * t.to_matrix();
            for &child in &entry.children {
                collect_recursive(
                    graph, child, m,
                    view_proj, viewport, camera_pos,
                    inside_billboard,
                    overlay_lines, positioned,
                );
            }
        }
        NodeData::Separator(_)
        | NodeData::Group(_) | NodeData::Environment(_) | NodeData::ShapeHints(_) | NodeData::Annotation(_) | NodeData::ResetTransform(_) | NodeData::Texture2Transform(_) | NodeData::MaterialBinding(_) | NodeData::IndexedLineSet(_) | NodeData::File(_) | NodeData::Decal(_) | NodeData::ExplodedView(_) | NodeData::ReflectionPlane(_)
        | NodeData::Lod(_)
        | NodeData::EventCallback(_)
        | NodeData::SectionPlane(_)
        | NodeData::Switch(_)
        | NodeData::MultipleCopy(_)
        | NodeData::HandlerNode(_) => {
            for &child in &entry.children {
                collect_recursive(
                    graph, child, model,
                    view_proj, viewport, camera_pos,
                    inside_billboard,
                    overlay_lines, positioned,
                );
            }
        }
        _ => {
            for &child in &entry.children {
                collect_recursive(
                    graph, child, model,
                    view_proj, viewport, camera_pos,
                    inside_billboard,
                    overlay_lines, positioned,
                );
            }
        }
    }
}

fn compute_billboard_facing(axis_aligned: bool, billboard_pos: Vec3, camera_pos: Vec3) -> Mat4 {
    let dir_to_cam = (camera_pos - billboard_pos).normalize();
    if axis_aligned {
        let fwd = Vec3::new(dir_to_cam.x, 0.0, dir_to_cam.z).normalize();
        let fwd = if fwd.length_squared() < 1e-6 { Vec3::NEG_Z } else { fwd };
        Mat4::look_at_rh(Vec3::ZERO, fwd, Vec3::Y)
    } else {
        let right = Vec3::Y.cross(dir_to_cam).normalize();
        let right = if right.length_squared() < 1e-6 { Vec3::X } else { right };
        let up = dir_to_cam.cross(right).normalize();
        Mat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            (-dir_to_cam).extend(0.0),
            Vec3::ZERO.extend(1.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;
    use rc3d_scene::{GroupNode, NodeData, SceneGraph, Text2Node, Text3Node, TransformNode};

    #[test]
    fn collect_text2_nodes_preserves_screen_command_fields() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Group(GroupNode));
        graph.add_child(
            root,
            NodeData::Text2(Text2Node {
                string: "screen label".to_string(),
                position: [12.0, 34.0],
                size: 18.0,
                color: [1.0, 0.5, 0.25, 1.0],
            }),
        );

        let result = collect_text_nodes(&graph, None, None, None);

        assert_eq!(result.overlay_lines.len(), 1);
        assert_eq!(result.overlay_lines[0], "screen label");
        assert!(result.positioned.is_empty());
    }

    #[test]
    fn collect_text3_nodes_projects_with_parent_transform() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Group(GroupNode));
        let transform = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(10.0, 20.0, 30.0))),
        );
        graph.add_child(
            transform,
            NodeData::Text3(Text3Node {
                string: "world label".to_string(),
                position: Vec3::new(1.0, 2.0, 3.0),
                size: 24.0,
                color: [0.25, 0.5, 1.0, 1.0],
            }),
        );

        // Without VP, Text3 falls through (no projection)
        let result = collect_text_nodes(&graph, None, None, None);
        assert!(result.positioned.is_empty());

        // With VP, should produce positioned text
        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0) * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::new(11.0, 22.0, 33.0), Vec3::Y);
        let result = collect_text_nodes(&graph, Some(vp), Some((800, 600)), Some(Vec3::new(0.0, 0.0, 5.0)));
        assert!(!result.positioned.is_empty());
        assert_eq!(result.positioned[0].string, "world label");
    }
}
