//! Text rendering pass for Text2/Text3 scene-graph nodes.
//! Collects text draw commands for rendering via the existing HUD glyphon system.

use rc3d_core::math::Mat4;
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

/// Collect all Text2/Text3 draw commands from the scene graph.
pub fn collect_text_nodes(graph: &SceneGraph) -> Vec<TextDrawCommand> {
    let mut cmds = Vec::new();
    for &root in graph.roots() {
        collect_recursive(graph, root, Mat4::IDENTITY, &mut cmds);
    }
    cmds
}

fn collect_recursive(
    graph: &SceneGraph,
    node: NodeId,
    model: Mat4,
    cmds: &mut Vec<TextDrawCommand>,
) {
    let Some(entry) = graph.get(node) else { return };
    match &entry.data {
        NodeData::Text2(t) => {
            cmds.push(TextDrawCommand {
                string: t.string.clone(),
                screen_pos: t.position,
                size: t.size,
                color: t.color,
                is_3d: false,
            });
        }
        NodeData::Text3(t) => {
            let world_pos = model.transform_point3(t.position);
            cmds.push(TextDrawCommand {
                string: t.string.clone(),
                screen_pos: [world_pos.x, world_pos.y],
                size: t.size,
                color: t.color,
                is_3d: true,
            });
        }
        NodeData::Transform(t) => {
            let m = model * t.to_matrix();
            for &child in &entry.children {
                collect_recursive(graph, child, m, cmds);
            }
        }
        NodeData::Separator(_)
        | NodeData::Group(_) | NodeData::Environment(_) | NodeData::ShapeHints(_) | NodeData::Annotation(_) | NodeData::ResetTransform(_) | NodeData::Texture2Transform(_) | NodeData::MaterialBinding(_) | NodeData::IndexedLineSet(_) | NodeData::File(_) | NodeData::Decal(_) | NodeData::ExplodedView(_) | NodeData::ReflectionPlane(_) | NodeData::Billboard(_)
        | NodeData::Lod(_)
        | NodeData::EventCallback(_)
        | NodeData::SectionPlane(_)
        | NodeData::Switch(_)
        | NodeData::MultipleCopy(_)
        | NodeData::HandlerNode(_) => {
            for &child in &entry.children {
                collect_recursive(graph, child, model, cmds);
            }
        }
        _ => {
            for &child in &entry.children {
                collect_recursive(graph, child, model, cmds);
            }
        }
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

        let commands = collect_text_nodes(&graph);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].string, "screen label");
        assert_eq!(commands[0].screen_pos, [12.0, 34.0]);
        assert_eq!(commands[0].size, 18.0);
        assert_eq!(commands[0].color, [1.0, 0.5, 0.25, 1.0]);
        assert!(!commands[0].is_3d);
    }

    #[test]
    fn collect_text3_nodes_applies_parent_transform() {
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

        let commands = collect_text_nodes(&graph);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].string, "world label");
        assert_eq!(commands[0].screen_pos, [11.0, 22.0]);
        assert_eq!(commands[0].size, 24.0);
        assert_eq!(commands[0].color, [0.25, 0.5, 1.0, 1.0]);
        assert!(commands[0].is_3d);
    }
}
