//! Text rendering pass for Text2/Text3 scene-graph nodes.
//! Collects text draw commands for rendering via the existing HUD glyphon system.

use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// A text draw command ready for rendering.
pub struct TextDrawCommand {
    pub string: String,
    pub screen_pos: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
    pub is_3d: bool,
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
        | NodeData::Group(_) | NodeData::IndexedLineSet(_) | NodeData::File(_) | NodeData::Decal(_) | NodeData::Billboard(_)
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
