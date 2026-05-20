//! Text rendering pass for Text2/Text3 scene-graph nodes.
//! Text2 → HUD overlay; Text3 → world-space quads (`world_label` pass).

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

use crate::world_label::{
    camera_billboard_basis, resolve_label_height_world, WorldLabelCommand,
};

/// A text draw command ready for rendering.
#[derive(Clone)]
pub struct TextDrawCommand {
    pub string: String,
    pub screen_pos: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
    pub is_3d: bool,
    /// When true, glyph is rotated by `baseline_angle_rad` (dimension labels use horizontal text).
    pub plane_aligned: bool,
    /// Screen-space baseline angle (radians); 0 = horizontal, readable regardless of camera orbit.
    pub baseline_angle_rad: f32,
    /// NDC depth at the label anchor (for glyphon depth-tested annotation text).
    pub clip_depth_ndc: f32,
}

/// Overlay lines (no positioning — rendered as HUD block) + positioned text entries.
pub(crate) struct TextCollection {
    pub overlay_lines: Vec<String>,
    pub positioned: Vec<TextDrawCommand>,
}

/// Collect Text2 (HUD) and Text3 (world quads) from the scene graph.
pub fn collect_text_nodes(
    graph: &SceneGraph,
    view_proj: Option<Mat4>,
    viewport: Option<(u32, u32)>,
    camera_pos: Option<Vec3>,
    depth_reversed_z: bool,
    world_labels: &mut Vec<WorldLabelCommand>,
) -> TextCollection {
    let mut overlay_lines = Vec::new();
    let mut positioned = Vec::new();
    for &root in graph.roots() {
        collect_recursive(
            graph,
            root,
            Mat4::IDENTITY,
            view_proj,
            viewport,
            camera_pos,
            depth_reversed_z,
            false,
            &mut overlay_lines,
            &mut positioned,
            world_labels,
        );
    }
    TextCollection {
        overlay_lines,
        positioned,
    }
}

fn collect_recursive(
    graph: &SceneGraph,
    node: NodeId,
    model: Mat4,
    view_proj: Option<Mat4>,
    viewport: Option<(u32, u32)>,
    camera_pos: Option<Vec3>,
    depth_reversed_z: bool,
    inside_billboard: bool,
    overlay_lines: &mut Vec<String>,
    positioned: &mut Vec<TextDrawCommand>,
    world_labels: &mut Vec<WorldLabelCommand>,
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
                        plane_aligned: false,
                        baseline_angle_rad: 0.0,
                        clip_depth_ndc: clip.z,
                    });
                }
            } else {
                overlay_lines.push(t.string.clone());
            }
        }
        NodeData::Text3(t) => {
            let world_pos = model.transform_point3(t.position);
            if let (Some(vp), Some((vw, vh)), Some(cam)) = (view_proj, viewport, camera_pos) {
                let clip = vp.project_point3(world_pos);
                if clip.z > 1.0 || clip.z < -1.0 {
                    return;
                }
                let (tangent, bitangent) = camera_billboard_basis(world_pos, cam);
                let at = t.position.into();
                let ext = t.size * 0.02;
                let height_world = resolve_label_height_world(
                    ext,
                    0.5,
                    t.size,
                    at,
                    bitangent,
                    model,
                    vp,
                    vw as f32,
                    vh as f32,
                    depth_reversed_z,
                );
                world_labels.push(WorldLabelCommand {
                    string: t.string.clone(),
                    model_matrix: model,
                    at,
                    tangent,
                    bitangent,
                    height_world,
                    screen_height_px: t.size,
                    color: t.color,
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
                    graph,
                    child,
                    new_model,
                    view_proj,
                    viewport,
                    camera_pos,
                    depth_reversed_z,
                    true,
                    overlay_lines,
                    positioned,
                    world_labels,
                );
            }
        }
        NodeData::Transform(t) => {
            let m = model * t.to_matrix();
            for &child in &entry.children {
                collect_recursive(
                    graph,
                    child,
                    m,
                    view_proj,
                    viewport,
                    camera_pos,
                    depth_reversed_z,
                    inside_billboard,
                    overlay_lines,
                    positioned,
                    world_labels,
                );
            }
        }
        NodeData::Separator(_)
        | NodeData::Group(_)
        | NodeData::Environment(_)
        | NodeData::ShapeHints(_)
        | NodeData::Annotation(_)
        | NodeData::ResetTransform(_)
        | NodeData::Texture2Transform(_)
        | NodeData::MaterialBinding(_)
        | NodeData::IndexedLineSet(_)
        | NodeData::File(_)
        | NodeData::Decal(_)
        | NodeData::ExplodedView(_)
        | NodeData::ReflectionPlane(_)
        | NodeData::Lod(_)
        | NodeData::EventCallback(_)
        | NodeData::SectionPlane(_)
        | NodeData::Switch(_)
        | NodeData::MultipleCopy(_)
        | NodeData::HandlerNode(_) => {
            for &child in &entry.children {
                collect_recursive(
                    graph,
                    child,
                    model,
                    view_proj,
                    viewport,
                    camera_pos,
                    depth_reversed_z,
                    inside_billboard,
                    overlay_lines,
                    positioned,
                    world_labels,
                );
            }
        }
        _ => {
            for &child in &entry.children {
                collect_recursive(
                    graph,
                    child,
                    model,
                    view_proj,
                    viewport,
                    camera_pos,
                    depth_reversed_z,
                    inside_billboard,
                    overlay_lines,
                    positioned,
                    world_labels,
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

        let mut world_labels = Vec::new();
        let result = collect_text_nodes(&graph, None, None, None, false, &mut world_labels);

        assert_eq!(result.overlay_lines.len(), 1);
        assert_eq!(result.overlay_lines[0], "screen label");
        assert!(result.positioned.is_empty());
        assert!(world_labels.is_empty());
    }

    #[test]
    fn collect_text3_nodes_emits_world_labels() {
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

        let mut world_labels = Vec::new();
        let result = collect_text_nodes(&graph, None, None, None, false, &mut world_labels);
        assert!(result.positioned.is_empty());
        assert!(world_labels.is_empty());

        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::new(11.0, 22.0, 33.0), Vec3::Y);
        let mut world_labels = Vec::new();
        let result = collect_text_nodes(
            &graph,
            Some(vp),
            Some((800, 600)),
            Some(Vec3::new(0.0, 0.0, 5.0)),
            false,
            &mut world_labels,
        );
        assert!(result.positioned.is_empty());
        assert_eq!(world_labels.len(), 1);
        assert_eq!(world_labels[0].string, "world label");
    }
}
