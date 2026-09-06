//! Text rendering pass for Text2/Text3 scene-graph nodes.
//! Text2 → HUD overlay; Text3 → world-space quads (`world_label` pass).

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::node_data::{FontNode, FontStyle};
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
    pub font_name: String,
    pub font_style: FontStyle,
}

/// Overlay lines (no positioning — rendered as HUD block) + positioned text entries.
pub(crate) struct TextCollection {
    pub overlay_lines: Vec<String>,
    pub positioned: Vec<TextDrawCommand>,
}

struct TextCollect<'a> {
    graph: &'a SceneGraph,
    view_proj: Option<Mat4>,
    viewport: Option<(u32, u32)>,
    camera_pos: Option<Vec3>,
    depth_reversed_z: bool,
    overlay_lines: &'a mut Vec<String>,
    positioned: &'a mut Vec<TextDrawCommand>,
    world_labels: &'a mut Vec<WorldLabelCommand>,
}

fn applied_size(node_size: f32, font: &Option<FontNode>) -> f32 {
    match font {
        Some(f) if f.size > 0.0 => f.size,
        _ => node_size,
    }
}

fn font_name(font: &Option<FontNode>) -> String {
    font.as_ref().map(|f| f.name.clone()).unwrap_or_default()
}

fn font_style(font: &Option<FontNode>) -> FontStyle {
    font.as_ref().map(|f| f.style).unwrap_or_default()
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
    let roots: Vec<NodeId> = graph.roots().to_vec();
    let mut ctx = TextCollect {
        graph,
        view_proj,
        viewport,
        camera_pos,
        depth_reversed_z,
        overlay_lines: &mut overlay_lines,
        positioned: &mut positioned,
        world_labels,
    };
    for root in roots {
        collect_recursive(&mut ctx, root, Mat4::IDENTITY, false, None);
    }
    TextCollection {
        overlay_lines,
        positioned,
    }
}

fn walk_children(
    ctx: &mut TextCollect<'_>,
    children: &[NodeId],
    model: Mat4,
    inside_billboard: bool,
    font: Option<FontNode>,
) {
    enum Step {
        Font { font: FontNode, kids: Vec<NodeId> },
        Xform { lm: Mat4, kids: Vec<NodeId> },
        Visit,
    }
    let mut font = font;
    let mut accum = model;
    for &child in children {
        let step = {
            let Some(ce) = ctx.graph.get(child) else {
                continue;
            };
            if let NodeData::Font(f) = &ce.data {
                Step::Font {
                    font: f.clone(),
                    kids: ce.children.clone(),
                }
            } else if let Some(lm) = ce.data.local_matrix() {
                Step::Xform {
                    lm,
                    kids: ce.children.clone(),
                }
            } else {
                Step::Visit
            }
        };
        match step {
            Step::Font { font: next, kids } => {
                font = Some(next);
                walk_children(ctx, &kids, accum, inside_billboard, font.clone());
            }
            Step::Xform { lm, kids } => {
                let nested = accum * lm;
                walk_children(ctx, &kids, nested, inside_billboard, font.clone());
                // Property-style Transform (no children) affects later siblings.
                // Parented Transform must not leak into the next sibling frame.
                if kids.is_empty() {
                    accum = nested;
                }
            }
            Step::Visit => {
                collect_recursive(ctx, child, accum, inside_billboard, font.clone());
            }
        }
    }
}

fn collect_recursive(
    ctx: &mut TextCollect<'_>,
    node: NodeId,
    model: Mat4,
    inside_billboard: bool,
    font: Option<FontNode>,
) {
    enum Kind {
        Text2(rc3d_scene::node_data::Text2Node),
        Text3(rc3d_scene::node_data::Text3Node),
        Font(FontNode),
        Billboard(bool),
        Xform(Option<Mat4>),
        Other,
    }
    let Some((kind, children)) = ctx.graph.get(node).map(|entry| {
        let children = entry.children.clone();
        let kind = match &entry.data {
            NodeData::Text2(t) => Kind::Text2(t.clone()),
            NodeData::Text3(t) => Kind::Text3(t.clone()),
            NodeData::Font(f) => Kind::Font(f.clone()),
            NodeData::Billboard(b) => Kind::Billboard(b.axis_aligned),
            NodeData::Transform(_) | NodeData::Rotation(_) | NodeData::RotationXYZ(_) => {
                Kind::Xform(entry.data.local_matrix())
            }
            _ => Kind::Other,
        };
        (kind, children)
    }) else {
        return;
    };
    match kind {
        Kind::Text2(t) => {
            let size = applied_size(t.size, &font);
            if inside_billboard {
                if let (Some(vp), Some((vw, vh))) = (ctx.view_proj, ctx.viewport) {
                    let world = model.transform_point3(Vec3::new(t.position[0], t.position[1], 0.0));
                    let clip = vp.project_point3(world);
                    if clip.z > 1.0 || clip.z < -1.0 {
                        return;
                    }
                    let sx = (clip.x * 0.5 + 0.5) * vw as f32;
                    let sy = (1.0 - (clip.y * 0.5 + 0.5)) * vh as f32;
                    ctx.positioned.push(TextDrawCommand {
                        string: t.string,
                        screen_pos: [sx, sy],
                        size,
                        color: t.color,
                        is_3d: false,
                        plane_aligned: false,
                        baseline_angle_rad: 0.0,
                        clip_depth_ndc: clip.z,
                        font_name: font_name(&font),
                        font_style: font_style(&font),
                    });
                }
            } else {
                ctx.overlay_lines.push(t.string);
            }
        }
        Kind::Text3(t) => {
            let size = applied_size(t.size, &font);
            let world_pos = model.transform_point3(t.position);
            if let (Some(vp), Some((vw, vh)), Some(cam)) =
                (ctx.view_proj, ctx.viewport, ctx.camera_pos)
            {
                let clip = vp.project_point3(world_pos);
                if clip.z > 1.0 || clip.z < -1.0 {
                    return;
                }
                let (tangent, bitangent) = if t.plane_aligned {
                    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
                } else {
                    camera_billboard_basis(world_pos, cam)
                };
                let at = t.position.into();
                let ext = size * 0.02;
                let height_world = resolve_label_height_world(
                    ext,
                    0.5,
                    size,
                    at,
                    bitangent,
                    model,
                    vp,
                    vw as f32,
                    vh as f32,
                    ctx.depth_reversed_z,
                );
                ctx.world_labels.push(WorldLabelCommand {
                    string: t.string,
                    model_matrix: model,
                    at,
                    tangent,
                    bitangent,
                    height_world,
                    screen_height_px: size,
                    color: t.color,
                    font_name: font_name(&font),
                    font_style: font_style(&font),
                });
            }
        }
        Kind::Font(f) => {
            walk_children(ctx, &children, model, inside_billboard, Some(f));
        }
        Kind::Billboard(axis_aligned) => {
            let billboard_pos = model.w_axis.truncate();
            let facing = if let Some(cam) = ctx.camera_pos {
                compute_billboard_facing(axis_aligned, billboard_pos, cam)
            } else {
                Mat4::IDENTITY
            };
            walk_children(ctx, &children, model * facing, true, font);
        }
        Kind::Xform(lm) => {
            let m = lm.map(|lm| model * lm).unwrap_or(model);
            walk_children(ctx, &children, m, inside_billboard, font);
        }
        Kind::Other => {
            walk_children(ctx, &children, model, inside_billboard, font);
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
                ..Default::default()
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
