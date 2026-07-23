//! Resolve annotation points against the scene graph.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use crate::node_data::{AnnotationElement, NodeData};
use crate::SceneGraph;

use super::point::AnnotationPoint;

/// Direct child `Transform` matrices multiplied (Coin3D group + child `Transform` pattern).
fn direct_child_transforms(graph: &SceneGraph, node: NodeId) -> Mat4 {
    let mut m = Mat4::IDENTITY;
    let Some(entry) = graph.get(node) else {
        return m;
    };
    for &child in &entry.children {
        if let Some(ce) = graph.get(child) {
            if let NodeData::Transform(t) = &ce.data {
                m *= t.to_matrix();
            }
        }
    }
    m
}

/// World transform of `node` (ancestor `Transform` chain, plus direct child transforms for groups).
pub fn node_world_matrix(graph: &SceneGraph, target: NodeId) -> Option<Mat4> {
    fn dfs(
        graph: &SceneGraph,
        node: NodeId,
        target: NodeId,
        cur: Mat4,
    ) -> Option<Option<Mat4>> {
        let entry = graph.get(node)?;
        let mut m = cur;
        if let NodeData::Transform(t) = &entry.data {
            m *= t.to_matrix();
        }
        if node == target {
            let frame = if matches!(entry.data, NodeData::Transform(_)) {
                m
            } else {
                m * direct_child_transforms(graph, node)
            };
            return Some(Some(frame));
        }
        for &child in &entry.children {
            if let Some(found) = dfs(graph, child, target, m)? {
                return Some(Some(found));
            }
        }
        None
    }

    for &root in graph.roots() {
        if let Some(m) = dfs(graph, root, target, Mat4::IDENTITY)? {
            return Some(m);
        }
    }
    None
}

/// First scene-graph node referenced by any point in the element.
pub fn bound_node_in_element(element: &AnnotationElement) -> Option<NodeId> {
    let from_point = |p: &AnnotationPoint| p.node;
    match element {
        AnnotationElement::Dimension { start, end, .. } => start.node.or(end.node),
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => {
            center.node.or(arm1.node).or(arm2.node)
        }
        AnnotationElement::RadialDimension { center, perimeter, .. } => center.node.or(perimeter.node),
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => {
            center.node.or(p1.node).or(p2.node)
        }
        AnnotationElement::Leader { anchor, .. } => from_point(anchor),
        AnnotationElement::Callout { anchor, .. } => from_point(anchor),
        AnnotationElement::Datum { position, .. } => from_point(position),
        AnnotationElement::GdtFeatureControlFrame { position, leader_target, .. } => {
            from_point(position).or(leader_target.as_ref().and_then(|lt| lt.node))
        }
        AnnotationElement::GdtDatumTarget { position, .. } => from_point(position),
        AnnotationElement::ChamferDimension { start, end, .. } => start.node.or(end.node),
        AnnotationElement::OrdinateDimension { feature, datum, .. } => feature.node.or(datum.node),
        AnnotationElement::SurfaceFinish { position, .. } => from_point(position),
        AnnotationElement::WeldSymbol { position, .. } => from_point(position),
        AnnotationElement::DatumIdentifier { position, .. } => from_point(position),
    }
}

/// Model matrix for projecting annotation geometry: `set_matrix * node_world` when node-bound.
pub fn effective_annotation_model(
    graph: &SceneGraph,
    set_matrix: Mat4,
    element: &AnnotationElement,
) -> Mat4 {
    if let Some(node) = bound_node_in_element(element) {
        if let Some(node_world) = node_world_matrix(graph, node) {
            return set_matrix * node_world;
        }
    }
    set_matrix
}

/// Strip node bindings; keep author-local coordinates (used with [`effective_annotation_model`]).
pub fn localize_element_points(element: &AnnotationElement) -> AnnotationElement {
    let loc = |p: &AnnotationPoint| AnnotationPoint::local(p.local);
    match element {
        AnnotationElement::Dimension {
            start,
            end,
            offset_dir,
            extension_len,
            arrow_size,
            label,
            label_mode,
            color,
        } => AnnotationElement::Dimension {
            start: loc(start),
            end: loc(end),
            offset_dir: *offset_dir,
            extension_len: *extension_len,
            arrow_size: *arrow_size,
            label: label.clone(),
            label_mode: label_mode.clone(),
            color: *color,
        },
        AnnotationElement::AngleDimension {
            center,
            arm1,
            arm2,
            radius,
            label,
            label_mode,
            color,
        } => AnnotationElement::AngleDimension {
            center: loc(center),
            arm1: loc(arm1),
            arm2: loc(arm2),
            radius: *radius,
            label: label.clone(),
            label_mode: label_mode.clone(),
            color: *color,
        },
        AnnotationElement::RadialDimension {
            center,
            perimeter,
            label,
            label_mode,
            arrow_size,
            color,
        } => AnnotationElement::RadialDimension {
            center: loc(center),
            perimeter: loc(perimeter),
            label: label.clone(),
            label_mode: label_mode.clone(),
            arrow_size: *arrow_size,
            color: *color,
        },
        AnnotationElement::DiameterDimension {
            center,
            p1,
            p2,
            label,
            label_mode,
            arrow_size,
            color,
        } => AnnotationElement::DiameterDimension {
            center: loc(center),
            p1: loc(p1),
            p2: loc(p2),
            label: label.clone(),
            label_mode: label_mode.clone(),
            arrow_size: *arrow_size,
            color: *color,
        },
        AnnotationElement::Leader {
            anchor,
            label_offset,
            text,
            color,
        } => AnnotationElement::Leader {
            anchor: loc(anchor),
            label_offset: *label_offset,
            text: text.clone(),
            color: *color,
        },
        AnnotationElement::Callout {
            anchor,
            label_offset,
            text,
            radius,
            color,
        } => AnnotationElement::Callout {
            anchor: loc(anchor),
            label_offset: *label_offset,
            text: text.clone(),
            radius: *radius,
            color: *color,
        },
        AnnotationElement::Datum { position, size, color } => AnnotationElement::Datum {
            position: loc(position),
            size: *size,
            color: *color,
        },
        AnnotationElement::GdtFeatureControlFrame {
            symbol,
            tolerance,
            diameter,
            datum_primary,
            datum_secondary,
            material_condition,
            position,
            leader_target,
            color,
        } => AnnotationElement::GdtFeatureControlFrame {
            symbol: *symbol,
            tolerance: *tolerance,
            diameter: *diameter,
            datum_primary: datum_primary.clone(),
            datum_secondary: datum_secondary.clone(),
            material_condition: *material_condition,
            position: loc(position),
            leader_target: leader_target.as_ref().map(loc),
            color: *color,
        },
        AnnotationElement::GdtDatumTarget {
            position,
            label,
            target_type,
            size,
            color,
        } => AnnotationElement::GdtDatumTarget {
            position: loc(position),
            label: label.clone(),
            target_type: *target_type,
            size: *size,
            color: *color,
        },
        AnnotationElement::ChamferDimension {
            start,
            end,
            offset_dir,
            extension_len,
            arrow_size,
            label,
            label_mode,
            color,
        } => AnnotationElement::ChamferDimension {
            start: loc(start),
            end: loc(end),
            offset_dir: *offset_dir,
            extension_len: *extension_len,
            arrow_size: *arrow_size,
            label: label.clone(),
            label_mode: label_mode.clone(),
            color: *color,
        },
        AnnotationElement::OrdinateDimension {
            feature,
            datum,
            axis_dir,
            jog_length,
            offset,
            label,
            label_mode,
            color,
        } => AnnotationElement::OrdinateDimension {
            feature: loc(feature),
            datum: loc(datum),
            axis_dir: *axis_dir,
            jog_length: *jog_length,
            offset: *offset,
            label: label.clone(),
            label_mode: label_mode.clone(),
            color: *color,
        },
        AnnotationElement::SurfaceFinish {
            ra_value,
            note,
            position,
            direction,
            color,
        } => AnnotationElement::SurfaceFinish {
            ra_value: *ra_value,
            note: note.clone(),
            position: loc(position),
            direction: *direction,
            color: *color,
        },
        AnnotationElement::WeldSymbol {
            weld_type,
            size,
            length,
            field_weld,
            arrow_side_text,
            other_side_text,
            position,
            arrow_dir,
            color,
        } => AnnotationElement::WeldSymbol {
            weld_type: *weld_type,
            size: *size,
            length: *length,
            field_weld: *field_weld,
            arrow_side_text: arrow_side_text.clone(),
            other_side_text: other_side_text.clone(),
            position: loc(position),
            arrow_dir: *arrow_dir,
            color: *color,
        },
        AnnotationElement::DatumIdentifier {
            label,
            position,
            size,
            filled,
            color,
        } => AnnotationElement::DatumIdentifier {
            label: label.clone(),
            position: loc(position),
            size: *size,
            filled: *filled,
            color: *color,
        },
    }
}

/// Prepare element + projection matrix for `pass_markup` (`scene_vp * model * local`).
pub fn prepare_annotation_for_render(
    graph: &SceneGraph,
    set_matrix: Mat4,
    element: &AnnotationElement,
) -> (AnnotationElement, Mat4) {
    let model = effective_annotation_model(graph, set_matrix, element);
    let element = localize_element_points(element);
    (element, model)
}

/// Legacy: bake bound points into set-local space (prefer [`prepare_annotation_for_render`]).
pub fn resolve_point(
    graph: &SceneGraph,
    set_matrix: Mat4,
    point: &AnnotationPoint,
) -> Option<[f32; 3]> {
    let world = if let Some(node) = point.node {
        let node_world = node_world_matrix(graph, node)?;
        let w = node_world * Vec3::from(point.local).extend(1.0);
        w.truncate()
    } else {
        return Some(point.local);
    };
    let set_inv = set_matrix.inverse();
    let local = set_inv * world.extend(1.0);
    Some(local.truncate().into())
}

fn resolve_pt(
    graph: &SceneGraph,
    set_matrix: Mat4,
    point: &AnnotationPoint,
) -> AnnotationPoint {
    AnnotationPoint::local(resolve_point(graph, set_matrix, point).unwrap_or(point.local))
}

/// Legacy resolver (bakes positions only). Rendering should use [`prepare_annotation_for_render`].
pub fn resolve_element(
    graph: &SceneGraph,
    set_matrix: Mat4,
    element: &AnnotationElement,
) -> AnnotationElement {
    match element {
        AnnotationElement::Dimension {
            start,
            end,
            offset_dir,
            extension_len,
            arrow_size,
            label,
            label_mode,
            color,
        } => AnnotationElement::Dimension {
            start: resolve_pt(graph, set_matrix, start),
            end: resolve_pt(graph, set_matrix, end),
            offset_dir: *offset_dir,
            extension_len: *extension_len,
            arrow_size: *arrow_size,
            label: label.clone(),
            label_mode: label_mode.clone(),
            color: *color,
        },
        AnnotationElement::AngleDimension {
            center,
            arm1,
            arm2,
            radius,
            label,
            label_mode,
            color,
        } => AnnotationElement::AngleDimension {
            center: resolve_pt(graph, set_matrix, center),
            arm1: resolve_pt(graph, set_matrix, arm1),
            arm2: resolve_pt(graph, set_matrix, arm2),
            radius: *radius,
            label: label.clone(),
            label_mode: label_mode.clone(),
            color: *color,
        },
        AnnotationElement::RadialDimension {
            center,
            perimeter,
            label,
            label_mode,
            arrow_size,
            color,
        } => AnnotationElement::RadialDimension {
            center: resolve_pt(graph, set_matrix, center),
            perimeter: resolve_pt(graph, set_matrix, perimeter),
            label: label.clone(),
            label_mode: label_mode.clone(),
            arrow_size: *arrow_size,
            color: *color,
        },
        AnnotationElement::DiameterDimension {
            center,
            p1,
            p2,
            label,
            label_mode,
            arrow_size,
            color,
        } => AnnotationElement::DiameterDimension {
            center: resolve_pt(graph, set_matrix, center),
            p1: resolve_pt(graph, set_matrix, p1),
            p2: resolve_pt(graph, set_matrix, p2),
            label: label.clone(),
            label_mode: label_mode.clone(),
            arrow_size: *arrow_size,
            color: *color,
        },
        AnnotationElement::Leader {
            anchor,
            label_offset,
            text,
            color,
        } => AnnotationElement::Leader {
            anchor: resolve_pt(graph, set_matrix, anchor),
            label_offset: *label_offset,
            text: text.clone(),
            color: *color,
        },
        AnnotationElement::Callout {
            anchor,
            label_offset,
            text,
            radius,
            color,
        } => AnnotationElement::Callout {
            anchor: resolve_pt(graph, set_matrix, anchor),
            label_offset: *label_offset,
            text: text.clone(),
            radius: *radius,
            color: *color,
        },
        AnnotationElement::Datum { position, size, color } => AnnotationElement::Datum {
            position: resolve_pt(graph, set_matrix, position),
            size: *size,
            color: *color,
        },
        AnnotationElement::SurfaceFinish {
            ra_value,
            note,
            position,
            direction,
            color,
        } => AnnotationElement::SurfaceFinish {
            ra_value: *ra_value,
            note: note.clone(),
            position: resolve_pt(graph, set_matrix, position),
            direction: *direction,
            color: *color,
        },
        AnnotationElement::WeldSymbol {
            weld_type,
            size,
            length,
            field_weld,
            arrow_side_text,
            other_side_text,
            position,
            arrow_dir,
            color,
        } => AnnotationElement::WeldSymbol {
            weld_type: *weld_type,
            size: *size,
            length: *length,
            field_weld: *field_weld,
            arrow_side_text: arrow_side_text.clone(),
            other_side_text: other_side_text.clone(),
            position: resolve_pt(graph, set_matrix, position),
            arrow_dir: *arrow_dir,
            color: *color,
        },
        AnnotationElement::DatumIdentifier {
            label,
            position,
            size,
            filled,
            color,
        } => AnnotationElement::DatumIdentifier {
            label: label.clone(),
            position: resolve_pt(graph, set_matrix, position),
            size: *size,
            filled: *filled,
            color: *color,
        },
        _ => element.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_data::{AnnotationLabelMode, SeparatorNode, TransformNode};

    #[test]
    fn effective_model_includes_transform_translation() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        let tf = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(3.0, 0.0, 0.0))),
        );

        let el = AnnotationElement::Dimension {
            start: AnnotationPoint::on_node(tf, [0.0, 0.0, 0.0]),
            end: AnnotationPoint::on_node(tf, [1.0, 0.0, 0.0]),
            offset_dir: [0.0, -1.0, 0.0],
            extension_len: 0.1,
            arrow_size: 0.05,
            label: String::new(),
            label_mode: AnnotationLabelMode::Auto,
            color: [1.0; 4],
        };

        let model = effective_annotation_model(&graph, Mat4::IDENTITY, &el);
        let world = model.transform_point3(Vec3::ZERO);
        assert!((world.x - 3.0).abs() < 1e-4);
    }

    #[test]
    fn prepare_keeps_local_coords_and_sets_model() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        let tf = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(-2.0, 1.0, 0.0))),
        );
        let el = AnnotationElement::Dimension {
            start: AnnotationPoint::on_node(tf, [-0.5, 0.0, 0.0]),
            end: AnnotationPoint::on_node(tf, [0.5, 0.0, 0.0]),
            offset_dir: [0.0, 1.0, 0.0],
            extension_len: 0.1,
            arrow_size: 0.05,
            label: String::new(),
            label_mode: AnnotationLabelMode::Auto,
            color: [1.0; 4],
        };
        let (loc_el, model) = prepare_annotation_for_render(&graph, Mat4::IDENTITY, &el);
        match loc_el {
            AnnotationElement::Dimension { start, .. } => {
                assert!(start.node.is_none());
                assert_eq!(start.local, [-0.5, 0.0, 0.0]);
            }
            _ => panic!("expected dimension"),
        }
        let w = model.transform_point3(Vec3::new(-0.5, 0.0, 0.0));
        assert!((w.x - (-2.5)).abs() < 1e-4);
        assert!((w.y - 1.0).abs() < 1e-4);
    }
}
