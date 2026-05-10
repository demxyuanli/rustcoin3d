//! Update [`LodNode::current_level`](rc3d_scene::node_data::LodNode) from camera distance.
//! Convention: `LodLevel::max_distance` is an **upper bound**; levels are ordered by
//! **increasing** `max_distance` (smallest threshold first for the finest content).

use rc3d_core::math::Vec3;
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// Recompute all `Lod` nodes under `root` using `camera_eye`.
/// Returns the number of LOD nodes found (0 = no LOD nodes in scene).
pub fn update_all_lod_nodes(graph: &mut SceneGraph, root: NodeId, camera_eye: Vec3) -> usize {
    let mut lod_count = 0usize;
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if let Some(entry) = graph.get(n) {
            for &c in entry.children.iter().rev() {
                stack.push(c);
            }
        }
        if let Some(entry) = graph.get(n) {
            if let NodeData::Lod(lod) = &entry.data {
                lod_count += 1;
                if lod.levels.is_empty() {
                    continue;
                }
                let dist = camera_to_lod_distance(graph, n, lod, camera_eye);
                if let Some(e) = graph.get_mut(n) {
                    if let NodeData::Lod(l2) = &mut e.data {
                        let mut picked = l2.levels.len() - 1;
                        for (i, lev) in l2.levels.iter().enumerate() {
                            if dist < lev.max_distance {
                                picked = i;
                                break;
                            }
                        }
                        l2.current_level = picked;
                    }
                }
            }
        }
    }
    lod_count
}

fn camera_to_lod_distance(
    graph: &SceneGraph,
    lod_id: NodeId,
    lod: &rc3d_scene::node_data::LodNode,
    camera_eye: Vec3,
) -> f32 {
    // Use finest level (first) children as a stable pivot for distance.
    if let Some(level0) = lod.levels.first() {
        let mut sum = Vec3::ZERO;
        let mut c = 0.0f32;
        for &ch in &level0.children {
            if let Some(m) = approximate_node_center(graph, ch) {
                sum += m;
                c += 1.0;
            }
        }
        if c > 0.0 {
            return (sum / c - camera_eye).length();
        }
    }
    approximate_node_center(graph, lod_id)
        .map(|p| (p - camera_eye).length())
        .unwrap_or(0.0)
}

fn approximate_node_center(graph: &SceneGraph, node: NodeId) -> Option<Vec3> {
    use rc3d_scene::NodeData;
    let entry = graph.get(node)?;
    match &entry.data {
        NodeData::Transform(t) => Some(t.translation),
        NodeData::Group(_) | NodeData::Environment(_) | NodeData::ShapeHints(_) | NodeData::Annotation(_) | NodeData::ResetTransform(_) | NodeData::Texture2Transform(_) | NodeData::MaterialBinding(_) | NodeData::IndexedLineSet(_) | NodeData::File(_) | NodeData::Decal(_) | NodeData::ExplodedView(_) | NodeData::ReflectionPlane(_) | NodeData::Billboard(_) | NodeData::Separator(_) | NodeData::Lod(_) | NodeData::Switch(_) | NodeData::MultipleCopy(_)
        | NodeData::Measurement(_) | NodeData::Markup(_) | NodeData::Text2(_) | NodeData::Text3(_) | NodeData::EventCallback(_) => {
            for &c in &entry.children {
                if let Some(p) = approximate_node_center(graph, c) {
                    return Some(p);
                }
            }
            None
        }
        _ => {
            for &c in &entry.children {
                if let Some(p) = approximate_node_center(graph, c) {
                    return Some(p);
                }
            }
            None
        }
    }
}
