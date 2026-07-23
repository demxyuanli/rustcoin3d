//! Two-pixel "box" selection: projects shape bounding-box centers to clip space.

use rc3d_actions::GetBoundingBoxAction;
use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

use rc3d_render::viewport::Viewport;

fn to_ndc(px: f32, py: f32, win_w: f32, win_h: f32) -> (f32, f32) {
    let x = 2.0 * px / win_w - 1.0;
    let y = -(2.0 * py / win_h - 1.0);
    (x, y)
}

fn is_shape_data(data: &NodeData) -> bool {
    matches!(
        data,
        NodeData::Cube(_)
            | NodeData::Sphere(_)
            | NodeData::Cone(_)
            | NodeData::Cylinder(_)
            | NodeData::IndexedFaceSet(_)
            | NodeData::Triangle(_)
    )
}

/// Selects shape nodes whose world-space AABB **center** projects into the 2D NDC band of the frustum
/// (screen-space `corner_a` to `corner_b` in the same space as `win_w`/`win_h`).
#[allow(clippy::too_many_arguments)]
pub fn select_nodes_in_screen_box(
    graph: &mut SceneGraph,
    root: NodeId,
    corner_a: (f32, f32),
    corner_b: (f32, f32),
    win_w: f32,
    win_h: f32,
    view: Mat4,
    proj: Mat4,
) {
    let mvp = proj * view;
    let (a_x, a_y) = to_ndc(corner_a.0, corner_a.1, win_w, win_h);
    let (b_x, b_y) = to_ndc(corner_b.0, corner_b.1, win_w, win_h);
    let min_x = a_x.min(b_x);
    let max_x = a_x.max(b_x);
    let min_y = a_y.min(b_y);
    let max_y = a_y.max(b_y);

    let mut stack = vec![root];
    let mut picked: Vec<NodeId> = Vec::new();
    while let Some(n) = stack.pop() {
        let Some(entry) = graph.get(n) else { continue };
        for &c in &entry.children {
            stack.push(c);
        }
        if !is_shape_data(&entry.data) {
            continue;
        }
        let mut bbox = GetBoundingBoxAction::new();
        bbox.apply(graph, n);
        if bbox.bounding_box.min.x > bbox.bounding_box.max.x {
            continue;
        }
        let c = 0.5 * (bbox.bounding_box.min + bbox.bounding_box.max);
        let p = mvp * c.extend(1.0);
        let wv = p.w.abs().max(1e-6);
        let nx = p.x / wv;
        let ny = p.y / wv;
        if nx >= min_x && nx <= max_x && ny >= min_y && ny <= max_y {
            picked.push(n);
        }
    }
    graph.clear_selection();
    graph.select_many(picked);
}

/// `corner_*` in **window** coordinates; `vport` defines a sub-rect in that window.
pub fn select_nodes_in_viewport_box(
    graph: &mut SceneGraph,
    root: NodeId,
    corner_a: (f32, f32),
    corner_b: (f32, f32),
    vport: &Viewport,
    view: Mat4,
    proj: Mat4,
) {
    let to_local = |px: f32, py: f32| (px - vport.rect.x as f32, py - vport.rect.y as f32);
    let a = to_local(corner_a.0, corner_a.1);
    let b = to_local(corner_b.0, corner_b.1);
    let w = vport.rect.width.max(1) as f32;
    let h = vport.rect.height.max(1) as f32;
    select_nodes_in_screen_box(graph, root, a, b, w, h, view, proj);
}
