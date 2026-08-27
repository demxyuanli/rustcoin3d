//! Engine-owned transform gizmo overlay and Hidden Line SVG hardcopy.

use std::path::Path;

use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_gizmo::{Gizmo, GizmoAxis, GizmoHandle, GizmoMode};
use rc3d_scene::node_data::{
    DraggerKind, DraggerNode, ManipMode, ManipSpace, NodeData, TransformManipNode,
};
use rc3d_scene::SceneGraph;

use crate::engine::Engine;

fn is_transform(graph: &SceneGraph, id: NodeId) -> bool {
    graph
        .get(id)
        .is_some_and(|e| matches!(e.data, NodeData::Transform(_)))
}

/// Coin3D-style: last `Transform` sibling before `id` under the same parent.
fn preceding_sibling_transform(graph: &SceneGraph, id: NodeId) -> Option<NodeId> {
    let parent = graph.get(id)?.parent?;
    let children = graph.children(parent)?;
    let mut last = None;
    for &child in children {
        if child == id {
            break;
        }
        if is_transform(graph, child) {
            last = Some(child);
        }
    }
    last
}

fn first_transform_child(graph: &SceneGraph, id: NodeId) -> Option<NodeId> {
    for &child in graph.children(id)? {
        if is_transform(graph, child) {
            return Some(child);
        }
    }
    None
}

/// Prefer a selected `Transform`, else ancestor, preceding sibling, or child.
pub fn find_transform_for_selection(graph: &SceneGraph) -> Option<NodeId> {
    for &id in graph.selected_nodes() {
        if is_transform(graph, id) {
            return Some(id);
        }
    }
    for &id in graph.selected_nodes() {
        let mut cur = id;
        while let Some(p) = graph.get(cur).and_then(|e| e.parent) {
            if is_transform(graph, p) {
                return Some(p);
            }
            cur = p;
        }
        if let Some(t) = preceding_sibling_transform(graph, id) {
            return Some(t);
        }
        if let Some(t) = first_transform_child(graph, id) {
            return Some(t);
        }
    }
    None
}

/// View/projection from a camera node, if `node` is perspective or orthographic.
pub fn camera_node_view_proj(graph: &SceneGraph, node: NodeId) -> Option<(Mat4, Mat4)> {
    match graph.get(node).map(|e| &e.data) {
        Some(NodeData::PerspectiveCamera(c)) => Some((c.view_matrix(), c.projection_matrix())),
        Some(NodeData::OrthographicCamera(c)) => Some((c.view_matrix(), c.projection_matrix())),
        _ => None,
    }
}

fn first_camera_matrices(graph: &SceneGraph, node: NodeId) -> Option<(Mat4, Mat4)> {
    if let Some(m) = camera_node_view_proj(graph, node) {
        return Some(m);
    }
    let entry = graph.get(node)?;
    for &child in &entry.children {
        if let Some(m) = first_camera_matrices(graph, child) {
            return Some(m);
        }
    }
    None
}

/// World view + projection for picking in a layout viewport.
pub fn viewport_pick_matrices(
    graph: &SceneGraph,
    vc: &crate::viewport::ViewportCamera,
    vport: &rc3d_render::viewport::Viewport,
) -> (Mat4, Mat4) {
    if let Some(m) = camera_node_view_proj(graph, vc.camera_node) {
        return m;
    }
    let aspect = vport.rect.aspect();
    let v = vc.controller.view_matrix();
    let p = match vport.projection_type {
        rc3d_render::viewport::ProjectionType::Perspective => {
            Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0)
        }
        rc3d_render::viewport::ProjectionType::Orthographic => {
            let height = vc.controller.distance * 1.2;
            let w = height * aspect;
            rc3d_render::shadow_map::orthographic_wgpu_rh(
                -w * 0.5,
                w * 0.5,
                -height * 0.5,
                height * 0.5,
                0.1,
                1000.0,
            )
        }
    };
    (v, p)
}

/// View/projection used to draw the last frame (or the first camera node).
pub fn scene_pick_matrices(engine: &Engine) -> (Mat4, Mat4) {
    let view = engine.world.collector.view_matrix;
    let proj = engine.world.collector.projection_matrix;
    if view != Mat4::IDENTITY && proj != Mat4::IDENTITY {
        return (view, proj);
    }
    for &root in engine.world.graph.roots() {
        if let Some(m) = first_camera_matrices(&engine.world.graph, root) {
            return m;
        }
    }
    let aspect = engine
        .renderer
        .as_ref()
        .map(|r| {
            r.config.width as f32 / r.config.height.max(1) as f32
        })
        .unwrap_or(1.0);
    (
        engine.controller.view_matrix(),
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect.max(1e-4), 0.1, 1000.0),
    )
}

fn manip_from_gizmo(mode: GizmoMode) -> ManipMode {
    match mode {
        GizmoMode::Translate => ManipMode::Translate,
        GizmoMode::Rotate => ManipMode::Rotate,
        GizmoMode::Scale => ManipMode::Scale,
    }
}

fn dragger_handles(kind: DraggerKind) -> &'static [GizmoHandle] {
    match kind {
        DraggerKind::TranslateX => &[GizmoHandle::TranslateArrow(GizmoAxis::X)],
        DraggerKind::TranslateY => &[GizmoHandle::TranslateArrow(GizmoAxis::Y)],
        DraggerKind::TranslateZ => &[GizmoHandle::TranslateArrow(GizmoAxis::Z)],
        DraggerKind::TranslateXY => &[GizmoHandle::TranslatePlane(GizmoAxis::XY)],
        DraggerKind::TranslateYZ => &[GizmoHandle::TranslatePlane(GizmoAxis::YZ)],
        DraggerKind::TranslateZX => &[GizmoHandle::TranslatePlane(GizmoAxis::ZX)],
        DraggerKind::RotateX => &[GizmoHandle::RotateRing(GizmoAxis::X)],
        DraggerKind::RotateY => &[GizmoHandle::RotateRing(GizmoAxis::Y)],
        DraggerKind::RotateZ => &[GizmoHandle::RotateRing(GizmoAxis::Z)],
        DraggerKind::ScaleX => &[GizmoHandle::ScaleHandle(GizmoAxis::X)],
        DraggerKind::ScaleY => &[GizmoHandle::ScaleHandle(GizmoAxis::Y)],
        DraggerKind::ScaleZ => &[GizmoHandle::ScaleHandle(GizmoAxis::Z)],
        DraggerKind::ScaleUniform => &[
            GizmoHandle::ScaleHandle(GizmoAxis::X),
            GizmoHandle::ScaleHandle(GizmoAxis::Y),
            GizmoHandle::ScaleHandle(GizmoAxis::Z),
        ],
    }
}

fn collect_handle_filter(graph: &SceneGraph, manip_id: NodeId) -> Option<Vec<GizmoHandle>> {
    let children = graph.children(manip_id)?;
    let mut handles = Vec::new();
    for &child in children {
        let Some(entry) = graph.get(child) else {
            continue;
        };
        let NodeData::Dragger(DraggerNode { kind, enabled: true }) = &entry.data else {
            continue;
        };
        for &h in dragger_handles(*kind) {
            if !handles.contains(&h) {
                handles.push(h);
            }
        }
    }
    if handles.is_empty() {
        None
    } else {
        Some(handles)
    }
}

fn has_dragger_children(graph: &SceneGraph, id: NodeId) -> bool {
    graph
        .children(id)
        .map(|c| {
            c.iter().any(|&ch| {
                graph
                    .get(ch)
                    .is_some_and(|e| matches!(e.data, NodeData::Dragger(_)))
            })
        })
        .unwrap_or(false)
}

fn resolve_manip_target(
    graph: &SceneGraph,
    manip_id: NodeId,
    node: &TransformManipNode,
) -> Option<NodeId> {
    if let Some(t) = node.target {
        if is_transform(graph, t) {
            return Some(t);
        }
    }
    preceding_sibling_transform(graph, manip_id)
}

fn manip_matches_selection(
    graph: &SceneGraph,
    manip_id: NodeId,
    node: &TransformManipNode,
) -> bool {
    let target = resolve_manip_target(graph, manip_id, node);
    let manip_parent = graph.parent(manip_id);
    for &sel in graph.selected_nodes() {
        if sel == manip_id {
            return true;
        }
        if target == Some(sel) {
            return true;
        }
        if graph.parent(sel) == Some(manip_id) {
            return true;
        }
        if manip_parent.is_some() && graph.parent(sel) == manip_parent {
            return true;
        }
    }
    false
}

fn apply_manip(
    gizmo: &mut Gizmo,
    graph: &SceneGraph,
    manip_id: NodeId,
    node: &TransformManipNode,
) -> bool {
    let Some(target) = resolve_manip_target(graph, manip_id, node) else {
        return false;
    };
    gizmo.target_node = Some(target);
    gizmo.update_target(graph);
    if node.size > 0.0 {
        gizmo.screen_scale = node.size;
    }
    gizmo.handle_filter = collect_handle_filter(graph, manip_id);
    gizmo.orientation = match node.space {
        ManipSpace::World => Mat4::IDENTITY,
        ManipSpace::Local => graph
            .get(target)
            .and_then(|e| match &e.data {
                NodeData::Transform(t) => Some(t.rotation),
                _ => None,
            })
            .unwrap_or(Mat4::IDENTITY),
    };
    gizmo.visible
}

fn clear_manip_override(gizmo: &mut Gizmo) {
    gizmo.handle_filter = None;
    gizmo.orientation = Mat4::IDENTITY;
}

fn collect_enabled_manips(graph: &SceneGraph) -> Vec<(NodeId, TransformManipNode)> {
    let mut out = Vec::new();
    for id in graph.all_node_ids() {
        if let Some(entry) = graph.get(id) {
            if let NodeData::TransformManip(n) = &entry.data {
                if n.enabled {
                    out.push((id, n.clone()));
                }
            }
        }
    }
    out
}

pub fn sync_gizmo_from_selection(gizmo: &mut Gizmo, graph: &SceneGraph) {
    let manips = collect_enabled_manips(graph);
    let selection_empty = graph.selected_nodes().is_empty();

    if selection_empty {
        if let Some((id, node)) = manips.first() {
            if apply_manip(gizmo, graph, *id, node) {
                return;
            }
        }
        clear_manip_override(gizmo);
        gizmo.target_node = None;
        gizmo.update_target(graph);
        return;
    }

    for (id, node) in &manips {
        if manip_matches_selection(graph, *id, node) && apply_manip(gizmo, graph, *id, node) {
            return;
        }
    }

    clear_manip_override(gizmo);
    gizmo.target_node = find_transform_for_selection(graph);
    gizmo.update_target(graph);
}

impl Engine {
    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo.mode = mode;
        let ids = self.world.graph.all_node_ids();
        for id in ids {
            if has_dragger_children(&self.world.graph, id) {
                continue;
            }
            if let Some(entry) = self.world.graph.get_mut(id) {
                if let NodeData::TransformManip(m) = &mut entry.data {
                    if m.enabled {
                        m.mode = manip_from_gizmo(mode);
                        break;
                    }
                }
            }
        }
    }

    pub fn set_lod_range_scale(&mut self, id: NodeId, scale: f32) -> bool {
        self.world.graph.set_lod_range_scale(id, scale)
    }

    /// Write Fast HLR edges as SVG (visible solid, hidden dashed).
    pub fn export_hidden_line_svg(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let (w, h) = self
            .renderer
            .as_ref()
            .map(|r| r.surface_size())
            .unwrap_or((800, 600));
        let view = self.world.collector.view_matrix;
        let proj = self.world.collector.projection_matrix;
        let cam = self.world.collector.camera_pos;
        let svg = rc3d_render::hidden_line_svg(
            &self.world.cached_draw_calls,
            view,
            proj,
            cam,
            w,
            h,
        );
        std::fs::write(path, svg)
    }
}
