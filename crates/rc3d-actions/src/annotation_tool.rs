//! Interactive annotation editing: drag points, undo/redo, axis locking.
//!
//! ```ignore
//! let mut tool = AnnotationTool::new();
//! tool.on_mouse_down(graph, vp, cursor, w, h);
//! tool.on_mouse_move(graph, vp, cursor, w, h, shift_held);
//! tool.on_mouse_up();
//! tool.undo(graph);
//! tool.redo(graph);
//! ```

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::annotation::AnnotationPoint;
use rc3d_scene::node_data::AnnotationElement;
use rc3d_scene::{NodeData, SceneGraph};

use crate::undo::{Command, CommandHistory};

// ── Tool state ──────────────────────────────────────────────────

/// Draggable handle on an annotation point.
#[derive(Clone, Debug)]
pub struct DragHandle {
    pub set_id: NodeId,
    pub elem_idx: usize,
    pub point_key: &'static str,
    pub plane_normal: Vec3,
}

/// Interactive annotation editing tool.
pub struct AnnotationTool {
    pub active_drag: Option<DragHandle>,
    pub history: CommandHistory,
    cursor: (f64, f64),
    win_w: u32,
    win_h: u32,
    camera_vp: Mat4,
    /// World-space position of the dragged point at drag start
    drag_anchor: Option<Vec3>,
}

impl AnnotationTool {
    pub fn new() -> Self {
        Self {
            active_drag: None,
            history: CommandHistory::new(128),
            cursor: (0.0, 0.0),
            win_w: 1,
            win_h: 1,
            camera_vp: Mat4::IDENTITY,
            drag_anchor: None,
        }
    }

    /// Set viewport dimensions and camera state.
    pub fn set_camera(&mut self, vp: Mat4, w: u32, h: u32) {
        self.camera_vp = vp;
        self.win_w = w.max(1);
        self.win_h = h.max(1);
    }

    pub fn is_dragging(&self) -> bool {
        self.active_drag.is_some()
    }

    pub fn can_undo(&self) -> bool { self.history.can_undo() }
    pub fn can_redo(&self) -> bool { self.history.can_redo() }

    // ── Event handlers ───────────────────────────────────────

    /// Try to pick and start dragging an annotation point.
    pub fn on_mouse_down(
        &mut self,
        graph: &SceneGraph,
        cursor: (f64, f64),
    ) -> bool {
        self.cursor = cursor;
        if let Some(handle) = pick_annotation_handle(graph, self.camera_vp, cursor, self.win_w, self.win_h) {
            // Save current state for undo
            if let Some(snap_cmd) = snapshot_command(graph, handle.set_id, handle.elem_idx) {
                self.history.execute(Box::new(snap_cmd), &mut SceneGraph::new()); // dummy execute, real state already saved
            }
            // Save anchor for relative drag
            self.drag_anchor = element_point_world(graph, handle.set_id, handle.elem_idx, handle.point_key);
            self.active_drag = Some(handle);
            true
        } else {
            false
        }
    }

    /// Update drag position. `shift_held` locks to nearest axis.
    pub fn on_mouse_move(
        &mut self,
        graph: &mut SceneGraph,
        cursor: (f64, f64),
        shift_held: bool,
    ) {
        self.cursor = cursor;
        let handle = match &self.active_drag {
            Some(h) => h.clone(),
            None => return,
        };

        let (ro, rd) = screen_ray(self.camera_vp, cursor.0, cursor.1, self.win_w, self.win_h);
        let plane_n = handle.plane_normal;

        // Get current anchor point for plane intersection
        let anchor = match self.drag_anchor {
            Some(a) => a,
            None => match element_point_world(graph, handle.set_id, handle.elem_idx, handle.point_key) {
                Some(a) => a,
                None => return,
            },
        };

        let mut new_world = match ray_plane_intersect(ro, rd, anchor, plane_n) {
            Some(p) => p,
            None => return,
        };

        // Shift-lock: project delta to nearest axis
        if shift_held {
            let delta = new_world - anchor;
            let abs = [delta.x.abs(), delta.y.abs(), delta.z.abs()];
            let axis = if abs[0] >= abs[1] && abs[0] >= abs[2] { 0 }
                else if abs[1] >= abs[2] { 1 }
                else { 2 };
            let mut snapped = Vec3::ZERO;
            snapped[axis] = delta[axis];
            new_world = anchor + snapped;
        }

        let new_local: [f32; 3] = new_world.into();
        apply_point_update(graph, handle.set_id, handle.elem_idx, handle.point_key, new_local);
    }

    pub fn on_mouse_up(&mut self) {
        self.active_drag = None;
        self.drag_anchor = None;
    }

    pub fn undo(&mut self, graph: &mut SceneGraph) -> bool {
        self.history.undo(graph)
    }

    pub fn redo(&mut self, graph: &mut SceneGraph) -> bool {
        self.history.redo(graph)
    }
}

// ── Ray + pick helpers ──────────────────────────────────────────

fn screen_ray(vp: Mat4, sx: f64, sy: f64, w: u32, h: u32) -> (Vec3, Vec3) {
    let ndc_x = (sx as f32 / w as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - (sy as f32 / h as f32) * 2.0;
    let inv_vp = vp.inverse();
    let near = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
    let far = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
    (near, (far - near).normalize())
}

fn point_to_ray_dist(ro: Vec3, rd: Vec3, pt: Vec3) -> f32 {
    let v = pt - ro;
    (v - rd * v.dot(rd)).length()
}

fn ray_plane_intersect(ro: Vec3, rd: Vec3, pp: Vec3, pn: Vec3) -> Option<Vec3> {
    let denom = rd.dot(pn);
    if denom.abs() < 1e-6 { return None; }
    let t = (pp - ro).dot(pn) / denom;
    if t > 0.0 { Some(ro + rd * t) } else { None }
}

/// Get a single element point in world space (annotation is in set-local space).
fn element_point_world(
    graph: &SceneGraph,
    set_id: NodeId,
    elem_idx: usize,
    point_key: &str,
) -> Option<Vec3> {
    let entry = graph.get(set_id)?;
    if let NodeData::AnnotationSet(ann) = &entry.data {
        ann.elements.get(elem_idx).and_then(|elem| {
            element_handles(elem).into_iter().find(|(k, _)| *k == point_key)
                .map(|(_, pt)| Vec3::from(pt))
        })
    } else { None }
}

fn pick_annotation_handle(
    graph: &SceneGraph,
    vp: Mat4,
    cursor: (f64, f64),
    w: u32, h: u32,
) -> Option<DragHandle> {
    let (ro, rd) = screen_ray(vp, cursor.0, cursor.1, w, h);
    let threshold = 0.35;
    let mut best: Option<DragHandle> = None;
    let mut best_dist = f32::INFINITY;
    for &root in graph.roots() {
        pick_recursive(graph, root, ro, rd, threshold, &mut best, &mut best_dist);
    }
    best
}

fn pick_recursive(
    graph: &SceneGraph, node: NodeId,
    ro: Vec3, rd: Vec3, threshold: f32,
    best: &mut Option<DragHandle>, best_dist: &mut f32,
) {
    let Some(entry) = graph.get(node) else { return };
    if let NodeData::AnnotationSet(ann) = &entry.data {
        if ann.visible {
            for (ei, elem) in ann.elements.iter().enumerate() {
                if let Some(plane_n) = element_plane_normal(elem) {
                    for (pk, pt_local) in element_handles(elem) {
                        let pt = Vec3::from(pt_local);
                        let dist = point_to_ray_dist(ro, rd, pt);
                        if dist < threshold && dist < *best_dist {
                            *best_dist = dist;
                            *best = Some(DragHandle { set_id: node, elem_idx: ei, point_key: pk, plane_normal: plane_n });
                        }
                    }
                }
            }
        }
    }
    for &child in &entry.children {
        pick_recursive(graph, child, ro, rd, threshold, best, best_dist);
    }
}

// ── Element handles ─────────────────────────────────────────────

fn element_handles(elem: &AnnotationElement) -> Vec<(&'static str, [f32; 3])> {
    let mut handles = match elem {
        AnnotationElement::Dimension { start, end, offset_dir, .. } => {
            let s = Vec3::from(start.coords());
            let e = Vec3::from(end.coords());
            let off = Vec3::from(*offset_dir);
            vec![
                ("start", start.coords()),
                ("end", end.coords()),
                ("offset_mid", ((s + e) * 0.5 + off).into()),
            ]
        }
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => vec![
            ("center", center.coords()),
            ("arm1", arm1.coords()),
            ("arm2", arm2.coords()),
        ],
        AnnotationElement::RadialDimension { center, perimeter, .. } => vec![
            ("center", center.coords()),
            ("perimeter", perimeter.coords()),
        ],
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => vec![
            ("center", center.coords()),
            ("p1", p1.coords()),
            ("p2", p2.coords()),
        ],
        AnnotationElement::GdtFeatureControlFrame { position, leader_target, .. } => {
            let mut pts = vec![("position", position.coords())];
            if let Some(ref tgt) = leader_target {
                pts.push(("leader_target", tgt.coords()));
            }
            pts
        }
        AnnotationElement::GdtDatumTarget { position, .. } => vec![
            ("position", position.coords()),
        ],
        AnnotationElement::ChamferDimension { start, end, .. } => vec![
            ("start", start.coords()),
            ("end", end.coords()),
        ],
        AnnotationElement::OrdinateDimension { feature, datum, .. } => vec![
            ("feature", feature.coords()),
            ("datum", datum.coords()),
        ],
        AnnotationElement::Leader { anchor, .. } => vec![("anchor", anchor.coords())],
        AnnotationElement::Callout { anchor, .. } => vec![("anchor", anchor.coords())],
        AnnotationElement::Datum { position, .. } => vec![("position", position.coords())],
        AnnotationElement::SurfaceFinish { position, .. } => vec![("position", position.coords())],
        AnnotationElement::WeldSymbol { position, .. } => vec![("position", position.coords())],
        AnnotationElement::DatumIdentifier { position, .. } => vec![("position", position.coords())],
    };
    if !handles.is_empty() {
        let sum: [f32; 3] = handles.iter().fold([0.0; 3], |acc, (_, p)| {
            [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
        });
        let n = handles.len() as f32;
        handles.push(("all", [sum[0] / n, sum[1] / n, sum[2] / n]));
    }
    handles
}

fn element_plane_normal(elem: &AnnotationElement) -> Option<Vec3> {
    match elem {
        AnnotationElement::Dimension { start, end, offset_dir, .. } => {
            let d = Vec3::from(end.coords()) - Vec3::from(start.coords());
            let o = Vec3::from(*offset_dir);
            let n = d.cross(o);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => {
            let c = Vec3::from(center.coords());
            let n = (Vec3::from(arm1.coords()) - c).cross(Vec3::from(arm2.coords()) - c);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::RadialDimension { center, perimeter, .. } => {
            let d = Vec3::from(perimeter.coords()) - Vec3::from(center.coords());
            if d.length_squared() > 1e-8 { Some(d.normalize()) } else { None }
        }
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => {
            let c = Vec3::from(center.coords());
            let n = (Vec3::from(p1.coords()) - c).cross(Vec3::from(p2.coords()) - c);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::ChamferDimension { start, end, offset_dir, .. } => {
            let d = Vec3::from(end.coords()) - Vec3::from(start.coords());
            let o = Vec3::from(*offset_dir);
            let n = d.cross(o);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::OrdinateDimension { feature, datum, axis_dir, .. } => {
            let d = Vec3::from(feature.coords()) - Vec3::from(datum.coords());
            let a = Vec3::from(*axis_dir);
            let n = d.cross(a);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::Datum { .. } | AnnotationElement::GdtDatumTarget { .. } => Some(Vec3::Z),
        AnnotationElement::SurfaceFinish { .. }
        | AnnotationElement::WeldSymbol { .. }
        | AnnotationElement::DatumIdentifier { .. } => Some(Vec3::Z),
        _ => None,
    }
}

// ── Point update ────────────────────────────────────────────────

fn apply_point_update(
    graph: &mut SceneGraph,
    set_id: NodeId,
    elem_idx: usize,
    point_key: &str,
    new_local: [f32; 3],
) {
    let Some(entry) = graph.get_mut(set_id) else { return };
    let NodeData::AnnotationSet(ref mut ann) = &mut entry.data else { return };
    let Some(elem) = ann.elements.get_mut(elem_idx) else { return };

    if point_key == "all" {
        let old_handles = element_handles(elem);
        let old_mid = old_handles.iter().find(|(k, _)| *k == "all")
            .map(|(_, p)| Vec3::from(*p)).unwrap_or(Vec3::ZERO);
        let delta = Vec3::from(new_local) - old_mid;
        shift_all_points(elem, delta);
        return;
    }

    match elem {
        AnnotationElement::Dimension { start, end, offset_dir, .. } => {
            let s = Vec3::from(start.coords());
            let e = Vec3::from(end.coords());
            match point_key {
                "start" => *start = AnnotationPoint::local(new_local),
                "end" => *end = AnnotationPoint::local(new_local),
                "offset_mid" => *offset_dir = (Vec3::from(new_local) - (s + e) * 0.5).into(),
                _ => {}
            }
        }
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => {
            let p = AnnotationPoint::local(new_local);
            match point_key { "center" => *center = p, "arm1" => *arm1 = p, "arm2" => *arm2 = p, _ => {} }
        }
        AnnotationElement::RadialDimension { center, perimeter, .. } => {
            let p = AnnotationPoint::local(new_local);
            match point_key { "center" => *center = p, "perimeter" => *perimeter = p, _ => {} }
        }
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => {
            let p = AnnotationPoint::local(new_local);
            match point_key { "center" => *center = p, "p1" => *p1 = p, "p2" => *p2 = p, _ => {} }
        }
        AnnotationElement::GdtFeatureControlFrame { position, leader_target, .. } => {
            let p = AnnotationPoint::local(new_local);
            match point_key {
                "position" => *position = p,
                "leader_target" => if let Some(ref mut t) = leader_target { *t = p; },
                _ => {}
            }
        }
        AnnotationElement::GdtDatumTarget { position, .. } => { *position = AnnotationPoint::local(new_local); }
        AnnotationElement::ChamferDimension { start, end, .. } => {
            let p = AnnotationPoint::local(new_local);
            match point_key { "start" => *start = p, "end" => *end = p, _ => {} }
        }
        AnnotationElement::OrdinateDimension { feature, datum, .. } => {
            let p = AnnotationPoint::local(new_local);
            match point_key { "feature" => *feature = p, "datum" => *datum = p, _ => {} }
        }
        AnnotationElement::Leader { anchor, .. } => { *anchor = AnnotationPoint::local(new_local); }
        AnnotationElement::Callout { anchor, .. } => { *anchor = AnnotationPoint::local(new_local); }
        AnnotationElement::Datum { position, .. } => { *position = AnnotationPoint::local(new_local); }
        AnnotationElement::SurfaceFinish { position, .. }
        | AnnotationElement::WeldSymbol { position, .. }
        | AnnotationElement::DatumIdentifier { position, .. } => {
            *position = AnnotationPoint::local(new_local);
        }
    }
}

fn shift_all_points(elem: &mut AnnotationElement, delta: Vec3) {
    let shift = |p: &mut AnnotationPoint| *p = AnnotationPoint::local((Vec3::from(p.coords()) + delta).into());
    match elem {
        AnnotationElement::Dimension { start, end, .. } => { shift(start); shift(end); }
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => { shift(center); shift(arm1); shift(arm2); }
        AnnotationElement::RadialDimension { center, perimeter, .. } => { shift(center); shift(perimeter); }
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => { shift(center); shift(p1); shift(p2); }
        AnnotationElement::GdtFeatureControlFrame { position, leader_target, .. } => { shift(position); if let Some(ref mut t) = leader_target { shift(t); } }
        AnnotationElement::GdtDatumTarget { position, .. } => { shift(position); }
        AnnotationElement::ChamferDimension { start, end, .. } => { shift(start); shift(end); }
        AnnotationElement::OrdinateDimension { feature, datum, .. } => { shift(feature); shift(datum); }
        AnnotationElement::Leader { anchor, .. } => { shift(anchor); }
        AnnotationElement::Callout { anchor, .. } => { shift(anchor); }
        AnnotationElement::Datum { position, .. } => { shift(position); }
        AnnotationElement::SurfaceFinish { position, .. } => { shift(position); }
        AnnotationElement::WeldSymbol { position, .. } => { shift(position); }
        AnnotationElement::DatumIdentifier { position, .. } => { shift(position); }
    }
}

// ── Undo command ────────────────────────────────────────────────

/// Captures an AnnotationElement's state for undo/redo.
#[derive(Debug)]
struct AnnotationPointCommand {
    set_id: NodeId,
    elem_idx: usize,
    old_element: AnnotationElement,
    new_element: AnnotationElement,
}

impl Command for AnnotationPointCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        apply_element(graph, self.set_id, self.elem_idx, &self.new_element);
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        apply_element(graph, self.set_id, self.elem_idx, &self.old_element);
    }
    fn description(&self) -> &str { "EditAnnotation" }
}

fn apply_element(graph: &mut SceneGraph, set_id: NodeId, elem_idx: usize, element: &AnnotationElement) {
    if let Some(entry) = graph.get_mut(set_id) {
        if let NodeData::AnnotationSet(ref mut ann) = &mut entry.data {
            if elem_idx < ann.elements.len() {
                ann.elements[elem_idx] = element.clone();
            }
        }
    }
}

fn snapshot_command(graph: &SceneGraph, set_id: NodeId, elem_idx: usize) -> Option<AnnotationPointCommand> {
    let entry = graph.get(set_id)?;
    let NodeData::AnnotationSet(ann) = &entry.data else { return None; };
    let old = ann.elements.get(elem_idx)?.clone();
    Some(AnnotationPointCommand {
        set_id,
        elem_idx,
        old_element: old.clone(),
        new_element: old,
    })
}
