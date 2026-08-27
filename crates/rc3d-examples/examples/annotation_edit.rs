//! Annotation drag-edit prototype: click & drag annotation points in 3D.
//!
//! Usage: cargo run -p rc3d-examples --example annotation_edit
//!
//! Controls:
//!   Left click annotation point → drag to move it
//!   Left click + drag elsewhere → camera orbit
//!   Scroll → zoom | Escape → quit

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_engine_api::{CameraController, Engine, EventRouteOpts};
use rc3d_examples::common::run_app;
use rc3d_scene::annotation::{AnnotationLabelMode, AnnotationPoint, AnnotationStyle};
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use std::cell::RefCell;
use std::rc::Rc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowAttributes;

/// A draggable handle on an annotation point.
#[derive(Clone, Debug)]
struct DragHandle {
    set_id: NodeId,
    elem_idx: usize,
    point_key: &'static str,
    /// Annotation-plane normal (model-local space)
    plane_normal: Vec3,
}

/// Snapshot of an AnnotationSet for undo/redo.
#[derive(Clone)]
struct AnnotationSnapshot {
    set_id: NodeId,
    elements: Vec<AnnotationElement>,
}

struct DragState {
    active: Option<DragHandle>,
    cursor: (f64, f64),
    win_w: u32,
    win_h: u32,
    camera_vp: Mat4,
    camera_eye: Vec3,
    shift_held: bool,
    undo_stack: Vec<AnnotationSnapshot>,
    redo_stack: Vec<AnnotationSnapshot>,
}

fn build_scene(graph: &mut SceneGraph) {
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(5.0, 4.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }),
    );

    // Gray cube at origin
    graph.add_child(
        root,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.5, 0.5, 0.6),
            roughness: 0.5,
            ..Default::default()
        }),
    );
    graph.add_child(root, NodeData::Cube(CubeNode { width: 2.0, height: 2.0, depth: 2.0 }));

    // Editable annotation — top and right face dimensions
    graph.add_child(
        root,
        NodeData::AnnotationSet(AnnotationSetNode {
            style: AnnotationStyle {
                unit_suffix: " mm".into(),
                ..AnnotationStyle::default()
            },
            elements: vec![
                AnnotationElement::Dimension {
                    start: [-1.0, 1.0, -1.0].into(),
                    end: [1.0, 1.0, -1.0].into(),
                    offset_dir: [0.0, 1.0, 0.0],
                    extension_len: 0.3,
                    arrow_size: 0.15,
                    label: "宽=2.0".into(),
                    label_mode: AnnotationLabelMode::Prefix,
                    color: [1.0, 0.5, 0.0, 1.0],
                },
                AnnotationElement::Dimension {
                    start: [1.0, 1.0, -1.0].into(),
                    end: [1.0, -1.0, -1.0].into(),
                    offset_dir: [1.0, 0.0, 0.0],
                    extension_len: 0.3,
                    arrow_size: 0.15,
                    label: "高=2.0".into(),
                    label_mode: AnnotationLabelMode::Prefix,
                    color: [1.0, 0.5, 0.0, 1.0],
                },
            ],
            visible: true,
            pmi: Vec::new(),
        }),
    );
}

struct AnnotationEditApp {
    engine: Option<Engine>,
    window: Option<winit::window::Window>,
    drag: Rc<RefCell<DragState>>,
}

impl ApplicationHandler for AnnotationEditApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(WindowAttributes::default().with_title("Annotation Edit"))
            .expect("failed to create window");
        let mut engine = Engine::new(&window);
        build_scene(engine.scene_mut());
        self.drag.borrow_mut().win_w = window.inner_size().width;
        self.drag.borrow_mut().win_h = window.inner_size().height;
        self.engine = Some(engine);
        self.window = Some(window);
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        let Some(window) = self.window.as_ref() else {
            return;
        };
        match &event {
            WindowEvent::RedrawRequested => {
                engine.render();
                window.request_redraw();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                engine.resize(size.width, size.height);
                self.drag.borrow_mut().win_w = size.width;
                self.drag.borrow_mut().win_h = size.height;
            }
            WindowEvent::CursorMoved { .. } => {
                engine.feed_input(&event);
                let cursor = engine.input.cursor_pos;
                let dragging = {
                    let mut ds = self.drag.borrow_mut();
                    ds.cursor = cursor;
                    let dragging = ds.active.is_some();
                    if dragging {
                        update_drag_position(engine.scene_mut(), &mut ds);
                    }
                    dragging
                };
                if !dragging {
                    engine.dispatch_routed_event(
                        &event,
                        EventRouteOpts {
                            left_orbit: engine.on_pick.is_none(),
                            pick_on_click: false,
                            camera: true,
                        },
                    );
                }
                window.request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left && *state == ElementState::Pressed {
                    let mut ds = self.drag.borrow_mut();
                    ds.camera_vp = compute_vp(
                        &engine.controller,
                        ds.win_w,
                        ds.win_h,
                        engine.scene(),
                    );
                    ds.camera_eye = engine.controller.eye_position();

                    if let Some(handle) = pick_annotation_handle(
                        engine.scene(),
                        ds.camera_vp,
                        ds.cursor,
                        ds.win_w,
                        ds.win_h,
                    ) {
                        save_undo_snapshot(engine.scene(), handle.set_id, &mut ds);
                        ds.active = Some(handle);
                    }
                }
                if *button == MouseButton::Left && *state == ElementState::Released {
                    self.drag.borrow_mut().active = None;
                }
                if self.drag.borrow().active.is_none() {
                    engine.handle_window_event(
                        &event,
                        EventRouteOpts {
                            left_orbit: engine.on_pick.is_none(),
                            pick_on_click: false,
                            camera: true,
                        },
                    );
                }
            }
            WindowEvent::MouseWheel { .. } => {
                engine.handle_window_event(
                    &event,
                    EventRouteOpts {
                        left_orbit: true,
                        pick_on_click: false,
                        camera: true,
                    },
                );
                window.request_redraw();
            }
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                engine.handle_window_event(
                    &event,
                    EventRouteOpts {
                        left_orbit: true,
                        pick_on_click: false,
                        camera: false,
                    },
                );
                use winit::keyboard::PhysicalKey;
                match &key_event.physical_key {
                    PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft)
                    | PhysicalKey::Code(winit::keyboard::KeyCode::ShiftRight) => {
                        self.drag.borrow_mut().shift_held =
                            key_event.state == ElementState::Pressed;
                    }
                    PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                        if key_event.state == ElementState::Pressed {
                            event_loop.exit();
                        }
                    }
                    PhysicalKey::Code(winit::keyboard::KeyCode::KeyZ) => {
                        if key_event.state == ElementState::Pressed && self.drag.borrow().shift_held {
                            let mut ds = self.drag.borrow_mut();
                            if let Some(snap) = ds.redo_stack.pop() {
                                save_undo_snapshot_no_clear(engine.scene(), snap.set_id, &mut ds);
                                apply_snapshot(engine.scene_mut(), &snap);
                                window.request_redraw();
                            }
                        } else if key_event.state == ElementState::Pressed {
                            let mut ds = self.drag.borrow_mut();
                            if let Some(snap) = ds.undo_stack.pop() {
                                if let Some(entry) = engine.scene().get(snap.set_id) {
                                    if let NodeData::AnnotationSet(ann) = &entry.data {
                                        ds.redo_stack.push(AnnotationSnapshot {
                                            set_id: snap.set_id,
                                            elements: ann.elements.clone(),
                                        });
                                    }
                                }
                                apply_snapshot(engine.scene_mut(), &snap);
                                window.request_redraw();
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {
                engine.handle_window_event(
                    &event,
                    EventRouteOpts {
                        left_orbit: engine.on_pick.is_none(),
                        pick_on_click: false,
                        camera: true,
                    },
                );
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn main() {
    let _ = env_logger::try_init();

    let drag = Rc::new(RefCell::new(DragState {
        active: None,
        cursor: (0.0, 0.0),
        win_w: 800,
        win_h: 600,
        camera_vp: Mat4::IDENTITY,
        camera_eye: Vec3::ZERO,
        shift_held: false,
        undo_stack: Vec::new(),
        redo_stack: Vec::new(),
    }));

    run_app(AnnotationEditApp {
        engine: None,
        window: None,
        drag,
    });
}

/// Build view-projection from camera controller + camera node projection.
fn compute_vp(ctrl: &CameraController, w: u32, h: u32, _graph: &SceneGraph) -> Mat4 {
    let aspect = w as f32 / h as f32;
    // Default perspective matching the example camera
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 1000.0);
    proj * ctrl.view_matrix()
}

/// Cast a ray from screen position into world space.
fn screen_ray(vp: Mat4, sx: f64, sy: f64, w: u32, h: u32) -> (Vec3, Vec3) {
    let ndc_x = (sx as f32 / w as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - (sy as f32 / h as f32) * 2.0;
    let inv_vp = vp.inverse();
    let near = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
    let far = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
    let dir = (far - near).normalize();
    (near, dir)
}

/// Distance from a point to a 3D ray.
fn point_to_ray_dist(ray_origin: Vec3, ray_dir: Vec3, pt: Vec3) -> f32 {
    let v = pt - ray_origin;
    let proj = ray_dir * v.dot(ray_dir);
    (v - proj).length()
}

/// Intersection of camera ray with a plane (returns point or None).
fn ray_plane_intersect(ro: Vec3, rd: Vec3, pp: Vec3, pn: Vec3) -> Option<Vec3> {
    let denom = rd.dot(pn);
    if denom.abs() < 1e-6 { return None; }
    let t = (pp - ro).dot(pn) / denom;
    if t > 0.0 { Some(ro + rd * t) } else { None }
}

/// Collect draggable point keys and model-local positions from an element.
/// The "all" key represents dragging the entire annotation (applies delta to all points).
fn element_handles(elem: &AnnotationElement) -> Vec<(&'static str, [f32; 3])> {
    let mut handles = match elem {
        AnnotationElement::Dimension { start, end, offset_dir, .. } => vec![
            ("start", start.coords()),
            ("end", end.coords()),
            ("offset_mid", {
                let s = Vec3::from(start.coords());
                let e = Vec3::from(end.coords());
                let off = Vec3::from(*offset_dir);
                ((s + e) * 0.5 + off).into()
            }),
        ],
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
        AnnotationElement::Leader { anchor, .. } => vec![
            ("anchor", anchor.coords()),
        ],
        AnnotationElement::Callout { anchor, .. } => vec![
            ("anchor", anchor.coords()),
        ],
        AnnotationElement::Datum { position, .. } => vec![
            ("position", position.coords()),
        ],
        AnnotationElement::GdtFeatureControlFrame { position, leader_target, .. } => {
            let mut h = vec![("position", position.coords())];
            if let Some(lt) = leader_target {
                h.push(("leader", lt.coords()));
            }
            h
        }
        AnnotationElement::GdtDatumTarget { position, .. } => vec![
            ("position", position.coords()),
        ],
        AnnotationElement::ChamferDimension { start, end, offset_dir, .. } => vec![
            ("start", start.coords()),
            ("end", end.coords()),
            ("offset_mid", {
                let s = Vec3::from(start.coords());
                let e = Vec3::from(end.coords());
                let off = Vec3::from(*offset_dir);
                ((s + e) * 0.5 + off).into()
            }),
        ],
        AnnotationElement::OrdinateDimension { feature, datum, .. } => vec![
            ("feature", feature.coords()),
            ("datum", datum.coords()),
        ],
        AnnotationElement::SurfaceFinish { position, .. }
        | AnnotationElement::WeldSymbol { position, .. }
        | AnnotationElement::DatumIdentifier { position, .. } => vec![
            ("position", position.coords()),
        ],
    };
    // Add "all" handle at average of all points for overall drag
    if !handles.is_empty() {
        let sum: [f32; 3] = handles.iter().fold([0.0; 3], |acc, (_, p)| {
            [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
        });
        let n = handles.len() as f32;
        handles.push(("all", [sum[0] / n, sum[1] / n, sum[2] / n]));
    }
    handles
}

/// Compute annotation plane normal from element.
fn element_plane_normal(elem: &AnnotationElement) -> Option<Vec3> {
    match elem {
        AnnotationElement::Dimension { start, end, offset_dir, .. } => {
            let s = Vec3::from(start.coords());
            let e = Vec3::from(end.coords());
            let off = Vec3::from(*offset_dir);
            let n = (e - s).cross(off);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => {
            let c = Vec3::from(center.coords());
            let a1 = Vec3::from(arm1.coords());
            let a2 = Vec3::from(arm2.coords());
            let n = (a1 - c).cross(a2 - c);
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
        AnnotationElement::Datum { .. } => Some(Vec3::Z),
        _ => None,
    }
}

/// Find the closest annotation handle within pick threshold.
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
    graph: &SceneGraph,
    node: NodeId,
    ro: Vec3, rd: Vec3,
    threshold: f32,
    best: &mut Option<DragHandle>,
    best_dist: &mut f32,
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
                            *best = Some(DragHandle {
                                set_id: node,
                                elem_idx: ei,
                                point_key: pk,
                                plane_normal: plane_n,
                            });
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

/// Update the dragged point to the cursor's intersection with the annotation plane.
/// When shift is held, the movement is locked to the nearest principal axis.
fn update_drag_position(
    graph: &mut SceneGraph,
    ds: &mut DragState,
) {
    let handle = match &ds.active {
        Some(ref h) => h.clone(),
        None => return,
    };
    let (ro, rd) = screen_ray(ds.camera_vp, ds.cursor.0, ds.cursor.1, ds.win_w, ds.win_h);

    // Find the current anchor point for plane intersection reference
    let entry = match graph.get(handle.set_id) {
        Some(e) => e,
        None => return,
    };
    let anchor_ws = match &entry.data {
        NodeData::AnnotationSet(ann) => {
            ann.elements.get(handle.elem_idx).and_then(|elem| {
                element_handles(elem).into_iter().find(|(k, _)| *k == handle.point_key)
            })
        }
        _ => None,
    };
    let anchor_ws = match anchor_ws {
        Some((_, pt)) => Vec3::from(pt),
        None => return,
    };

    // Intersect camera ray with the annotation plane (through the anchor point)
    let mut new_world = match ray_plane_intersect(ro, rd, anchor_ws, handle.plane_normal) {
        Some(p) => p,
        None => return,
    };

    // Shift-lock: project world-space delta to nearest principal axis
    if ds.shift_held {
        let delta = new_world - anchor_ws;
        let abs = [delta.x.abs(), delta.y.abs(), delta.z.abs()];
        let axis = if abs[0] >= abs[1] && abs[0] >= abs[2] { 0 }
            else if abs[1] >= abs[2] { 1 }
            else { 2 };
        let mut snapped = Vec3::ZERO;
        snapped[axis] = delta[axis];
        new_world = anchor_ws + snapped;
    }

    let new_local: [f32; 3] = new_world.into();
    update_element_point(graph, handle.set_id, handle.elem_idx, handle.point_key, new_local);
}

/// Save the current state of an AnnotationSet to the undo stack (clears redo).
fn save_undo_snapshot(graph: &SceneGraph, set_id: NodeId, ds: &mut DragState) {
    if let Some(entry) = graph.get(set_id) {
        if let NodeData::AnnotationSet(ann) = &entry.data {
            ds.undo_stack.push(AnnotationSnapshot {
                set_id,
                elements: ann.elements.clone(),
            });
            ds.redo_stack.clear();
        }
    }
}

/// Save without clearing redo stack (used during undo/redo navigation).
fn save_undo_snapshot_no_clear(graph: &SceneGraph, set_id: NodeId, ds: &mut DragState) {
    if let Some(entry) = graph.get(set_id) {
        if let NodeData::AnnotationSet(ann) = &entry.data {
            ds.undo_stack.push(AnnotationSnapshot {
                set_id,
                elements: ann.elements.clone(),
            });
        }
    }
}

/// Restore an AnnotationSet from a snapshot.
fn apply_snapshot(graph: &mut SceneGraph, snap: &AnnotationSnapshot) {
    if let Some(entry) = graph.get_mut(snap.set_id) {
        if let NodeData::AnnotationSet(ref mut ann) = &mut entry.data {
            ann.elements = snap.elements.clone();
        }
    }
}

fn update_element_point(
    graph: &mut SceneGraph,
    set_id: NodeId,
    elem_idx: usize,
    point_key: &str,
    new_local: [f32; 3],
) {
    let Some(entry) = graph.get_mut(set_id) else { return };
    if let NodeData::AnnotationSet(ref mut ann) = &mut entry.data {
        let Some(elem) = ann.elements.get_mut(elem_idx) else { return };
        // "all" handle: apply delta to every point
        if point_key == "all" {
            let old_handles = element_handles(elem);
            let old_mid = old_handles.iter().find(|(k, _)| *k == "all")
                .map(|(_, p)| Vec3::from(*p)).unwrap_or(Vec3::ZERO);
            let delta = Vec3::from(new_local) - old_mid;
            let _first_pt = old_handles.first().map(|(_, p)| Vec3::from(*p) + delta).unwrap_or(Vec3::ZERO);
            // Apply to all points via concrete key handling
            // We need to shift all points by delta
            match elem {
                AnnotationElement::Dimension { ref mut start, ref mut end, .. } => {
                    let sp: [f32; 3] = (Vec3::from(start.coords()) + delta).into();
                    let ep: [f32; 3] = (Vec3::from(end.coords()) + delta).into();
                    *start = AnnotationPoint::local(sp);
                    *end = AnnotationPoint::local(ep);
                }
                AnnotationElement::AngleDimension { ref mut center, ref mut arm1, ref mut arm2, .. } => {
                    *center = AnnotationPoint::local((Vec3::from(center.coords()) + delta).into());
                    *arm1 = AnnotationPoint::local((Vec3::from(arm1.coords()) + delta).into());
                    *arm2 = AnnotationPoint::local((Vec3::from(arm2.coords()) + delta).into());
                }
                AnnotationElement::RadialDimension { ref mut center, ref mut perimeter, .. } => {
                    *center = AnnotationPoint::local((Vec3::from(center.coords()) + delta).into());
                    *perimeter = AnnotationPoint::local((Vec3::from(perimeter.coords()) + delta).into());
                }
                AnnotationElement::DiameterDimension { ref mut center, ref mut p1, ref mut p2, .. } => {
                    *center = AnnotationPoint::local((Vec3::from(center.coords()) + delta).into());
                    *p1 = AnnotationPoint::local((Vec3::from(p1.coords()) + delta).into());
                    *p2 = AnnotationPoint::local((Vec3::from(p2.coords()) + delta).into());
                }
                AnnotationElement::Leader { ref mut anchor, .. } => {
                    *anchor = AnnotationPoint::local((Vec3::from(anchor.coords()) + delta).into());
                }
                AnnotationElement::Callout { ref mut anchor, .. } => {
                    *anchor = AnnotationPoint::local((Vec3::from(anchor.coords()) + delta).into());
                }
                AnnotationElement::Datum { ref mut position, .. } => {
                    *position = AnnotationPoint::local((Vec3::from(position.coords()) + delta).into());
                }
                AnnotationElement::GdtFeatureControlFrame { ref mut position, ref mut leader_target, .. } => {
                    *position = AnnotationPoint::local((Vec3::from(position.coords()) + delta).into());
                    if let Some(ref mut lt) = leader_target {
                        *lt = AnnotationPoint::local((Vec3::from(lt.coords()) + delta).into());
                    }
                }
                AnnotationElement::GdtDatumTarget { ref mut position, .. } => {
                    *position = AnnotationPoint::local((Vec3::from(position.coords()) + delta).into());
                }
                AnnotationElement::ChamferDimension { ref mut start, ref mut end, .. } => {
                    *start = AnnotationPoint::local((Vec3::from(start.coords()) + delta).into());
                    *end = AnnotationPoint::local((Vec3::from(end.coords()) + delta).into());
                }
                AnnotationElement::OrdinateDimension { ref mut feature, ref mut datum, .. } => {
                    *feature = AnnotationPoint::local((Vec3::from(feature.coords()) + delta).into());
                    *datum = AnnotationPoint::local((Vec3::from(datum.coords()) + delta).into());
                }
                AnnotationElement::SurfaceFinish { ref mut position, .. }
                | AnnotationElement::WeldSymbol { ref mut position, .. }
                | AnnotationElement::DatumIdentifier { ref mut position, .. } => {
                    *position = AnnotationPoint::local((Vec3::from(position.coords()) + delta).into());
                }
            }
            return;
        }
        match elem {
            AnnotationElement::Dimension { ref mut start, ref mut end, ref mut offset_dir, .. } => {
                let s = Vec3::from(start.coords());
                let e = Vec3::from(end.coords());
                match point_key {
                    "start" => *start = AnnotationPoint::local(new_local),
                    "end" => *end = AnnotationPoint::local(new_local),
                    "offset_mid" => *offset_dir = (Vec3::from(new_local) - (s + e) * 0.5).into(),
                    _ => {}
                }
            }
            AnnotationElement::AngleDimension { ref mut center, ref mut arm1, ref mut arm2, .. } => {
                let p = AnnotationPoint::local(new_local);
                match point_key {
                    "center" => *center = p,
                    "arm1" => *arm1 = p,
                    "arm2" => *arm2 = p,
                    _ => {}
                }
            }
            AnnotationElement::RadialDimension { ref mut center, ref mut perimeter, .. } => {
                let p = AnnotationPoint::local(new_local);
                match point_key {
                    "center" => *center = p,
                    "perimeter" => *perimeter = p,
                    _ => {}
                }
            }
            AnnotationElement::DiameterDimension { ref mut center, ref mut p1, ref mut p2, .. } => {
                let p = AnnotationPoint::local(new_local);
                match point_key {
                    "center" => *center = p,
                    "p1" => *p1 = p,
                    "p2" => *p2 = p,
                    _ => {}
                }
            }
            AnnotationElement::Leader { ref mut anchor, .. } => {
                *anchor = AnnotationPoint::local(new_local);
            }
            AnnotationElement::Callout { ref mut anchor, .. } => {
                *anchor = AnnotationPoint::local(new_local);
            }
            AnnotationElement::Datum { ref mut position, .. } => {
                *position = AnnotationPoint::local(new_local);
            }
            AnnotationElement::GdtFeatureControlFrame { ref mut position, ref mut leader_target, .. } => {
                match point_key {
                    "position" => *position = AnnotationPoint::local(new_local),
                    "leader" => { *leader_target = Some(AnnotationPoint::local(new_local)); }
                    _ => {}
                }
            }
            AnnotationElement::GdtDatumTarget { ref mut position, .. } => {
                *position = AnnotationPoint::local(new_local);
            }
            AnnotationElement::ChamferDimension { ref mut start, ref mut end, ref mut offset_dir, .. } => {
                let s = Vec3::from(start.coords());
                let e = Vec3::from(end.coords());
                match point_key {
                    "start" => *start = AnnotationPoint::local(new_local),
                    "end" => *end = AnnotationPoint::local(new_local),
                    "offset_mid" => *offset_dir = (Vec3::from(new_local) - (s + e) * 0.5).into(),
                    _ => {}
                }
            }
            AnnotationElement::OrdinateDimension { ref mut feature, ref mut datum, .. } => {
                let p = AnnotationPoint::local(new_local);
                match point_key {
                    "feature" => *feature = p,
                    "datum" => *datum = p,
                    _ => {}
                }
            }
            AnnotationElement::SurfaceFinish { ref mut position, .. }
            | AnnotationElement::WeldSymbol { ref mut position, .. }
            | AnnotationElement::DatumIdentifier { ref mut position, .. } => {
                *position = AnnotationPoint::local(new_local);
            }
        }
    }
}
