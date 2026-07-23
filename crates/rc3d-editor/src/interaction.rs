//! Editor interaction: mouse-down, mouse-up, cursor-move, box-select, gizmo drag.
//!
//! These functions were extracted from `rc3d-app::App` and refactored to operate
//! on an `EditorContext` (which wraps `&mut Engine`) plus explicit window/input
//! parameters.

use rc3d_actions::{Ray, RayPickAction};
use rc3d_core::math::{Mat4, Quat};
use rc3d_core::NodeId;
use rc3d_engine_api::{Engine, InputState};

use crate::context::EditorContext;
use crate::gizmo;

/// Build a pick ray from the current cursor position, using viewport cameras
/// if available, otherwise falling back to the main camera controller.
pub fn build_pick_ray(
    engine: &Engine,
    input: &InputState,
    window: &winit::window::Window,
) -> Option<Ray> {
    let s = window.inner_size();
    let (lx, ly, vw, vh, v, p) = pointer_pick_frame(
        engine,
        input.cursor_pos.0 as f32,
        input.cursor_pos.1 as f32,
        s.width,
        s.height,
    )?;
    Some(Ray::from_screen_point(lx, ly, vw, vh, v, p))
}

/// Determine the viewport-local pick frame for a cursor position.
fn pointer_pick_frame(
    engine: &Engine,
    cx: f32,
    cy: f32,
    surface_w: u32,
    surface_h: u32,
) -> Option<(f32, f32, f32, f32, Mat4, Mat4)> {
    if !engine.viewport_cameras.cameras.is_empty() {
        let r = engine.renderer.as_ref()?;
        let vp_ref = r
            .viewport_layout()
            .viewport_at(cx, cy)
            .or_else(|| r.viewport_layout().active())
            .or_else(|| r.viewport_layout().viewports.first())?;
        if let Some(vc) = engine.viewport_cameras.find(vp_ref.id) {
            let (v, p) = gizmo::pick_view_proj(&engine.world.graph, vc, vp_ref);
            let lx = cx - vp_ref.rect.x as f32;
            let ly = cy - vp_ref.rect.y as f32;
            let vw = vp_ref.rect.width.max(1) as f32;
            let vh = vp_ref.rect.height.max(1) as f32;
            return Some((lx, ly, vw, vh, v, p));
        }
    }
    // Fallback: main camera controller
    let vw = surface_w.max(1) as f32;
    let vh = surface_h.max(1) as f32;
    let aspect = vw / vh.max(1.0);
    let v = engine.controller.view_matrix();
    let p = Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0);
    Some((cx, cy, vw, vh, v, p))
}

/// Perform a ray pick and toggle selection on the hit node.
pub fn do_pick(engine: &mut Engine, input: &InputState, window: &winit::window::Window) {
    let Some(ray) = build_pick_ray(engine, input, window) else {
        return;
    };
    let mut picker = RayPickAction::new(ray);
    rc3d_actions::apply_to_all_roots(&mut picker, &engine.world.graph);

    if let Some(hit) = picker.hits.first() {
        engine.world.graph.toggle_selection(hit.node);
        log::info!(
            "Pick hit: node={:?}, point={:?}, selected={}",
            hit.node,
            hit.point,
            engine.world.graph.is_selected(hit.node)
        );
    }
}

/// Compute active camera view + projection matrices.
pub fn active_camera_matrices(engine: &Engine, width: f32, height: f32) -> (Mat4, Mat4) {
    let aspect = width / height.max(1.0);
    let proj = Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0);
    if let Some(vc) = engine.viewport_cameras.active() {
        (vc.controller.view_matrix(), proj)
    } else {
        (engine.controller.view_matrix(), proj)
    }
}

/// Handle left mouse button press.
pub fn on_left_down(
    ctx: &mut EditorContext,
    window: &winit::window::Window,
    input: &InputState,
    editor_ui_enabled: bool,
) {
    // Viewport splitter drag
    if let Some(ref r) = ctx.engine.renderer {
        if r.viewport_layout().viewports.len() > 1 {
            let s = window.inner_size();
            if let Some(axis) = r.viewport_layout().splitter_hit(
                input.cursor_pos.0 as f32,
                input.cursor_pos.1 as f32,
                s.width,
                s.height,
            ) {
                ctx.interaction.view_split_drag = Some(axis);
                return;
            }
        }
    }
    // Viewport click to activate – extract vp_id first to avoid double borrow
    let activate_vp: Option<rc3d_render::viewport::ViewportId> =
        if let Some(ref r) = ctx.engine.renderer {
            if r.viewport_layout().viewports.len() > 1 {
                r.viewport_layout()
                    .viewport_at(input.cursor_pos.0 as f32, input.cursor_pos.1 as f32)
                    .map(|v| v.id)
            } else {
                None
            }
        } else {
            None
        };
    if let Some(vp_id) = activate_vp {
        // Access engine fields directly to allow split borrows
        let vcs = &mut ctx.engine.viewport_cameras;
        let layout = ctx.engine.renderer.as_mut().unwrap().viewport_layout_mut();
        vcs.set_active(vp_id, layout);
    }

    let left_orbit_enabled = !editor_ui_enabled && !ctx.interaction.measurement_mode;

    if left_orbit_enabled && !input.ctrl_pressed {
        ctx.interaction.left_pick_arm_pos = Some(input.cursor_pos);
        ctx.interaction.left_drag_suppresses_pick = false;
        return;
    }
    if input.ctrl_pressed {
        ctx.interaction.box_select_drag = true;
        ctx.interaction.box_select_anchor = (
            input.cursor_pos.0 as f32,
            input.cursor_pos.1 as f32,
        );
        return;
    }
    if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
        gizmo::sync_gizmo_from_selection(
            &mut ctx.interaction.gizmo,
            &ctx.engine.world.graph,
        );
        if ctx.interaction.gizmo.visible {
            if let Some((h, _)) = ctx.interaction.gizmo.hit_test(&ray) {
                if let Some(n) = ctx.interaction.gizmo.target_node {
                    if let Some(e) = ctx.engine.world.graph.get(n) {
                        if let rc3d_scene::NodeData::Transform(t) = &e.data {
                            let old = Mat4::from_scale_rotation_translation(
                                t.scale,
                                Quat::from_mat4(&t.rotation),
                                t.translation,
                            );
                            ctx.interaction.gizmo_pending_transform = Some((n, old));
                        }
                    }
                }
                ctx.interaction.gizmo.start_drag(&ray, h);
                ctx.interaction.gizmo_dragging = true;
                return;
            }
        }
    }
    // Fallback: pick
    do_pick(ctx.engine, input, window);
    gizmo::sync_gizmo_from_selection(
        &mut ctx.interaction.gizmo,
        &ctx.engine.world.graph,
    );
}

/// Handle left mouse button release.
pub fn on_left_up(
    ctx: &mut EditorContext,
    window: &winit::window::Window,
    input: &InputState,
    editor_ui_enabled: bool,
) {
    let left_orbit_enabled = !editor_ui_enabled && !ctx.interaction.measurement_mode;

    if left_orbit_enabled
        && !input.ctrl_pressed
        && !ctx.interaction.box_select_drag
        && !ctx.interaction.gizmo_dragging
    {
        if ctx.interaction.left_pick_arm_pos.is_some() {
            let _ = ctx.interaction.left_pick_arm_pos.take();
            if !ctx.interaction.left_drag_suppresses_pick {
                do_pick(ctx.engine, input, window);
                gizmo::sync_gizmo_from_selection(
                    &mut ctx.interaction.gizmo,
                    &ctx.engine.world.graph,
                );
            }
            ctx.interaction.left_drag_suppresses_pick = false;
        }
    } else {
        ctx.interaction.left_pick_arm_pos = None;
        ctx.interaction.left_drag_suppresses_pick = false;
    }
    if ctx.interaction.view_split_drag.is_some() {
        ctx.interaction.view_split_drag = None;
    }
    if ctx.interaction.box_select_drag {
        ctx.interaction.box_select_drag = false;

        // Gather all data we need before mutating world.graph
        let s = window.inner_size();
        let roots = ctx.engine.world.graph.roots().to_vec();
        let has_vp_cams = !ctx.engine.viewport_cameras.cameras.is_empty();
        let vp_id_opt = ctx.engine.viewport_cameras.active().map(|vc| vc.viewport_id);
        let cam_node_opt = ctx.engine.viewport_cameras.active().map(|vc| vc.camera_node);
        let anchor = ctx.interaction.box_select_anchor;
        let cursor = (input.cursor_pos.0 as f32, input.cursor_pos.1 as f32);

        // Now use the data without holding borrows on ctx.engine
        if has_vp_cams {
            if let (Some(vp_id), Some(cam_node)) = (vp_id_opt, cam_node_opt) {
                if let Some(r) = ctx.engine.renderer.as_ref() {
                    if let Some(avp) = r.viewport_layout().viewports.iter().find(|v| v.id == vp_id)
                    {
                        let (v, p) = pick_view_proj_for_node(
                            &ctx.engine.world.graph,
                            cam_node,
                            avp,
                        );
                        // Clone viewport rect info since avp borrows from r
                        let vp_rect = avp.rect;
                        let vp_projection_type = avp.projection_type;
                        let vp_id_copy = avp.id;

                        // Build a temporary viewport with copied data
                        let temp_vp = rc3d_render::viewport::Viewport {
                            id: vp_id_copy,
                            rect: vp_rect,
                            projection_type: vp_projection_type,
                            name: String::new(),
                            camera_node: None,
                            is_active: false,
                        };

                        for &root in &roots {
                            crate::box_select::select_nodes_in_viewport_box(
                                &mut ctx.engine.world.graph,
                                root,
                                anchor,
                                cursor,
                                &temp_vp,
                                v,
                                p,
                            );
                        }
                    }
                }
            }
        } else {
            let (v, p) = active_camera_matrices(ctx.engine, s.width as f32, s.height as f32);
            for &root in &roots {
                crate::box_select::select_nodes_in_screen_box(
                    &mut ctx.engine.world.graph,
                    root,
                    anchor,
                    cursor,
                    s.width as f32,
                    s.height as f32,
                    v,
                    p,
                );
            }
        }
        gizmo::sync_gizmo_from_selection(
            &mut ctx.interaction.gizmo,
            &ctx.engine.world.graph,
        );
        return;
    }
    if ctx.interaction.gizmo_dragging {
        ctx.interaction.gizmo.end_drag();
        if let Some((n, old_mat)) = ctx.interaction.gizmo_pending_transform.take() {
            if let Some(e) = ctx.engine.world.graph.get(n) {
                if let rc3d_scene::NodeData::Transform(t) = &e.data {
                    let new_mat = Mat4::from_scale_rotation_translation(
                        t.scale,
                        Quat::from_mat4(&t.rotation),
                        t.translation,
                    );
                    let (old_scale, old_rot, old_trans) = old_mat.to_scale_rotation_translation();
                    let (new_scale, new_rot, new_trans) = new_mat.to_scale_rotation_translation();

                    if (new_trans - old_trans).length_squared() > 1e-8 {
                        ctx.push_command(
                            crate::commands::EditorCommand::CommitTransformTranslation {
                                node: n,
                                old: old_trans.to_array(),
                                new: new_trans.to_array(),
                            },
                        );
                    }
                    if (new_scale - old_scale).length_squared() > 1e-8 {
                        ctx.push_command(
                            crate::commands::EditorCommand::CommitTransformScale {
                                node: n,
                                old: old_scale.to_array(),
                                new: new_scale.to_array(),
                            },
                        );
                    }
                    let old_q: [f32; 4] = [old_rot.x, old_rot.y, old_rot.z, old_rot.w];
                    let new_q: [f32; 4] = [new_rot.x, new_rot.y, new_rot.z, new_rot.w];
                    if (old_q[0] - new_q[0]).abs() > 1e-6
                        || (old_q[1] - new_q[1]).abs() > 1e-6
                        || (old_q[2] - new_q[2]).abs() > 1e-6
                        || (old_q[3] - new_q[3]).abs() > 1e-6
                    {
                        ctx.push_command(
                            crate::commands::EditorCommand::CommitTransformRotationQuat {
                                node: n,
                                old: old_q,
                                new: new_q,
                            },
                        );
                    }
                }
            }
        }
        ctx.interaction.gizmo_dragging = false;
    }
}

/// Like `gizmo::pick_view_proj` but takes camera node directly instead of ViewportCamera.
fn pick_view_proj_for_node(
    graph: &rc3d_scene::SceneGraph,
    camera_node: NodeId,
    vport: &rc3d_render::viewport::Viewport,
) -> (Mat4, Mat4) {
    if let Some(e) = graph.get(camera_node) {
        match &e.data {
            rc3d_scene::NodeData::PerspectiveCamera(c) => {
                return (c.view_matrix(), c.projection_matrix())
            }
            rc3d_scene::NodeData::OrthographicCamera(c) => {
                return (c.view_matrix(), c.projection_matrix())
            }
            _ => {}
        }
    }
    // Fallback
    let aspect = vport.rect.aspect();
    let v = Mat4::look_at_rh(
        rc3d_core::math::Vec3::new(0.0, 0.0, 5.0),
        rc3d_core::math::Vec3::ZERO,
        rc3d_core::math::Vec3::Y,
    );
    let p = match vport.projection_type {
        rc3d_render::viewport::ProjectionType::Perspective => {
            Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0)
        }
        rc3d_render::viewport::ProjectionType::Orthographic => {
            let height = 5.0 * 1.2;
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

/// Handle cursor movement during drag operations.
pub fn on_cursor_moved(
    ctx: &mut EditorContext,
    window: &winit::window::Window,
    input: &InputState,
) {
    if let Some(axis) = ctx.interaction.view_split_drag {
        let s = window.inner_size();
        if let Some(ref mut r) = ctx.engine.renderer {
            r.viewport_layout_mut().apply_split_drag(
                axis,
                input.cursor_pos.0 as f32,
                input.cursor_pos.1 as f32,
                s.width,
                s.height,
            );
            r.viewport_layout_mut().rebuild(s.width, s.height);
        }
        ctx.engine
            .viewport_cameras
            .remap_viewport_ids_from_layout(
                ctx.engine.renderer.as_ref().unwrap().viewport_layout(),
            );
        window.request_redraw();
        return;
    }
    if ctx.interaction.gizmo_dragging {
        if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
            if let Some((n, old_mat)) = ctx.interaction.gizmo_pending_transform {
                if let Some(d) = ctx.interaction.gizmo.drag_delta(&ray) {
                    let new_mat = d * old_mat;
                    let (scale, rot, trans) = new_mat.to_scale_rotation_translation();
                    if let Some(e) = ctx.engine.world.graph.get_mut(n) {
                        if let rc3d_scene::NodeData::Transform(t) = &mut e.data {
                            t.translation = trans;
                            t.rotation = Mat4::from_quat(rot);
                            t.scale = scale;
                        }
                    }
                }
            }
        }
        window.request_redraw();
        return;
    }
    if !ctx.interaction.gizmo.visible {
        return;
    }
    if ctx.engine.renderer.is_some() {
        if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
            gizmo::sync_gizmo_from_selection(
                &mut ctx.interaction.gizmo,
                &ctx.engine.world.graph,
            );
            ctx.interaction.gizmo.hovered =
                ctx.interaction.gizmo.hit_test(&ray).map(|(h, _)| h);
        }
    }
}

/// Fit the camera to the bounding box of all selected nodes.
pub fn fit_selection_to_view(engine: &mut Engine) {
    if engine.world.graph.selected_nodes().is_empty() {
        return;
    }
    use rc3d_actions::GetBoundingBoxAction;
    let mut aabb: Option<rc3d_core::Aabb> = None;
    for &id in engine.world.graph.selected_nodes() {
        let mut a = GetBoundingBoxAction::new();
        a.apply(&engine.world.graph, id);
        if a.bounding_box.min.x <= a.bounding_box.max.x {
            aabb = Some(match aabb {
                Some(p) => p.union(&a.bounding_box),
                None => a.bounding_box,
            });
        }
    }
    if let Some(b) = aabb {
        if let Some(vc) = engine.viewport_cameras.active_mut() {
            vc.controller.fit_bounds(&b, 60.0f32.to_radians());
        } else {
            engine.controller.fit_bounds(&b, 60.0f32.to_radians());
        }
    }
}
