//! Editor interaction: mouse-down, mouse-up, cursor-move, box-select, gizmo drag.
//!
//! These functions operate on an `EditorContext` (which wraps `&mut Engine`)
//! plus explicit window/input parameters.

use rc3d_actions::{Ray, RayPickAction};
use rc3d_core::math::{Mat4, Quat};
use rc3d_core::NodeId;
use rc3d_engine_api::{Engine, InputState};

use crate::commands::EditorCommand;
use crate::context::EditorContext;
use crate::gizmo;
use crate::ui::types::SelectKind;

/// Build a pick ray from the current cursor position, using viewport cameras
/// if available, otherwise falling back to the main camera controller.
pub fn build_pick_ray(
    engine: &Engine,
    input: &InputState,
    window: &winit::window::Window,
) -> Option<Ray> {
    let s = window.inner_size();
    engine
        .pointer_pick_frame(
            input.cursor_pos.0 as f32,
            input.cursor_pos.1 as f32,
            s.width,
            s.height,
        )
        .map(|(lx, ly, vw, vh, v, p)| Ray::from_screen_point(lx, ly, vw, vh, v, p))
}

/// Perform a ray pick and toggle selection on the hit node.
pub fn do_pick(
    engine: &mut Engine,
    input: &InputState,
    window: &winit::window::Window,
    locked: &std::collections::HashSet<NodeId>,
) {
    let Some(ray) = build_pick_ray(engine, input, window) else {
        return;
    };
    let mut picker = RayPickAction::new(ray);
    rc3d_actions::apply_to_all_roots(&mut picker, &engine.world.graph);

    if let Some(hit) = picker.hits.iter().find(|h| !locked.contains(&h.node)) {
        if !input.shift_pressed {
            engine.world.graph.clear_selection();
        }
        engine.world.graph.select(hit.node);
        log::info!(
            "Pick hit: node={:?}, point={:?}, selected={}",
            hit.node,
            hit.point,
            engine.world.graph.is_selected(hit.node)
        );
    } else if !input.shift_pressed {
        engine.world.graph.clear_selection();
    }
}

/// Compute active camera view + projection matrices.
pub fn active_camera_matrices(engine: &Engine, _width: f32, _height: f32) -> (Mat4, Mat4) {
    rc3d_engine_api::scene_pick_matrices(engine)
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
                ctx.interaction.view_split_hover = Some(axis);
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

    if ctx.interaction.markup.tool.is_drawing() {
        ctx.push_command(EditorCommand::MarkupMouseDown {
            screen_pos: screen_xy(input),
        });
        return;
    }
    if ctx.interaction.measurement_mode {
        if let Some(world) = pick_measure_world(ctx.engine, input, window) {
            ctx.push_command(EditorCommand::MeasurementPick { world });
        }
        return;
    }

    let left_orbit_enabled = !editor_ui_enabled && !ctx.interaction.measurement_mode;

    if left_orbit_enabled && !input.ctrl_pressed && !input.alt_pressed {
        ctx.interaction.left_pick_arm_pos = Some(input.cursor_pos);
        ctx.interaction.left_drag_suppresses_pick = false;
        return;
    }
    let tool_lasso = ctx.interaction.select_kind == SelectKind::Lasso;
    let tool_box = ctx.interaction.select_kind == SelectKind::Box;
    if input.alt_pressed || tool_lasso {
        let p = (input.cursor_pos.0 as f32, input.cursor_pos.1 as f32);
        ctx.interaction.lasso_drag = true;
        ctx.interaction.lasso_points.clear();
        ctx.interaction.lasso_points.push(p);
        return;
    }
    if input.ctrl_pressed || tool_box {
        ctx.interaction.box_select_drag = true;
        ctx.interaction.box_select_anchor = (input.cursor_pos.0 as f32, input.cursor_pos.1 as f32);
        return;
    }
    if ctx.interaction.section_edit_mode {
        if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
            let mut best: Option<(rc3d_core::NodeId, f32, [f32; 4])> = None;
            for (id, plane) in crate::section_edit::enabled_section_planes(&ctx.engine.world.graph)
            {
                if let Some(d) = crate::section_edit::hit_test(&ray, plane) {
                    if best.as_ref().map_or(true, |(_, bd, _)| d < *bd) {
                        best = Some((id, d, plane));
                    }
                }
            }
            if let Some((id, _, plane)) = best {
                ctx.interaction.section_drag = Some((id, plane));
                ctx.interaction.section_hovered = Some(id);
                return;
            }
        }
    }
    if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
        gizmo::sync_gizmo_from_selection(&mut ctx.engine.gizmo, &ctx.engine.world.graph);
        if ctx.engine.gizmo.visible {
            let locked_target = ctx
                .engine
                .gizmo
                .target_node
                .is_some_and(|n| ctx.interaction.locked_nodes.contains(&n));
            if !locked_target {
                if let Some((h, _)) = ctx.engine.gizmo.hit_test(&ray) {
                    if let Some(n) = ctx.engine.gizmo.target_node {
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
                    ctx.engine.gizmo.start_drag(&ray, h);
                    ctx.interaction.gizmo_dragging = true;
                    return;
                }
            }
        }
    }
    // Fallback: pick
    do_pick(ctx.engine, input, window, &ctx.interaction.locked_nodes);
    gizmo::sync_gizmo_from_selection(&mut ctx.engine.gizmo, &ctx.engine.world.graph);
}

/// Handle left mouse button release.
pub fn on_left_up(
    ctx: &mut EditorContext,
    window: &winit::window::Window,
    input: &InputState,
    editor_ui_enabled: bool,
) {
    if ctx.interaction.markup.tool.is_drawing() {
        ctx.push_command(EditorCommand::MarkupMouseUp {
            screen_pos: screen_xy(input),
        });
        return;
    }

    let left_orbit_enabled = !editor_ui_enabled && !ctx.interaction.measurement_mode;

    if left_orbit_enabled
        && !input.ctrl_pressed
        && !input.alt_pressed
        && !ctx.interaction.box_select_drag
        && !ctx.interaction.lasso_drag
        && ctx.interaction.section_drag.is_none()
        && !ctx.interaction.gizmo_dragging
    {
        if ctx.interaction.left_pick_arm_pos.is_some() {
            let _ = ctx.interaction.left_pick_arm_pos.take();
            if !ctx.interaction.left_drag_suppresses_pick {
                do_pick(ctx.engine, input, window, &ctx.interaction.locked_nodes);
                gizmo::sync_gizmo_from_selection(&mut ctx.engine.gizmo, &ctx.engine.world.graph);
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
    if ctx.interaction.section_drag.take().is_some() {
        return;
    }
    if ctx.interaction.lasso_drag {
        ctx.interaction.lasso_drag = false;
        let points = std::mem::take(&mut ctx.interaction.lasso_points);
        apply_lasso_selection(ctx, window, &points);
        gizmo::sync_gizmo_from_selection(&mut ctx.engine.gizmo, &ctx.engine.world.graph);
        return;
    }
    if ctx.interaction.box_select_drag {
        ctx.interaction.box_select_drag = false;

        // Gather all data we need before mutating world.graph
        let s = window.inner_size();
        let roots = ctx.engine.world.graph.roots().to_vec();
        let has_vp_cams = !ctx.engine.viewport_cameras.cameras.is_empty();
        let vp_id_opt = ctx
            .engine
            .viewport_cameras
            .active()
            .map(|vc| vc.viewport_id);
        let cam_node_opt = ctx
            .engine
            .viewport_cameras
            .active()
            .map(|vc| vc.camera_node);
        let anchor = ctx.interaction.box_select_anchor;
        let cursor = (input.cursor_pos.0 as f32, input.cursor_pos.1 as f32);

        // Now use the data without holding borrows on ctx.engine
        if has_vp_cams {
            if let (Some(vp_id), Some(cam_node)) = (vp_id_opt, cam_node_opt) {
                if let Some(r) = ctx.engine.renderer.as_ref() {
                    if let Some(avp) = r.viewport_layout().viewports.iter().find(|v| v.id == vp_id)
                    {
                        let (v, p) =
                            pick_view_proj_for_node(&ctx.engine.world.graph, cam_node, avp);
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
        gizmo::sync_gizmo_from_selection(&mut ctx.engine.gizmo, &ctx.engine.world.graph);
        return;
    }
    if ctx.interaction.gizmo_dragging {
        ctx.engine.gizmo.end_drag();
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
                        ctx.push_command(crate::commands::EditorCommand::CommitTransformScale {
                            node: n,
                            old: old_scale.to_array(),
                            new: new_scale.to_array(),
                        });
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

/// Like [`rc3d_engine_api::viewport_pick_matrices`] but takes a camera node
/// and a look-at fallback when the node is not a camera.
fn pick_view_proj_for_node(
    graph: &rc3d_scene::SceneGraph,
    camera_node: NodeId,
    vport: &rc3d_render::viewport::Viewport,
) -> (Mat4, Mat4) {
    if let Some(m) = rc3d_engine_api::camera_node_view_proj(graph, camera_node) {
        return m;
    }
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
        ctx.interaction.view_split_hover = Some(axis);
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
        ctx.engine.viewport_cameras.remap_viewport_ids_from_layout(
            ctx.engine.renderer.as_ref().unwrap().viewport_layout(),
        );
        window.request_redraw();
        return;
    }
    ctx.interaction.view_split_hover = if let Some(ref r) = ctx.engine.renderer {
        if r.viewport_layout().viewports.len() > 1 {
            let s = window.inner_size();
            r.viewport_layout().splitter_hit(
                input.cursor_pos.0 as f32,
                input.cursor_pos.1 as f32,
                s.width,
                s.height,
            )
        } else {
            None
        }
    } else {
        None
    };
    if ctx.interaction.markup.tool.is_drawing() && !ctx.interaction.markup.click_points.is_empty() {
        ctx.push_command(EditorCommand::MarkupMouseMove {
            screen_pos: screen_xy(input),
        });
        window.request_redraw();
        return;
    }
    if ctx.interaction.lasso_drag {
        let p = (input.cursor_pos.0 as f32, input.cursor_pos.1 as f32);
        if let Some(&(lx, ly)) = ctx.interaction.lasso_points.last() {
            let dx = p.0 - lx;
            let dy = p.1 - ly;
            if dx * dx + dy * dy >= 16.0 {
                ctx.interaction.lasso_points.push(p);
            }
        } else {
            ctx.interaction.lasso_points.push(p);
        }
        window.request_redraw();
        return;
    }
    if let Some((id, start_plane)) = ctx.interaction.section_drag {
        if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
            if let Some(pt) = crate::section_edit::drag_point_on_normal(&ray, start_plane) {
                let n = rc3d_core::math::Vec3::new(start_plane[0], start_plane[1], start_plane[2]);
                let plane = crate::section_edit::plane_from_point(n, pt);
                crate::section_edit::set_section_plane(&mut ctx.engine.world.graph, id, plane);
            }
        }
        window.request_redraw();
        return;
    }
    if ctx.interaction.section_edit_mode {
        ctx.interaction.section_hovered = None;
        if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
            let mut best: Option<(rc3d_core::NodeId, f32)> = None;
            for (id, plane) in crate::section_edit::enabled_section_planes(&ctx.engine.world.graph)
            {
                if let Some(d) = crate::section_edit::hit_test(&ray, plane) {
                    if best.as_ref().map_or(true, |(_, bd)| d < *bd) {
                        best = Some((id, d));
                    }
                }
            }
            ctx.interaction.section_hovered = best.map(|(id, _)| id);
        }
    }
    if ctx.interaction.gizmo_dragging {
        if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
            if let Some((n, old_mat)) = ctx.interaction.gizmo_pending_transform {
                if let Some(d) = ctx.engine.gizmo.drag_delta(&ray) {
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
    if !ctx.engine.gizmo.visible {
        return;
    }
    if ctx.engine.renderer.is_some() {
        if let Some(ray) = build_pick_ray(ctx.engine, input, window) {
            gizmo::sync_gizmo_from_selection(&mut ctx.engine.gizmo, &ctx.engine.world.graph);
            ctx.engine.gizmo.hovered = ctx.engine.gizmo.hit_test(&ray).map(|(h, _)| h);
        }
    }
}

fn screen_xy(input: &InputState) -> [f32; 2] {
    [input.cursor_pos.0 as f32, input.cursor_pos.1 as f32]
}

fn pick_measure_world(
    engine: &Engine,
    input: &InputState,
    window: &winit::window::Window,
) -> Option<[f32; 3]> {
    let ray = build_pick_ray(engine, input, window)?;
    let mut picker = RayPickAction::new(ray.clone());
    rc3d_actions::apply_to_all_roots(&mut picker, &engine.world.graph);
    if let Some(hit) = picker.hits.first() {
        return Some(hit.point.to_array());
    }
    crate::measurement::ground_hit(&ray).map(|p| p.to_array())
}

fn apply_lasso_selection(
    ctx: &mut EditorContext,
    window: &winit::window::Window,
    points: &[(f32, f32)],
) {
    if points.len() < 3 {
        return;
    }
    let s = window.inner_size();
    let roots = ctx.engine.world.graph.roots().to_vec();
    let has_vp_cams = !ctx.engine.viewport_cameras.cameras.is_empty();
    let vp_id_opt = ctx
        .engine
        .viewport_cameras
        .active()
        .map(|vc| vc.viewport_id);
    let cam_node_opt = ctx
        .engine
        .viewport_cameras
        .active()
        .map(|vc| vc.camera_node);

    if has_vp_cams {
        if let (Some(vp_id), Some(cam_node)) = (vp_id_opt, cam_node_opt) {
            if let Some(r) = ctx.engine.renderer.as_ref() {
                if let Some(avp) = r.viewport_layout().viewports.iter().find(|v| v.id == vp_id) {
                    let (v, p) = pick_view_proj_for_node(&ctx.engine.world.graph, cam_node, avp);
                    let temp_vp = rc3d_render::viewport::Viewport {
                        id: avp.id,
                        rect: avp.rect,
                        projection_type: avp.projection_type,
                        name: String::new(),
                        camera_node: None,
                        is_active: false,
                    };
                    for &root in &roots {
                        crate::box_select::select_nodes_in_viewport_lasso(
                            &mut ctx.engine.world.graph,
                            root,
                            points,
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
            crate::box_select::select_nodes_in_screen_lasso(
                &mut ctx.engine.world.graph,
                root,
                points,
                s.width as f32,
                s.height as f32,
                v,
                p,
            );
        }
    }
}

/// Refresh section-plane overlay lines on the engine (call once per frame).
pub fn sync_section_overlay(
    engine: &mut Engine,
    interaction: &crate::context::EditorInteractionState,
) {
    if interaction.section_edit_mode {
        engine.overlay_line_batches =
            crate::section_edit::widget_batches(&engine.world.graph, interaction.section_hovered);
    } else {
        engine.overlay_line_batches.clear();
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
        fit_aabb_to_view(engine, &b);
    }
}

/// Fit the camera to the bounding box of the whole scene.
pub fn fit_scene_to_view(engine: &mut Engine) {
    if let Some(b) = rc3d_scene::GetBoundingBoxAction::compute_scene_aabb(&engine.world.graph) {
        fit_aabb_to_view(engine, &b);
    }
}

fn fit_aabb_to_view(engine: &mut Engine, b: &rc3d_core::Aabb) {
    if let Some(vc) = engine.viewport_cameras.active_mut() {
        vc.controller.fit_bounds(b, 60.0f32.to_radians());
    } else {
        engine.controller.fit_bounds(b, 60.0f32.to_radians());
    }
}
