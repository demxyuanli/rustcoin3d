use rc3d_actions::{Action, SetRotationCommand, SetScaleCommand, SetTranslationCommand};
use rc3d_core::math::{Mat4, Quat};

use super::{gizmo_support, App};

impl App {
    pub(super) fn editor_on_left_down(&mut self) {
        if let (Some(r), Some(w)) = (&mut self.renderer, &self.window) {
            if r.viewport_layout.viewports.len() > 1 {
                let s = w.inner_size();
                if let Some(axis) = r.viewport_layout.splitter_hit(
                    self.cursor_pos.0 as f32,
                    self.cursor_pos.1 as f32,
                    s.width,
                    s.height,
                ) {
                    self.view_split_drag = Some(axis);
                    return;
                }
            }
        }
        if let (Some(r), _w) = (&mut self.renderer, &self.window) {
            if r.viewport_layout.viewports.len() > 1 {
                if let Some(vp) = r
                    .viewport_layout
                    .viewport_at(self.cursor_pos.0 as f32, self.cursor_pos.1 as f32)
                {
                    self.viewport_cameras
                        .set_active(vp.id, &mut r.viewport_layout);
                }
            }
        }
        if self.camera_left_orbit_enabled() && !self.ctrl_pressed {
            self.left_pick_arm_pos = Some(self.cursor_pos);
            self.left_drag_suppresses_pick = false;
            return;
        }
        if self.ctrl_pressed {
            self.box_select_drag = true;
            self.box_select_anchor = (self.cursor_pos.0 as f32, self.cursor_pos.1 as f32);
            return;
        }
        if let Some(ray) = self.build_pick_ray() {
            gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
            if self.gizmo.visible {
                if let Some((h, _)) = self.gizmo.hit_test(&ray) {
                    if let Some(n) = self.gizmo.target_node {
                        if let Some(e) = self.world.graph.get(n) {
                            if let rc3d_scene::NodeData::Transform(t) = &e.data {
                                let old = Mat4::from_scale_rotation_translation(
                                    t.scale,
                                    Quat::from_mat4(&t.rotation),
                                    t.translation,
                                );
                                self.gizmo_pending_transform = Some((n, old));
                            }
                        }
                    }
                    self.gizmo.start_drag(&ray, h);
                    self.gizmo_dragging = true;
                    return;
                }
            }
        }
        let can_pick = true;
        if can_pick {
            self.do_pick();
            gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
        }
    }

    pub(super) fn editor_on_left_up(&mut self) {
        if self.camera_left_orbit_enabled()
            && !self.ctrl_pressed
            && !self.box_select_drag
            && !self.gizmo_dragging
        {
            if self.left_pick_arm_pos.is_some() {
                let _ = self.left_pick_arm_pos.take();
                if !self.left_drag_suppresses_pick {
                    self.do_pick();
                    gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
                }
                self.left_drag_suppresses_pick = false;
            }
        } else {
            self.left_pick_arm_pos = None;
            self.left_drag_suppresses_pick = false;
        }
        if self.view_split_drag.is_some() {
            self.view_split_drag = None;
        }
        if self.box_select_drag {
            self.box_select_drag = false;
            if let (Some(r), Some(w)) = (self.renderer.as_ref(), self.window.as_ref()) {
                let s = w.inner_size();
                let roots = self.world.graph.roots().to_vec();
                if !self.viewport_cameras.cameras.is_empty() {
                    if let (Some(vc), Some(_)) =
                        (self.viewport_cameras.active(), self.build_pick_ray())
                    {
                        if let Some(avp) = r
                            .viewport_layout
                            .viewports
                            .iter()
                            .find(|v| v.id == vc.viewport_id)
                        {
                            let (v, p) = gizmo_support::pick_view_proj(&self.world.graph, vc, avp);
                            for &root in &roots {
                                super::box_select::select_nodes_in_viewport_box(
                                    &mut self.world.graph,
                                    root,
                                    self.box_select_anchor,
                                    (self.cursor_pos.0 as f32, self.cursor_pos.1 as f32),
                                    avp,
                                    v,
                                    p,
                                );
                            }
                        }
                    }
                } else {
                    let (v, p) = self.active_camera_matrices(s.width as f32, s.height as f32);
                    for &root in &roots {
                        super::box_select::select_nodes_in_screen_box(
                            &mut self.world.graph,
                            root,
                            self.box_select_anchor,
                            (self.cursor_pos.0 as f32, self.cursor_pos.1 as f32),
                            s.width as f32,
                            s.height as f32,
                            v,
                            p,
                        );
                    }
                }
                gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
            }
            return;
        }
        if self.gizmo_dragging {
            self.gizmo.end_drag();
            if let Some((n, old_mat)) = self.gizmo_pending_transform.take() {
                if let Some(e) = self.world.graph.get(n) {
                    if let rc3d_scene::NodeData::Transform(t) = &e.data {
                        let new_mat = Mat4::from_scale_rotation_translation(
                            t.scale,
                            Quat::from_mat4(&t.rotation),
                            t.translation,
                        );
                        let (old_scale, _old_rot, old_trans) =
                            old_mat.to_scale_rotation_translation();
                        let (new_scale, _new_rot, new_trans) =
                            new_mat.to_scale_rotation_translation();
                        if (new_trans - old_trans).length_squared() > 1e-8 {
                            self.command_history.execute(
                                Box::new(SetTranslationCommand {
                                    node: n,
                                    old_value: old_trans,
                                    new_value: new_trans,
                                }),
                                &mut self.world.graph,
                            );
                        }
                        if (new_scale - old_scale).length_squared() > 1e-8 {
                            self.command_history.execute(
                                Box::new(SetScaleCommand {
                                    node: n,
                                    old_value: old_scale,
                                    new_value: new_scale,
                                }),
                                &mut self.world.graph,
                            );
                        }
                        let old_rot_cols = Mat4::from_quat(_old_rot).to_cols_array_2d();
                        let new_rot_cols = Mat4::from_quat(_new_rot).to_cols_array_2d();
                        if old_rot_cols != new_rot_cols {
                            self.command_history.execute(
                                Box::new(SetRotationCommand {
                                    node: n,
                                    old_value: Mat4::from_quat(_old_rot),
                                    new_value: Mat4::from_quat(_new_rot),
                                }),
                                &mut self.world.graph,
                            );
                        }
                    }
                }
            }
            self.gizmo_dragging = false;
        }
    }

    pub(super) fn editor_on_cursor_moved(&mut self) {
        if let Some(axis) = self.view_split_drag {
            if let (Some(r), Some(w)) = (&mut self.renderer, &self.window) {
                let s = w.inner_size();
                r.viewport_layout.apply_split_drag(
                    axis,
                    self.cursor_pos.0 as f32,
                    self.cursor_pos.1 as f32,
                    s.width,
                    s.height,
                );
                r.viewport_layout.rebuild(s.width, s.height);
            }
            self.sync_viewport_camera_ids();
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            return;
        }
        if self.gizmo_dragging {
            if let Some(ray) = self.build_pick_ray() {
                if let Some((n, old_mat)) = &self.gizmo_pending_transform {
                    if let Some(d) = self.gizmo.drag_delta(
                        &ray,
                        Mat4::IDENTITY,
                        Mat4::IDENTITY,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ) {
                        let new_mat = d * *old_mat;
                        let (scale, rot, trans) = new_mat.to_scale_rotation_translation();
                        if let Some(e) = self.world.graph.get_mut(*n) {
                            if let rc3d_scene::NodeData::Transform(t) = &mut e.data {
                                t.translation = trans;
                                t.rotation = Mat4::from_quat(rot);
                                t.scale = scale;
                            }
                        }
                    }
                }
            }
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            return;
        }
        if !self.gizmo.visible {
            return;
        }
        if self.renderer.is_some() {
            if let (Some(ray), Some(_r)) = (self.build_pick_ray(), self.renderer.as_ref()) {
                gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
                self.gizmo.hovered = self.gizmo.hit_test(&ray).map(|(h, _)| h);
            }
        }
    }

    pub(super) fn fit_selection_to_view(&mut self) {
        if self.world.graph.selected_nodes().is_empty() {
            return;
        }
        use rc3d_actions::GetBoundingBoxAction;
        let mut aabb: Option<rc3d_core::Aabb> = None;
        for &id in self.world.graph.selected_nodes() {
            let mut a = GetBoundingBoxAction::new();
            a.apply(&self.world.graph, id);
            if a.bounding_box.min.x <= a.bounding_box.max.x {
                aabb = Some(match aabb {
                    Some(p) => p.union(&a.bounding_box),
                    None => a.bounding_box,
                });
            }
        }
        if let Some(b) = aabb {
            if let Some(cc) = &mut self.camera_controller {
                cc.fit_bounds(&b, 60.0f32.to_radians());
            } else if let Some(vc) = self.viewport_cameras.active_mut() {
                vc.controller.fit_bounds(&b, 60.0f32.to_radians());
            }
        }
    }
}
