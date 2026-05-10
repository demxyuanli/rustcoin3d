use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::event_loop::ActiveEventLoop;
use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};
use winit::window::WindowAttributes;

use rc3d_actions::{update_all_lod_nodes, Action, SectionPlaneAction};
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::{DisplayMode, NodeId};
use rc3d_gizmo::GizmoMode;
use rc3d_render::{AdaptiveControl, DrawCall, Renderer};

use crate::editor_ui::{EditorCommand, EditorUi, EditorUiContext, RenderFeatureFlags};

use super::streaming_lod::stream_step_ms;

use crate::adaptive_quality::AdaptiveQualityMode;

use super::gizmo_support;
use super::App;
pub(crate) fn resumed(app: &mut App, event_loop: &ActiveEventLoop) {
        if app.state.window.is_none() {
            let window = event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title(app.state.window_title.clone())
                        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
                        .with_resizable(true)
                        .with_maximized(true)
                        .with_visible(false),
                )
                .expect("failed to create window");
            let renderer = pollster::block_on(Renderer::new(&window));
            app.state.window = Some(window);
            app.state.renderer = Some(renderer);
            if app.state.editor_ui_enabled {
                if let (Some(window), Some(renderer)) = (&app.state.window, &app.state.renderer) {
                    app.state.editor_ui = Some(EditorUi::new(window, renderer));
                }
            }
            if let Some(renderer) = &mut app.state.renderer {
                renderer.set_display_mode(app.state.initial_display_mode);
                if app.state.enable_hdr_post_processing {
                    renderer.set_hdr_post_processing(true);
                }
                if app.state.editor_ui_enabled {
                    renderer.set_hud_enabled(false);
                }
                if let Some((vig, chr, bloom, grain)) =
                    app.state.initial_post_effect_params.take()
                {
                    renderer.set_post_effect_params(vig, chr, bloom, grain);
                }
            }
            if let Some(window) = &app.state.window {
                window.set_visible(true);
            }
        }
}

pub(crate) fn window_event(
    app: &mut App,
    event_loop: &ActiveEventLoop,
    _window_id: winit::window::WindowId,
    event: WindowEvent,
) {
        let cursor_pos_before_event = app.input.cursor_pos;
        let ui_consumed = if app.state.editor_ui_enabled {
            if let (Some(ui), Some(window)) = (&mut app.state.editor_ui, &app.state.window) {
                ui.on_window_event(window, &event)
            } else {
                false
            }
        } else {
            false
        };

        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                app.state.adaptive_last_interaction = Instant::now();
                let size = *physical_size;
                if let Some(renderer) = &mut app.state.renderer {
                    renderer.resize(size.width, size.height);
                }
                app.sync_viewport_camera_ids();
                if let Some(window) = &app.state.window {
                    if let Some(ui) = &mut app.state.editor_ui {
                        ui.resize(size.width, size.height, window.scale_factor() as f32);
                    }
                }
                if let Some(window) = &app.state.window {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                app.state.adaptive_last_interaction = Instant::now();
                let prev = app.input.cursor_pos;
                app.input.cursor_pos = (position.x, position.y);
                if app.camera_left_orbit_enabled() {
                    if let Some((ax, ay)) = app.editor.left_pick_arm_pos {
                        let dx = app.input.cursor_pos.0 - ax;
                        let dy = app.input.cursor_pos.1 - ay;
                        if dx * dx + dy * dy > 9.0 {
                            app.editor.left_drag_suppresses_pick = true;
                        }
                    }
                }
                if !app.editor.measurement_mode {
                    app.editor_on_cursor_moved();
                }
                if !ui_consumed
                    && !app.editor.measurement_mode
                    && !app.should_block_handle_event_pointer_dispatch(true)
                {
                    let dx = (app.input.cursor_pos.0 - prev.0) as f32;
                    let dy = (app.input.cursor_pos.1 - prev.1) as f32;
                    app.dispatch_handle_event_for_pointer(rc3d_actions::Event::MouseMove {
                        x: 0.0,
                        y: 0.0,
                        dx,
                        dy,
                    });
                }
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if ui_consumed {
                    if let Some(window) = &app.state.window {
                        window.request_redraw();
                    }
                    return;
                }
                if key_event.state == winit::event::ElementState::Pressed && !key_event.repeat {
                    let key = key_event.physical_key;
                    if let winit::keyboard::PhysicalKey::Code(code) = key {
                        app.state.adaptive_last_interaction = Instant::now();
                        if let Some(hook) = &mut app.panel_overlay_key_hook {
                            hook(code);
                        }
                    }
                    match key {
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyW) => {
                            if app.state.camera_controller.as_ref().map(|c| c.walk_mode).unwrap_or(false) {
                                if let Some(ref mut cc) = app.state.camera_controller { cc.walk(0.1, 0.0, 0.0, 0.15); }
                            } else if let Some(renderer) = &mut app.state.renderer {
                                renderer.set_display_mode(DisplayMode::Wireframe);
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyS) => {
                            if app.state.camera_controller.as_ref().map(|c| c.walk_mode).unwrap_or(false) {
                                if let Some(ref mut cc) = app.state.camera_controller { cc.walk(-0.1, 0.0, 0.0, 0.15); }
                            } else if let Some(renderer) = &mut app.state.renderer {
                                renderer.set_display_mode(DisplayMode::Shaded);
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyE) => {
                            if let Some(renderer) = &mut app.state.renderer {
                                renderer.set_display_mode(DisplayMode::ShadedWithEdges);
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyH) => {
                            if let Some(renderer) = &mut app.state.renderer {
                                renderer.set_display_mode(DisplayMode::HiddenLine);
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyK) => {
                            if let Some(renderer) = &mut app.state.renderer {
                                renderer.set_display_mode(DisplayMode::Flat);
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyX) => {
                            app.editor.axis_clip[0] = !app.editor.axis_clip[0];
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyY) => {
                            if app.input.ctrl_pressed {
                                let _ = app.editor.command_history.redo(&mut app.state.world.graph);
                            } else {
                                app.editor.axis_clip[1] = !app.editor.axis_clip[1];
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyZ) => {
                            if app.input.ctrl_pressed {
                                let _ = app.editor.command_history.undo(&mut app.state.world.graph);
                            } else {
                                app.editor.axis_clip[2] = !app.editor.axis_clip[2];
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyF) => {
                            app.fit_selection_to_view();
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyT) => {
                            app.editor.gizmo.mode = GizmoMode::Translate;
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyR) => {
                            app.editor.gizmo.mode = GizmoMode::Rotate;
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyG) => {
                            app.editor.gizmo.mode = GizmoMode::Scale;
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyP) => {
                            app.editor.section_edit_mode = !app.editor.section_edit_mode;
                        }
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::BracketLeft,
                        ) => {
                            if app.editor.section_edit_mode {
                                app.nudge_section_planes(-0.04);
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::BracketRight,
                        ) => {
                            if app.editor.section_edit_mode {
                                app.nudge_section_planes(0.04);
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                            app.state.world.graph.clear_selection();
                            app.editor.measurements.clear();
                            app.editor.measurement_first_point = None;
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyM) => {
                            app.editor.measurement_mode = !app.editor.measurement_mode;
                            app.editor.measurement_first_point = None;
                            log::info!("Measurement mode: {}", app.editor.measurement_mode);
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyN) => {
                            app.editor.grid_enabled = !app.editor.grid_enabled;
                            if let Some(renderer) = &mut app.state.renderer {
                                renderer.set_grid_enabled(app.editor.grid_enabled);
                            }
                            log::info!("Grid: {}", if app.editor.grid_enabled { "on" } else { "off" });
                        }
                        // Camera bookmarks: Ctrl+Digit = save, Digit = recall
                        k @ winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::Digit1
                            | winit::keyboard::KeyCode::Digit2
                            | winit::keyboard::KeyCode::Digit3
                            | winit::keyboard::KeyCode::Digit4
                            | winit::keyboard::KeyCode::Digit5
                            | winit::keyboard::KeyCode::Digit6
                            | winit::keyboard::KeyCode::Digit7
                            | winit::keyboard::KeyCode::Digit8
                            | winit::keyboard::KeyCode::Digit9,
                        ) => {
                            let code = match k {
                                winit::keyboard::PhysicalKey::Code(c) => c,
                                _ => winit::keyboard::KeyCode::Digit1,
                            };
                            let slot = super::bookmark_slot_from_key(code);
                            if app.input.ctrl_pressed {
                                if let Some(ctrl) = app.active_camera_controller_mut() {
                                    ctrl.save_bookmark(slot, "bookmark");
                                    log::info!("Saved camera bookmark to slot {}", slot + 1);
                                }
                            } else {
                                if let Some(ctrl) = app.active_camera_controller_mut() {
                                    ctrl.recall_bookmark(slot);
                                    log::info!("Recalling camera bookmark {}", slot + 1);
                                }
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyI) => {
                            if let Some(renderer) = &mut app.state.renderer {
                                renderer.cycle_ibl_preset();
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyC) => {
                            if let Some(renderer) = &mut app.state.renderer {
                                let (w, h) = renderer.surface_size();
                                
                                renderer.viewport_layout_mut().cycle_layout(w, h);
                            }
                            app.sync_viewport_camera_ids();
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyV) => {
                            if let Some(ref mut cc) = app.state.camera_controller { cc.toggle_walk_mode(); }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Tab) => {
                            if let Some(renderer) = &mut app.state.renderer {
                                renderer.viewport_layout_mut().cycle_active();
                                app.state.viewport_cameras.active_viewport =
                                    renderer.viewport_layout().active_id;
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyA) => {
                            if app.input.ctrl_pressed {
                                let roots: Vec<NodeId> = app.state.world.graph.roots().to_vec();
                                app.state.world.graph.select_many(roots);
                            } else if let Some(ref mut cc) = app.state.camera_controller {
                                if cc.walk_mode { cc.walk(0.0, -0.1, 0.0, 0.15); }
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyD) => {
                            if app.input.ctrl_pressed {
                                let selected: Vec<NodeId> =
                                    app.state.world.graph.selected_nodes().iter().copied().collect();
                                for id in selected {
                                    let _ = app.state.editor_commands.push_back(
                                        EditorCommand::DuplicateNode(id),
                                    );
                                }
                            } else if let Some(ref mut cc) = app.state.camera_controller {
                                if cc.walk_mode { cc.walk(0.0, 0.1, 0.0, 0.15); }
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::Delete
                            | winit::keyboard::KeyCode::Backspace,
                        ) => {
                            let selected: Vec<NodeId> =
                                app.state.world.graph.selected_nodes().iter().copied().collect();
                            for id in selected {
                                let _ = app.state.editor_commands.push_back(
                                    EditorCommand::DeleteNode(id),
                                );
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::Backquote,
                        ) => {
                            if let Some(ref mut ui) = app.state.editor_ui {
                                ui.toggle_console();
                            }
                        }
                        _ => {}
                    }
                    if let Some(window) = &app.state.window {
                        window.request_redraw();
                    }
                }
                if let WindowEvent::KeyboardInput { event: kb, .. } = &event {
                    if kb.state == winit::event::ElementState::Pressed {
                        let evt = rc3d_actions::Event::KeyPress {
                            key: format!("{:?}", kb.logical_key),
                        };
                        if let (Some(_r), Some(active)) =
                            (&app.state.renderer, app.state.viewport_cameras.active())
                        {
                            let view = active.controller.view_matrix();
                            let proj =
                                Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 1000.0);
                            let ctx = rc3d_actions::EventContext::new(evt, view, proj);
                            let mut action = rc3d_actions::HandleEventAction::new(ctx);
                            action.apply_non_pointer_only(
                                &app.state.world.graph,
                                app.state.world
                                    .graph
                                    .roots()
                                    .first()
                                    .copied()
                                    .unwrap_or(NodeId::default()),
                            );
                        }
                        if let Some(window) = &app.state.window {
                            window.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                app.state.adaptive_last_interaction = Instant::now();
                app.input.shift_pressed = mods.state().shift_key();
                app.input.ctrl_pressed = mods.state().control_key();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if ui_consumed {
                    if let Some(window) = &app.state.window {
                        window.request_redraw();
                    }
                    return;
                }
                if !app.editor.measurement_mode && !app.should_block_handle_event_pointer_dispatch(false) {
                    use winit::event::ElementState;
                    let btn = match *button {
                        MouseButton::Left => Some(0u8),
                        MouseButton::Middle => Some(1),
                        MouseButton::Right => Some(2),
                        _ => None,
                    };
                    if let Some(b) = btn {
                        let evt = match *state {
                            ElementState::Pressed => rc3d_actions::Event::ButtonPress {
                                button: b,
                                x: 0.0,
                                y: 0.0,
                            },
                            ElementState::Released => rc3d_actions::Event::ButtonRelease {
                                button: b,
                                x: 0.0,
                                y: 0.0,
                            },
                        };
                        app.dispatch_handle_event_for_pointer(evt);
                    }
                }
                if *button == MouseButton::Left {
                    app.state.adaptive_last_interaction = Instant::now();
                    if *state == winit::event::ElementState::Pressed {
                        if let Some(hook) = &mut app.panel_overlay_mouse_hook {
                            if let Some(window) = app.state.window.as_ref() {
                                let size = window.inner_size();
                                if hook(
                                    app.input.cursor_pos.0 as f32,
                                    app.input.cursor_pos.1 as f32,
                                    size.width,
                                    size.height,
                                ) {
                                    if let Some(window) = &app.state.window {
                                        window.request_redraw();
                                    }
                                    return;
                                }
                            }
                        }
                        if app.editor.measurement_mode {
                            app.do_measure_pick();
                        } else {
                            app.editor_on_left_down();
                        }
                    } else if *state == winit::event::ElementState::Released {
                        app.editor_on_left_up();
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if ui_consumed {
                    if let Some(window) = &app.state.window {
                        window.request_redraw();
                    }
                    return;
                }
                app.state.adaptive_last_interaction = Instant::now();
                if !app.editor.measurement_mode && !app.should_block_handle_event_pointer_dispatch(false) {
                    let (dx, dy) = match delta {
                        MouseScrollDelta::LineDelta(x, y) => (*x, *y),
                        MouseScrollDelta::PixelDelta(pos) => {
                            (pos.x as f32 / 50.0, pos.y as f32 / 50.0)
                        }
                    };
                    app.dispatch_handle_event_for_scroll(dx, dy);
                }
            }
            WindowEvent::RedrawRequested => {
                app.poll_pending_graph_load();
                if app.lod.preview_mode_active
                    && app.state.renderer.is_some()
                    && app.lod.stream_next_tick.is_none()
                {
                    app.lod.stream_next_tick =
                        Some(Instant::now() + Duration::from_millis(stream_step_ms()));
                }
                app.tick_mesh_stream();

                // Legacy camera controller: update scene-graph camera node
                if let Some(ref ctrl) = app.state.camera_controller {
                    let aspect = app.state
                        .renderer
                        .as_ref()
                        .map(|r| r.config.width as f32 / r.config.height.max(1) as f32)
                        .unwrap_or(1.0);
                    let roots: Vec<NodeId> = app.state.world.graph.roots().to_vec();
                    for &root in &roots {
                        App::update_camera_recursive(ctrl, &mut app.state.world.graph, root, aspect);
                    }
                }

                // Snapshot viewport layout for camera updates (before renderer is mutably borrowed)
                if !app.state.viewport_cameras.cameras.is_empty() {
                    let viewports: Vec<_> = app.state
                        .renderer
                        .as_ref()
                        .map(|r| {
                            r.viewport_layout()
                                .viewports
                                .iter()
                                .map(|v| (v.id, v.rect.aspect()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    for vc in &app.state.viewport_cameras.cameras {
                        if let Some(&(_, aspect)) =
                            viewports.iter().find(|&&(id, _)| id == vc.viewport_id)
                        {
                            vc.controller.update_camera_node(
                                &mut app.state.world.graph,
                                vc.camera_node,
                                aspect,
                            );
                        }
                    }
                }

                let selected_set = app.state.world.graph.selected_nodes().clone();
                let selected_count = selected_set.len();
                // Snapshot bookmarks outside of renderer/editor borrow scope
                let bookmarks: [(bool, &'static str); 9] = {
                    let mut bm = [(false, ""); 9];
                    if let Some(ctrl) = app.active_camera_controller_mut() {
                        for (i, slot) in ctrl.bookmarks.iter().enumerate() {
                            if slot.is_some() {
                                bm[i] = (true, "saved");
                            }
                        }
                    }
                    bm
                };
                if app.state.editor_ui_enabled {
                    if let (Some(ui), Some(window), Some(renderer)) =
                        (&mut app.state.editor_ui, &app.state.window, &app.state.renderer)
                    {
                        let ui_ctx = EditorUiContext {
                            selected: selected_set,
                            display_mode_label: format!("{:?}", renderer.display_mode()),
                            ibl_label: renderer.ibl_preset_name().to_string(),
                            ibl_preset: renderer.ibl_preset,
                            gizmo_mode: app.editor.gizmo.mode,
                            layout_mode: renderer.viewport_layout().layout_mode,
                            layout_mode_label: format!(
                                "{:?}",
                                renderer.viewport_layout().layout_mode
                            ),
                            active_viewport_label: format!(
                                "{:?}",
                                renderer.viewport_layout().active_id
                            ),
                            smoothed_fps: app.state.fps_tracker.smoothed_fps(),
                            frame_time_ms: app.state.last_frame_time_ms,
                            diagnostics: app.state.last_render_stats.diagnostics.clone(),
                            hidden_nodes: app.state.hidden_nodes.clone(),
                            render_features: RenderFeatureFlags {
                                taa: renderer.enable_taa,
                                motion_blur: renderer.enable_motion_blur,
                                ssr: renderer.enable_ssr,
                                color_grading: renderer.enable_color_grading,
                                dof: renderer.enable_dof,
                                volumetric_fog: renderer.enable_volumetric_fog,
                                cluster_lights: renderer.enable_cluster_lights,
                                omni_shadows: renderer.enable_omni_shadows,
                                xray: renderer.xray_mode,
                            },
                            hdr_enabled: renderer.hdr_post_processing,
                            vsync_enabled: matches!(
                                renderer.config.present_mode,
                                wgpu::PresentMode::AutoVsync
                            ),
                            grid_enabled: app.editor.grid_enabled,
                            hud_enabled: renderer.hud_enabled,
                            outline_width: renderer.outline_width,
                            outline_color: renderer.outline_color,
                            xray_mode: renderer.xray_mode,
                            adaptive_quality_mode: app.state.adaptive_quality_mode,
                            adaptive_quality_name: renderer.adaptive_quality_name().to_string(),
                            bookmarks,
                            selected_count,
                        };
                        ui.run(window, &app.state.world.graph, renderer, &ui_ctx);
                        for cmd in ui.take_commands() {
                            app.state.editor_commands.push_back(cmd);
                        }
                    }
                    app.apply_editor_commands();
                }

                if let (Some(renderer), Some(window)) = (&mut app.state.renderer, &app.state.window) {
                    let roots_lod: Vec<NodeId> = app.state.world.graph.roots().to_vec();
                    if renderer.has_lod_nodes() {
                        let mut total_lod = 0usize;
                        for &r in &roots_lod {
                            total_lod += update_all_lod_nodes(&mut app.state.world.graph, r, app.state.last_camera_eye);
                        }
                        renderer.note_lod_scan(total_lod);
                    }
                    {
                        let mut section = SectionPlaneAction::new();
                        for &r in &roots_lod {
                            section.apply(&app.state.world.graph, r);
                        }
                        let mut merged = section.planes;
                        let mut merged_caps = section.cap_tints;
                        if app.editor.axis_clip[0] {
                            merged.push([1.0, 0.0, 0.0, 0.0]);
                            merged_caps.push(None);
                        }
                        if app.editor.axis_clip[1] {
                            merged.push([0.0, 1.0, 0.0, 0.0]);
                            merged_caps.push(None);
                        }
                        if app.editor.axis_clip[2] {
                            merged.push([0.0, 0.0, 1.0, 0.0]);
                            merged_caps.push(None);
                        }
                        renderer.set_clip_planes(merged, merged_caps);
                    }
                    let dirty_roots = rc3d_render::dirty_flags::collect_dirty_roots(&app.state.world.graph);
                    let has_cache = !app.state.world.cached_draw_calls.is_empty();
                    let can_skip = dirty_roots.is_empty() && has_cache;
                    let camera_only = !can_skip
                        && has_cache
                        && dirty_roots.iter().all(|&id| {
                            app.state.world.graph.get(id).map_or(false, |e| {
                                matches!(e.data,
                                    rc3d_scene::NodeData::PerspectiveCamera(_)
                                    | rc3d_scene::NodeData::OrthographicCamera(_))
                                && e.dirty_flags & rc3d_scene::node_entry::dirty_flags::GEOMETRY == 0
                            })
                        });

                    if can_skip {
                        // Nothing changed — reuse cached draw calls as-is (no clone needed
                        // if collector already holds the right data from last frame).
                        if app.state.world.collector.draw_calls.is_empty() {
                            app.state.world.collector.draw_calls = app.state.world.cached_draw_calls.clone();
                        }
                    } else if camera_only {
                        let cam_w = renderer.config.width.max(1) as f32;
                        let cam_h = renderer.config.height.max(1) as f32;
                        let aspect = cam_w / cam_h.max(1.0);
                        let vp_proj = Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0);
                        let vp_view = if let Some(vc) = app.state.viewport_cameras.active() {
                            vc.controller.view_matrix()
                        } else if let Some(ctrl) = &app.state.camera_controller {
                            ctrl.view_matrix()
                        } else {
                            Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
                        };
                        let cam_pos = app.state.camera_controller.as_ref()
                            .map(|c| c.eye_position())
                            .or_else(|| app.state.viewport_cameras.active().map(|vc| vc.controller.eye_position()))
                            .unwrap_or(Vec3::new(0.0, 0.0, 5.0));
                        app.state.world.collector.view_matrix = vp_view;
                        app.state.world.collector.projection_matrix = vp_proj;
                        app.state.world.collector.camera_pos = cam_pos;
                        // Apply camera in-place to collector (avoid clone in hot camera path)
                        rc3d_render::render_action::apply_world_camera(
                            &mut app.state.world.collector.draw_calls,
                            vp_view, vp_proj, cam_pos,
                        );
                        app.state.world.cached_draw_calls = app.state.world.collector.draw_calls.clone();
                        drop(dirty_roots);
                        rc3d_render::dirty_flags::clear_all_dirty_flags(&mut app.state.world.graph);
                    } else {
                        // Full traversal needed — reset collector and run engines/materials
                        app.state.world.evaluate_engines();
                        renderer.set_materials(app.state.world.materials.clone());
                        app.state.world.collector.draw_calls.clear();
                        app.state.world.collector.state = rc3d_actions::State::new();
                        app.state.world.collector.camera_pos = Vec3::new(0.0, 0.0, 5.0);
                        app.state.world.collector.view_matrix = Mat4::IDENTITY;
                        app.state.world.collector.projection_matrix = Mat4::IDENTITY;
                        app.state.world.collector.projection_orthographic = false;
                        app.state.world.collector.global_display_mode = renderer.display_mode();
                        app.state.world.collector.material_library = Some(app.state.world.materials.clone());
                        app.state.world.collector.set_hidden_nodes(&app.state.hidden_nodes);

                        let _t0 = std::time::Instant::now();
                        app.state.world.collector.reserve_draw_calls(
                            app.state.world.cached_draw_calls.len());
                        for &root in app.state.world.graph.roots() {
                            app.state.world.collector.traverse(&app.state.world.graph, root);
                        }
                        let traversal_ms = _t0.elapsed().as_secs_f64() * 1000.0;
                        if traversal_ms > 5.0 {
                            log::info!("Scene traversal: {:.2}ms ({} nodes, {} dirty roots)",
                                traversal_ms, app.state.world.graph.roots().len(), dirty_roots.len());
                        }
                        app.state.world.cached_draw_calls = app.state.world.collector.draw_calls.clone();
                        drop(dirty_roots);
                        rc3d_render::dirty_flags::clear_all_dirty_flags(&mut app.state.world.graph);
                    }
                    renderer.set_grid_enabled(app.editor.grid_enabled);
                    app.state.last_camera_eye = app.state.world.collector.camera_pos;
                    gizmo_support::sync_gizmo_from_selection(&mut app.editor.gizmo, &app.state.world.graph);

                    if !app.editor.measurements.is_empty() {
                        let vp = app.state.world.collector.projection_matrix
                            * app.state.world.collector.view_matrix;
                        let depth_reversed_z = rc3d_core::depth_reversed_z_from_projection(
                            app.state.world.collector.projection_matrix,
                        );
                        for &(p1, p2, _dist) in &app.editor.measurements {
                            app.state.world.collector.draw_calls.push(DrawCall {
                                vertices: Arc::new(Vec::new()),
                                indices: None,
                                edge_positions: Arc::new(vec![p1.to_array(), p2.to_array()]),
                                mvp: vp,
                                model_matrix: Mat4::IDENTITY,
                                camera_pos: app.state.world.collector.camera_pos,
                                light_set_id: 0,
                                overlay_color: Some([1.0, 1.0, 0.0, 1.0]),
                                display_mode: DisplayMode::ShadedWithEdges,
                                projection_orthographic: app.state
                                    .world
                                    .collector
                                    .projection_orthographic,
                                depth_reversed_z,
                                node_type_label: Arc::from("Measurement"),
                                ..Default::default()
                            });
                        }
                    }
                    if app.editor.gizmo.visible {
                        let vp = app.state.world.collector.projection_matrix
                            * app.state.world.collector.view_matrix;
                        let depth_reversed_z = rc3d_core::depth_reversed_z_from_projection(
                            app.state.world.collector.projection_matrix,
                        );
                        for (line_verts, color) in app.editor.gizmo.generate_lines() {
                            let mut ep: Vec<[f32; 3]> = Vec::new();
                            for c in line_verts.chunks_exact(2) {
                                ep.push(c[0].position);
                                ep.push(c[1].position);
                            }
                            if !ep.is_empty() {
                                app.state.world.collector.draw_calls.push(DrawCall {
                                    vertices: Arc::new(Vec::new()),
                                    indices: None,
                                    edge_positions: Arc::new(ep),
                                    mvp: vp,
                                    model_matrix: Mat4::IDENTITY,
                                    camera_pos: app.state.world.collector.camera_pos,
                                    light_set_id: 0,
                                    overlay_color: Some(color),
                                    display_mode: DisplayMode::ShadedWithEdges,
                                    projection_orthographic: app.state
                                        .world
                                        .collector
                                        .projection_orthographic,
                                    depth_reversed_z,
                                    node_type_label: Arc::from("Gizmo"),
                                    ..Default::default()
                                });
                            }
                        }
                    }

                    // Only re-collect markup and transfer tables on full traversal
                    if !can_skip && !camera_only {
                        let markup_root = app.state.world.graph.roots().first().copied().unwrap_or_default();
                        renderer.collect_markup_vertices(&app.state.world.graph, markup_root);
                        renderer.set_effect_commands(
                            std::mem::take(&mut app.state.world.collector.effect_commands),
                        );
                        let light_sets =
                            std::mem::take(&mut app.state.world.collector.light_sets);
                        renderer.set_light_sets(light_sets);
                    }

                    let cam_interacting = app.state.camera_controller
                        .as_ref()
                        .map_or(false, |c| c.middle_orbit_held || c.left_orbit_held || c.panning)
                        || app.state.viewport_cameras.active()
                            .map_or(false, |vc| vc.controller.middle_orbit_held || vc.controller.left_orbit_held || vc.controller.panning);
                    renderer.interaction_active = cam_interacting;

                    if !app.state.world.collector.draw_calls.is_empty() {
                        let mut overlay = None;
                        if let Some(ui) = &mut app.state.editor_ui {
                            overlay = Some(ui as *mut EditorUi);
                        }
                        let stats = if let Some(ui_ptr) = overlay {
                            let device = renderer.device.clone();
                            let queue = renderer.queue.clone();
                            renderer.render_draw_calls_with_overlay(
                                &app.state.world.collector.draw_calls,
                                &app.state.world.graph,
                                Some(&mut |encoder, view| {
                                    // SAFETY: ui_ptr points to app.state.editor_ui and is valid for this frame scope.
                                    let ui = unsafe { &mut *ui_ptr };
                                    ui.paint(&device, &queue, encoder, view);
                                }),
                            )
                        } else {
                            renderer.render_draw_calls(
                                &app.state.world.collector.draw_calls,
                                &app.state.world.graph,
                            )
                        };
                        let now = Instant::now();
                        let frame_time_ms =
                            now.duration_since(app.state.last_frame_time).as_secs_f32() * 1000.0;
                        app.state.last_frame_time = now;
                        app.state.last_frame_time_ms = frame_time_ms;
                        app.state.fps_tracker.push(frame_time_ms);
                        if renderer.frame_counter() % 60 == 0 {
                            log::info!(
                                "CPU frame time: {:.1}ms ({:.0}fps)",
                                frame_time_ms,
                                1000.0 / frame_time_ms.max(0.001)
                            );
                        }
                        app.state.last_render_stats = stats;
                        let idle_for_secs = app.state.adaptive_last_interaction.elapsed().as_secs_f32();
                        let has_dynamic_scene = app.state.world.engines.is_some();
                        let allow_downgrade = idle_for_secs < 0.35 || has_dynamic_scene;
                        let lock_idle = idle_for_secs >= 2.0 && !has_dynamic_scene;
                        let adaptive_control = match app.state.adaptive_quality_mode {
                            AdaptiveQualityMode::Off => AdaptiveControl::Disabled,
                            AdaptiveQualityMode::On => AdaptiveControl::Dynamic { allow_downgrade },
                            AdaptiveQualityMode::AutoIdleLock => {
                                if lock_idle {
                                    AdaptiveControl::Locked
                                } else {
                                    AdaptiveControl::Dynamic { allow_downgrade }
                                }
                            }
                        };
                        renderer.report_frame_time_ms(frame_time_ms, adaptive_control);
                        let mut mode_name = format!(
                            "{:?} | IBL:{}",
                            renderer.display_mode(),
                            renderer.ibl_preset_name()
                        );
                        if let Some(text_hook) = &app.panel_overlay_text_hook {
                            let overlay = text_hook();
                            if !overlay.is_empty() {
                                mode_name.push('\n');
                                mode_name.push_str(&overlay);
                            }
                        }
                        renderer.update_hud(
                            app.state.fps_tracker.smoothed_fps(),
                            app.state.fps_tracker.average_frame_ms(),
                            &app.state.last_render_stats,
                            &mode_name,
                        );
                        let quality = renderer.adaptive_quality_name().to_string();
                        app.state.fps_tracker.maybe_log(&app.state.last_render_stats, &quality);
                        let perf_active = renderer.performance_mode_active();
                        if perf_active != app.state.perf_mode_last {
                            app.state.perf_mode_last = perf_active;
                            if perf_active {
                                window.set_title("rustcoin3d [Performance Mode]");
                                log::warn!("Performance mode is enabled");
                            } else {
                                window.set_title("rustcoin3d");
                                log::info!("Performance mode is disabled");
                            }
                        }
                        if app.lod.preview_mode_active {
                            window.set_title("rustcoin3d [Stream mesh]");
                        }
                    } else {
                        match renderer.surface.get_current_texture() {
                            Ok(tex) => drop(tex),
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                // Surface needs reconfiguration; handled on next resize event
                            }
                            Err(e) => log::warn!("Surface error during no-op frame: {:?}", e),
                        }
                    }
                    if app.state.continuous_redraw {
                        window.request_redraw();
                    }
                }
                if app.lod.preview_mode_active {
                    if let Some(w) = &app.state.window {
                        w.request_redraw();
                    }
                }
            }
            _ => {}
        }

        // Tick fly-to camera animation (outside renderer borrow scope)
        {
            let dt_s = app.state.last_frame_time_ms / 1000.0;
            let mut flying = false;
            if let Some(ref mut ctrl) = app.state.camera_controller {
                flying = ctrl.tick_fly(dt_s);
            }
            if !flying {
                if let Some(vc) = app.state.viewport_cameras.active_mut() {
                    vc.controller.tick_fly(dt_s);
                }
            }
        }

        // Camera control (legacy single-camera path)
        let block_orbit = app.editor.gizmo_dragging || app.editor.box_select_drag || app.editor.view_split_drag.is_some();
        let cursor_for_camera_delta: (f64, f64) = match &event {
            WindowEvent::CursorMoved { .. } => cursor_pos_before_event,
            _ => app.input.cursor_pos,
        };
        let left_orbit = app.camera_left_orbit_enabled();
        if !ui_consumed {
            if let Some(ref mut ctrl) = app.state.camera_controller {
                if !app.editor.measurement_mode && !block_orbit {
                    App::dispatch_camera_event(ctrl, &event, &cursor_for_camera_delta, left_orbit);
                    if let Some(window) = &app.state.window {
                        match event {
                            WindowEvent::CursorMoved { .. }
                            | WindowEvent::MouseInput { .. }
                            | WindowEvent::MouseWheel { .. }
                            | WindowEvent::KeyboardInput { .. } => window.request_redraw(),
                            _ => {}
                        }
                    }
                }
            }

            // Viewport camera set (new multi-viewport path).
            // Wheel: cursor viewport; middle/right: lock to press viewport until release.
            if !app.state.viewport_cameras.cameras.is_empty() && !app.editor.measurement_mode && !block_orbit {
                app.dispatch_multi_viewport_camera(&event, &cursor_for_camera_delta, left_orbit);
            }
        }
    }
