//! winit host: window lifecycle, event routing and frame scheduling.
//! Frame presentation lives in `present.rs`, redraw policy in `redraw.rs`,
//! command/prefs handling in `host_cmds.rs`.

use rc3d_core::DisplayMode;
use rc3d_editor::{
    apply_command, interaction, Editor, EditorContext, EditorInteractionState, EditorSession,
};
use rc3d_engine_api::{CameraController, Engine, EventRouteOpts};
use rc3d_scene::SceneGraph;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::CursorIcon;

use crate::host::HostState;
use crate::host_cmds;
use crate::present::{apply_scene_region_from_editor, present_frame, FrameDeps, FrameOutcome};
use crate::redraw::{
    route_scene_pointer, schedule_pointer_redraw, schedule_redraw, should_keep_redrawing,
    viewport_splitter_cursor, PointerPhase, RedrawKind,
};
use crate::win_shell;

pub struct StudioApp {
    engine: Option<Engine>,
    editor: Option<Editor>,
    window: Option<winit::window::Window>,
    state: HostState,
    on_resize_edge: bool,
    cad_matrix: bool,
}

impl StudioApp {
    pub fn new(graph: SceneGraph, cad_matrix: bool) -> Self {
        let prefs = crate::prefs::load();
        let mut session = EditorSession::default();
        session.ui_theme = prefs.theme;
        session.ui_locale = prefs.locale;
        session.keymap = prefs.keymap.clone();
        session.file_dialog_dir = prefs.last_folder.clone();
        let mut pending = graph;
        if let Some(path) = &prefs.last_document {
            if let Ok(g) = rc3d_editor::document::load_native_scene(path) {
                pending = g;
                session.document_path = Some(path.clone());
            }
        }
        Self {
            engine: None,
            editor: None,
            window: None,
            state: HostState {
                interaction: EditorInteractionState::default(),
                session,
                pending_graph: Some(pending),
                cursor_pos: None,
                recent: prefs.recent,
                last_autosave: std::time::Instant::now(),
                last_prefs_save: std::time::Instant::now(),
                pending_redraw: RedrawKind::Full,
                scene_presented: false,
                last_ui_pointer_redraw: std::time::Instant::now(),
                active_case: None,
                docks_resizing: false,
                viewport_split_resizing: false,
                splash_start: std::time::Instant::now(),
            },
            on_resize_edge: false,
            cad_matrix,
        }
    }
}

impl ApplicationHandler for StudioApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(win_shell::window_attributes())
            .expect("failed to create window");
        win_shell::apply_after_create(&window);
        // Splash shows immediately: engine + editor are created now (device,
        // pipelines, egui fonts), but the scene graph upload is deferred to the
        // first `Ready` frame so the splash is visible while it runs.
        let mut engine = Engine::new(&window);
        engine.controller = CameraController::new(rc3d_core::math::Vec3::ZERO, 10.0);
        engine.set_display_mode(DisplayMode::Shaded);
        engine.add_overlay_viewport(rc3d_editor::ui::nav_cube::nav_cube_overlay());
        let mut editor = Editor::new(&window, &engine);
        editor.enable_studio_shell("rustcoin3d Studio");
        editor.set_splash(rc3d_editor::ui::splash::SplashState::starting());
        {
            let p = crate::prefs::load();
            let chrome = editor.chrome_mut();
            chrome.side_tab = p.side_tab;
            chrome.bottom_tab = p.bottom_tab;
            chrome.workspace = p.workspace;
            chrome.tool_strip_pos = p.tool_strip_pos;
            chrome.side_dock_width = p.side_dock_width;
            chrome.bottom_dock_height = p.bottom_dock_height;
            chrome.inspector_ratio = p.inspector_ratio;
            engine
                .viewport_layout_mut()
                .set_quad_splits(p.viewport_h_split, p.viewport_v_split);
            engine.set_layout_mode(p.viewport_layout_mode);
        }
        self.engine = Some(engine);
        self.editor = Some(editor);
        self.window = Some(window);
        if self.cad_matrix {
            let engine = self.engine.as_mut().expect("engine");
            let report = crate::cad_matrix::run(engine);
            let path = std::path::Path::new("target/cad-matrix-report.txt");
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            match std::fs::write(path, &report.text) {
                Ok(()) => log::info!("CAD matrix report written to {}", path.display()),
                Err(e) => log::error!("CAD matrix report write failed: {e}"),
            }
            for line in report.text.lines() {
                if line.starts_with("FAIL") || line.starts_with("result:") {
                    log::warn!("{line}");
                } else {
                    log::info!("{line}");
                }
            }
            std::process::exit(if report.passed { 0 } else { 1 });
        }
        if let Some(window) = self.window.as_ref() {
            // Pacing starts when the first frame can run, not when the app
            // struct was built (engine init would eat the splash budget).
            self.state.splash_start = std::time::Instant::now();
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if let WindowEvent::CursorMoved { position, .. } = &event {
            self.state.cursor_pos = Some(*position);
            if let Some(dir) = win_shell::resize_direction(
                *position,
                window.inner_size(),
                window.scale_factor(),
                window.is_maximized(),
            ) {
                window.set_cursor(win_shell::cursor_for_resize(dir));
                self.on_resize_edge = true;
            } else if self.on_resize_edge {
                window.set_cursor(CursorIcon::Default);
                self.on_resize_edge = false;
            }
        }
        if let WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } = &event
        {
            if let Some(pos) = self.state.cursor_pos {
                if let Some(dir) = win_shell::resize_direction(
                    pos,
                    window.inner_size(),
                    window.scale_factor(),
                    window.is_maximized(),
                ) {
                    let _ = window.drag_resize_window(dir);
                    return;
                }
            }
        }
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        let egui_consumed = editor.handle_event(window, &event);
        let is_pointer = matches!(
            event,
            WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseInput { .. }
                | WindowEvent::MouseWheel { .. }
        );
        let keep_for_host = matches!(
            event,
            WindowEvent::Resized(_)
                | WindowEvent::ScaleFactorChanged { .. }
                | WindowEvent::RedrawRequested
        );
        if egui_consumed && !is_pointer && !keep_for_host {
            schedule_redraw(&mut self.state, window, RedrawKind::UiOnly);
            return;
        }
        match event {
            WindowEvent::RedrawRequested => {
                let outcome = present_frame(
                    FrameDeps {
                        engine,
                        editor,
                        window,
                        event_loop,
                    },
                    &mut self.state,
                );
                if outcome == FrameOutcome::EarlyReturn {
                    return;
                }
                if should_keep_redrawing(engine, &self.state.interaction, editor) {
                    schedule_redraw(&mut self.state, window, RedrawKind::Full);
                }
            }
            WindowEvent::CloseRequested => {
                host_cmds::save_prefs(
                    &self.state.session,
                    editor,
                    engine,
                    &self.state.recent,
                );
                if self.state.session.dirty {
                    editor.request_close_prompt();
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if size.width >= 2 && size.height >= 2 {
                    engine.resize(size.width, size.height);
                    editor.resize(size.width, size.height, window.scale_factor() as f32);
                }
                self.state.scene_presented = false;
                schedule_redraw(&mut self.state, window, RedrawKind::Full);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = window.inner_size();
                if size.width >= 2 && size.height >= 2 {
                    engine.resize(size.width, size.height);
                    editor.resize(size.width, size.height, window.scale_factor() as f32);
                }
                self.state.scene_presented = false;
                schedule_redraw(&mut self.state, window, RedrawKind::Full);
            }
            WindowEvent::CursorMoved { .. } => {
                if self.on_resize_edge {
                    return;
                }
                engine.feed_input(&event);
                apply_scene_region_from_editor(engine, editor);
                if route_scene_pointer(engine, editor, &self.state.interaction, PointerPhase::Move)
                {
                    let input = engine.input;
                    let cmds = {
                        let mut ctx = EditorContext::with_interaction(
                            engine,
                            std::mem::take(&mut self.state.interaction),
                        );
                        interaction::on_cursor_moved(&mut ctx, window, &input);
                        let cmds = std::mem::take(&mut ctx.commands);
                        self.state.interaction = ctx.interaction;
                        cmds
                    };
                    for cmd in cmds {
                        apply_command(
                            engine,
                            &mut self.state.interaction,
                            &mut self.state.session,
                            cmd,
                        );
                    }
                    engine.dispatch_routed_event(&event, EventRouteOpts::editor());
                } else if self.state.interaction.view_split_drag.is_none() {
                    self.state.interaction.view_split_hover = None;
                }
                if let Some(cursor) = viewport_splitter_cursor(
                    self.state
                        .interaction
                        .view_split_drag
                        .or(self.state.interaction.view_split_hover),
                ) {
                    window.set_cursor(cursor);
                } else {
                    window.set_cursor(CursorIcon::Default);
                }
                schedule_pointer_redraw(
                    &mut self.state,
                    window,
                    engine,
                    editor,
                    PointerPhase::Move,
                    true,
                );
            }
            WindowEvent::MouseInput { state, button, .. } => {
                engine.feed_input(&event);
                apply_scene_region_from_editor(engine, editor);
                let phase = match state {
                    ElementState::Pressed => PointerPhase::Press,
                    ElementState::Released => PointerPhase::Release,
                };
                if route_scene_pointer(engine, editor, &self.state.interaction, phase) {
                    let input = engine.input;
                    if button == MouseButton::Left {
                        let cmds = {
                            let mut ctx = EditorContext::with_interaction(
                                engine,
                                std::mem::take(&mut self.state.interaction),
                            );
                            match state {
                                ElementState::Pressed => {
                                    interaction::on_left_down(&mut ctx, window, &input, true);
                                }
                                ElementState::Released => {
                                    interaction::on_left_up(&mut ctx, window, &input, true);
                                }
                            }
                            let cmds = std::mem::take(&mut ctx.commands);
                            self.state.interaction = ctx.interaction;
                            cmds
                        };
                        for cmd in cmds {
                            apply_command(
                                engine,
                                &mut self.state.interaction,
                                &mut self.state.session,
                                cmd,
                            );
                        }
                    }
                    engine.dispatch_routed_event(&event, EventRouteOpts::editor());
                }
                schedule_pointer_redraw(
                    &mut self.state,
                    window,
                    engine,
                    editor,
                    phase,
                    false,
                );
            }
            WindowEvent::MouseWheel { .. } => {
                apply_scene_region_from_editor(engine, editor);
                if route_scene_pointer(
                    engine,
                    editor,
                    &self.state.interaction,
                    PointerPhase::Wheel,
                ) {
                    engine.handle_window_event(&event, EventRouteOpts::editor());
                }
                schedule_pointer_redraw(
                    &mut self.state,
                    window,
                    engine,
                    editor,
                    PointerPhase::Wheel,
                    false,
                );
            }
            WindowEvent::ModifiersChanged(_) => {
                engine.feed_input(&event);
                schedule_redraw(&mut self.state, window, RedrawKind::UiOnly);
            }
            WindowEvent::KeyboardInput {
                event: ref key, ..
            } if key.state == ElementState::Pressed => {
                engine.handle_window_event(
                    &event,
                    EventRouteOpts {
                        left_orbit: false,
                        pick_on_click: false,
                        camera: false,
                    },
                );
                if let PhysicalKey::Code(code) = key.physical_key {
                    host_cmds::dispatch_key(
                        engine,
                        editor,
                        &mut self.state.interaction,
                        &mut self.state.session,
                        &mut self.state.recent,
                        &mut self.state.active_case,
                        code,
                    );
                }
                // Keys apply commands immediately (not via editor queue) — Full until that path is unified.
                schedule_redraw(&mut self.state, window, RedrawKind::Full);
            }
            _ => {
                engine.handle_window_event(&event, EventRouteOpts::editor());
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Before the first painted splash frame reveals the window it is
        // hidden, and a hidden window receives no WM_PAINT — so neither
        // `request_redraw` nor Poll would deliver `RedrawRequested`. Drive
        // those frames directly from the wake tick. Once visible, Poll keeps
        // the splash animating via normal RedrawRequested events and direct
        // presentation would double-present, so skip it.
        let splash_active = self
            .editor
            .as_ref()
            .is_some_and(|e| !e.splash().done);
        if splash_active {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            let window_hidden = self
                .window
                .as_ref()
                .is_some_and(|w| w.is_visible() != Some(true));
            if window_hidden {
                if let (Some(engine), Some(editor), Some(window)) =
                    (self.engine.as_mut(), self.editor.as_mut(), self.window.as_ref())
                {
                    let _ = present_frame(
                        FrameDeps {
                            engine,
                            editor,
                            window,
                            event_loop,
                        },
                        &mut self.state,
                    );
                }
            }
            return;
        }
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
        if let (Some(engine), Some(editor)) =
            (self.engine.as_mut(), self.editor.as_mut())
        {
            host_cmds::maybe_autosave(engine, &self.state.session, &mut self.state.last_autosave);
            host_cmds::maybe_prefs(
                &self.state.session,
                editor,
                engine,
                &self.state.recent,
                &mut self.state.last_prefs_save,
            );
            if should_keep_redrawing(engine, &self.state.interaction, editor) {
                if let Some(window) = self.window.as_ref() {
                    schedule_redraw(&mut self.state, window, RedrawKind::Full);
                }
            }
        }
    }
}

