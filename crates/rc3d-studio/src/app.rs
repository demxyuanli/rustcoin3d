use rc3d_core::DisplayMode;
use rc3d_editor::{
    apply_command, interaction, CaptionAction, Editor, EditorCommand, EditorContext,
    EditorInteractionState, EditorSession,
};
use rc3d_engine_api::{sync_gizmo_from_selection, CameraController, Engine, EventRouteOpts};
use rc3d_gizmo::GizmoMode;
use rc3d_render::viewport::ViewportRect;
use rc3d_scene::SceneGraph;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::CursorIcon;

use crate::document::DocumentWebView;
use crate::ui_ctx;
use crate::win_shell;

pub struct StudioApp {
    pending_graph: Option<SceneGraph>,
    engine: Option<Engine>,
    editor: Option<Editor>,
    window: Option<winit::window::Window>,
    interaction: EditorInteractionState,
    session: EditorSession,
    docs: Option<DocumentWebView>,
    cursor_pos: Option<PhysicalPosition<f64>>,
    on_resize_edge: bool,
}

impl StudioApp {
    pub fn new(graph: SceneGraph) -> Self {
        let prefs = crate::prefs::load();
        let mut session = EditorSession::default();
        session.ui_theme = prefs.theme;
        session.ui_locale = prefs.locale;
        Self {
            pending_graph: Some(graph),
            engine: None,
            editor: None,
            window: None,
            interaction: EditorInteractionState::default(),
            session,
            docs: None,
            cursor_pos: None,
            on_resize_edge: false,
        }
    }
}

fn sync_surface_to_window(
    engine: &mut Engine,
    editor: &mut Editor,
    window: &winit::window::Window,
) -> bool {
    let size = window.inner_size();
    if size.width < 2 || size.height < 2 {
        return false;
    }
    let (ew, eh) = engine.input.window_size;
    if ew != size.width || eh != size.height {
        engine.resize(size.width, size.height);
        editor.resize(size.width, size.height, window.scale_factor() as f32);
    }
    true
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
        let graph = self.pending_graph.take().expect("demo scene");
        let mut engine = Engine::new(&window);
        engine.load_scene(graph);
        engine.controller = CameraController::new(rc3d_core::math::Vec3::ZERO, 10.0);
        engine.set_display_mode(DisplayMode::Shaded);
        if let Some(r) = engine.renderer.as_mut() {
            r.set_hud_enabled(false);
        }
        let mut editor = Editor::new(&window, &engine);
        editor.enable_studio_shell("rustcoin3d Studio");
        let docs = DocumentWebView::new(&window);
        self.engine = Some(engine);
        self.editor = Some(editor);
        self.docs = Some(docs);
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
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if let WindowEvent::CursorMoved { position, .. } = &event {
            self.cursor_pos = Some(*position);
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
            if let Some(pos) = self.cursor_pos {
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
            return;
        }
        match event {
            WindowEvent::RedrawRequested => {
                if !sync_surface_to_window(engine, editor, window) {
                    window.request_redraw();
                    return;
                }
                for cmd in editor.take_commands() {
                    let persist = matches!(
                        cmd,
                        EditorCommand::SetUiTheme(_) | EditorCommand::SetUiLocale(_)
                    );
                    apply_command(engine, &mut self.interaction, &mut self.session, cmd);
                    if persist {
                        crate::prefs::save(self.session.ui_theme, self.session.ui_locale);
                    }
                }
                if editor.close_after_save() {
                    if !self.session.dirty {
                        event_loop.exit();
                        return;
                    }
                    editor.clear_close_after_save();
                }
                editor.sync_document_chrome(self.session.window_title(), self.session.dirty);
                let ui_ctx = ui_ctx::build_ui_ctx(engine, &self.session);
                editor.set_window_maximized(window.is_maximized());
                editor.render(window, engine, &ui_ctx);
                match editor.take_caption_action() {
                    Some(CaptionAction::Close) => {
                        event_loop.exit();
                        return;
                    }
                    Some(CaptionAction::Minimize) => {
                        window.set_minimized(true);
                        window.request_redraw();
                        return;
                    }
                    Some(CaptionAction::ToggleMaximize) => {
                        window.set_maximized(!window.is_maximized());
                        let _ = sync_surface_to_window(engine, editor, window);
                        window.request_redraw();
                        return;
                    }
                    Some(CaptionAction::Drag) => {
                        let _ = window.drag_window();
                    }
                    Some(CaptionAction::ShowSystemMenu) => {
                        if let Some(pos) = self.cursor_pos {
                            window.show_window_menu(pos);
                        }
                    }
                    None => {}
                }
                if let Some(r) = editor.scene_pixel_rect() {
                    engine.apply_scene_region(ViewportRect {
                        x: r.x,
                        y: r.y,
                        width: r.width,
                        height: r.height,
                    });
                }
                if let Some(docs) = self.docs.as_ref() {
                    docs.sync(editor.document_pixel_rect(), editor.document_html_visible());
                }
                interaction::sync_section_overlay(engine, &self.interaction);
                let device = engine.wgpu_device().clone();
                let queue = engine.wgpu_queue().clone();
                engine.render_with_overlay(Some(&mut |enc, view| {
                    editor.paint(&device, &queue, enc, view);
                }));
                window.request_redraw();
            }
            WindowEvent::CloseRequested => {
                if self.session.dirty {
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
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = window.inner_size();
                if size.width >= 2 && size.height >= 2 {
                    engine.resize(size.width, size.height);
                    editor.resize(size.width, size.height, window.scale_factor() as f32);
                }
            }
            WindowEvent::CursorMoved { .. } => {
                if self.on_resize_edge {
                    return;
                }
                engine.feed_input(&event);
                if !route_scene_pointer(engine, editor, &self.interaction, PointerPhase::Move) {
                    return;
                }
                let input = engine.input;
                let cmds = {
                    let mut ctx = EditorContext::with_interaction(
                        engine,
                        std::mem::take(&mut self.interaction),
                    );
                    interaction::on_cursor_moved(&mut ctx, window, &input);
                    let cmds = std::mem::take(&mut ctx.commands);
                    self.interaction = ctx.interaction;
                    cmds
                };
                for cmd in cmds {
                    apply_command(engine, &mut self.interaction, &mut self.session, cmd);
                }
                engine.dispatch_routed_event(&event, EventRouteOpts::editor());
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let phase = match state {
                    ElementState::Pressed => PointerPhase::Press,
                    ElementState::Released => PointerPhase::Release,
                };
                if !route_scene_pointer(engine, editor, &self.interaction, phase) {
                    return;
                }
                engine.feed_input(&event);
                let input = engine.input;
                if button == MouseButton::Left {
                    let cmds = {
                        let mut ctx = EditorContext::with_interaction(
                            engine,
                            std::mem::take(&mut self.interaction),
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
                        self.interaction = ctx.interaction;
                        cmds
                    };
                    for cmd in cmds {
                        apply_command(engine, &mut self.interaction, &mut self.session, cmd);
                    }
                }
                engine.dispatch_routed_event(&event, EventRouteOpts::editor());
            }
            WindowEvent::MouseWheel { .. } => {
                if !route_scene_pointer(engine, editor, &self.interaction, PointerPhase::Wheel) {
                    return;
                }
                engine.handle_window_event(&event, EventRouteOpts::editor());
            }
            WindowEvent::ModifiersChanged(_) => {
                engine.feed_input(&event);
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
                let ctrl = engine.input.ctrl_pressed;
                let shift = engine.input.shift_pressed;
                match key.physical_key {
                    PhysicalKey::Code(KeyCode::KeyN) if ctrl => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::NewScene,
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyO) if ctrl => {
                        if let Some(path) = rc3d_editor::pick_open_scene() {
                            apply_command(
                                engine,
                                &mut self.interaction,
                                &mut self.session,
                                EditorCommand::OpenScene(path),
                            );
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyS) if ctrl && shift => {
                        if let Some(path) = rc3d_editor::pick_save_scene() {
                            apply_command(
                                engine,
                                &mut self.interaction,
                                &mut self.session,
                                EditorCommand::SaveSceneAs(path),
                            );
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyS) if ctrl => {
                        if self.session.document_path.is_some() {
                            apply_command(
                                engine,
                                &mut self.interaction,
                                &mut self.session,
                                EditorCommand::SaveScene,
                            );
                        } else if let Some(path) = rc3d_editor::pick_save_scene() {
                            apply_command(
                                engine,
                                &mut self.interaction,
                                &mut self.session,
                                EditorCommand::SaveSceneAs(path),
                            );
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyZ) if ctrl => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::Undo,
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyY) if ctrl => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::Redo,
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyT) => engine.set_gizmo_mode(GizmoMode::Translate),
                    PhysicalKey::Code(KeyCode::KeyR) => engine.set_gizmo_mode(GizmoMode::Rotate),
                    PhysicalKey::Code(KeyCode::KeyG) => engine.set_gizmo_mode(GizmoMode::Scale),
                    PhysicalKey::Code(KeyCode::KeyP) => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::ToggleSectionEdit,
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyF) => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::FitSelection,
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyW) if !ctrl => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::SetDisplayMode(
                                rc3d_editor::EditorDisplayMode::Wireframe,
                            ),
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyS) if !ctrl => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::SetDisplayMode(rc3d_editor::EditorDisplayMode::Shaded),
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyE) => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::SetDisplayMode(
                                rc3d_editor::EditorDisplayMode::ShadedWithEdges,
                            ),
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyH) => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::SetDisplayMode(
                                rc3d_editor::EditorDisplayMode::HiddenLine,
                            ),
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyL) => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::SetDisplayMode(
                                rc3d_editor::EditorDisplayMode::FlatWithEdge,
                            ),
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyI) => {
                        apply_command(
                            engine,
                            &mut self.interaction,
                            &mut self.session,
                            EditorCommand::CycleIbl,
                        );
                    }
                    PhysicalKey::Code(KeyCode::Escape) => {
                        engine.world.graph.clear_selection();
                        sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
                    }
                    _ => {}
                }
            }
            _ => {
                engine.handle_window_event(&event, EventRouteOpts::editor());
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

#[derive(Clone, Copy)]
enum PointerPhase {
    Move,
    Press,
    Release,
    Wheel,
}

fn scene_pointer_captured(engine: &Engine, interaction: &EditorInteractionState) -> bool {
    let c = &engine.controller;
    c.middle_orbit_held
        || c.left_orbit_held
        || c.panning
        || interaction.gizmo_dragging
        || interaction.box_select_drag
        || interaction.lasso_drag
        || interaction.section_drag.is_some()
        || interaction.view_split_drag.is_some()
        || !interaction.markup.click_points.is_empty()
}

fn route_scene_pointer(
    engine: &Engine,
    editor: &rc3d_editor::Editor,
    interaction: &EditorInteractionState,
    phase: PointerPhase,
) -> bool {
    let captured = scene_pointer_captured(engine, interaction);
    let (px, py) = (
        engine.input.cursor_pos.0 as f32,
        engine.input.cursor_pos.1 as f32,
    );
    if editor.nav_cube_blocks_scene_pointer(px, py) && !captured {
        return false;
    }
    let in_scene = engine.pointer_in_scene_region();
    match phase {
        PointerPhase::Wheel | PointerPhase::Press => in_scene,
        PointerPhase::Move | PointerPhase::Release => in_scene || captured,
    }
}
