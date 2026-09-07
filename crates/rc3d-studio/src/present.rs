//! Per-frame presentation: command drain, egui render, caption actions,
//! prefs persistence edges and the UiOnly/Full present decision.

use rc3d_editor::{CaptionAction, Editor, EditorCommand};
use rc3d_engine_api::Engine;
use winit::event_loop::ActiveEventLoop;

use crate::host::HostState;
use crate::host_cmds;
use crate::redraw::{self, merge_redraw, schedule_redraw, RedrawKind};

/// What the frame decided; `EarlyReturn` skips the caller's keep-redrawing check.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameOutcome {
    Done,
    EarlyReturn,
}

pub(crate) struct FrameDeps<'a> {
    pub(crate) engine: &'a mut Engine,
    pub(crate) editor: &'a mut Editor,
    pub(crate) window: &'a winit::window::Window,
    pub(crate) event_loop: &'a ActiveEventLoop,
}

pub(crate) fn sync_surface_to_window(
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

/// Returns `true` when the 3D region changed this frame (dock resize / open / close).
pub(crate) fn apply_scene_region_from_editor(engine: &mut Engine, editor: &Editor) -> bool {
    let Some(r) = editor.scene_pixel_rect() else {
        return false;
    };
    engine.apply_scene_region(rc3d_render::viewport::ViewportRect {
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
    })
}

/// Load the pending scene graph once (first splash frame), before any present
/// so the very first rendered frame already contains real geometry.
fn load_pending_scene(engine: &mut Engine, state: &mut HostState) {
    if let Some(graph) = state.pending_graph.take() {
        engine.load_scene(graph);
    }
}

/// Minimum time the splash stays up: long enough to be perceived, short
/// enough not to feel like stalling.
pub(crate) const SPLASH_MIN_SHOW: std::time::Duration =
    std::time::Duration::from_millis(700);

/// One splash frame. The splash advances Device -> Scene -> Ready over at
/// least `SPLASH_MIN_SHOW`, then flips `done` so the next frame paints the
/// real workspace. Frames during the splash are driven by `about_to_wait`
/// polling because a hidden window receives no `WM_PAINT`, so
/// `request_redraw` alone never delivers `RedrawRequested` on Windows.
pub(crate) fn splash_step(
    editor: &mut Editor,
    state: &mut HostState,
    now: std::time::Instant,
) {
    use rc3d_editor::ui::splash::{SplashStage, SplashState};
    let cur = editor.splash();
    if cur.done {
        return;
    }
    // Measure the minimum show from the first painted splash frame, not from
    // window creation: slow device/scene init could otherwise consume the
    // whole budget and skip the splash entirely.
    if !cur.painted {
        state.splash_start = now;
        return;
    }
    let elapsed = now.saturating_duration_since(state.splash_start);
    // Continuous bar: discrete per-stage steps stalled at 75% for the whole
    // second half of the show.
    let frac = (elapsed.as_secs_f32() / SPLASH_MIN_SHOW.as_secs_f32()).clamp(0.0, 1.0);
    let stage = if frac >= 1.0 {
        SplashStage::Ready
    } else if frac < 0.5 {
        SplashStage::Device
    } else {
        SplashStage::Scene
    };
    // Keep `painted`: it is set by the egui draw pass and gates the window
    // reveal, so replacing the whole state must not clear it.
    editor.set_splash(SplashState {
        stage,
        progress: frac,
        done: frac >= 1.0,
        painted: cur.painted,
    });
}

/// Drain editor commands, render egui chrome and present UiOnly or Full.
pub(crate) fn present_frame(deps: FrameDeps<'_>, state: &mut HostState) -> FrameOutcome {
    let FrameDeps {
        engine,
        editor,
        window,
        event_loop,
    } = deps;
    if !sync_surface_to_window(engine, editor, window) {
        schedule_redraw(state, window, RedrawKind::Full);
        return FrameOutcome::EarlyReturn;
    }
    let mut redraw = state.pending_redraw;
    state.pending_redraw = RedrawKind::None;
    // Splash lifecycle: load the scene on the first frame so the reveal frame
    // already contains geometry; the stage bar is time driven. The first frames
    // run while the window is still hidden (no WM_PAINT), so `about_to_wait`
    // drives them directly until the splash painted frame reveals the window.
    let splash_active = !editor.splash().done;
    if splash_active {
        load_pending_scene(engine, state);
        splash_step(editor, state, std::time::Instant::now());
        redraw = RedrawKind::Full;
    }
    // No scene film yet → Full. Bare request_repaint (egui menus) → UiOnly, never Full.
    if !state.scene_presented {
        redraw = RedrawKind::Full;
    } else if redraw == RedrawKind::None {
        redraw = RedrawKind::UiOnly;
    }
    let mut need_scene = false;
    for cmd in editor.take_commands() {
        need_scene |= cmd.needs_scene_redraw();
        let persist = matches!(
            cmd,
            EditorCommand::SetUiTheme(_)
                | EditorCommand::SetUiLocale(_)
                | EditorCommand::BindKey { .. }
        );
        host_cmds::apply_editor_cmd(
            engine,
            &mut state.interaction,
            &mut state.session,
            &mut state.recent,
            &mut state.active_case,
            cmd,
        );
        if persist {
            host_cmds::save_prefs(&state.session, editor, engine, &state.recent);
            state.last_prefs_save = std::time::Instant::now();
        }
    }
    if need_scene {
        redraw = RedrawKind::Full;
    }
    if editor.close_after_save() {
        if !state.session.dirty {
            host_cmds::save_prefs(&state.session, editor, engine, &state.recent);
            event_loop.exit();
            return FrameOutcome::EarlyReturn;
        }
        editor.clear_close_after_save();
    }
    editor.sync_document_chrome(state.session.window_title(), state.session.dirty);
    let ui_ctx = crate::ui_ctx::build_ui_ctx(
        engine,
        &state.session,
        &state.interaction,
        state.active_case.as_ref(),
    );
    editor.set_window_maximized(window.is_maximized());
    let egui_wants_repaint = editor.render(window, engine, &ui_ctx);
    engine.compositor = editor.compositor_graph().clone();
    if egui_wants_repaint {
        // Menus/popups open on click and need a settle frame; without this,
        // Wait + no pointer move leaves the dropdown invisible until move.
        merge_redraw(&mut state.pending_redraw, RedrawKind::UiOnly);
        window.request_redraw();
    }
    // The splash paints no widgets, so it never sets `repaint_delay`; without
    // an explicit pump the loop stalls on the current stage (the bar froze at
    // 75%) and `done` is never reached.
    if !editor.splash().done {
        window.request_redraw();
    }
    match editor.take_caption_action() {
        Some(CaptionAction::Close) => {
            host_cmds::save_prefs(&state.session, editor, engine, &state.recent);
            event_loop.exit();
            return FrameOutcome::EarlyReturn;
        }
        Some(CaptionAction::Minimize) => {
            window.set_minimized(true);
            schedule_redraw(state, window, RedrawKind::UiOnly);
            return FrameOutcome::EarlyReturn;
        }
        Some(CaptionAction::ToggleMaximize) => {
            window.set_maximized(!window.is_maximized());
            let _ = sync_surface_to_window(engine, editor, window);
            schedule_redraw(state, window, RedrawKind::Full);
            return FrameOutcome::EarlyReturn;
        }
        Some(CaptionAction::Drag) => {
            let _ = window.drag_window();
        }
        Some(CaptionAction::ShowSystemMenu) => {
            if let Some(pos) = state.cursor_pos {
                window.show_window_menu(pos);
            }
        }
        None => {}
    }
    let region_changed = apply_scene_region_from_editor(engine, editor);
    persist_on_resize_edges(engine, editor, state);
    host_cmds::maybe_autosave(engine, &state.session, &mut state.last_autosave);
    host_cmds::maybe_prefs(
        &state.session,
        editor,
        engine,
        &state.recent,
        &mut state.last_prefs_save,
    );
    let device = engine.wgpu_device().clone();
    let queue = engine.wgpu_queue().clone();
    // Region changed (dock drag / panel open / close): the retained film
    // has the stale size, so a UiOnly blit would show a stale image. Full-render.
    let mut do_full = redraw == RedrawKind::Full || region_changed;
    if !do_full {
        engine.set_interaction_active(false);
        let ok = engine.present_ui_overlay_only(Some(&mut |enc, view| {
            editor.paint(&device, &queue, enc, view);
        }));
        if !ok {
            do_full = true;
        }
    }
    if do_full {
        present_full_frame(engine, editor, state, window, &device, &queue);
        state.scene_presented = true;
        // Reveal as soon as a frame has real pixels in the swapchain.
        // Splash phase: that frame paints only the splash (build_ui returns
        // early), so the user sees the splash first — `painted` guards the
        // very first frames where egui has no screen size yet and the surface
        // would still be un-presented white. After `done` flips, the same
        // path reveals the finished workspace when the splash is disabled.
        let splash = editor.splash();
        let revealed = if splash.done { true } else { splash.painted };
        if revealed && window.is_visible() != Some(true) {
            window.set_visible(true);
        }
    }
    FrameOutcome::Done
}

/// Dock/splitter drag falling edges persist the final geometry immediately so
/// a crash or quick exit cannot lose it (`maybe_prefs` is throttled).
fn persist_on_resize_edges(engine: &mut Engine, editor: &Editor, state: &mut HostState) {
    let resizing_now =
        editor.chrome().side_dock_resizing || editor.chrome().bottom_dock_resizing;
    if state.docks_resizing && !resizing_now {
        host_cmds::save_prefs(&state.session, editor, engine, &state.recent);
        state.last_prefs_save = std::time::Instant::now();
    }
    state.docks_resizing = resizing_now;
    let split_resizing_now = state.interaction.view_split_drag.is_some();
    if state.viewport_split_resizing && !split_resizing_now {
        host_cmds::save_prefs(&state.session, editor, engine, &state.recent);
        state.last_prefs_save = std::time::Instant::now();
    }
    state.viewport_split_resizing = split_resizing_now;
}

/// Full scene render with nav-cube / section overlays and egui paint on top.
fn present_full_frame(
    engine: &mut Engine,
    editor: &mut Editor,
    state: &HostState,
    window: &winit::window::Window,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let ppp = window.scale_factor() as f32;
    let cube_px = (rc3d_editor::ui::nav_cube::SIZE * ppp).round() as u32;
    let cube_margin = (rc3d_editor::ui::nav_cube::MARGIN * ppp).round() as u32;
    engine.layout_overlay_top_right(
        rc3d_engine_api::OVERLAY_NAV_CUBE,
        cube_px.max(1),
        cube_margin,
    );
    if let Some(ov) = engine.overlay_viewport_mut(rc3d_engine_api::OVERLAY_NAV_CUBE) {
        ov.clear_color = [0.0, 0.0, 0.0, 0.0];
        rc3d_editor::ui::nav_cube::sync_overlay(
            ov,
            state.session.ui_locale,
            state.session.ui_theme,
            editor.nav_cube_hover_slot(),
        );
    }
    rc3d_editor::interaction::sync_section_overlay(engine, &state.interaction);
    engine.set_interaction_active(redraw::scene_interaction_active(
        engine,
        &state.interaction,
        editor,
    ));
    engine.render_with_overlay(Some(&mut |enc, view| {
        editor.paint(device, queue, enc, view);
    }));
}
