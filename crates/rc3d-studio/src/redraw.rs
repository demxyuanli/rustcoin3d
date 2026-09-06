//! Redraw scheduling: pending-kind merging, pointer throttling and the
//! "does the scene need another frame" predicates.

use rc3d_editor::{Editor, EditorInteractionState};
use rc3d_engine_api::Engine;
use winit::window::CursorIcon;

use crate::host::HostState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RedrawKind {
    None,
    UiOnly,
    Full,
}

pub(crate) const UI_POINTER_REDRAW_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(16);

pub(crate) fn merge_redraw(pending: &mut RedrawKind, kind: RedrawKind) {
    *pending = match (*pending, kind) {
        (RedrawKind::Full, _) | (_, RedrawKind::Full) => RedrawKind::Full,
        (RedrawKind::UiOnly, RedrawKind::UiOnly) => RedrawKind::UiOnly,
        (RedrawKind::None, k) => k,
        (a, RedrawKind::None) => a,
    };
}

/// Merge `kind` into `state.pending_redraw` and request a window redraw.
pub(crate) fn schedule_redraw(state: &mut HostState, window: &winit::window::Window, kind: RedrawKind) {
    merge_redraw(&mut state.pending_redraw, kind);
    window.request_redraw();
}

pub(crate) fn viewport_splitter_cursor(
    axis: Option<rc3d_render::viewport::ViewportSplitAxis>,
) -> Option<CursorIcon> {
    use rc3d_render::viewport::ViewportSplitAxis;
    use winit::window::ResizeDirection;
    match axis {
        Some(ViewportSplitAxis::HorizontalFraction) => Some(CursorIcon::from(ResizeDirection::East)),
        Some(ViewportSplitAxis::VerticalFraction) => Some(CursorIcon::from(ResizeDirection::South)),
        None => None,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum PointerPhase {
    Move,
    Press,
    Release,
    Wheel,
}

pub(crate) fn schedule_pointer_redraw(
    state: &mut HostState,
    window: &winit::window::Window,
    engine: &Engine,
    editor: &Editor,
    phase: PointerPhase,
    throttle_ui_move: bool,
) {
    let kind = if pointer_needs_scene_frame(engine, editor, &state.interaction, phase) {
        RedrawKind::Full
    } else {
        RedrawKind::UiOnly
    };
    // Always cap UiOnly pointer moves (including over floating chrome). Without this,
    // high-rate CursorMoved over the tool strip/menus floods fullscreen blit + egui.
    if throttle_ui_move
        && kind == RedrawKind::UiOnly
        && state.last_ui_pointer_redraw.elapsed() < UI_POINTER_REDRAW_INTERVAL
    {
        return;
    }
    state.last_ui_pointer_redraw = std::time::Instant::now();
    merge_redraw(&mut state.pending_redraw, kind);
    window.request_redraw();
}

pub(crate) fn scene_interaction_active(
    engine: &Engine,
    interaction: &EditorInteractionState,
    editor: &Editor,
) -> bool {
    scene_pointer_captured(engine, interaction) || editor.chrome().nav_cube_dragging
}

fn camera_wants_continuous_redraw(engine: &Engine) -> bool {
    if engine.controller.walk_mode || engine.controller.is_flying() {
        return true;
    }
    engine
        .viewport_cameras
        .active()
        .is_some_and(|vc| vc.controller.walk_mode || vc.controller.is_flying())
}

fn pointer_needs_scene_frame(
    engine: &Engine,
    editor: &Editor,
    interaction: &EditorInteractionState,
    phase: PointerPhase,
) -> bool {
    if scene_interaction_active(engine, interaction, editor) {
        return true;
    }
    if camera_wants_continuous_redraw(engine) {
        return true;
    }
    // Floating UI / nav cube: never Full from idle pointer motion.
    if editor.egui_blocks_scene_pointer()
        || editor.nav_cube_blocks_scene_pointer(
            engine.input.cursor_pos.0 as f32,
            engine.input.cursor_pos.1 as f32,
        )
    {
        return false;
    }
    match phase {
        // Idle hover over the film stays UiOnly; Press/Wheel/captured drag drive Full.
        PointerPhase::Move => false,
        PointerPhase::Press | PointerPhase::Wheel => engine.pointer_in_scene_region(),
        PointerPhase::Release => {
            engine.pointer_in_scene_region() || scene_pointer_captured(engine, interaction)
        }
    }
}

pub(crate) fn should_keep_redrawing(
    engine: &Engine,
    interaction: &EditorInteractionState,
    editor: &Editor,
) -> bool {
    scene_interaction_active(engine, interaction, editor)
        || camera_wants_continuous_redraw(engine)
}

pub(crate) fn scene_pointer_captured(
    engine: &Engine,
    interaction: &EditorInteractionState,
) -> bool {
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
        || !interaction.measure.points.is_empty()
}

pub(crate) fn route_scene_pointer(
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
    if (editor.nav_cube_blocks_scene_pointer(px, py) || editor.egui_blocks_scene_pointer())
        && !captured
    {
        return false;
    }
    let in_scene = engine.pointer_in_scene_region();
    match phase {
        PointerPhase::Wheel | PointerPhase::Press => in_scene,
        PointerPhase::Move | PointerPhase::Release => in_scene || captured,
    }
}
