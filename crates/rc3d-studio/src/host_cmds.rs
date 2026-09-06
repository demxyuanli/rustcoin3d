//! Host-side command application and persistence: editor command dispatch,
//! keymap handling, prefs saving and throttled autosave.

use rc3d_editor::{
    apply_command, keymap, Editor, EditorCommand, EditorInteractionState, EditorSession,
    KeyAction, KeyChord,
};
use rc3d_engine_api::{sync_gizmo_from_selection, Engine};
use winit::keyboard::KeyCode;

use crate::cases::ActiveCase;

/// Apply one editor command on the host, with case-runner interception and
/// recent-file bookkeeping.
pub(crate) fn apply_editor_cmd(
    engine: &mut Engine,
    interaction: &mut EditorInteractionState,
    session: &mut EditorSession,
    recent: &mut Vec<std::path::PathBuf>,
    active_case: &mut Option<ActiveCase>,
    cmd: EditorCommand,
) {
    match &cmd {
        EditorCommand::LoadCase(id) => {
            *active_case = crate::cases::load_case(id, engine, interaction, session);
            return;
        }
        EditorCommand::SetCaseParam { id, value } => {
            if let Some(active) = active_case.as_mut() {
                crate::cases::set_case_param(active, engine, id, *value);
            }
            return;
        }
        EditorCommand::RunCaseStep(step) => {
            if let Some(active) = active_case.as_mut() {
                crate::cases::run_case_step(active, engine, interaction, *step);
            }
            return;
        }
        EditorCommand::NewScene | EditorCommand::OpenScene(_) => {
            *active_case = None;
        }
        _ => {}
    }
    let file = matches!(
        cmd,
        EditorCommand::NewScene
            | EditorCommand::OpenScene(_)
            | EditorCommand::SaveScene
            | EditorCommand::SaveSceneAs(_)
    );
    apply_command(engine, interaction, session, cmd);
    if file {
        if let Some(path) = session.document_path.clone() {
            crate::prefs::push_recent(recent, path);
        }
    }
}

/// Resolve a pressed key through the keymap capture / shortcut table.
pub(crate) fn dispatch_key(
    engine: &mut Engine,
    editor: &mut Editor,
    interaction: &mut EditorInteractionState,
    session: &mut EditorSession,
    recent: &mut Vec<std::path::PathBuf>,
    active_case: &mut Option<ActiveCase>,
    code: KeyCode,
) {
    let ctrl = engine.input.ctrl_pressed;
    let shift = engine.input.shift_pressed;
    let alt = engine.input.alt_pressed;
    if editor.chrome().keymap_capture.is_some() {
        let action = editor.chrome_mut().keymap_capture.take().expect("capture");
        if code != KeyCode::Escape {
            apply_command(
                engine,
                interaction,
                session,
                EditorCommand::BindKey {
                    action,
                    chord: KeyChord {
                        ctrl,
                        shift,
                        alt,
                        code,
                    },
                },
            );
            save_prefs(session, editor, engine, recent);
        }
        return;
    }
    let Some(action) = session.keymap.resolve(code, ctrl, shift, alt) else {
        return;
    };
    match action {
        KeyAction::OpenScene => {
            if let Some(path) =
                rc3d_editor::pick_open_scene(session.file_dialog_dir.as_deref())
            {
                apply_editor_cmd(
                    engine,
                    interaction,
                    session,
                    recent,
                    active_case,
                    EditorCommand::OpenScene(path),
                );
            }
        }
        KeyAction::SaveScene => {
            if session.document_path.is_some() {
                apply_editor_cmd(
                    engine,
                    interaction,
                    session,
                    recent,
                    active_case,
                    EditorCommand::SaveScene,
                );
            } else if let Some(path) =
                rc3d_editor::pick_save_scene(session.file_dialog_dir.as_deref())
            {
                apply_editor_cmd(
                    engine,
                    interaction,
                    session,
                    recent,
                    active_case,
                    EditorCommand::SaveSceneAs(path),
                );
            }
        }
        KeyAction::SaveSceneAs => {
            if let Some(path) =
                rc3d_editor::pick_save_scene(session.file_dialog_dir.as_deref())
            {
                apply_editor_cmd(
                    engine,
                    interaction,
                    session,
                    recent,
                    active_case,
                    EditorCommand::SaveSceneAs(path),
                );
            }
        }
        KeyAction::FlyWalk => {
            let on = !engine.controller.walk_mode;
            apply_command(engine, interaction, session, EditorCommand::SetWalkMode(on));
        }
        KeyAction::Cancel => {
            let busy = !interaction.markup.click_points.is_empty()
                || (interaction.measurement_mode && !interaction.measure.points.is_empty());
            apply_command(engine, interaction, session, EditorCommand::CancelTool);
            if !busy {
                engine.world.graph.clear_selection();
                sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
            }
        }
        other => {
            if let Some(cmd) = keymap::command_for(other) {
                apply_editor_cmd(engine, interaction, session, recent, active_case, cmd);
            }
        }
    }
}

pub(crate) fn save_prefs(
    session: &EditorSession,
    editor: &Editor,
    engine: &Engine,
    recent: &[std::path::PathBuf],
) {
    let layout = engine
        .renderer
        .as_ref()
        .expect("renderer not initialized")
        .viewport_layout();
    crate::prefs::save(&crate::prefs::UiPrefs {
        theme: session.ui_theme,
        locale: session.ui_locale,
        keymap: session.keymap.clone(),
        side_tab: editor.chrome().side_tab,
        bottom_tab: editor.chrome().bottom_tab,
        workspace: editor.chrome().workspace,
        tool_strip_pos: editor.chrome().tool_strip_pos,
        side_dock_width: editor.chrome().side_dock_width,
        bottom_dock_height: editor.chrome().bottom_dock_height,
        inspector_ratio: editor.chrome().inspector_ratio,
        viewport_layout_mode: engine
            .renderer
            .as_ref()
            .expect("renderer not initialized")
            .viewport_layout()
            .layout_mode,
        viewport_h_split: layout.quad_h_split,
        viewport_v_split: layout.quad_v_split,
        last_document: session.document_path.clone(),
        last_folder: session.file_dialog_dir.clone(),
        recent: recent.to_vec(),
    });
}

pub(crate) fn maybe_prefs(
    session: &EditorSession,
    editor: &Editor,
    engine: &Engine,
    recent: &[std::path::PathBuf],
    last: &mut std::time::Instant,
) {
    if last.elapsed() < std::time::Duration::from_secs(2) {
        return;
    }
    save_prefs(session, editor, engine, recent);
    *last = std::time::Instant::now();
}

pub(crate) fn maybe_autosave(
    engine: &Engine,
    session: &EditorSession,
    last: &mut std::time::Instant,
) {
    if !session.dirty {
        return;
    }
    let Some(path) = session.document_path.as_ref() else {
        return;
    };
    if last.elapsed() < std::time::Duration::from_secs(30) {
        return;
    }
    crate::prefs::write_autosave(&engine.world.graph, path);
    *last = std::time::Instant::now();
}
