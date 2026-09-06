//! Status bar, unsaved-changes dialog and the viewport context menu.

use rc3d_scene::SceneGraph;

use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::theme::{chrome_panel_frame, ThemePalette};
use crate::ui::types::{CanvasTool, EditorChromeState, EditorDisplayMode, EditorUiContext};

pub(super) fn flush_panel(panel: egui::Panel, pal: &ThemePalette, inner: egui::Margin) -> egui::Panel {
    panel
        .show_separator_line(false)
        .frame(chrome_panel_frame(pal, inner))
}

/// Bottom status bar: document, perf/mode labels, tool hint, measure/markup state.
pub(super) fn draw_status_bar(ui: &mut egui::Ui, ui_ctx: &EditorUiContext) {
    let loc = ui_ctx.ui_locale;
    ui.horizontal(|ui| {
        let doc = crate::document::document_display_name(ui_ctx.document_path.as_deref());
        let dirty = if ui_ctx.document_dirty { "*" } else { "" };
        ui.label(format!("{doc}{dirty}"));
        ui.separator();
        // (label key, value) pairs rendered as `key: value` columns.
        let pairs: Vec<(&str, String)> = vec![
            ("status.fps", format!("{:.1}", ui_ctx.smoothed_fps)),
            ("status.frame", format!("{:.2} ms", ui_ctx.frame_time_ms)),
            ("status.mode", ui_ctx.display_mode_label.clone()),
            ("status.ibl", ui_ctx.ibl_label.clone()),
            ("status.layout", ui_ctx.layout_mode_label.clone()),
            ("status.vp", ui_ctx.active_viewport_label.clone()),
            ("status.sel", ui_ctx.selected_count.to_string()),
            (
                "status.aq",
                format!(
                    "{} ({})",
                    super::menus::aq_mode_label(loc, ui_ctx.adaptive_quality_mode),
                    ui_ctx.adaptive_quality_name
                ),
            ),
            (
                "status.tier",
                super::menus::cad_tier_label(loc, ui_ctx.cad_display_tier).to_string(),
            ),
        ];
        ui.label(
            pairs
                .iter()
                .map(|(k, v)| format!("{}: {v}", t(loc, k)))
                .collect::<Vec<_>>()
                .join("  "),
        );
        ui.separator();
        ui.label(tool_hint(loc, ui_ctx));
        match ui_ctx.canvas_tool {
            CanvasTool::Measure => {
                ui.separator();
                let needed = ui_ctx.measure_needed.max(1);
                let text = if ui_ctx.measure_label.is_empty() {
                    format!(
                        "{} {}/{}",
                        t(loc, "status.measure"),
                        ui_ctx.measure_points,
                        needed
                    )
                } else {
                    format!(
                        "{} {}  {}/{}",
                        t(loc, "status.measure"),
                        ui_ctx.measure_label,
                        ui_ctx.measure_points,
                        needed
                    )
                };
                ui.label(text);
            }
            CanvasTool::Markup => {
                ui.separator();
                ui.label(t(loc, "status.markup"));
            }
            _ => {}
        }
    });
}

/// Modal "unsaved changes" prompt shown on caption close with dirty state.
pub(super) fn draw_close_prompt(
    ctx: &egui::Context,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    egui::Window::new(t(loc, "dialog.unsaved"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0_f32, 0.0_f32))
        .show(ctx, |ui| {
            ui.label(t(loc, "dialog.unsaved_body"));
            ui.add_space(8.0_f32);
            ui.horizontal(|ui| {
                if ui.button(t(loc, "dialog.save")).clicked() {
                    if ui_ctx.document_path.is_some() {
                        push(EditorCommand::SaveScene);
                        chrome.caption.close_after_save = true;
                        chrome.caption.close_prompt = false;
                    } else if let Some(path) =
                        crate::document::pick_save_scene(ui_ctx.file_dialog_dir.as_deref())
                    {
                        push(EditorCommand::SaveSceneAs(path));
                        chrome.caption.close_after_save = true;
                        chrome.caption.close_prompt = false;
                    }
                }
                if ui.button(t(loc, "dialog.dont_save")).clicked() {
                    chrome.caption.close_prompt = false;
                    chrome.caption.action = Some(crate::ui::types::CaptionAction::Close);
                }
                if ui.button(t(loc, "dialog.cancel")).clicked() {
                    chrome.caption.close_prompt = false;
                }
            });
        });
}

/// Right-click menu inside the 3D viewport. Any action click clears
/// `context_menu_pos`; clicking elsewhere also dismisses it.
pub(super) fn draw_scene_context_menu(
    ctx: &egui::Context,
    _graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    context_menu_pos: &mut Option<egui::Pos2>,
    _chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let pos = match *context_menu_pos {
        Some(p) => p,
        None => return,
    };
    let loc = ui_ctx.ui_locale;
    let pal = ui_ctx.ui_theme.palette();
    egui::Window::new("##context_menu")
        .fixed_pos(pos)
        .resizable(false)
        .title_bar(false)
        .auto_sized()
        .show(ctx, |ui| {
            ui.set_min_width(180.0);
            ui.menu_button(t(loc, "menu.add"), |ui| {
                super::menus::create_node_menu(ui, None, loc, &mut |c| push(c));
            });
            ui.menu_button(t(loc, "menu.view"), |ui| {
                use rc3d_engine_api::camera::ViewPreset;
                let presets = [
                    (ViewPreset::Top, "view.top"),
                    (ViewPreset::Front, "view.front"),
                    (ViewPreset::Right, "view.right"),
                    (ViewPreset::Iso, "view.iso"),
                ];
                for (preset, key) in presets {
                    if ui.button(t(loc, key)).clicked() {
                        push(EditorCommand::SetViewPreset(preset));
                        *context_menu_pos = None;
                    }
                }
            });
            if super::icons::icon_menu_button(
                ui,
                super::icons::Icon::Grid,
                t(loc, "ctx.grid"),
                &pal,
            )
            .clicked()
            {
                push(EditorCommand::SetGridEnabled(!ui_ctx.grid_enabled));
                *context_menu_pos = None;
            }
            if super::icons::icon_menu_button(
                ui,
                super::icons::Icon::Shape,
                t(loc, "ctx.wire"),
                &pal,
            )
            .clicked()
            {
                push(EditorCommand::SetDisplayMode(EditorDisplayMode::Wireframe));
                *context_menu_pos = None;
            }
            ui.separator();
            if super::menus::selection_context_actions(ui, loc, &pal, &mut |c| push(c)) {
                *context_menu_pos = None;
            }
            ui.menu_button(t(loc, "menu.display"), |ui| {
                for (mode, key) in [
                    (EditorDisplayMode::Wireframe, "shade.wireframe"),
                    (EditorDisplayMode::Shaded, "shade.shaded"),
                    (EditorDisplayMode::ShadedWithEdges, "shade.shaded_edges"),
                    (EditorDisplayMode::HiddenLine, "shade.hidden"),
                    (EditorDisplayMode::Flat, "shade.flat"),
                ] {
                    if ui.button(t(loc, key)).clicked() {
                        push(EditorCommand::SetDisplayMode(mode));
                        *context_menu_pos = None;
                    }
                }
            });
        });

    if ctx.input(|i| {
        i.pointer.button_clicked(egui::PointerButton::Primary)
            || i.pointer.button_clicked(egui::PointerButton::Secondary)
    }) && context_menu_pos.is_some()
    {
        *context_menu_pos = None;
    }
}

fn tool_hint(loc: crate::ui::i18n::UiLocale, ui_ctx: &EditorUiContext) -> String {
    let tool = match ui_ctx.canvas_tool {
        CanvasTool::Select => match ui_ctx.select_kind {
            crate::ui::types::SelectKind::Pick => t(loc, "tools.select"),
            crate::ui::types::SelectKind::Box => t(loc, "tools.box_select"),
            crate::ui::types::SelectKind::Lasso => t(loc, "tools.lasso"),
        },
        CanvasTool::Transform => t(loc, "tools.gizmo"),
        CanvasTool::Measure => t(loc, "status.measure"),
        CanvasTool::Section => t(loc, "tools.section"),
        CanvasTool::Markup => t(loc, "status.markup"),
        CanvasTool::Walk => t(loc, "tools.walk"),
    };
    format!("{}  {}", t(loc, "status.tool"), tool)
}
